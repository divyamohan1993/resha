from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form, Request
from fastapi.responses import StreamingResponse
from ..core.limiter import limiter
from ..schemas.resume import AnalysisRequest, AnalysisResponse, AIAnalysisResult, FeedbackRequest
from ..schemas.shortlist import ShortlistRequest, ShortlistResponse, ReasoningLoopResponse
from ..services.ai_engine import ai_service
from ..services.llm_agent import llm_agent
from ..services.onboard_ai import onboard_ai
from ..services.local_llm import local_llm
from ..services.hybrid_agent import hybrid_agent
from ..core.config import get_settings, Settings
from ..services.parser import parser_service
from ..db.audit import AuditLogger
from ..core.logging import get_logger
import json
import asyncio
from typing import AsyncGenerator, Any

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)
audit_logger = AuditLogger()



def _safe_float_score(value: Any) -> float:
    """Safely convert a score to float, removing % signs and handling strings."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.replace("%", "").strip()
        try:
            return float(clean)
        except ValueError:
            return 0.0
    return 0.0


def calculate_consensus_score(
    onboard_score: float,
    local_llm_score: float = None,
    gemini_score: float = None,
    local_llm_available: bool = False,
    gemini_available: bool = False
) -> tuple[float, str]:
    """
    Calculate an OPTIMISTIC consensus score with anti-hallucination safeguards.
    
    === BALANCED SCORING APPROACH ===
    
    This scoring system is designed to be OPTIMISTIC for qualified candidates while
    still preventing wild hallucinations. Key principles:
    
    1. DETERMINISTIC ANCHOR (Onboard AI):
       - TF-IDF based scoring provides a baseline (35% weight)
       - Serves as a reference point but doesn't dominate
       - Scientific basis: Information Retrieval theory (Salton, 1975)
    
    2. LLM DEVIATION BOUNDING (±40%):
       - LLM scores are bounded to deviate max ±40% from onboard anchor
       - Allows high-quality candidates to score well even with low TF-IDF match
       - Prevents extreme hallucinations while being more generous
    
    3. LLM AGREEMENT BONUS:
       - When both LLMs agree on high scores (>75%), gives a bonus
       - Rewards confidence from multiple AI models
       - Scientific basis: Ensemble voting agreement
    
    4. CROSS-VALIDATION (when 2 LLMs available):
       - If Local LLM and Gemini disagree by >20%, reduce confidence
       - Agreement between independent LLMs increases trust
       - Scientific basis: Inter-rater reliability (Cohen's Kappa concept)
    
    5. CONFIDENCE-WEIGHTED AVERAGING:
       - Onboard AI: 35% weight (reference anchor)
       - LLMs combined: 65% weight (contextual understanding)
       - Gives LLMs more influence for nuanced matching
    
    6. SOFTENED CONSERVATIVE BIAS:
       - Only applies when LLMs are >20% more positive than onboard
       - Mild correction (15% of excess) instead of harsh penalty
       - Prevents wild over-optimism while allowing fair scoring
    
    Args:
        onboard_score: Score from Onboard AI (0-100) - Reference anchor
        local_llm_score: Score from Local LLM (0-100), None if unavailable
        gemini_score: Score from Gemini (0-100), None if unavailable
        local_llm_available: Whether Local LLM provided valid result
        gemini_available: Whether Gemini provided valid result
    
    Returns:
        tuple: (final_score as decimal 0-1, model_info string, evidence dict)
    """
    
    # === MECHANISM 1: Deterministic Anchor ===
    # Onboard AI is our immutable ground truth (but with less weight to allow LLM influence)
    onboard_anchor = max(0.0, min(100.0, _safe_float_score(onboard_score)))
    
    # === MECHANISM 2: LLM Deviation Bounding ===
    # Scientific basis: Reject outliers beyond acceptable deviation
    # OPTIMISTIC ADJUSTMENT: Increased from ±25% to ±40% to let high-quality candidates shine
    MAX_DEVIATION = 40.0  # LLMs can deviate max ±40% from anchor
    
    def bound_llm_to_anchor(llm_score: float, anchor: float, source: str) -> float:
        """Bound LLM score to prevent hallucination, but allow more room for qualified candidates."""
        if llm_score is None:
            return None
        
        raw_score = max(0.0, min(100.0, _safe_float_score(llm_score)))
        lower_bound = max(0.0, anchor - MAX_DEVIATION)
        upper_bound = min(100.0, anchor + MAX_DEVIATION)
        
        bounded = max(lower_bound, min(upper_bound, raw_score))
        
        deviation = abs(raw_score - bounded)
        if deviation > 5:
            logger.warning(f"[ANTI-HALLUCINATION] {source} bounded: {raw_score:.1f}% -> {bounded:.1f}% (anchor={anchor:.1f}%, deviation={deviation:.1f}%)")
        
        return bounded
    
    local_bounded = None
    if local_llm_available and local_llm_score is not None:
        local_bounded = bound_llm_to_anchor(local_llm_score, onboard_anchor, "LocalLLM")
    
    gemini_bounded = None
    if gemini_available and gemini_score is not None:
        gemini_bounded = bound_llm_to_anchor(gemini_score, onboard_anchor, "Gemini")
    
    # === MECHANISM 3: Cross-Validation (Inter-LLM Agreement) ===
    llm_agreement_penalty = 0.0
    llm_agreement_bonus = 0.0  # NEW: Bonus when LLMs agree on high scores
    if local_bounded is not None and gemini_bounded is not None:
        llm_disagreement = abs(local_bounded - gemini_bounded)
        if llm_disagreement > 20:
            # LLMs significantly disagree - reduce confidence in both
            llm_agreement_penalty = (llm_disagreement - 20) * 0.3  # Penalty scales with disagreement
            logger.info(f"[CROSS-VALIDATION] LLM disagreement={llm_disagreement:.1f}%, penalty={llm_agreement_penalty:.1f}%")
        elif llm_disagreement <= 10 and local_bounded >= 75 and gemini_bounded >= 75:
            # OPTIMISTIC: Both LLMs strongly agree on high score - give bonus
            llm_agreement_bonus = min(8.0, (local_bounded + gemini_bounded) / 2 - 75) * 0.5
            logger.info(f"[OPTIMISTIC] LLM agreement bonus: +{llm_agreement_bonus:.1f}% (both LLMs gave high scores)")
    
    # === MECHANISM 4: Confidence-Weighted Averaging ===
    # OPTIMISTIC ADJUSTMENT: Reduced onboard weight from 50% to 35%
    # This gives LLMs (which can understand context better) more influence
    available_scores = [onboard_anchor]
    weights = [0.35]  # Onboard gets 35% (down from 50%, giving LLMs more weight)
    
    if local_bounded is not None and gemini_bounded is not None:
        # Both LLMs available - they share the remaining 65%
        llm_avg = (local_bounded + gemini_bounded) / 2.0
        available_scores.append(llm_avg)
        weights.append(0.65)
        model_info = "3-Tier Optimistic (Onboard + Local + Gemini)"
    elif local_bounded is not None:
        available_scores.append(local_bounded)
        weights.append(0.65)
        model_info = "2-Tier Optimistic (Onboard + Local LLM)"
    elif gemini_bounded is not None:
        available_scores.append(gemini_bounded)
        weights.append(0.65)
        model_info = "2-Tier Optimistic (Onboard + Gemini)"
    else:
        # Only onboard - fully deterministic, but apply a small boost for isolated mode
        weights = [1.0]
        model_info = "Deterministic (Onboard Only)"
    
    # Calculate weighted average
    consensus_pct = sum(s * w for s, w in zip(available_scores, weights))
    
    # Track all adjustments for evidence
    adjustments = []
    
    # Apply LLM agreement bonus (if applicable)
    if llm_agreement_bonus > 0:
        consensus_pct += llm_agreement_bonus
        adjustments.append({
            "type": "LLM_AGREEMENT_BONUS",
            "amount": llm_agreement_bonus,
            "reason": f"Both LLMs agreed on high scores (Local: {local_bounded:.1f}%, Gemini: {gemini_bounded:.1f}%)"
        })
    
    # Apply LLM disagreement penalty
    if llm_agreement_penalty > 0:
        consensus_pct -= llm_agreement_penalty
        adjustments.append({
            "type": "LLM_DISAGREEMENT_PENALTY",
            "amount": -llm_agreement_penalty,
            "reason": f"Local LLM and Gemini disagreed by {abs((local_bounded or 0) - (gemini_bounded or 0)):.1f}%"
        })
    
    # === MECHANISM 5: Conservative Bias (SOFTENED) ===
    # OPTIMISTIC ADJUSTMENT: Only apply when gap is very large (>20% instead of >12%)
    # And reduce the penalty multiplier from 0.25 to 0.15
    conservative_penalty = 0
    if consensus_pct > onboard_anchor + 20:
        excess = consensus_pct - onboard_anchor - 20  # Only penalize the excess beyond 20%
        conservative_penalty = excess * 0.15  # Reduced from 0.25
        consensus_pct -= conservative_penalty
        adjustments.append({
            "type": "CONSERVATIVE_BIAS",
            "amount": -conservative_penalty,
            "reason": f"LLMs were significantly more positive than anchor - applying mild correction"
        })
        logger.info(f"[CONSERVATIVE BIAS] Mild reduction of {conservative_penalty:.1f}%")
    
    # === MECHANISM 6: Statistical Bounds Check ===
    all_scores = [s for s in [onboard_anchor, local_bounded, gemini_bounded] if s is not None]
    statistical_adjustment = 0
    if len(all_scores) > 1:
        mean_score = sum(all_scores) / len(all_scores)
        if abs(consensus_pct - mean_score) > 15:
            old_consensus = consensus_pct
            consensus_pct = (consensus_pct + mean_score) / 2.0
            statistical_adjustment = consensus_pct - old_consensus
            adjustments.append({
                "type": "STATISTICAL_CORRECTION",
                "amount": statistical_adjustment,
                "reason": f"Consensus was {abs(old_consensus - mean_score):.1f}% away from tier mean ({mean_score:.1f}%)"
            })
            logger.warning(f"[STATISTICAL CHECK] Adjusted consensus from {old_consensus:.1f}% to {consensus_pct:.1f}%")
    
    # === FINAL: Strict 0-100 Clamp ===
    consensus_pct = max(0.0, min(100.0, consensus_pct))
    final_score = consensus_pct / 100.0
    
    # === BUILD EVIDENCE/PROOF DICTIONARY ===
    evidence = {
        "calculation_method": "Optimistic Multi-Tier Consensus",
        "scoring_approach": [
            "Deterministic TF-IDF Anchor (35% weight)",
            "LLM Contextual Analysis (65% weight)",
            "LLM Deviation Bounding (±40%)",
            "LLM Agreement Bonus (when both agree on high scores)",
            "Cross-Validation Between LLMs",
            "Softened Conservative Correction"
        ],
        "tier_scores": {
            "onboard_ai": {
                "raw_score": onboard_anchor,
                "weight": weights[0] if len(weights) > 0 else 1.0,
                "contribution": onboard_anchor * (weights[0] if len(weights) > 0 else 1.0),
                "source": "Deterministic TF-IDF + Pattern Matching",
                "note": "Reference anchor (35% weight) - provides baseline for LLM scoring"
            }
        },
        "weighted_calculation": {
            "formula": " + ".join([f"({s:.1f} × {w:.2f})" for s, w in zip(available_scores, weights)]),
            "raw_weighted_sum": sum(s * w for s, w in zip(available_scores, weights)),
            "final_after_adjustments": consensus_pct
        },
        "adjustments_applied": adjustments,
        "final_score_proof": {
            "raw_consensus": sum(s * w for s, w in zip(available_scores, weights)),
            "total_adjustments": sum(a["amount"] for a in adjustments),
            "final_percentage": round(consensus_pct, 2),
            "final_decimal": round(final_score, 4),
            "decision_threshold": 70.0,
            "decision": "SHORTLIST" if consensus_pct >= 70 else "REJECT"
        }
    }
    
    # Add LLM tier scores if available
    if local_bounded is not None:
        raw_local = max(0.0, min(100.0, _safe_float_score(local_llm_score))) if local_llm_score else 0
        evidence["tier_scores"]["local_llm"] = {
            "raw_score": raw_local,
            "bounded_score": local_bounded,
            "was_grounded": abs(raw_local - local_bounded) > 1,
            "grounding_amount": raw_local - local_bounded if abs(raw_local - local_bounded) > 1 else 0,
            "source": "Ollama Local LLM (CPU inference)",
            "note": f"Bounded to ±40% of anchor ({onboard_anchor:.1f}%)"
        }
    
    if gemini_bounded is not None:
        raw_gemini = max(0.0, min(100.0, _safe_float_score(gemini_score))) if gemini_score else 0
        evidence["tier_scores"]["gemini"] = {
            "raw_score": raw_gemini,
            "bounded_score": gemini_bounded,
            "was_grounded": abs(raw_gemini - gemini_bounded) > 1,
            "grounding_amount": raw_gemini - gemini_bounded if abs(raw_gemini - gemini_bounded) > 1 else 0,
            "source": "Google Gemini Cloud LLM",
            "note": f"Bounded to ±40% of anchor ({onboard_anchor:.1f}%)"
        }
    
    # Logging for transparency
    logger.info(f"[OPTIMISTIC CONSENSUS] Onboard={onboard_anchor:.1f}%, Local={local_bounded or 'N/A'}, Gemini={gemini_bounded or 'N/A'} -> Final={consensus_pct:.1f}%")
    
    return final_score, model_info, evidence

@router.get("/health/live")
async def health_live():
    return {"status": "ok", "version": settings.VERSION}

@router.get("/health/ready")
async def health_ready():
    # In a real app, check DB connection here
    return {"status": "ready"}

@router.post("/analyze", response_model=AnalysisResponse)
@limiter.limit("5/minute")
async def analyze_resume(request: Request, request_data: AnalysisRequest):
    """
    Analyze a resume text against a job description.
    
    Uses the same 3-tier consensus mechanism as development mode:
    1. Tier 1: Onboard AI (TF-IDF + Pattern Matching) - Fast, local
    2. Tier 2: Local LLM (Ollama Gemma3) - CPU-based inference
    3. Tier 3: Cloud LLM (Gemini) - Advanced reasoning, fallback
    
    The final result uses weighted consensus from all available tiers:
    - Onboard AI: 20% weight (fast preliminary analysis)
    - Local LLM: 40% weight (if available, real LLM reasoning)
    - Cloud LLM: 40% weight (advanced cloud-based analysis)
    
    If any tier fails, weights are redistributed to available tiers.
    
    Args:
        request: FastAPI Request object (used for rate limiting).
        request_data: JSON payload containing resume_text and job_description.
        
    Returns:
        AnalysisResponse: Score, decision, and detailed AI analysis.
        
    Raises:
        HTTPException: If analysis fails.
    """
    logger.info("Received analysis request (using 3-tier consensus mechanism)")
    try:
        # Tier 1: Run Onboard AI analysis (TF-IDF + Pattern Matching) - Always available
        onboard_result = onboard_ai.analyze(request_data.resume_text, request_data.job_description)
        preliminary = onboard_result.get("preliminary_analysis", {})
        onboard_score = _safe_float_score(preliminary.get("score", 0))  # 0-100 scale
        
        # Tier 2: Run Local LLM analysis (Ollama - Gemma3:1b or 4b)
        local_llm_score = None
        local_llm_decision = None
        local_llm_available = False
        try:
            local_result = await local_llm.evaluate(request_data.resume_text, request_data.job_description)
            if local_result.get("decision") != "Error":
                local_llm_available = True
                local_llm_score = _safe_float_score(local_result.get("match_percentage", 0))
                local_llm_decision = local_result.get("decision")
                logger.info(f"Local LLM (Ollama) result: {local_llm_decision} ({local_llm_score}%)")
        except Exception as e:
            logger.warning(f"Local LLM evaluation failed: {e}")
        
        # Tier 3: Run Gemini LLM for advanced analysis
        gemini_result = llm_agent.evaluate(request_data.resume_text, request_data.job_description)
        gemini_available = gemini_result.get("decision") != "Error"
        gemini_score = _safe_float_score(gemini_result.get("match_percentage", 0)) if gemini_available else None
        gemini_decision = gemini_result.get("decision", "Error")
        
        # Calculate consensus score using strict algorithm
        final_score, model_info, score_evidence = calculate_consensus_score(
            onboard_score=onboard_score,
            local_llm_score=local_llm_score,
            gemini_score=gemini_score,
            local_llm_available=local_llm_available,
            gemini_available=gemini_available
        )
        
        # For display purposes, clamp individual scores
        onboard_score = max(0, min(100, onboard_score))
        local_llm_score = max(0, min(100, local_llm_score)) if local_llm_score else 0
        gemini_score = max(0, min(100, gemini_score)) if gemini_score else 0
        
        # Determine final decision based on consensus score (70% threshold)
        decision = "SHORTLIST" if final_score >= 0.70 else "REJECT"
        
        # Get matched/missing requirements and candidate profile - prefer Gemini's analysis, fallback to local LLM
        # Candidate profile extraction (LLM-determined for dynamic UI updates)
        candidate_profile = {}
        if gemini_available:
            matched_kw = gemini_result.get("matched_requirements", [])
            missing_kw = gemini_result.get("missing_requirements", [])
            summary = gemini_result.get("summary", "")
            candidate_profile = gemini_result.get("candidate_profile", {})
        elif local_llm_available:
            matched_kw = local_result.get("matched_requirements", [])
            missing_kw = local_result.get("missing_requirements", [])
            summary = local_result.get("summary", "")
            candidate_profile = local_result.get("candidate_profile", {})
        else:
            matched_kw = preliminary.get("skills", {}).get("matched", [])
            missing_kw = preliminary.get("skills", {}).get("missing", [])
            summary = f"Analysis based on onboard AI scoring ({onboard_score:.1f}%)"
        
        # Get candidate info - prefer LLM-determined values, fallback to onboard pattern matching
        semantic_score = preliminary.get("semantic_similarity", 0)
        
        # LLM-determined candidate type and experience (dynamic based on resume)
        if candidate_profile:
            candidate_type = candidate_profile.get("candidate_type", "Unknown")
            years_exp = candidate_profile.get("years_experience", 0)
            # Ensure years_exp is a number
            try:
                years_exp = int(years_exp) if years_exp else 0
            except (ValueError, TypeError):
                years_exp = 0
        else:
            # Fallback to onboard pattern matching
            candidate_type_raw = preliminary.get("experience", {}).get("level", "Unknown")
            years_exp = preliminary.get("experience", {}).get("years", 0)
            # Normalize candidate type
            if candidate_type_raw in ["junior", "entry"]:
                candidate_type = "Fresher"
            elif candidate_type_raw in ["mid", "senior", "lead"]:
                candidate_type = "Experienced"
            else:
                candidate_type = candidate_type_raw.title() if candidate_type_raw else "Unknown"
        
        # LLM-determined entities (organizations and locations from resume)
        organizations = candidate_profile.get("organizations", []) if candidate_profile else []
        locations = candidate_profile.get("locations", []) if candidate_profile else []
        entities = {
            "ORG": organizations,
            "GPE": locations
        }
        
        # Audit
        resume_hash = ai_service.hash_text(request_data.resume_text)
        audit_logger.log_decision(resume_hash, final_score, decision, filename="Manual Input")
        
        # Build detailed response with all tier scores
        details = {
            "onboard_score": round(onboard_score, 2),
            "local_llm_score": round(local_llm_score, 2) if local_llm_score else "N/A",
            "local_llm_model": local_llm.model_name if local_llm_available else "N/A",
            "gemini_score": round(gemini_score, 2) if gemini_available else "N/A",
            "gemini_decision": gemini_decision,
            "consensus_score": round(final_score * 100, 2),
            "semantic_similarity": round(semantic_score, 4),
            "tiers_used": sum([1, local_llm_available, gemini_available]),
            "analysis_method": model_info
        }
        
        return AnalysisResponse(
            decision=decision,
            score=round(final_score, 4),
            ai_analysis=AIAnalysisResult(
                cosine_similarity_score=semantic_score,
                matched_keywords=matched_kw,
                missing_keywords=missing_kw,
                candidate_type=candidate_type,
                years_experience=years_exp,
                entities=entities,  # LLM-determined organizations and locations
                contact_info={},
                details=details,
                verification={}
            ),
            meta={
                "threshold_used": 0.70,  # 70% threshold as per Task B
                "model": model_info,
                "gemini_summary": summary
            },
            reasoning_loop={
                "agent_type": "multi_tier_consensus",
                "tiers_executed": {
                    "tier_1_onboard_ai": {
                        "status": "complete",
                        "score": round(onboard_score, 2),
                        "source": "TF-IDF + Pattern Matching"
                    },
                    "tier_2_local_llm": {
                        "status": "complete" if local_llm_available else "skipped",
                        "score": round(local_llm_score, 2) if local_llm_score else None,
                        "model": local_llm.model_name if local_llm_available else None,
                        "decision": local_llm_decision if local_llm_available else None
                    },
                    "tier_3_gemini": {
                        "status": "complete" if gemini_available else "error",
                        "score": round(gemini_score, 2) if gemini_available else None,
                        "decision": gemini_decision
                    }
                },
                "consensus_calculation": score_evidence,
                "final_decision": {
                    "decision": decision,
                    "score_percentage": round(final_score * 100, 2),
                    "threshold": 70,
                    "reasoning": summary
                },
                "candidate_profile": candidate_profile,
                "matched_requirements": matched_kw,
                "missing_requirements": missing_kw
            }
        )
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Analysis processing failed")

@router.post("/analyze-file", response_model=AnalysisResponse)
@limiter.limit("5/minute")
async def analyze_resume_file(
    request: Request,
    file: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Analyze a resume file (PDF, DOCX, TXT) against a job description.
    
    Args:
        request: FastAPI Request object.
        file: The resume file to upload (max 5MB).
        job_description: The job description text.
        
    Returns:
        AnalysisResponse: Score, decision, and detailed AI analysis.
    """
    logger.info(f"Received file analysis request: {file.filename}")
    
    # Validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is missing")
    
    allowed_types = ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "text/plain"]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"Invalid file type. Allowed: PDF, DOCX, TXT. Got: {file.content_type}")
    
    # Check size
    await file.seek(0, 2)
    size = file.tell()
    await file.seek(0)
    
    if size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE/1024/1024}MB")

    try:
        # Extract Text
        resume_text = await parser_service.extract_text(file)
        
        # Use the same 3-tier consensus mechanism as /api/analyze
        # Tier 1: Onboard AI analysis
        onboard_result = onboard_ai.analyze(resume_text, job_description)
        preliminary = onboard_result.get("preliminary_analysis", {})
        onboard_score = _safe_float_score(preliminary.get("score", 0))
        
        # Tier 2: Local LLM analysis (Ollama - Gemma3)
        local_llm_score = None
        local_llm_available = False
        try:
            local_result = await local_llm.evaluate(resume_text, job_description)
            if local_result.get("decision") != "Error":
                local_llm_available = True
                local_llm_score = _safe_float_score(local_result.get("match_percentage", 0))
                logger.info(f"Local LLM result: {local_result.get('decision')} ({local_llm_score}%)")
        except Exception as e:
            logger.warning(f"Local LLM evaluation failed: {e}")
        
        # Tier 3: Gemini LLM analysis
        gemini_result = llm_agent.evaluate(resume_text, job_description)
        gemini_available = gemini_result.get("decision") != "Error"
        gemini_score = _safe_float_score(gemini_result.get("match_percentage", 0)) if gemini_available else None
        
        # Calculate consensus score using strict algorithm
        final_score, model_info, score_evidence = calculate_consensus_score(
            onboard_score=onboard_score,
            local_llm_score=local_llm_score,
            gemini_score=gemini_score,
            local_llm_available=local_llm_available,
            gemini_available=gemini_available
        )
        
        # For display purposes, clamp individual scores (with fail-safe type conversion)
        onboard_score = max(0, min(100, float(onboard_score) if onboard_score else 0))
        local_llm_score = max(0, min(100, float(local_llm_score))) if local_llm_score else 0
        gemini_score = max(0, min(100, float(gemini_score))) if gemini_score else 0
        
        decision = "SHORTLIST" if final_score >= 0.70 else "REJECT"
        
        # Get matched/missing requirements and candidate profile - prefer Gemini's analysis, fallback to local LLM
        # Candidate profile extraction (LLM-determined for dynamic UI updates)
        candidate_profile = {}
        if gemini_available:
            matched_kw = gemini_result.get("matched_requirements", [])
            missing_kw = gemini_result.get("missing_requirements", [])
            summary = gemini_result.get("summary", "")
            candidate_profile = gemini_result.get("candidate_profile", {})
        elif local_llm_available:
            matched_kw = local_result.get("matched_requirements", [])
            missing_kw = local_result.get("missing_requirements", [])
            summary = local_result.get("summary", "")
            candidate_profile = local_result.get("candidate_profile", {})
        else:
            matched_kw = preliminary.get("skills", {}).get("matched", [])
            missing_kw = preliminary.get("skills", {}).get("missing", [])
            summary = f"Analysis based on onboard AI scoring ({onboard_score:.1f}%)"
        
        # Get candidate info - prefer LLM-determined values, fallback to onboard pattern matching
        semantic_score = preliminary.get("semantic_similarity", 0)
        
        # LLM-determined candidate type and experience (dynamic based on resume)
        if candidate_profile:
            candidate_type = candidate_profile.get("candidate_type", "Unknown")
            years_exp = candidate_profile.get("years_experience", 0)
            # Ensure years_exp is a number
            try:
                years_exp = int(years_exp) if years_exp else 0
            except (ValueError, TypeError):
                years_exp = 0
        else:
            # Fallback to onboard pattern matching
            candidate_type_raw = preliminary.get("experience", {}).get("level", "Unknown")
            years_exp = preliminary.get("experience", {}).get("years", 0)
            # Normalize candidate type
            if candidate_type_raw in ["junior", "entry"]:
                candidate_type = "Fresher"
            elif candidate_type_raw in ["mid", "senior", "lead"]:
                candidate_type = "Experienced"
            else:
                candidate_type = candidate_type_raw.title() if candidate_type_raw else "Unknown"
        
        # LLM-determined entities (organizations and locations from resume)
        organizations = candidate_profile.get("organizations", []) if candidate_profile else []
        locations = candidate_profile.get("locations", []) if candidate_profile else []
        entities = {
            "ORG": organizations,
            "GPE": locations
        }
        
        # Audit
        resume_hash = ai_service.hash_text(resume_text)
        audit_logger.log_decision(resume_hash, final_score, decision, filename=file.filename)
        
        # Build detailed response
        details = {
            "onboard_score": round(onboard_score, 2),
            "local_llm_score": round(local_llm_score, 2) if local_llm_score else "N/A",
            "local_llm_model": local_llm.model_name if local_llm_available else "N/A",
            "gemini_score": round(gemini_score, 2) if gemini_available else "N/A",
            "consensus_score": round(final_score * 100, 2),
            "semantic_similarity": round(semantic_score, 4),
            "tiers_used": sum([1, local_llm_available, gemini_available]),
            "analysis_method": model_info
        }
        
        return AnalysisResponse(
            decision=decision,
            score=round(final_score, 4),
            ai_analysis=AIAnalysisResult(
                cosine_similarity_score=semantic_score,
                matched_keywords=matched_kw,
                missing_keywords=missing_kw,
                candidate_type=candidate_type,
                years_experience=years_exp,
                entities=entities,  # LLM-determined organizations and locations
                contact_info={},
                details=details,
                verification={}
            ),
            meta={
                "threshold_used": 0.70,
                "model": model_info,
                "gemini_summary": summary,
                "filename": file.filename
            },
            reasoning_loop={
                "agent_type": "multi_tier_consensus",
                "tiers_executed": {
                    "tier_1_onboard_ai": {
                        "status": "complete",
                        "score": round(onboard_score, 2),
                        "source": "TF-IDF + Pattern Matching"
                    },
                    "tier_2_local_llm": {
                        "status": "complete" if local_llm_available else "skipped",
                        "score": round(local_llm_score, 2) if local_llm_score else None,
                        "model": local_llm.model_name if local_llm_available else None
                    },
                    "tier_3_gemini": {
                        "status": "complete" if gemini_available else "error",
                        "score": round(gemini_score, 2) if gemini_available else None
                    }
                },
                "consensus_calculation": score_evidence,
                "final_decision": {
                    "decision": decision,
                    "score_percentage": round(final_score * 100, 2),
                    "threshold": 70,
                    "reasoning": summary
                },
                "candidate_profile": candidate_profile,
                "matched_requirements": matched_kw,
                "missing_requirements": missing_kw,
                "source_file": file.filename
            }
        )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"File analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis processing failed: {str(e)}")

@router.get("/history")
async def get_history():
    return audit_logger.get_history()

@router.post("/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Endpoint for Learning Mode. 
    HR submits rating (1-5) and relevant keywords to adjust.
    """
    logger.info(f"Received feedback: Score {feedback.score}")
    try:
        ai_service.learn_feedback(feedback.keywords, feedback.score)
        return {"status": "Feedback processed", "message": "Neural weights updated successfully."}
    except Exception as e:
        logger.error(f"Feedback processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# TASK B: LLM-BASED SHORTLISTING ENDPOINT (Gemini AI)
# =============================================================================

@router.post("/shortlist", response_model=ShortlistResponse)
@limiter.limit("10/minute")
async def shortlist_resume(request: Request, request_data: ShortlistRequest):
    """
    AI-Powered Resume Shortlisting Agent (Task B Implementation)
    
    This endpoint uses Gemini AI with a strict system instruction to:
    1. Take a Resume (plain text) as input
    2. Take a Job Description (JD) as input  
    3. Use an AI model + prompt template to evaluate the match
    4. Return a final decision: Shortlisted (if >= 70% match) or Rejected (if < 70%)
    5. Return a JSON summary explaining why it was shortlisted or rejected
    
    The AI agent follows this strict system instruction:
    "You are a Resume Shortlisting AI. You must analyze skills, experience, projects, 
    education, and role fit. If the resume closely matches 70% or more of the JD 
    requirements, output 'Shortlisted'. Otherwise output 'Rejected'. You must follow 
    the evaluation strictly."
    
    Args:
        request: FastAPI Request object (used for rate limiting).
        request_data: JSON payload containing resume_text and job_description.
        
    Returns:
        ShortlistResponse: Decision, match_percentage, reasoning, and requirements analysis.
        
    Raises:
        HTTPException: If analysis fails.
    """
    logger.info("Received LLM shortlist request")
    try:
        # Call LLM Agent for evaluation
        result = llm_agent.evaluate(request_data.resume_text, request_data.job_description)
        
        # Check for errors
        if result.get("decision") == "Error":
            raise HTTPException(status_code=500, detail=result.get("summary", "LLM evaluation failed"))
        
        # Audit log
        resume_hash = ai_service.hash_text(request_data.resume_text)
        audit_logger.log_decision(
            resume_hash, 
            _safe_float_score(result.get("match_percentage", 0)) / 100,  # Normalize to 0-1 for consistency
            result.get("decision"), 
            filename="LLM Shortlist"
        )
        
        return ShortlistResponse(
            decision=result.get("decision", "Rejected"),
            match_percentage=result.get("match_percentage", 0),
            reasoning=result.get("reasoning", {}),
            matched_requirements=result.get("matched_requirements", []),
            missing_requirements=result.get("missing_requirements", []),
            summary=result.get("summary", ""),
            agent_metadata=result.get("agent_metadata")
        )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"LLM shortlist failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"LLM analysis processing failed: {str(e)}")


@router.post("/shortlist/reasoning-loop", response_model=ReasoningLoopResponse)
@limiter.limit("5/minute")
async def shortlist_reasoning_loop(request: Request, request_data: ShortlistRequest):
    """
    Get the complete reasoning loop of the AI agent.
    
    This endpoint exposes the internal step-by-step reasoning process:
    - Stage 1: Input Processing
    - Stage 2: Prompt Construction  
    - Stage 3: LLM Inference
    - Stage 4: Decision Output
    
    Useful for debugging, auditing, and understanding the agent's decision-making.
    
    Returns:
        ReasoningLoopResponse: Full reasoning loop in JSON format.
    """
    logger.info("Received reasoning loop request")
    try:
        result = llm_agent.get_reasoning_loop_json(
            request_data.resume_text, 
            request_data.job_description
        )
        return result
    except Exception as e:
        logger.error(f"Reasoning loop failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reasoning loop failed: {str(e)}")


# =============================================================================
# DEVELOPMENT MODE ENDPOINTS (Real-time Chain-of-Thought Reasoning)
# =============================================================================

@router.get("/dev/config")
async def get_dev_config():
    """
    Get development mode configuration.
    Returns current development mode status, AI settings, and hybrid agent status.
    """
    # Get hybrid agent status for complete picture
    hybrid_status = await hybrid_agent.get_status()
    
    return {
        "development_mode": settings.DEVELOPMENT_MODE,
        "api_prefix": settings.API_PREFIX,
        "version": settings.VERSION,
        "threshold": settings.THRESHOLD_SCORE,
        "models": {
            "onboard": "TF-IDF + Pattern Matching",
            "local_llm": hybrid_status.get("tiers", {}).get("local_llm", {}),
            "cloud_llm": hybrid_status.get("tiers", {}).get("cloud_llm", {})
        },
        "hybrid_agent": hybrid_status,
        "features": [
            "real_streaming_cot",
            "multi_model_consensus",
            "offline_capable",
            "data_privacy"
        ]
    }


async def _stream_reasoning_steps(resume_text: str, job_description: str) -> AsyncGenerator[str, None]:
    """
    Generator function that streams chain-of-thought reasoning steps.
    Uses the same 3-tier consensus mechanism as the main screen:
    1. Tier 1: Onboard AI (TF-IDF + Pattern Matching)
    2. Tier 2: Local LLM (Ollama Gemma3)
    3. Tier 3: Cloud LLM (Gemini)
    
    Each step is sent as a Server-Sent Event for real-time display.
    """
    
    # Phase 1: Initialize
    yield f"data: {json.dumps({'phase': 'init', 'step': 0, 'name': 'Initializing Analysis', 'action': 'Setting up 3-tier AI analysis pipeline', 'status': 'running', 'progress': 0})}\n\n"
    await asyncio.sleep(0.3)
    
    # Tier scores for consensus
    onboard_score = 0
    local_llm_score = None
    gemini_score = 0
    local_llm_available = False
    gemini_available = False
    local_llm_result = None
    gemini_result = None
    
    # Phase 2: Tier 1 - Onboard AI Analysis (Fast, Local)
    yield f"data: {json.dumps({'phase': 'onboard', 'step': 1, 'name': '🔧 Onboard AI Active', 'action': 'Running lightweight local analysis (TF-IDF + Pattern Matching)', 'status': 'running', 'progress': 5})}\n\n"
    await asyncio.sleep(0.2)
    
    # Run onboard analysis
    preliminary = {}
    try:
        onboard_result = onboard_ai.analyze(resume_text, job_description)
        reasoning_chain = onboard_result.get("reasoning_chain", [])
        preliminary = onboard_result.get("preliminary_analysis", {})
        onboard_score = _safe_float_score(preliminary.get("score", 0))
        
        # Stream each step of the onboard reasoning chain
        total_onboard_steps = len(reasoning_chain)
        for idx, step in enumerate(reasoning_chain):
            progress = 10 + int((idx / max(total_onboard_steps, 1)) * 25)  # 10-35% range
            step_data = {
                'phase': 'onboard',
                'step': step.get('step', idx + 1),
                'name': f"📊 {step.get('name', 'Analysis Step')}",
                'action': step.get('action', ''),
                'details': step.get('details', ''),
                'data': step.get('data'),
                'status': step.get('status', 'complete'),
                'progress': progress
            }
            yield f"data: {json.dumps(step_data)}\n\n"
            await asyncio.sleep(0.25)
        
        # Send preliminary results
        prelim_decision = preliminary.get('decision', 'Unknown')
        onboard_complete_data = {
            'phase': 'onboard_complete',
            'step': 10,
            'name': '✅ Onboard Analysis Complete',
            'action': 'Preliminary scoring finished',
            'details': f"Preliminary Score: {onboard_score:.1f}% - {prelim_decision}",
            'data': preliminary,
            'status': 'complete',
            'progress': 35
        }
        yield f"data: {json.dumps(onboard_complete_data)}\n\n"
        await asyncio.sleep(0.3)
        
    except Exception as e:
        yield f"data: {json.dumps({'phase': 'error', 'step': -1, 'name': '❌ Onboard AI Error', 'action': str(e), 'status': 'error', 'progress': 35})}\n\n"
        preliminary = {}
    
    # Phase 3: Tier 2 - Local LLM Analysis (Ollama Gemma3)
    local_llm_connect_data = {
        'phase': 'local_llm', 'step': 11, 'name': '🖥️ Local LLM Connecting',
        'action': f'Initiating Ollama ({local_llm.model_name}) for CPU-based inference',
        'status': 'running', 'progress': 40
    }
    yield f"data: {json.dumps(local_llm_connect_data)}\n\n"
    await asyncio.sleep(0.3)
    
    try:
        local_llm_result = await local_llm.evaluate(resume_text, job_description)
        if local_llm_result.get("decision") != "Error":
            local_llm_available = True
            local_llm_score = _safe_float_score(local_llm_result.get("match_percentage", 0))
            
            local_llm_complete_data = {
                'phase': 'local_llm', 'step': 12, 'name': '✅ Local LLM Complete',
                'action': f"Decision: {local_llm_result.get('decision')} ({local_llm_score}% match)",
                'details': local_llm_result.get('summary', ''),
                'status': 'complete', 'progress': 50
            }
            yield f"data: {json.dumps(local_llm_complete_data)}\n\n"
        else:
            local_llm_unavail_data = {
                'phase': 'local_llm', 'step': 12, 'name': '⚠️ Local LLM Unavailable',
                'action': 'Ollama not running or no model available',
                'details': 'Proceeding with Gemini only',
                'status': 'complete', 'progress': 50
            }
            yield f"data: {json.dumps(local_llm_unavail_data)}\n\n"
    except Exception as e:
        local_llm_skip_data = {
            'phase': 'local_llm', 'step': 12, 'name': '⚠️ Local LLM Skipped',
            'action': str(e), 'status': 'complete', 'progress': 50
        }
        yield f"data: {json.dumps(local_llm_skip_data)}\n\n"
    await asyncio.sleep(0.3)
    
    # Phase 4: Tier 3 - Gemini AI Analysis (Cloud, Deep Analysis)
    yield f"data: {json.dumps({'phase': 'gemini', 'step': 13, 'name': '🤖 Gemini AI Connecting', 'action': 'Initiating cloud-based advanced LLM analysis', 'status': 'running', 'progress': 55})}\n\n"
    await asyncio.sleep(0.3)
    
    yield f"data: {json.dumps({'phase': 'gemini', 'step': 14, 'name': '📝 Prompt Construction', 'action': 'Building strict evaluation prompt with system instruction', 'details': 'System: You are a Resume Shortlisting AI. You must analyze skills, experience, projects, education, and role fit...', 'status': 'running', 'progress': 60})}\n\n"
    await asyncio.sleep(0.3)
    
    yield f"data: {json.dumps({'phase': 'gemini', 'step': 15, 'name': '🚀 API Request', 'action': 'Sending request to Gemini 3 Flash Preview', 'status': 'running', 'progress': 65})}\n\n"
    await asyncio.sleep(0.2)
    
    # Call Gemini
    try:
        gemini_result = llm_agent.evaluate(resume_text, job_description)
        
        if gemini_result.get("decision") != "Error":
            gemini_available = True
            gemini_score = _safe_float_score(gemini_result.get("match_percentage", 0))
            
            yield f"data: {json.dumps({'phase': 'gemini', 'step': 16, 'name': '📥 Response Received', 'action': 'Parsing Gemini API response', 'status': 'running', 'progress': 75})}\n\n"
            await asyncio.sleep(0.2)
            
            # Extract reasoning details
            reasoning = gemini_result.get("reasoning", {})
            
            # Stream each reasoning component
            reasoning_steps = [
                ("skills_analysis", "🎯 Skills Analysis", 80),
                ("experience_analysis", "📈 Experience Analysis", 84),
                ("education_analysis", "🎓 Education Analysis", 88),
                ("projects_analysis", "💼 Projects Analysis", 92),
                ("role_fit_analysis", "✨ Role Fit Analysis", 96)
            ]
            
            step_num = 17
            for key, name, progress in reasoning_steps:
                if key in reasoning and reasoning[key]:
                    yield f"data: {json.dumps({'phase': 'gemini_reasoning', 'step': step_num, 'name': name, 'action': reasoning[key], 'status': 'complete', 'progress': progress})}\n\n"
                    step_num += 1
                    await asyncio.sleep(0.2)
        else:
            yield f"data: {json.dumps({'phase': 'gemini', 'step': 16, 'name': '⚠️ Gemini Error', 'action': gemini_result.get('summary', 'API Error'), 'status': 'complete', 'progress': 75})}\n\n"
            reasoning = {}
        
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        yield f"data: {json.dumps({'phase': 'gemini', 'step': 16, 'name': '⚠️ Gemini Error', 'action': str(e), 'status': 'complete', 'progress': 75})}\n\n"
        reasoning = {}
    
    # Phase 5: Calculate Consensus Score (same as main screen)
    yield f"data: {json.dumps({'phase': 'consensus', 'step': 25, 'name': '🎯 Calculating Consensus', 'action': 'Combining scores from all available tiers', 'status': 'running', 'progress': 97})}\n\n"
    await asyncio.sleep(0.2)
    
    # Use centralized consensus function for strict 0-100% range
    final_score_decimal, model_info, score_evidence = calculate_consensus_score(
        onboard_score=onboard_score,
        local_llm_score=local_llm_score,
        gemini_score=gemini_score,
        local_llm_available=local_llm_available,
        gemini_available=gemini_available
    )
    
    # Convert to percentage for display
    consensus_score = final_score_decimal * 100  # Now guaranteed to be 0-100
    
    # Clamp individual scores for display (with fail-safe type conversion)
    onboard_score = max(0, min(100, float(onboard_score) if onboard_score else 0))
    local_llm_score = max(0, min(100, float(local_llm_score))) if local_llm_score else 0
    gemini_score = max(0, min(100, float(gemini_score))) if gemini_score else 0
    
    # Determine final decision based on consensus score
    final_decision = "Shortlisted" if consensus_score >= 70 else "Rejected"
    
    # Get matched/missing requirements from best available source
    if gemini_available:
        matched_reqs = gemini_result.get("matched_requirements", [])
        missing_reqs = gemini_result.get("missing_requirements", [])
        summary = gemini_result.get("summary", "")
    elif local_llm_available:
        matched_reqs = local_llm_result.get("matched_requirements", [])
        missing_reqs = local_llm_result.get("missing_requirements", [])
        summary = local_llm_result.get("summary", "")
    else:
        matched_reqs = preliminary.get("skills", {}).get("matched", [])
        missing_reqs = preliminary.get("skills", {}).get("missing", [])
        summary = f"Analysis based on onboard AI scoring ({onboard_score:.1f}%)"
    
    # Final result combining all tiers
    final_result = {
        "decision": final_decision,
        "match_percentage": round(consensus_score, 1),
        "reasoning": reasoning if gemini_available else {},
        "matched_requirements": matched_reqs,
        "missing_requirements": missing_reqs,
        "summary": summary,
        "consensus_breakdown": {
            "onboard_score": round(onboard_score, 1),
            "local_llm_score": round(local_llm_score, 1) if local_llm_score else "N/A",
            "gemini_score": round(gemini_score, 1) if gemini_available else "N/A",
            "tiers_used": sum([1, local_llm_available, gemini_available]),
            "model_info": model_info
        },
        "score_evidence": score_evidence,  # Full proof/evidence for the score calculation
        "preliminary_analysis": preliminary,
        "agent_metadata": {
            "model": model_info,
            "threshold": 70,
            "consensus_mechanism": "scientifically_grounded_multi_tier"
        }
    }
    
    final_data = {
        'phase': 'final',
        'step': 26,
        'name': '🏆 FINAL DECISION',
        'action': f'{final_decision} ({consensus_score:.0f}% match)',
        'details': summary,
        'status': 'complete',
        'progress': 100,
        'result': final_result
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    
    # End stream
    yield f"data: {json.dumps({'phase': 'done', 'status': 'complete'})}\n\n"


@router.post("/dev/analyze-stream")
async def analyze_with_streaming_reasoning(request: Request, request_data: ShortlistRequest):
    """
    Development Mode: Analyze resume with real-time chain-of-thought streaming.
    
    This endpoint streams reasoning steps using Server-Sent Events (SSE) for
    real-time visualization of the AI's decision-making process:
    
    1. Onboard AI Analysis (Local, Fast):
       - TF-IDF based semantic similarity
       - Pattern-based skill extraction
       - Experience level detection
       - Education verification
       - Preliminary scoring (40% skills, 25% experience, 10% education, 25% semantic)
    
    2. Gemini AI Analysis (Cloud, Deep):
       - Advanced LLM reasoning
       - Strict 70% threshold evaluation
       - Detailed skills/experience/education/projects/role fit analysis
       - Final decision with JSON reasoning
    
    Both analyses are combined for a comprehensive evaluation visible in real-time.
    
    Returns:
        StreamingResponse: Server-Sent Events stream of reasoning steps
    """
    if not settings.DEVELOPMENT_MODE:
        raise HTTPException(
            status_code=403, 
            detail="Development mode is not enabled. Set DEVELOPMENT_MODE=true in .env"
        )
    
    logger.info("Received development mode streaming analysis request")
    
    return StreamingResponse(
        _stream_reasoning_steps(request_data.resume_text, request_data.job_description),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("/dev/analyze")
@limiter.limit("10/minute")
async def analyze_development_mode(request: Request, request_data: ShortlistRequest):
    """
    Development Mode: Full analysis with combined Onboard AI + Gemini reasoning.
    
    This is the non-streaming version that returns the complete result at once.
    Use /dev/analyze-stream for real-time reasoning display.
    
    Returns:
        JSON with both onboard and Gemini analysis results plus full reasoning chain.
    """
    if not settings.DEVELOPMENT_MODE:
        raise HTTPException(
            status_code=403,
            detail="Development mode is not enabled. Set DEVELOPMENT_MODE=true in .env"
        )
    
    logger.info("Received development mode analysis request")
    
    try:
        # Run onboard analysis
        onboard_result = onboard_ai.analyze(request_data.resume_text, request_data.job_description)
        
        # Run Gemini analysis
        gemini_result = llm_agent.evaluate(request_data.resume_text, request_data.job_description)
        
        # Combine results
        return {
            "onboard_analysis": onboard_result,
            "gemini_analysis": gemini_result,
            "final_decision": gemini_result.get("decision", "Error"),
            "final_match_percentage": gemini_result.get("match_percentage", 0),
            "combined_summary": f"Onboard Score: {onboard_result.get('preliminary_analysis', {}).get('score', 0):.1f}% | Gemini Score: {gemini_result.get('match_percentage', 0)}%",
            "development_mode": True
        }
        
    except Exception as e:
        logger.error(f"Development mode analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# =============================================================================
# RESHA HYBRID AI AGENT ENDPOINTS (Production-Ready Multi-Model Analysis)
# =============================================================================

@router.post("/dev/hybrid-analyze")
@limiter.limit("10/minute")
async def hybrid_analyze(request: Request, request_data: ShortlistRequest):
    """
    Resha Hybrid AI Analysis.
    
    This endpoint uses the HybridAIAgent which combines:
    1. Tier 1 - Onboard AI (TF-IDF + Pattern): Ultra-fast preliminary analysis
    2. Tier 2 - Local LLM (Ollama): CPU-based inference with real reasoning
    3. Tier 3 - Cloud LLM (Gemini): Advanced cloud-based analysis (fallback)
    
    Features:
    - Multi-model weighted consensus for higher accuracy
    - Automatic fallback with graceful degradation
    - Complete audit trail and explainability
    - Offline-capable (when using local LLM only mode)
    
    Query Parameters:
    - mode: "fast" | "local" | "cloud" | "smart" | "consensus" (default: "smart")
    """
    if not settings.DEVELOPMENT_MODE:
        raise HTTPException(
            status_code=403,
            detail="Development mode is not enabled. Set DEVELOPMENT_MODE=true in .env"
        )
    
    # Get mode from query params
    mode = request.query_params.get("mode", "smart")
    if mode not in ["fast", "local", "cloud", "smart", "consensus"]:
        mode = "smart"
    
    logger.info(f"Received hybrid analysis request (mode: {mode})")
    
    try:
        result = await hybrid_agent.analyze(
            request_data.resume_text,
            request_data.job_description,
            mode=mode
        )
        
        # Convert to JSON-serializable format
        return {
            "final_decision": result.final_decision,
            "match_percentage": result.final_match_percentage,
            "confidence": result.final_confidence,
            "consensus_reached": result.consensus_reached,
            "tier_results": [
                {
                    "tier": r.tier.value,
                    "decision": r.decision,
                    "match_percentage": r.match_percentage,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms,
                    "reasoning": r.reasoning,
                    "summary": r.summary
                }
                for r in result.tier_results
            ],
            "reasoning_chain": result.reasoning_chain,
            "summary": result.summary,
            "metadata": result.metadata
        }
        
    except Exception as e:
        logger.error(f"Hybrid analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hybrid analysis failed: {str(e)}")


@router.post("/dev/local-llm")
@limiter.limit("10/minute")
async def local_llm_analyze(request: Request, request_data: ShortlistRequest):
    """
    Local LLM Analysis (CPU-Only, Offline-Capable).
    
    Uses Ollama with lightweight models optimized for CPU inference:
    - Phi-3 Mini (3.8B): Best quality/speed ratio
    - Qwen2.5 (3B): Excellent for structured JSON output
    - Gemma (2B): Ultra-lightweight
    - TinyLlama (1.1B): Fastest
    
    Prerequisites:
    1. Install Ollama: https://ollama.ai
    2. Start Ollama: ollama serve
    3. Pull a model: ollama pull phi3:mini
    
    Returns structured analysis with REAL chain-of-thought (not simulated).
    """
    if not settings.DEVELOPMENT_MODE:
        raise HTTPException(
            status_code=403,
            detail="Development mode is not enabled. Set DEVELOPMENT_MODE=true in .env"
        )
    
    logger.info("Received local LLM analysis request")
    
    try:
        result = await local_llm.evaluate(
            request_data.resume_text,
            request_data.job_description
        )
        
        if result.get("decision") == "Error":
            raise HTTPException(
                status_code=503,
                detail=result.get("summary", "Local LLM not available")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Local LLM analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Local LLM analysis failed: {str(e)}")


@router.get("/dev/status")
async def get_ai_status():
    """
    Get comprehensive AI system status.
    
    Returns status of all inference tiers:
    - Onboard AI: Always available
    - Local LLM (Ollama): Requires Ollama service running
    - Cloud LLM (Gemini): Requires GEMINI_API_KEY configured
    
    Useful for monitoring and debugging.
    """
    try:
        hybrid_status = await hybrid_agent.get_status()
        local_status = await local_llm.get_service_status()
        
        return {
            "status": "operational",
            "development_mode": settings.DEVELOPMENT_MODE,
            "hybrid_agent": hybrid_status,
            "local_llm": local_status,
            "cloud_llm": {
                "configured": bool(settings.GEMINI_API_KEY),
                "model": "gemini-3-flash-preview"
            },
            "recommendations": _get_recommendations(hybrid_status, local_status)
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {
            "status": "degraded",
            "error": str(e)
        }


def _get_recommendations(hybrid_status: dict, local_status: dict) -> list:
    """Generate recommendations based on system status."""
    recommendations = []
    
    if not local_status.get("available"):
        recommendations.append({
            "type": "warning",
            "message": "Local LLM (Ollama) not available",
            "action": "Install Ollama (https://ollama.ai) and run: ollama serve && ollama pull phi3:mini"
        })
    
    if not settings.GEMINI_API_KEY:
        recommendations.append({
            "type": "info",
            "message": "Cloud LLM (Gemini) not configured",
            "action": "Set GEMINI_API_KEY in .env for cloud-based analysis"
        })
    
    if local_status.get("available") and not local_status.get("preferred_model"):
        recommendations.append({
            "type": "warning",
            "message": "No local LLM models installed",
            "action": "Run: ollama pull phi3:mini (or gemma:2b for lighter model)"
        })
    
    if not recommendations:
        recommendations.append({
            "type": "success",
            "message": "All systems operational",
            "action": None
        })
    
    return recommendations


@router.post("/dev/hybrid-stream")
async def hybrid_stream_analysis(request: Request, request_data: ShortlistRequest):
    """
    Stream REAL chain-of-thought reasoning from hybrid analysis.
    
    This endpoint provides GENUINE streaming output:
    1. Quick onboard analysis results
    2. Real streaming tokens from local LLM (if available)
    3. Consolidated multi-model decision
    
    Uses Server-Sent Events (SSE) for real-time display.
    """
    if not settings.DEVELOPMENT_MODE:
        raise HTTPException(
            status_code=403,
            detail="Development mode is not enabled. Set DEVELOPMENT_MODE=true in .env"
        )
    
    logger.info("Received hybrid streaming analysis request")
    
    async def stream_generator():
        try:
            async for event in hybrid_agent.stream_analysis(
                request_data.resume_text,
                request_data.job_description
            ):
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("phase") != "local_llm_streaming":
                    await asyncio.sleep(0.05)
        except Exception as e:
            logger.error(f"Hybrid streaming error: {e}")
            yield f"data: {json.dumps({'phase': 'error', 'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'phase': 'done', 'status': 'complete'})}\n\n"
    
    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# =============================================================================
# LLM WARMUP AND CACHE MANAGEMENT ENDPOINTS
# =============================================================================

@router.post("/dev/warmup")
async def warmup_llm():
    """
    Warmup the local LLM by preloading the model into RAM.
    
    Call this endpoint:
    - After service restart
    - Before expecting heavy traffic
    - When you notice slow first requests
    
    This sends a minimal request to Ollama to load the model into memory,
    preventing cold-start delays on real analysis requests.
    
    Returns:
        Success status and warmup details
    """
    logger.info("Received warmup request")
    try:
        result = await local_llm.warmup()
        return result
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return {"success": False, "error": str(e)}


@router.get("/dev/cache")
async def get_cache_status():
    """
    Get LLM response cache status.
    
    Shows:
    - Current cache size
    - Maximum cache capacity
    - Cache hit/miss statistics
    """
    status = await local_llm.get_service_status()
    return {
        "cache_size": status.get("cache_size", 0),
        "cache_max_size": status.get("cache_max_size", 100),
        "warmed_up_models": status.get("warmed_up_models", []),
        "message": "Cache is active. Repeated analyses will be served instantly from cache."
    }


@router.delete("/dev/cache")
async def clear_cache():
    """
    Clear the LLM response cache.
    
    Use this when:
    - You want fresh analysis results
    - Cache data is stale
    - Testing or debugging
    """
    logger.info("Clearing LLM cache")
    result = local_llm.clear_cache()
    return result

