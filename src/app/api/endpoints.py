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
from typing import AsyncGenerator

router = APIRouter()
settings = get_settings()
logger = get_logger(__name__)
audit_logger = AuditLogger()

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
    
    Args:
        request: FastAPI Request object (used for rate limiting).
        request_data: JSON payload containing resume_text and job_description.
        
    Returns:
        AnalysisResponse: Score, decision, and detailed AI analysis.
        
    Raises:
        HTTPException: If analysis fails.
    """
    logger.info("Received analysis request")
    try:
        # AI Processing
        result = ai_service.analyze_candidate(request_data.resume_text, request_data.job_description)
        score = result["score"]
        decision = result["decision"]
        
        # Audit
        resume_hash = ai_service.hash_text(request_data.resume_text)
        audit_logger.log_decision(resume_hash, score, decision, filename="Manual Input")
        
        return AnalysisResponse(
            decision=decision,
            score=round(score, 4),
            ai_analysis=AIAnalysisResult(
                cosine_similarity_score=result["semantic_score"],
                matched_keywords=result["matched_keywords"],
                missing_keywords=result["missing_keywords"],
                candidate_type=result["candidate_type"],
                years_experience=result["years_experience"],
                entities=result["entities"],
                contact_info=result["contact"],
                details=result["details"],
                verification=result.get("verification", {})
            ),
            meta={
                "threshold_used": settings.THRESHOLD_SCORE,
                "model": "SBERT (all-MiniLM-L6-v2) + Spacy + RLHF"
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
        
        # AI Processing
        result = ai_service.analyze_candidate(resume_text, job_description)
        score = result["score"]
        decision = result["decision"]
        
        # Audit
        resume_hash = ai_service.hash_text(resume_text)
        audit_logger.log_decision(resume_hash, score, decision, filename=file.filename)
        
        return AnalysisResponse(
            decision=decision,
            score=round(score, 4),
            ai_analysis=AIAnalysisResult(
                cosine_similarity_score=result["semantic_score"],
                matched_keywords=result["matched_keywords"],
                missing_keywords=result["missing_keywords"],
                candidate_type=result["candidate_type"],
                years_experience=result["years_experience"],
                entities=result["entities"],
                contact_info=result["contact"],
                details=result["details"],
                verification=result.get("verification", {})
            ),
            meta={
                "threshold_used": settings.THRESHOLD_SCORE,
                "model": "SBERT (all-MiniLM-L6-v2) + Spacy + RLHF",
                "filename": file.filename
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
            result.get("match_percentage", 0) / 100,  # Normalize to 0-1 for consistency
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
            "enterprise_security"
        ]
    }


async def _stream_reasoning_steps(resume_text: str, job_description: str) -> AsyncGenerator[str, None]:
    """
    Generator function that streams chain-of-thought reasoning steps.
    Each step is sent as a Server-Sent Event for real-time display.
    """
    
    # Phase 1: Initialize
    yield f"data: {json.dumps({'phase': 'init', 'step': 0, 'name': 'Initializing Analysis', 'action': 'Setting up AI analysis pipeline', 'status': 'running', 'progress': 0})}\n\n"
    await asyncio.sleep(0.3)
    
    # Phase 2: Onboard AI Analysis (Fast, Local)
    yield f"data: {json.dumps({'phase': 'onboard', 'step': 1, 'name': '🔧 Onboard AI Active', 'action': 'Running lightweight local analysis (TF-IDF + Pattern Matching)', 'status': 'running', 'progress': 5})}\n\n"
    await asyncio.sleep(0.2)
    
    # Run onboard analysis
    try:
        onboard_result = onboard_ai.analyze(resume_text, job_description)
        reasoning_chain = onboard_result.get("reasoning_chain", [])
        preliminary = onboard_result.get("preliminary_analysis", {})
        
        # Stream each step of the onboard reasoning chain
        total_onboard_steps = len(reasoning_chain)
        for idx, step in enumerate(reasoning_chain):
            progress = 10 + int((idx / max(total_onboard_steps, 1)) * 40)  # 10-50% range
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
            await asyncio.sleep(0.25)  # Visible pacing
        
        # Send preliminary results
        prelim_score = preliminary.get('score', 0)
        prelim_decision = preliminary.get('decision', 'Unknown')
        onboard_complete_data = {
            'phase': 'onboard_complete',
            'step': 10,
            'name': '✅ Onboard Analysis Complete',
            'action': 'Preliminary scoring finished',
            'details': f"Preliminary Score: {prelim_score:.1f}% - {prelim_decision}",
            'data': preliminary,
            'status': 'complete',
            'progress': 50
        }
        yield f"data: {json.dumps(onboard_complete_data)}\n\n"
        await asyncio.sleep(0.3)
        
    except Exception as e:
        yield f"data: {json.dumps({'phase': 'error', 'step': -1, 'name': '❌ Onboard AI Error', 'action': str(e), 'status': 'error', 'progress': 50})}\n\n"
        preliminary = {}
    
    # Phase 3: Gemini AI Analysis (Cloud, Deep Analysis)
    yield f"data: {json.dumps({'phase': 'gemini', 'step': 11, 'name': '🤖 Gemini AI Connecting', 'action': 'Initiating cloud-based advanced LLM analysis', 'status': 'running', 'progress': 55})}\n\n"
    await asyncio.sleep(0.3)
    
    yield f"data: {json.dumps({'phase': 'gemini', 'step': 12, 'name': '📝 Prompt Construction', 'action': 'Building strict evaluation prompt with system instruction', 'details': 'System: You are a Resume Shortlisting AI. You must analyze skills, experience, projects, education, and role fit...', 'status': 'running', 'progress': 60})}\n\n"
    await asyncio.sleep(0.3)
    
    yield f"data: {json.dumps({'phase': 'gemini', 'step': 13, 'name': '🚀 API Request', 'action': 'Sending request to Gemini 3 Flash Preview', 'status': 'running', 'progress': 65})}\n\n"
    await asyncio.sleep(0.2)
    
    # Call Gemini
    try:
        gemini_result = llm_agent.evaluate(resume_text, job_description)
        
        yield f"data: {json.dumps({'phase': 'gemini', 'step': 14, 'name': '📥 Response Received', 'action': 'Parsing Gemini API response', 'status': 'running', 'progress': 75})}\n\n"
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
        
        step_num = 15
        for key, name, progress in reasoning_steps:
            if key in reasoning and reasoning[key]:
                yield f"data: {json.dumps({'phase': 'gemini_reasoning', 'step': step_num, 'name': name, 'action': reasoning[key], 'status': 'complete', 'progress': progress})}\n\n"
                step_num += 1
                await asyncio.sleep(0.2)
        
        # Final Decision
        final_decision = gemini_result.get("decision", "Error")
        match_percentage = gemini_result.get("match_percentage", 0)
        summary = gemini_result.get("summary", "")
        
        final_result = {
            "decision": final_decision,
            "match_percentage": match_percentage,
            "reasoning": reasoning,
            "matched_requirements": gemini_result.get("matched_requirements", []),
            "missing_requirements": gemini_result.get("missing_requirements", []),
            "summary": summary,
            "preliminary_analysis": preliminary,
            "agent_metadata": gemini_result.get("agent_metadata")
        }
        
        final_data = {
            'phase': 'final',
            'step': 20,
            'name': '🏆 FINAL DECISION',
            'action': f'{final_decision} ({match_percentage}% match)',
            'details': summary,
            'status': 'complete',
            'progress': 100,
            'result': final_result
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        
    except Exception as e:
        logger.error(f"Gemini analysis failed: {e}")
        
        # Fallback to onboard-only decision
        fallback_decision = "Shortlisted" if preliminary.get("score", 0) >= 70 else "Rejected"
        fallback_data = {
            'phase': 'fallback',
            'step': 20,
            'name': '⚠️ Fallback Decision (Gemini Error)',
            'action': f'{fallback_decision} based on onboard analysis only',
            'details': str(e),
            'status': 'complete',
            'progress': 100,
            'result': {
                'decision': fallback_decision,
                'match_percentage': preliminary.get('score', 0),
                'preliminary_analysis': preliminary,
                'error': str(e)
            }
        }
        yield f"data: {json.dumps(fallback_data)}\n\n"
    
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
# HYBRID AI AGENT ENDPOINTS (Enterprise-Grade Multi-Model Analysis)
# =============================================================================

@router.post("/dev/hybrid-analyze")
@limiter.limit("10/minute")
async def hybrid_analyze(request: Request, request_data: ShortlistRequest):
    """
    Enterprise-Grade Hybrid AI Analysis.
    
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


