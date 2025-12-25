"""
Hybrid AI Agent Orchestrator

This module implements a sophisticated multi-model AI agent that orchestrates:
1. Local LLM (Ollama) - Fast, private, CPU-only inference
2. Cloud LLM (Gemini) - Advanced reasoning, fallback capability
3. Onboard AI - Ultra-fast pattern matching and TF-IDF

Architecture:
                    ┌─────────────────────────────────────┐
                    │      HYBRID AI AGENT ORCHESTRATOR    │
                    ├─────────────────────────────────────┤
                    │                                     │
     ┌──────────────┼──────────────┬──────────────────────┤
     │              │              │                      │
     ▼              ▼              ▼                      │
┌─────────┐   ┌─────────┐   ┌─────────────┐              │
│ TIER 1  │   │ TIER 2  │   │   TIER 3    │              │
│ Onboard │──▶│  Local  │──▶│   Cloud     │              │
│   AI    │   │   LLM   │   │   (Gemini)  │              │
│(TF-IDF) │   │(Ollama) │   │             │              │
└─────────┘   └─────────┘   └─────────────┘              │
     │              │              │                      │
     └──────────────┴──────────────┴──────────────────────┘
                         │
                         ▼
                   FINAL DECISION
                   (Consensus + Confidence)

Enterprise Features:
- Multi-model consensus for higher accuracy
- Automatic fallback with graceful degradation
- Real-time streaming chain-of-thought
- Complete audit trail and explainability
- Zero external dependency mode (offline capable)
"""

import asyncio
import json
from typing import Dict, Any, Optional, AsyncGenerator, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from ..core.logging import get_logger
from .local_llm import local_llm, LocalLLMService
from .onboard_ai import onboard_ai, OnboardAI
from .llm_agent import llm_agent, LLMShortlistingAgent

logger = get_logger(__name__)


class InferenceTier(Enum):
    """Inference tier for the hybrid agent."""
    ONBOARD = "onboard"       # Ultra-fast, pattern-based
    LOCAL_LLM = "local_llm"   # Local LLM (Ollama)
    CLOUD_LLM = "cloud_llm"   # Cloud LLM (Gemini)
    HYBRID = "hybrid"         # Multi-model consensus


@dataclass
class AnalysisResult:
    """Structured analysis result from any inference tier."""
    tier: InferenceTier
    decision: str
    match_percentage: float
    confidence: float
    reasoning: Dict[str, Any]
    matched_requirements: List[str]
    missing_requirements: List[str]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_output: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class HybridResult:
    """Combined result from hybrid analysis."""
    final_decision: str
    final_match_percentage: float
    final_confidence: float
    consensus_reached: bool
    tier_results: List[AnalysisResult]
    reasoning_chain: List[Dict[str, Any]]
    summary: str
    metadata: Dict[str, Any]


class HybridAIAgent:
    """
    Enterprise-grade Hybrid AI Agent for resume shortlisting.
    
    This agent combines multiple inference engines:
    1. Onboard AI (Tier 1): Ultra-fast deterministic analysis
    2. Local LLM (Tier 2): CPU-based LLM inference via Ollama  
    3. Cloud LLM (Tier 3): Gemini API for advanced reasoning
    
    The agent uses intelligent routing and consensus:
    - Simple cases: Onboard AI is sufficient
    - Standard cases: Local LLM with Ollama
    - Complex/ambiguous cases: Cloud LLM or multi-model consensus
    """
    
    def __init__(self):
        self.onboard: OnboardAI = onboard_ai
        self.local_llm: LocalLLMService = local_llm
        self.cloud_llm: LLMShortlistingAgent = llm_agent
        
        # Configuration
        self.consensus_threshold = 0.85  # Agreement required for consensus
        self.confidence_threshold = 0.70  # Minimum confidence for single-tier decision
        self.ambiguity_range = (55, 85)  # Match % range that triggers multi-model
        
    async def _analyze_onboard(
        self, 
        resume_text: str, 
        job_description: str
    ) -> AnalysisResult:
        """Tier 1: Ultra-fast onboard analysis."""
        import time
        start = time.time()
        
        result = self.onboard.analyze(resume_text, job_description)
        preliminary = result.get("preliminary_analysis", {})
        
        score = preliminary.get("score", 0)
        decision = "Shortlisted" if score >= 70 else "Rejected"
        
        # Calculate confidence based on score distance from threshold
        distance_from_threshold = abs(score - 70)
        confidence = min(0.95, 0.50 + (distance_from_threshold / 100))
        
        return AnalysisResult(
            tier=InferenceTier.ONBOARD,
            decision=decision,
            match_percentage=score,
            confidence=confidence,
            reasoning={
                "skills_analysis": f"Matched {preliminary.get('skills', {}).get('match_rate', 0):.1f}% of required skills",
                "experience_analysis": f"{preliminary.get('experience', {}).get('years', 0)} years (level: {preliminary.get('experience', {}).get('level', 'unknown')})",
                "education_analysis": f"Degree: {preliminary.get('education', {}).get('degree_level', 'unknown')}",
                "projects_analysis": "Pattern-based analysis (limited depth)",
                "role_fit_analysis": f"Preliminary score: {score:.1f}%"
            },
            matched_requirements=preliminary.get("skills", {}).get("matched", []),
            missing_requirements=preliminary.get("skills", {}).get("missing", []),
            summary=f"Onboard AI: {decision} ({score:.1f}% match, {confidence:.0%} confidence)",
            metadata={"method": "tfidf_pattern", "reasoning_chain": result.get("reasoning_chain", [])},
            latency_ms=(time.time() - start) * 1000
        )
    
    async def _analyze_local_llm(
        self,
        resume_text: str,
        job_description: str
    ) -> AnalysisResult:
        """Tier 2: Local LLM analysis via Ollama."""
        import time
        start = time.time()
        
        result = await self.local_llm.evaluate(resume_text, job_description)
        
        decision = result.get("decision", "Error")
        match_pct = result.get("match_percentage", 0)
        
        # Local LLM typically has good confidence for clear cases
        if decision == "Error":
            confidence = 0.0
        else:
            distance = abs(match_pct - 70)
            confidence = min(0.90, 0.60 + (distance / 100))
        
        return AnalysisResult(
            tier=InferenceTier.LOCAL_LLM,
            decision=decision,
            match_percentage=match_pct,
            confidence=confidence,
            reasoning=result.get("reasoning", {}),
            matched_requirements=result.get("matched_requirements", []),
            missing_requirements=result.get("missing_requirements", []),
            summary=result.get("summary", ""),
            metadata=result.get("agent_metadata", {}),
            latency_ms=(time.time() - start) * 1000
        )
    
    async def _analyze_cloud_llm(
        self,
        resume_text: str,
        job_description: str
    ) -> AnalysisResult:
        """Tier 3: Cloud LLM analysis via Gemini."""
        import time
        start = time.time()
        
        result = self.cloud_llm.evaluate(resume_text, job_description)
        
        decision = result.get("decision", "Error")
        match_pct = result.get("match_percentage", 0)
        
        # Cloud LLM has highest confidence
        if decision == "Error":
            confidence = 0.0
        else:
            distance = abs(match_pct - 70)
            confidence = min(0.95, 0.70 + (distance / 100))
        
        return AnalysisResult(
            tier=InferenceTier.CLOUD_LLM,
            decision=decision,
            match_percentage=match_pct,
            confidence=confidence,
            reasoning=result.get("reasoning", {}),
            matched_requirements=result.get("matched_requirements", []),
            missing_requirements=result.get("missing_requirements", []),
            summary=result.get("summary", ""),
            metadata=result.get("agent_metadata", {}),
            latency_ms=(time.time() - start) * 1000
        )
    
    def _compute_consensus(
        self,
        results: List[AnalysisResult]
    ) -> Tuple[str, float, float, bool]:
        """
        Compute weighted consensus from multiple analysis results.
        
        Returns: (decision, match_percentage, confidence, consensus_reached)
        """
        if not results:
            return "Error", 0.0, 0.0, False
        
        # Filter out errors
        valid_results = [r for r in results if r.decision != "Error"]
        if not valid_results:
            return "Error", 0.0, 0.0, False
        
        # Weight by tier (higher tier = more weight)
        tier_weights = {
            InferenceTier.ONBOARD: 1.0,
            InferenceTier.LOCAL_LLM: 2.0,
            InferenceTier.CLOUD_LLM: 3.0,
        }
        
        # Calculate weighted decision
        shortlist_score = 0.0
        reject_score = 0.0
        total_weight = 0.0
        weighted_match = 0.0
        
        for r in valid_results:
            weight = tier_weights.get(r.tier, 1.0) * r.confidence
            total_weight += weight
            weighted_match += r.match_percentage * weight
            
            if r.decision == "Shortlisted":
                shortlist_score += weight
            else:
                reject_score += weight
        
        # Determine consensus
        final_match = weighted_match / total_weight if total_weight > 0 else 0
        shortlist_ratio = shortlist_score / total_weight if total_weight > 0 else 0
        
        if shortlist_ratio > 0.5:
            final_decision = "Shortlisted"
            agreement = shortlist_ratio
        else:
            final_decision = "Rejected"
            agreement = 1 - shortlist_ratio
        
        consensus_reached = agreement >= self.consensus_threshold
        final_confidence = agreement * max(r.confidence for r in valid_results)
        
        return final_decision, final_match, final_confidence, consensus_reached
    
    async def analyze(
        self,
        resume_text: str,
        job_description: str,
        mode: str = "smart"
    ) -> HybridResult:
        """
        Analyze resume with hybrid multi-model approach.
        
        Modes:
        - "fast": Onboard AI only
        - "local": Local LLM only
        - "cloud": Cloud LLM only
        - "smart": Intelligent tier selection based on complexity
        - "consensus": All tiers with weighted consensus
        """
        reasoning_chain = []
        results: List[AnalysisResult] = []
        
        # Step 1: Always run onboard for initial assessment
        reasoning_chain.append({
            "step": 1,
            "phase": "onboard",
            "action": "Running ultra-fast onboard analysis (TF-IDF + Pattern)",
            "status": "running"
        })
        
        onboard_result = await self._analyze_onboard(resume_text, job_description)
        results.append(onboard_result)
        
        reasoning_chain.append({
            "step": 1,
            "phase": "onboard",
            "action": f"Onboard: {onboard_result.decision} ({onboard_result.match_percentage:.1f}%)",
            "confidence": onboard_result.confidence,
            "latency_ms": onboard_result.latency_ms,
            "status": "complete"
        })
        
        if mode == "fast":
            # Fast mode: Return onboard result immediately
            return self._build_hybrid_result(results, reasoning_chain)
        
        # Step 2: Determine if we need higher-tier analysis
        needs_llm = (
            mode in ["local", "cloud", "consensus"] or
            (mode == "smart" and (
                onboard_result.confidence < self.confidence_threshold or
                self.ambiguity_range[0] <= onboard_result.match_percentage <= self.ambiguity_range[1]
            ))
        )
        
        if needs_llm and mode != "cloud":
            # Try local LLM first
            reasoning_chain.append({
                "step": 2,
                "phase": "local_llm",
                "action": "Running local LLM analysis (Ollama)",
                "status": "running"
            })
            
            local_result = await self._analyze_local_llm(resume_text, job_description)
            
            if local_result.decision != "Error":
                results.append(local_result)
                reasoning_chain.append({
                    "step": 2,
                    "phase": "local_llm",
                    "action": f"Local LLM: {local_result.decision} ({local_result.match_percentage:.1f}%)",
                    "confidence": local_result.confidence,
                    "latency_ms": local_result.latency_ms,
                    "model": local_result.metadata.get("model", "unknown"),
                    "status": "complete"
                })
            else:
                reasoning_chain.append({
                    "step": 2,
                    "phase": "local_llm",
                    "action": f"Local LLM unavailable: {local_result.summary}",
                    "status": "skipped"
                })
        
        # Step 3: Cloud LLM for consensus or fallback
        needs_cloud = (
            mode in ["cloud", "consensus"] or
            (mode == "smart" and len(results) == 1)  # Local failed, need cloud
        )
        
        if needs_cloud:
            reasoning_chain.append({
                "step": 3,
                "phase": "cloud_llm",
                "action": "Running cloud LLM analysis (Gemini)",
                "status": "running"
            })
            
            cloud_result = await self._analyze_cloud_llm(resume_text, job_description)
            
            if cloud_result.decision != "Error":
                results.append(cloud_result)
                reasoning_chain.append({
                    "step": 3,
                    "phase": "cloud_llm",
                    "action": f"Cloud LLM: {cloud_result.decision} ({cloud_result.match_percentage:.1f}%)",
                    "confidence": cloud_result.confidence,
                    "latency_ms": cloud_result.latency_ms,
                    "model": cloud_result.metadata.get("model", "gemini"),
                    "status": "complete"
                })
            else:
                reasoning_chain.append({
                    "step": 3,
                    "phase": "cloud_llm",
                    "action": f"Cloud LLM error: {cloud_result.summary}",
                    "status": "error"
                })
        
        return self._build_hybrid_result(results, reasoning_chain)
    
    def _build_hybrid_result(
        self,
        results: List[AnalysisResult],
        reasoning_chain: List[Dict[str, Any]]
    ) -> HybridResult:
        """Build the final hybrid result with consensus."""
        decision, match_pct, confidence, consensus = self._compute_consensus(results)
        
        # Use the best result's detailed reasoning
        best_result = max(results, key=lambda r: r.confidence) if results else None
        
        # Compile summary
        tier_summaries = [f"{r.tier.value}: {r.decision}" for r in results]
        
        metadata = {
            "tiers_used": [r.tier.value for r in results],
            "total_latency_ms": sum(r.latency_ms for r in results),
            "consensus_reached": consensus,
            "analysis_timestamp": datetime.utcnow().isoformat(),
            "agent_version": "3.0.0-hybrid"
        }
        
        return HybridResult(
            final_decision=decision,
            final_match_percentage=match_pct,
            final_confidence=confidence,
            consensus_reached=consensus,
            tier_results=results,
            reasoning_chain=reasoning_chain,
            summary=f"Hybrid Analysis: {decision} ({match_pct:.1f}% match, {confidence:.0%} confidence). Tiers: {', '.join(tier_summaries)}",
            metadata=metadata
        )
    
    async def stream_analysis(
        self,
        resume_text: str,
        job_description: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real-time chain-of-thought analysis.
        
        This method provides GENUINE streaming output:
        1. First, quick onboard analysis results
        2. Then, real streaming tokens from local LLM
        3. Finally, consolidated decision
        """
        # Phase 1: Onboard Analysis
        yield {
            "phase": "init",
            "step": 0,
            "name": "🔧 Hybrid AI Agent Initializing",
            "action": "Setting up multi-model analysis pipeline",
            "status": "running",
            "progress": 0
        }
        
        await asyncio.sleep(0.2)
        
        # Onboard analysis
        onboard_result = await self._analyze_onboard(resume_text, job_description)
        
        # Stream onboard reasoning chain
        for step in onboard_result.metadata.get("reasoning_chain", []):
            yield {
                "phase": "onboard",
                "step": step.get("step", 0),
                "name": f"📊 {step.get('name', 'Analysis')}",
                "action": step.get("action", ""),
                "details": step.get("details", ""),
                "data": step.get("data"),
                "status": "complete",
                "progress": 5 + (step.get("step", 0) * 4)
            }
            await asyncio.sleep(0.15)
        
        yield {
            "phase": "onboard_complete",
            "step": 10,
            "name": "✅ Onboard Analysis Complete",
            "action": f"Preliminary: {onboard_result.decision} ({onboard_result.match_percentage:.1f}%)",
            "confidence": onboard_result.confidence,
            "latency_ms": onboard_result.latency_ms,
            "status": "complete",
            "progress": 40
        }
        
        # Phase 2: Local LLM (Real Streaming)
        yield {
            "phase": "local_llm_init",
            "step": 11,
            "name": "🤖 Local LLM Starting",
            "action": "Initializing Ollama for CPU-based inference",
            "status": "running",
            "progress": 42
        }
        
        local_llm_success = False
        local_result = None
        
        try:
            # Stream actual LLM tokens
            accumulated = ""
            async for event in self.local_llm.stream_chain_of_thought(resume_text, job_description):
                if event.get("phase") == "streaming":
                    accumulated += event.get("token", "")
                    yield {
                        "phase": "local_llm_streaming",
                        "step": 12,
                        "name": "💭 Real-time Reasoning",
                        "token": event.get("token", ""),
                        "section": event.get("section", ""),
                        "accumulated_length": len(accumulated),
                        "status": "streaming",
                        "progress": min(85, 45 + (len(accumulated) // 100))
                    }
                elif event.get("phase") == "complete":
                    local_result = event.get("result")
                    local_llm_success = True
                    yield {
                        "phase": "local_llm_complete",
                        "step": 13,
                        "name": "✅ Local LLM Complete",
                        "action": f"Local: {local_result.get('decision')} ({local_result.get('match_percentage')}%)",
                        "result": local_result,
                        "status": "complete",
                        "progress": 90
                    }
                elif event.get("phase") == "error":
                    yield {
                        "phase": "local_llm_error",
                        "step": 13,
                        "name": "⚠️ Local LLM Unavailable",
                        "action": event.get("content", "Ollama not running"),
                        "status": "skipped",
                        "progress": 50
                    }
        except Exception as e:
            yield {
                "phase": "local_llm_error",
                "step": 13,
                "name": "⚠️ Local LLM Error",
                "action": str(e),
                "status": "error",
                "progress": 50
            }
        
        # Phase 3: Final Decision
        results = [onboard_result]
        if local_result:
            from .local_llm import local_llm as llm_service
            results.append(AnalysisResult(
                tier=InferenceTier.LOCAL_LLM,
                decision=local_result.get("decision", "Error"),
                match_percentage=local_result.get("match_percentage", 0),
                confidence=0.85,
                reasoning=local_result.get("reasoning", {}),
                matched_requirements=local_result.get("matched_requirements", []),
                missing_requirements=local_result.get("missing_requirements", []),
                summary=local_result.get("summary", ""),
                metadata=local_result.get("agent_metadata", {}),
            ))
        
        decision, match_pct, confidence, consensus = self._compute_consensus(results)
        
        yield {
            "phase": "final",
            "step": 20,
            "name": "🏆 FINAL DECISION",
            "action": f"{decision} ({match_pct:.1f}% match)",
            "details": f"Confidence: {confidence:.0%} | Consensus: {'Yes' if consensus else 'No'}",
            "tiers_used": [r.tier.value for r in results],
            "result": {
                "decision": decision,
                "match_percentage": match_pct,
                "confidence": confidence,
                "consensus": consensus,
                "reasoning": results[-1].reasoning if results else {},
                "matched_requirements": results[-1].matched_requirements if results else [],
                "missing_requirements": results[-1].missing_requirements if results else [],
            },
            "status": "complete",
            "progress": 100
        }
        
        yield {
            "phase": "done",
            "status": "complete"
        }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get the status of all inference tiers."""
        local_status = await self.local_llm.get_service_status()
        cloud_configured = bool(self.cloud_llm.gemini_api_key)
        
        return {
            "agent": "hybrid_ai",
            "version": "3.0.0",
            "tiers": {
                "onboard": {
                    "available": True,
                    "method": "TF-IDF + Pattern Matching",
                    "latency": "~50ms"
                },
                "local_llm": {
                    "available": local_status.get("available", False),
                    "backend": "ollama",
                    "model": local_status.get("preferred_model"),
                    "latency": "~2-5s"
                },
                "cloud_llm": {
                    "available": cloud_configured,
                    "backend": "gemini",
                    "model": "gemini-3-flash-preview",
                    "latency": "~1-3s"
                }
            },
            "features": [
                "multi_model_consensus",
                "real_streaming_cot",
                "automatic_fallback",
                "offline_capable",
                "enterprise_audit"
            ]
        }


# Singleton instance
hybrid_agent = HybridAIAgent()
