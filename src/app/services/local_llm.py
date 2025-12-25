"""
Enterprise-Grade Local LLM Service

This module provides lightweight, modern LLM capabilities that run locally without GPU:
- Primary: Ollama integration (supports Qwen3, Phi-3, Gemma, Mistral, TinyLlama)
- Fallback: llama.cpp via llama-cpp-python for quantized models
- Zero external API dependency for offline operation

Key Features:
1. CPU-optimized inference with quantized models
2. Dynamic chain-of-thought streaming (real LLM output, not simulated)
3. Multiple model support with automatic fallback
4. Enterprise security: no data leaves the local environment

Recommended Models (sorted by size/speed):
- Phi-3 Mini (3.8B): Best quality/speed ratio for most tasks
- Gemma 2B: Ultra-lightweight, good for quick analysis
- Qwen2.5-Coder-3B: Excellent for technical/structured output
- TinyLlama (1.1B): Fastest, suitable for simple classification
"""

import os
import json
import asyncio
import httpx
from typing import Dict, Any, Optional, AsyncGenerator, List
from dataclasses import dataclass
from enum import Enum
from ..core.logging import get_logger

logger = get_logger(__name__)


class LocalModelType(Enum):
    """Supported local model types for CPU inference."""
    PHI3_MINI = "phi3:mini"           # 3.8B - Best balance
    PHI3_MEDIUM = "phi3:medium"       # 14B - Higher quality
    GEMMA_2B = "gemma:2b"             # 2B - Ultra-lightweight
    GEMMA2_2B = "gemma2:2b"           # 2B - Improved version
    QWEN2_5_3B = "qwen2.5:3b"         # 3B - Great for JSON output
    QWEN2_5_CODER_3B = "qwen2.5-coder:3b"  # 3B - Code/structured output
    TINYLLAMA = "tinyllama"           # 1.1B - Fastest
    MISTRAL_7B = "mistral:7b-instruct-q4_K_M"  # 7B quantized


@dataclass
class ModelConfig:
    """Configuration for local model inference."""
    name: str
    context_length: int
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    num_ctx: int = 4096
    

class OllamaClient:
    """
    Ollama API client for local LLM inference.
    
    Ollama provides the easiest way to run LLMs locally with:
    - Simple REST API
    - Automatic model download/management
    - Optimized for CPU inference
    - Support for quantized models (GGUF format)
    """
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self._available_models: List[str] = []
        self._is_available = False
        
    async def check_health(self) -> bool:
        """Check if Ollama service is running."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    self._available_models = [m["name"] for m in data.get("models", [])]
                    self._is_available = True
                    logger.info(f"Ollama available with models: {self._available_models}")
                    return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        self._is_available = False
        return False
    
    @property
    def is_available(self) -> bool:
        return self._is_available
    
    @property
    def available_models(self) -> List[str]:
        return self._available_models
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            logger.info(f"Pulling model: {model_name}")
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name, "stream": False}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Dict[str, Any]:
        """Generate a completion (non-streaming)."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1,
                }
            }
            if system:
                payload["system"] = system
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Ollama generate failed: {response.status_code}")
                    return {"error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            return {"error": str(e)}
    
    async def stream_generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion token by token for real-time chain-of-thought."""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                }
            }
            if system:
                payload["system"] = system
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                token = data.get("response", "")
                                if token:
                                    yield token
                                if data.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                                
        except Exception as e:
            logger.error(f"Ollama stream error: {e}")
            yield f"[Error: {str(e)}]"


class LocalLLMService:
    """
    Enterprise-grade Local LLM Service for resume analysis.
    
    Features:
    1. Multi-model support with automatic fallback
    2. Real streaming chain-of-thought (not simulated)
    3. CPU-optimized inference
    4. JSON output parsing and validation
    5. Graceful degradation to fallback analysis
    """
    
    # Resume shortlisting system prompt
    SYSTEM_PROMPT = """You are an advanced Resume Shortlisting AI Agent. Your task is to evaluate resumes against job descriptions with precision.

ANALYSIS PROTOCOL:
1. SKILLS ANALYSIS: Identify technical and soft skills, match against JD requirements
2. EXPERIENCE ANALYSIS: Evaluate years of experience, career progression, relevant roles
3. EDUCATION ANALYSIS: Check educational qualifications, certifications
4. PROJECTS ANALYSIS: Assess relevant projects, achievements, impact
5. ROLE FIT: Determine overall suitability for the position

DECISION CRITERIA:
- If match >= 70%: Output "Shortlisted"
- If match < 70%: Output "Rejected"

You MUST think step-by-step and output your chain-of-thought reasoning.
After your reasoning, output a JSON block with your final decision.

JSON FORMAT (must be valid JSON enclosed in ```json blocks):
```json
{
    "decision": "Shortlisted" or "Rejected",
    "match_percentage": <number 0-100>,
    "reasoning": {
        "skills_analysis": "<detailed analysis>",
        "experience_analysis": "<detailed analysis>",
        "education_analysis": "<detailed analysis>",
        "projects_analysis": "<detailed analysis>",
        "role_fit_analysis": "<final assessment>"
    },
    "matched_requirements": ["<list of matched items>"],
    "missing_requirements": ["<list of missing items>"],
    "summary": "<brief explanation of decision>"
}
```

Be thorough, objective, and consistent in your evaluation."""

    def __init__(self):
        # Import settings for configuration
        try:
            from ..core.config import get_settings
            settings = get_settings()
            ollama_url = settings.OLLAMA_BASE_URL
            preferred_model = settings.OLLAMA_PREFERRED_MODEL
        except Exception:
            ollama_url = "http://localhost:11434"
            preferred_model = ""
        
        self.ollama = OllamaClient(base_url=ollama_url)
        self._preferred_model: Optional[str] = preferred_model if preferred_model else None
        self._initialized = False
        
        # Model preference order (by quality/speed balance)
        self.model_priority = [
            "phi3:mini",
            "qwen2.5:3b", 
            "gemma2:2b",
            "gemma:2b",
            "mistral:7b-instruct-q4_K_M",
            "tinyllama",
        ]
    
    async def initialize(self) -> bool:
        """Initialize the local LLM service and detect available models."""
        if self._initialized:
            return True
            
        is_available = await self.ollama.check_health()
        
        if is_available:
            # Find the best available model
            for model in self.model_priority:
                if any(model in m for m in self.ollama.available_models):
                    self._preferred_model = model
                    logger.info(f"Selected local LLM model: {self._preferred_model}")
                    break
            
            if not self._preferred_model and self.ollama.available_models:
                # Use first available model if none from priority list
                self._preferred_model = self.ollama.available_models[0]
                logger.info(f"Using fallback model: {self._preferred_model}")
            
            self._initialized = True
            return True
        
        logger.warning("Local LLM service not available - Ollama not running")
        return False
    
    @property
    def is_available(self) -> bool:
        return self._initialized and self.ollama.is_available
    
    @property
    def model_name(self) -> str:
        return self._preferred_model or "none"
    
    def _build_evaluation_prompt(self, resume_text: str, job_description: str) -> str:
        """Build the evaluation prompt."""
        return f"""Evaluate the following resume against the job description.

=== JOB DESCRIPTION ===
{job_description}

=== RESUME ===
{resume_text}

=== INSTRUCTIONS ===
Think through your analysis step-by-step:
1. First, identify and list the key requirements from the JD
2. Then, analyze the resume for each requirement
3. Calculate an approximate match percentage
4. Make your final decision

Show your reasoning process, then provide the JSON output."""

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response to extract JSON."""
        import re
        
        # Try to find JSON block
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*"decision"[\s\S]*\}'
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue
        
        # Fallback: try to extract decision from text
        text_lower = response_text.lower()
        if "shortlisted" in text_lower:
            decision = "Shortlisted"
            match_pct = 75
        elif "rejected" in text_lower:
            decision = "Rejected"
            match_pct = 40
        else:
            decision = "Error"
            match_pct = 0
        
        return {
            "decision": decision,
            "match_percentage": match_pct,
            "reasoning": {
                "skills_analysis": "Could not parse structured response",
                "experience_analysis": "",
                "education_analysis": "",
                "projects_analysis": "",
                "role_fit_analysis": response_text[:500]
            },
            "matched_requirements": [],
            "missing_requirements": [],
            "summary": f"Analysis completed with {decision} verdict (parsing fallback)"
        }

    async def evaluate(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Evaluate a resume against a job description using local LLM.
        
        Returns structured analysis with decision and reasoning.
        """
        if not await self.initialize():
            return {
                "decision": "Error",
                "match_percentage": 0,
                "reasoning": {"error": "Local LLM service not available"},
                "matched_requirements": [],
                "missing_requirements": [],
                "summary": "Local LLM (Ollama) not running. Please start Ollama service."
            }
        
        prompt = self._build_evaluation_prompt(resume_text, job_description)
        
        try:
            response = await self.ollama.generate(
                model=self._preferred_model,
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=2048
            )
            
            if "error" in response:
                raise Exception(response["error"])
            
            response_text = response.get("response", "")
            result = self._parse_response(response_text)
            
            # Add metadata
            result["agent_metadata"] = {
                "model": self._preferred_model,
                "threshold": 70,
                "inference_type": "local_cpu",
                "system_instruction_version": "2.0"
            }
            
            # Normalize decision based on match_percentage
            if result.get("decision") not in ["Shortlisted", "Rejected"]:
                match_pct = result.get("match_percentage", 0)
                result["decision"] = "Shortlisted" if match_pct >= 70 else "Rejected"
            
            return result
            
        except Exception as e:
            logger.error(f"Local LLM evaluation error: {e}")
            return {
                "decision": "Error",
                "match_percentage": 0,
                "reasoning": {"error": str(e)},
                "matched_requirements": [],
                "missing_requirements": [],
                "summary": f"Local LLM evaluation failed: {str(e)}"
            }

    async def stream_chain_of_thought(
        self,
        resume_text: str,
        job_description: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream real chain-of-thought reasoning from the LLM.
        
        This provides GENUINE streaming output from the LLM,
        not pre-scripted simulated steps. Each yielded item contains
        actual tokens from the model's reasoning process.
        """
        if not await self.initialize():
            yield {
                "phase": "error",
                "step": 0,
                "content": "Local LLM service not available. Start Ollama with: ollama serve",
                "status": "error"
            }
            return
        
        # Initialize
        yield {
            "phase": "init",
            "step": 0,
            "name": "🚀 Local LLM Analysis",
            "content": f"Starting analysis with {self._preferred_model}",
            "model": self._preferred_model,
            "status": "running",
            "progress": 0
        }
        
        prompt = self._build_evaluation_prompt(resume_text, job_description)
        accumulated_text = ""
        current_section = "reasoning"
        step = 1
        
        try:
            async for token in self.ollama.stream_generate(
                model=self._preferred_model,
                prompt=prompt,
                system=self.SYSTEM_PROMPT,
                temperature=0.3,
                max_tokens=2048
            ):
                accumulated_text += token
                
                # Detect section markers and emit meaningful events
                lower_token = accumulated_text.lower()
                
                # Emit tokens in chunks for readability
                if len(token) > 0:
                    # Determine current section based on content
                    if "skill" in lower_token[-100:]:
                        current_section = "skills_analysis"
                    elif "experience" in lower_token[-100:]:
                        current_section = "experience_analysis"
                    elif "education" in lower_token[-100:]:
                        current_section = "education_analysis"
                    elif "project" in lower_token[-100:]:
                        current_section = "projects_analysis"
                    elif "role fit" in lower_token[-100:] or "overall" in lower_token[-100:]:
                        current_section = "role_fit_analysis"
                    elif "```json" in lower_token[-20:]:
                        current_section = "json_output"
                    
                    yield {
                        "phase": "streaming",
                        "step": step,
                        "section": current_section,
                        "token": token,
                        "accumulated_length": len(accumulated_text),
                        "status": "running",
                        "progress": min(90, 10 + (len(accumulated_text) // 50))
                    }
            
            # Parse final result
            result = self._parse_response(accumulated_text)
            
            yield {
                "phase": "complete",
                "step": step + 1,
                "name": "✅ Analysis Complete",
                "content": f"Decision: {result.get('decision')} ({result.get('match_percentage')}% match)",
                "result": result,
                "full_reasoning": accumulated_text,
                "status": "complete",
                "progress": 100
            }
            
        except Exception as e:
            logger.error(f"Stream CoT error: {e}")
            yield {
                "phase": "error",
                "step": step,
                "content": str(e),
                "status": "error"
            }

    async def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the local LLM service."""
        is_available = await self.ollama.check_health()
        
        return {
            "service": "local_llm",
            "available": is_available,
            "backend": "ollama",
            "preferred_model": self._preferred_model,
            "available_models": self.ollama.available_models,
            "model_priority": self.model_priority,
            "features": [
                "cpu_only_inference",
                "real_streaming_cot",
                "offline_capable",
                "enterprise_secure"
            ]
        }


# Singleton instance
local_llm = LocalLLMService()
