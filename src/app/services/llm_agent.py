"""
LLM-based Resume Shortlisting Agent

This module implements the AI agent logic as specified in Task B requirements:
- Uses Gemini AI with a strict system instruction prompt
- Evaluates resumes against job descriptions using 70% match threshold
- Returns structured JSON with reasoning and decision

Agent Architecture:
1. Input Processing: Receives resume_text and job_description
2. Prompt Construction: Builds a strict evaluation prompt
3. LLM Inference: Calls Gemini API for analysis
4. Response Parsing: Extracts decision and reasoning from LLM output
5. Output Generation: Returns structured JSON response
"""

import os
import json
import re
from typing import Optional
from ..core.logging import get_logger

logger = get_logger(__name__)

# System Instruction as specified in Task B
SYSTEM_INSTRUCTION = """You are a Resume Shortlisting AI. You must analyze skills, experience, projects, education, and role fit. If the resume closely matches 70% or more of the JD requirements, output 'Shortlisted'. Otherwise output 'Rejected'. You must follow the evaluation strictly.

IMPORTANT: You MUST respond with a valid JSON object in this exact format:
{
    "decision": "Shortlisted" or "Rejected",
    "match_percentage": <number between 0-100>,
    "candidate_profile": {
        "candidate_type": "Fresher" or "Experienced" or "Senior",
        "years_experience": <number of years as integer>,
        "organizations": ["<list of companies/universities mentioned in resume>"],
        "locations": ["<list of cities/countries mentioned in resume>"]
    },
    "reasoning": {
        "skills_analysis": "<analysis of skills match>",
        "experience_analysis": "<analysis of experience match>",
        "education_analysis": "<analysis of education match>",
        "projects_analysis": "<analysis of relevant projects>",
        "role_fit_analysis": "<overall role fit assessment>"
    },
    "matched_requirements": ["<list of matched requirements>"],
    "missing_requirements": ["<list of missing requirements>"],
    "summary": "<brief summary explaining the decision>"
}

Respond ONLY with the JSON object, no additional text before or after."""


class LLMShortlistingAgent:
    """
    AI Agent for Resume Shortlisting using LLM (Gemini).
    
    This agent follows a strict evaluation protocol:
    1. Analyzes skills, experience, projects, education, and role fit
    2. Uses 70% match threshold for shortlisting decisions
    3. Provides detailed reasoning in JSON format
    """
    
    def __init__(self):
        # Try to get API key from environment or settings
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.gemini_api_key:
            try:
                from ..core.config import get_settings
                settings = get_settings()
                self.gemini_api_key = settings.GEMINI_API_KEY
            except Exception:
                pass
        
        self.model_name = "gemini-3-flash-preview"  # Using Gemini 3 Flash Preview
        self._gemini_configured = False
        self._client = None
        
        if self.gemini_api_key:
            self._configure_gemini()
        else:
            logger.warning("GEMINI_API_KEY not set. LLM agent will not function.")
    
    def _configure_gemini(self):
        """Configure Google Gemini API using google-genai."""
        try:
            from google import genai
            self._client = genai.Client(api_key=self.gemini_api_key)
            self._gemini_configured = True
            logger.info(f"Gemini API configured successfully with model: {self.model_name}")
        except ImportError:
            logger.error("google-genai package not installed. Run: pip install google-genai")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {e}")
    
    def _build_evaluation_prompt(self, resume_text: str, job_description: str) -> str:
        """Build the evaluation prompt for the LLM."""
        return f"""Evaluate the following resume against the job description.

=== JOB DESCRIPTION ===
{job_description}

=== RESUME ===
{resume_text}

=== EVALUATION INSTRUCTIONS ===
1. Analyze how well the candidate's SKILLS match the required skills in the JD
2. Analyze how the candidate's EXPERIENCE aligns with the job requirements
3. Evaluate the candidate's EDUCATION relevance to the role
4. Assess any PROJECTS that demonstrate relevant capabilities
5. Determine overall ROLE FIT

If the overall match is 70% or higher, the decision should be "Shortlisted".
If the overall match is below 70%, the decision should be "Rejected".

Be strict and objective in your evaluation."""
    
    def _parse_llm_response(self, response_text: str) -> dict:
        """Parse the LLM response to extract JSON."""
        try:
            # Try to find JSON in the response
            # First, try direct JSON parsing
            cleaned = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return error structure
            logger.error(f"Failed to parse LLM response as JSON: {response_text[:500]}")
            return {
                "decision": "Error",
                "match_percentage": 0,
                "reasoning": {
                    "error": "Failed to parse LLM response"
                },
                "matched_requirements": [],
                "missing_requirements": [],
                "summary": "Error in processing"
            }
    
    def evaluate(self, resume_text: str, job_description: str) -> dict:
        """
        Evaluate a resume against a job description using Gemini AI.
        
        Args:
            resume_text: Plain text of the candidate's resume
            job_description: Plain text of the job description
            
        Returns:
            dict: Structured response with decision, match_percentage, 
                  reasoning, and requirements analysis
        """
        if not self._gemini_configured:
            return {
                "decision": "Error",
                "match_percentage": 0,
                "reasoning": {
                    "error": "Gemini API not configured. Please set GEMINI_API_KEY environment variable."
                },
                "matched_requirements": [],
                "missing_requirements": [],
                "summary": "LLM agent not available - API key not configured"
            }
        
        try:
            # Build the prompt
            user_prompt = self._build_evaluation_prompt(resume_text, job_description)
            
            # Call Gemini API using google-genai client
            logger.info("Calling Gemini API for resume evaluation...")
            
            # Combine system instruction and user prompt
            full_prompt = f"{SYSTEM_INSTRUCTION}\n\n{user_prompt}"
            
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": 0.2,  # Low temperature for consistent evaluation
                    "max_output_tokens": 2048
                }
            )
            
            response_text = response.text
            logger.info("Gemini API response received")
            
            # Parse the response
            result = self._parse_llm_response(response_text)
            
            # Validate decision
            if result.get("decision") not in ["Shortlisted", "Rejected", "Error"]:
                # Normalize decision based on match percentage
                match_pct = result.get("match_percentage", 0)
                result["decision"] = "Shortlisted" if match_pct >= 70 else "Rejected"
            
            # Add reasoning loop metadata
            result["agent_metadata"] = {
                "model": self.model_name,
                "threshold": 70,
                "system_instruction_version": "1.0"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error during LLM evaluation: {e}")
            return {
                "decision": "Error",
                "match_percentage": 0,
                "reasoning": {
                    "error": str(e)
                },
                "matched_requirements": [],
                "missing_requirements": [],
                "summary": f"Error during evaluation: {str(e)}"
            }
    
    def get_reasoning_loop_json(self, resume_text: str, job_description: str) -> dict:
        """
        Get the complete reasoning loop as JSON.
        
        This method returns the full agent reasoning process including:
        - Input processing
        - Evaluation steps
        - Decision making
        - Final output
        
        Returns:
            dict: Complete reasoning loop in JSON format
        """
        # Stage 1: Input Processing
        reasoning_loop = {
            "stage_1_input_processing": {
                "resume_length": len(resume_text),
                "jd_length": len(job_description),
                "resume_preview": resume_text[:200] + "..." if len(resume_text) > 200 else resume_text,
                "jd_preview": job_description[:200] + "..." if len(job_description) > 200 else job_description
            },
            "stage_2_prompt_construction": {
                "system_instruction": SYSTEM_INSTRUCTION[:200] + "...",
                "evaluation_criteria": [
                    "skills_analysis",
                    "experience_analysis", 
                    "education_analysis",
                    "projects_analysis",
                    "role_fit_analysis"
                ],
                "threshold": "70% match required for Shortlisted"
            }
        }
        
        # Stage 3: LLM Inference
        evaluation_result = self.evaluate(resume_text, job_description)
        
        reasoning_loop["stage_3_llm_inference"] = {
            "model_used": self.model_name,
            "api_status": "success" if evaluation_result.get("decision") != "Error" else "error",
            "raw_evaluation": evaluation_result
        }
        
        # Stage 4: Decision Output
        reasoning_loop["stage_4_decision_output"] = {
            "final_decision": evaluation_result.get("decision"),
            "match_percentage": evaluation_result.get("match_percentage"),
            "summary": evaluation_result.get("summary")
        }
        
        return reasoning_loop


# Singleton instance
llm_agent = LLMShortlistingAgent()
