"""
Pydantic schemas for the LLM-based shortlisting endpoint.
These schemas match the exact Task B requirements.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List


class ShortlistRequest(BaseModel):
    """Request schema for the shortlist endpoint."""
    resume_text: str = Field(
        ..., 
        min_length=50, 
        description="The raw plain text of the candidate's resume."
    )
    job_description: str = Field(
        ..., 
        min_length=50, 
        description="The raw plain text of the job description (JD)."
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "John Doe - Software Engineer with 5 years experience in Python, AWS, Docker, and Kubernetes. Worked at Tech Corp building microservices...",
                "job_description": "We are looking for a Senior Software Engineer with expertise in Python, cloud technologies (AWS/GCP), containerization (Docker/K8s), and 5+ years of experience..."
            }
        }


class ReasoningDetails(BaseModel):
    """Detailed reasoning from AI analysis."""
    skills_analysis: str = ""
    experience_analysis: str = ""
    education_analysis: str = ""
    projects_analysis: str = ""
    role_fit_analysis: str = ""


class AgentMetadata(BaseModel):
    """Metadata about the AI agent processing."""
    model: str = "gemini-2.0-flash"
    threshold: int = 70
    system_instruction_version: str = "1.0"


class ShortlistResponse(BaseModel):
    """
    Response schema matching Task B requirements.
    
    Returns:
    - decision: "Shortlisted" or "Rejected"
    - match_percentage: How closely the resume matches the JD (0-100)
    - reasoning: Detailed analysis of skills, experience, education, projects, role fit
    - matched_requirements: List of JD requirements the candidate meets
    - missing_requirements: List of JD requirements the candidate lacks
    - summary: Brief explanation of the decision
    """
    decision: str = Field(..., description="'Shortlisted' if match >= 70%, otherwise 'Rejected'")
    match_percentage: float = Field(..., ge=0, le=100, description="Percentage match between resume and JD")
    reasoning: Dict[str, str] = Field(default_factory=dict, description="Detailed reasoning from AI analysis")
    matched_requirements: List[str] = Field(default_factory=list, description="JD requirements the candidate meets")
    missing_requirements: List[str] = Field(default_factory=list, description="JD requirements the candidate lacks")
    summary: str = Field(..., description="Brief explanation of the decision")
    agent_metadata: Optional[AgentMetadata] = None


class ReasoningLoopResponse(BaseModel):
    """
    Full reasoning loop output showing the agent's step-by-step process.
    This exposes the internal reasoning chain for debugging and transparency.
    """
    stage_1_input_processing: Dict[str, Any]
    stage_2_prompt_construction: Dict[str, Any]
    stage_3_llm_inference: Dict[str, Any]
    stage_4_decision_output: Dict[str, Any]
