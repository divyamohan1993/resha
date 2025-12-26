from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List

class AnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=50, description="The raw natural language text of the candidate resume.")
    job_description: str = Field(..., min_length=50, description="The raw natural language text of the job description.")
    
    class Config:
        json_schema_extra = {
            "example": {
                "resume_text": "Experienced Python Developer with 5 years in AWS...",
                "job_description": "Looking for a Senior Software Engineer with Python and Cloud experience..."
            }
        }

class AIAnalysisResult(BaseModel):
    cosine_similarity_score: float
    matched_keywords: list[str] = []
    missing_keywords: list[str] = []
    candidate_type: str = "Unknown"
    years_experience: float = 0.0
    entities: Dict[str, list[str]] = {}
    contact_info: Dict[str, Optional[str]] = {}
    details: Dict[str, Any] = {}
    verification: Dict[str, Any] = {}

class AnalysisResponse(BaseModel):
    decision: str
    score: float
    ai_analysis: AIAnalysisResult
    meta: Dict[str, Any]
    reasoning_loop: Optional[Dict[str, Any]] = None  # Agent's reasoning loop JSON output

class FeedbackRequest(BaseModel):
    resume_id: Optional[str] = None # Hash or filename
    score: int = Field(..., ge=1, le=5) # 1-5 rating
    keywords: List[str] = [] # Keywords to reinforce
