"""
Lightweight Onboard AI Service

This module provides lightweight, on-device AI analysis using:
- Sentence embeddings (SBERT - already available)
- Keyword extraction
- Semantic similarity scoring

This is used BEFORE calling Gemini to provide quick preliminary analysis,
reducing API calls and providing instant feedback in development mode.
"""

import re
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from ..core.logging import get_logger

logger = get_logger(__name__)

class OnboardAI:
    """
    Lightweight on-device AI for preliminary resume analysis.
    Uses TF-IDF and pattern matching - no external API calls needed.
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=200,
            ngram_range=(1, 2)  # Include bigrams
        )
        
        # Common skill patterns for different domains
        self.skill_categories = {
            "programming_languages": [
                "python", "java", "javascript", "typescript", "c++", "c#", 
                "go", "rust", "ruby", "php", "swift", "kotlin", "scala"
            ],
            "web_frameworks": [
                "react", "angular", "vue", "django", "flask", "fastapi",
                "express", "nextjs", "spring", "rails", "laravel"
            ],
            "databases": [
                "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
                "cassandra", "dynamodb", "firebase", "sqlite", "oracle"
            ],
            "cloud_devops": [
                "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
                "jenkins", "ci/cd", "devops", "linux", "nginx", "ansible"
            ],
            "ai_ml": [
                "machine learning", "deep learning", "nlp", "tensorflow", 
                "pytorch", "keras", "scikit-learn", "pandas", "numpy",
                "computer vision", "neural network", "llm", "transformers"
            ],
            "soft_skills": [
                "leadership", "communication", "teamwork", "problem solving",
                "agile", "scrum", "project management", "time management"
            ]
        }
        
        # Experience level indicators
        self.experience_patterns = {
            "senior": ["senior", "lead", "principal", "staff", "architect", "director"],
            "mid": ["mid", "intermediate", "mid-level", "mid level"],
            "junior": ["junior", "entry", "graduate", "intern", "fresher", "trainee"]
        }
        
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills categorized by type."""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skill_categories.items():
            matches = []
            for skill in skills:
                if skill in text_lower:
                    matches.append(skill)
            if matches:
                found_skills[category] = matches
                
        return found_skills
    
    def extract_years_experience(self, text: str) -> int:
        """Extract years of experience from text."""
        patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience\s*(?:of\s*)?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*(?:in|with)',
        ]
        
        years = 0
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                years = max(years, max(int(m) for m in matches))
        
        return years
    
    def detect_experience_level(self, text: str, years: int) -> str:
        """Detect experience level from text and years."""
        text_lower = text.lower()
        
        # Check explicit indicators first
        for level, patterns in self.experience_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return level
        
        # Fall back to years of experience
        if years >= 7:
            return "senior"
        elif years >= 3:
            return "mid"
        else:
            return "junior"
    
    def calculate_tfidf_similarity(self, resume: str, jd: str) -> float:
        """Calculate TF-IDF based similarity between resume and JD."""
        try:
            # Fit on both documents
            tfidf_matrix = self.vectorizer.fit_transform([resume, jd])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            return 0.0
    
    def extract_education(self, text: str) -> Dict[str, Any]:
        """Extract education information."""
        text_lower = text.lower()
        education = {
            "degree_level": "unknown",
            "fields": []
        }
        
        # Degree levels
        if any(d in text_lower for d in ["phd", "ph.d", "doctorate", "doctoral"]):
            education["degree_level"] = "PhD"
        elif any(d in text_lower for d in ["master", "m.s", "m.tech", "mba", "msc"]):
            education["degree_level"] = "Masters"
        elif any(d in text_lower for d in ["bachelor", "b.s", "b.tech", "bsc", "b.e"]):
            education["degree_level"] = "Bachelors"
        
        # Fields of study
        fields = ["computer science", "software engineering", "information technology",
                  "electrical engineering", "data science", "mathematics", "physics"]
        education["fields"] = [f for f in fields if f in text_lower]
        
        return education
    
    def analyze(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        """
        Complete onboard AI analysis with chain-of-thought reasoning.
        Returns detailed step-by-step analysis for development mode display.
        """
        reasoning_chain = []
        
        # Step 1: Preprocessing
        reasoning_chain.append({
            "step": 1,
            "name": "Preprocessing",
            "action": "Tokenizing and normalizing text inputs",
            "details": f"Resume length: {len(resume_text)} chars, JD length: {len(job_description)} chars",
            "status": "complete"
        })
        
        # Step 2: Extract Resume Skills
        resume_skills = self.extract_skills(resume_text)
        skill_count = sum(len(v) for v in resume_skills.values())
        reasoning_chain.append({
            "step": 2,
            "name": "Resume Skill Extraction",
            "action": "Identifying technical and soft skills from resume",
            "details": f"Found {skill_count} skills across {len(resume_skills)} categories",
            "data": resume_skills,
            "status": "complete"
        })
        
        # Step 3: Extract JD Requirements
        jd_skills = self.extract_skills(job_description)
        jd_skill_count = sum(len(v) for v in jd_skills.values())
        reasoning_chain.append({
            "step": 3,
            "name": "JD Requirement Extraction",
            "action": "Parsing required skills from job description",
            "details": f"Found {jd_skill_count} required skills across {len(jd_skills)} categories",
            "data": jd_skills,
            "status": "complete"
        })
        
        # Step 4: Skills Matching
        matched_skills = []
        missing_skills = []
        for category, skills in jd_skills.items():
            resume_cat_skills = resume_skills.get(category, [])
            for skill in skills:
                if skill in resume_cat_skills:
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
        
        skill_match_rate = len(matched_skills) / max(len(matched_skills) + len(missing_skills), 1) * 100
        reasoning_chain.append({
            "step": 4,
            "name": "Skill Matching Analysis",
            "action": "Comparing candidate skills against job requirements",
            "details": f"Match rate: {skill_match_rate:.1f}% ({len(matched_skills)}/{len(matched_skills) + len(missing_skills)})",
            "data": {"matched": matched_skills, "missing": missing_skills},
            "status": "complete"
        })
        
        # Step 5: Experience Analysis
        resume_years = self.extract_years_experience(resume_text)
        jd_years = self.extract_years_experience(job_description)
        experience_level = self.detect_experience_level(resume_text, resume_years)
        
        experience_match = "adequate" if resume_years >= jd_years else "insufficient"
        reasoning_chain.append({
            "step": 5,
            "name": "Experience Analysis",
            "action": "Evaluating years of experience and seniority level",
            "details": f"Candidate: {resume_years} years ({experience_level}), Required: {jd_years}+ years",
            "data": {
                "candidate_years": resume_years,
                "required_years": jd_years,
                "level": experience_level,
                "match": experience_match
            },
            "status": "complete"
        })
        
        # Step 6: Education Check
        education = self.extract_education(resume_text)
        reasoning_chain.append({
            "step": 6,
            "name": "Education Verification",
            "action": "Checking educational qualifications",
            "details": f"Degree: {education['degree_level']}, Fields: {', '.join(education['fields']) or 'Not specified'}",
            "data": education,
            "status": "complete"
        })
        
        # Step 7: Semantic Similarity
        tfidf_score = self.calculate_tfidf_similarity(resume_text, job_description)
        reasoning_chain.append({
            "step": 7,
            "name": "Semantic Similarity (TF-IDF)",
            "action": "Computing document similarity using TF-IDF vectors",
            "details": f"Similarity score: {tfidf_score * 100:.1f}%",
            "data": {"score": tfidf_score},
            "status": "complete"
        })
        
        # Step 8: Calculate Preliminary Score
        # Weighted scoring
        weights = {
            "skill_match": 0.40,
            "experience": 0.25,
            "education": 0.10,
            "semantic": 0.25
        }
        
        skill_score = skill_match_rate / 100
        exp_score = 1.0 if resume_years >= jd_years else resume_years / max(jd_years, 1)
        edu_score = 1.0 if education["degree_level"] != "unknown" else 0.5
        
        preliminary_score = (
            weights["skill_match"] * skill_score +
            weights["experience"] * min(exp_score, 1.0) +
            weights["education"] * edu_score +
            weights["semantic"] * tfidf_score
        ) * 100
        
        preliminary_decision = "Likely Shortlist" if preliminary_score >= 70 else "Likely Reject"
        
        reasoning_chain.append({
            "step": 8,
            "name": "Preliminary Scoring",
            "action": "Computing weighted preliminary match score",
            "details": f"Preliminary score: {preliminary_score:.1f}% -> {preliminary_decision}",
            "data": {
                "score": preliminary_score,
                "decision": preliminary_decision,
                "breakdown": {
                    "skill_contribution": skill_score * weights["skill_match"] * 100,
                    "experience_contribution": min(exp_score, 1.0) * weights["experience"] * 100,
                    "education_contribution": edu_score * weights["education"] * 100,
                    "semantic_contribution": tfidf_score * weights["semantic"] * 100
                }
            },
            "status": "complete"
        })
        
        # Step 9: Prepare for Gemini (if needed)
        reasoning_chain.append({
            "step": 9,
            "name": "Gemini AI Handoff",
            "action": "Preparing context for advanced LLM analysis",
            "details": "Packaging preliminary analysis for Gemini API",
            "status": "pending"
        })
        
        return {
            "reasoning_chain": reasoning_chain,
            "preliminary_analysis": {
                "score": preliminary_score,
                "decision": preliminary_decision,
                "skills": {
                    "matched": matched_skills,
                    "missing": missing_skills,
                    "match_rate": skill_match_rate
                },
                "experience": {
                    "years": resume_years,
                    "required": jd_years,
                    "level": experience_level
                },
                "education": education,
                "semantic_similarity": tfidf_score
            }
        }


# Singleton instance
onboard_ai = OnboardAI()
