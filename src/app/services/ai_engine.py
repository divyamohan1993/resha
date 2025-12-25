import re
import math
import json
import os
import hashlib
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import CountVectorizer
from ..core.logging import get_logger
from .extractor import extractor
import numpy as np

# Try importing googlesearch, handle if not installed (though we added to requirements)
try:
    from googlesearch import search
except ImportError:
    search = None

logger = get_logger(__name__)

class AIService:
    def __init__(self):
        logger.info("Loading SBERT model...")
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SBERT model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SBERT model: {e}")
            self.model = None

        # Learning Mode: Persistent Weights
        self.weights_file = "data/learning_weights.json"
        self.keyword_weights = {}
        self.load_weights()

    def load_weights(self):
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, 'r') as f:
                    self.keyword_weights = json.load(f)
                logger.info("Loaded learned weights from persistent storage.")
            except Exception as e:
                logger.error(f"Failed to load weights: {e}")
        else:
            logger.info("No existing learning data found. Starting fresh.")

    def save_weights(self):
        try:
            os.makedirs("data", exist_ok=True)
            with open(self.weights_file, 'w') as f:
                json.dump(self.keyword_weights, f)
        except Exception as e:
            logger.error(f"Failed to save learning weights: {e}")

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return " ".join(text.split())

    def get_semantic_score(self, resume_text: str, jd_text: str) -> float:
        if not self.model:
            return 0.0
        embeddings = self.model.encode([resume_text, jd_text], convert_to_tensor=True)
        score = util.cos_sim(embeddings[0], embeddings[1])
        return float(score[0][0])

    def compare_skills(self, resume_text: str, jd_text: str) -> dict:
        cv = CountVectorizer(stop_words='english', max_features=50)
        try:
            cv.fit([jd_text])
            jd_keywords = set(cv.vocabulary_.keys())
            
            cv_res = CountVectorizer(stop_words='english', max_features=100)
            cv_res.fit([resume_text])
            resume_keywords = set(cv_res.vocabulary_.keys())
            
            matched = list(jd_keywords.intersection(resume_keywords))
            missing = list(jd_keywords - resume_keywords)
            
            return {"matched": matched, "missing": missing, "resume_keywords": list(resume_keywords)}
        except:
            return {"matched": [], "missing": [], "resume_keywords": []}

    def verify_authenticity(self, entities: dict, contact: dict) -> dict:
        """
        Automated verification using web searches.
        """
        verification_score = 0.5 # Neutral start
        checks = []
        
        # 1. Email Domain Check
        email = contact.get("email", "")
        if email:
            domain = email.split('@')[-1]
            if domain in ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]:
                 checks.append(f"Public Email Provider ({domain}) - Standard")
            else:
                 checks.append(f"Professional/Custom Domain ({domain}) - +Trust")
                 verification_score += 0.1

        # 2. Web Presence Check (Using Google Search)
        # We verify if the candidate + top org exists online
        name = "Candidate" # We don't have name extraction perfectly yet, assuming 'entities' has PERSON? 
        # Extracting generic PERSON usually fails in simple models, but let's try ORG
        
        orgs = entities.get("ORG", [])
        if orgs and search:
            top_org = orgs[0]
            query = f'"{top_org}" employee LinkedIn'
            try:
                # Perform a shallow search (1-2 results) to see if the Org is real/popular
                results = list(search(query, num_results=1))
                if results:
                    checks.append(f"Organization '{top_org}' verified online.")
                    verification_score += 0.1
            except Exception as e:
                checks.append("Web search rate limited or failed.")

        return {
            "score": min(verification_score, 1.0),
            "checks": checks
        }

    def learn_feedback(self, resume_keywords: list, rating: int):
        """
        Reinforcement Learning from Human Feedback (RLHF) - Light Version
        rating: 1 (Bad) to 5 (Good)
        """
        # Calculate impact factor
        impact = 0.0
        if rating >= 4:
            impact = 0.05 # Boost
        elif rating <= 2:
            impact = -0.05 # Penalize
        
        if impact == 0: return

        # Update weights for keywords present in this resume
        for word in resume_keywords:
            current = self.keyword_weights.get(word, 1.0)
            new_weight = current + impact
            # Clamp weights to avoid runaway
            new_weight = max(0.5, min(new_weight, 2.0)) 
            self.keyword_weights[word] = new_weight
            
        self.save_weights()
        logger.info(f"Updated neural weights based on feedback (Rating: {rating})")

    def calculate_learned_bonus(self, keywords: list) -> float:
        if not keywords: return 0.0
        
        total_bonus = 0.0
        count = 0
        for word in keywords:
            weight = self.keyword_weights.get(word, 1.0)
            if weight != 1.0:
                # If weight is 1.2, bonus is 0.2. If 0.8, penalty is -0.2
                total_bonus += (weight - 1.0)
                count += 1
        
        # Normalize: don't let bonus exceed +/- 20%
        if count == 0: return 0.0
        avg_bonus = total_bonus / len(keywords) # Average influence
        return max(-0.2, min(avg_bonus, 0.2))

    def analyze_candidate(self, resume_text: str, jd_text: str) -> dict:
        logger.info("Starting Advanced AI Analysis")
        
        # 1. Metadata & Extraction
        years_exp = extractor.estimate_experience_years(resume_text)
        candidate_type = extractor.detect_candidate_type(resume_text, years_exp)
        entities = extractor.extract_entities(resume_text)
        contact = extractor.extract_contact_info(resume_text)
        
        # 2. Verification
        verification = self.verify_authenticity(entities, contact)
        
        # 3. Semantic & Keyword Analysis
        semantic_score = self.get_semantic_score(resume_text, jd_text)
        skills_analysis = self.compare_skills(resume_text, jd_text)
        
        # 4. Apply Learning (RLHF)
        learned_bonus = self.calculate_learned_bonus(skills_analysis["resume_keywords"])
        
        # 5. Final Scoring
        final_score = 0.0
        details = {}
        
        # Base Algorithm
        if candidate_type == "Fresher":
            final_score = semantic_score * 0.8 + (0.2 if len(skills_analysis["matched"]) > 0 else 0)
        else:
            exp_bonus = min(years_exp / 10.0, 0.2)
            final_score = (semantic_score * 0.7) + exp_bonus

        # Apply Learned Bonus & Verification Boost
        final_score += learned_bonus
        if verification["score"] > 0.6:
            final_score += 0.05 # Trust bonus
            
        # Normalize
        final_score = min(max(final_score, 0.0), 1.0)

        details = {
            "Years_Exp": years_exp,
            "Semantic_Match": round(semantic_score, 2),
            "Learned_Bias": round(learned_bonus, 3),
            "Trust_Score": round(verification["score"], 2)
        }

        logger.info(f"Analysis Complete. Score: {final_score}")

        return {
            "score": final_score,
            "decision": "SHORTLIST" if final_score >= 0.55 else "REJECT",
            "candidate_type": candidate_type,
            "years_experience": years_exp,
            "semantic_score": semantic_score,
            "matched_keywords": skills_analysis["matched"],
            "missing_keywords": skills_analysis["missing"],
            "entities": entities,
            "contact": contact,
            "details": details,
            "verification": verification
        }

    def hash_text(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

ai_service = AIService()
