import re
import spacy
from typing import Dict, List, Any
from ..core.logging import get_logger

logger = get_logger(__name__)

# Load Spacy Model (Cached)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("Spacy model not found. downloading...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class ResumeExtractor:
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        doc = nlp(text)
        entities = {
            "ORG": [],
            "PERSON": [],
            "GPE": [],
            "DATE": [],
            "EDU": [] # Custom if possible, but standard NER models don't have EDU
        }
        
        for ent in doc.ents:
            if ent.label_ in entities:
                entities[ent.label_].append(ent.text)
                
        # Dedupe
        for k in entities:
            entities[k] = list(set(entities[k]))
            
        return entities

    def estimate_experience_years(self, text: str) -> float:
        # Regex for "X years experience", "X+ years", etc.
        # This is heuristic.
        regex = r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)"
        matches = re.findall(regex, text, re.IGNORECASE)
        if not matches:
            return 0.0
        
        # Take the maximum number found that is reasonable (< 50)
        years = [float(x) for x in matches if float(x) < 50]
        return max(years) if years else 0.0

    def detect_candidate_type(self, text: str, years_exp: float) -> str:
        # Heuristic: If exp < 2 years, Fresher.
        # Or if "Student", "Graduate", "Fresher" in text likely Fresher.
        
        if years_exp > 2:
            return "Experienced"
            
        lower_text = text.lower()
        fresher_keywords = ["fresh graduate", "recent graduate", "final year", "student", "internship"]
        if any(k in lower_text for k in fresher_keywords):
            return "Fresher"
            
        return "Fresher" if years_exp <= 2 else "Experienced"

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        email_regex = r"[\w\.-]+@[\w\.-]+\.\w+"
        phone_regex = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        
        emails = re.findall(email_regex, text)
        phones = re.findall(phone_regex, text)
        
        return {
            "email": emails[0] if emails else None,
            "phone": phones[0] if phones else None
        }

extractor = ResumeExtractor()
