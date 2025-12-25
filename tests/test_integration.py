from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.main import app
from src.app.core.config import get_settings

client = TestClient(app)
settings = get_settings()

def test_health_check_live():
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_unauthorized():
    # Attempt without API Key
    response = client.post("/api/analyze", json={
        "resume_text": "Sample",
        "job_description": "Sample"
    })
    assert response.status_code == 403

def test_analyze_authorized():
    # Mock AI Service to avoid actual heavy lifting/GPU usage
    with patch("src.app.services.ai_engine.ai_service.analyze_candidate") as mock_analyze:
        mock_analyze.return_value = {
            "score": 0.85,
            "decision": "Shortlist",
            "semantic_score": 0.8,
            "matched_keywords": ["Python"],
            "missing_keywords": [],
            "candidate_type": "Experienced",
            "years_experience": 5,
            "entities": {},
            "contact": {},
            "details": {}
        }
        
        # Mock hashing to avoid DB errors if DB not inited (though AuditLogger uses SQLite)
        # AuditLogger usually needs DB setup.
        # But we can patch AuditLogger too.
        
        with patch("src.app.api.endpoints.AuditLogger") as MockAudit:
            mock_audit_instance = MockAudit.return_value
            
            headers = {"x-api-key": settings.API_KEY}
            payload = {
                "resume_text": "Experienced Python Developer...",
                "job_description": "Looking for Python expert..."
            }
            
            response = client.post("/api/analyze", json=payload, headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert data["decision"] == "Shortlist"
            assert data["score"] == 0.85

def test_file_upload_too_large():
    # Mock file size check
    with patch("src.app.services.parser.parser_service.extract_text") as mock_extract:
         # Actually validation happens before parser.
         # We need to simulate a large file. TestClient allows sending files.
         # But constructing a fake 6MB file in memory is okay.
         
         large_content = b"a" * (settings.MAX_UPLOAD_SIZE + 100)
         files = {"file": ("large.txt", large_content, "text/plain")}
         headers = {"x-api-key": settings.API_KEY}
         data = {"job_description": "test"}
         
         response = client.post("/api/analyze-file", files=files, data=data, headers=headers)
         assert response.status_code == 400
         assert "File too large" in response.json()["detail"]
