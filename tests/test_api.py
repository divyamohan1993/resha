from fastapi.testclient import TestClient
from src.main import app
import io

client = TestClient(app)

def test_health_live():
    response = client.get("/api/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_analyze_flow():
    payload = {
        "resume_text": "Python developer with AWS and Docker experience " * 5, # Make it long enough
        "job_description": "Looking for Python AWS Docker experts " * 5
    }
    response = client.post("/api/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "decision" in data
    assert "ai_analysis" in data
    assert data["ai_analysis"]["cosine_similarity_score"] > 0

def test_analyze_file_flow():
    # Simulate a TXT file
    file_content = b"Python developer with AWS and Docker experience. Expert in Kubernetes."
    files = {'file': ('resume.txt', file_content, 'text/plain')}
    data = {'job_description': 'Looking for Python AWS Docker Kubernetes experts'}
    
    response = client.post("/api/analyze-file", files=files, data=data)
    assert response.status_code == 200
    data = response.json()
    assert "score" in data
    assert "decision" in data
    assert data["meta"]["filename"] == "resume.txt"

def test_history():
    # Trigger at least one action to ensure history
    test_analyze_file_flow()
    
    response = client.get("/api/history")
    assert response.status_code == 200
    history = response.json()
    assert isinstance(history, list)
    assert len(history) > 0
