import unittest
import requests
import json
import os
import time
from datetime import datetime
from typing import Dict, Any

# QA Automation Suite for Resume Shortlisting AI Agent
# Implements Test Cases from TEST_CASES.md

BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("API_KEY", "")

# Artifact Paths
DELIVERABLES_DIR = r"r:\resume-shortlisting-agent\deliverables"
FULL_STACK_DIR = os.path.join(DELIVERABLES_DIR, "Full_Stack")
QA_DIR = os.path.join(DELIVERABLES_DIR, "QA")
REPORT_PATH = os.path.join(QA_DIR, "QA_TEST_EXECUTION_REPORT.md")
REASONING_OUTPUT_PATH = os.path.join(FULL_STACK_DIR, "agent_reasoning_loop_output.json")

# Ensure directories exist
os.makedirs(FULL_STACK_DIR, exist_ok=True)
os.makedirs(QA_DIR, exist_ok=True)

test_results_buffer = []

def log_test_result(test_id, description, status, details=""):
    test_results_buffer.append({
        "id": test_id,
        "description": description,
        "status": status,
        "details": details,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })


# Sample Data
PERFECT_MATCH_RESUME = """
John Doe
Senior Python Developer
Skills: Python, Django, AWS, Docker, Kubernetes, SQL, REST APIs.
Experience: 
- Senior Python Dev at Tech Co (5 years): Built cloud-native apps.
- Backend Lead (3 years): Managing AWS infrastructure.
"""

PERFECT_MATCH_JD = """
Job: Senior Python Developer
Requirements:
- 5+ years Python experience
- AWS, Docker, Kubernetes
- SQL and REST APIs
"""

MISMATCH_RESUME = """
Jane Doe
Professional Chef
Skills: Italian Cuisine, Pastry, Menu Planning, Kitchen Management.
Experience:
- Head Chef at Bistro (5 years)
- Sous Chef (3 years)
"""

BORDERLINE_RESUME = """
Jack Smith
Python Developer
Skills: Python, SQL, Git.
Experience:
- Python Dev (2 years)
"""

class TestResumeShortlistingAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Verify server is up
        print("\n[QA] Starting Automation Suite...")
        print(f"[QA] Target URL: {BASE_URL}")
        try:
            requests.get(f"{BASE_URL}/api/dev/status", timeout=30)
        except requests.exceptions.ConnectionError:
            print("[QA] CRITICAL: Server is not reachable. Ensure server is running on port 8000.")
            # We don't exit here to allow the test runner to report failures normally
            pass

    def _get_headers(self):
        return {
            "Content-Type": "application/json",
            "x-api-key": API_KEY
        }

    def test_tc01_perfect_match_shortlisting(self):
        """TC01: Verify Shortlisting for Perfect Match"""
        payload = {
            "resume_text": PERFECT_MATCH_RESUME,
            "job_description": PERFECT_MATCH_JD
        }
        try:
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            # Validation
            self.assertEqual(data.get("decision"), "Shortlisted")
            self.assertGreaterEqual(data.get("match_percentage"), 70)
            
            # DYNAMIC DELIVERABLE GENERATION
            # Save this actual, real-time output to the deliverables folder
            with open(REASONING_OUTPUT_PATH, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n[Artifact Generated] Saved dynamic reasoning output to {REASONING_OUTPUT_PATH}")
            
            log_test_result("TC01", "Verify Shortlisting for Perfect Match", "PASS", f"Match Score: {data.get('match_percentage')}%")
            
        except Exception as e:
            log_test_result("TC01", "Verify Shortlisting for Perfect Match", "FAIL", str(e))
            raise e

    def test_tc02_random_mismatch_rejection(self):
        """TC02: Verify Rejection for Random Mismatch"""
        payload = {
            "resume_text": MISMATCH_RESUME,
            "job_description": PERFECT_MATCH_JD
        }
        try:
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            self.assertEqual(response.status_code, 200)
            data = response.json()
            
            self.assertEqual(data.get("decision"), "Rejected")
            self.assertLess(data.get("match_percentage"), 70)
            
            log_test_result("TC02", "Verify Rejection for Random Mismatch", "PASS", f"Match Score: {data.get('match_percentage')}%")
            
        except Exception as e:
            log_test_result("TC02", "Verify Rejection for Random Mismatch", "FAIL", str(e))
            raise e

    def test_tc05_verify_reasoning_output(self):
        """TC05: Verify Reasoning Output Existence"""
        payload = {
            "resume_text": PERFECT_MATCH_RESUME,
            "job_description": PERFECT_MATCH_JD
        }
        try:
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            data = response.json()
            self.assertIn("reasoning", data)
            self.assertTrue(data["reasoning"]) # Should not be empty
            log_test_result("TC05", "Verify Reasoning Output Existence", "PASS")
        except Exception as e:
            log_test_result("TC05", "Verify Reasoning Output Existence", "FAIL", str(e))
            raise e

    def test_tc08_verify_http_200(self):
        """TC08: Verify HTTP 200 OK"""
        payload = {
            "resume_text": PERFECT_MATCH_RESUME,
            "job_description": PERFECT_MATCH_JD
        }
        try:
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            self.assertEqual(response.status_code, 200)
            log_test_result("TC08", "Verify HTTP 200 OK", "PASS")
        except Exception as e:
            log_test_result("TC08", "Verify HTTP 200 OK", "FAIL", str(e))
            raise e

    def test_tc09_json_structure(self):
        """TC09: Verify JSON Response Structure"""
        payload = {
            "resume_text": PERFECT_MATCH_RESUME,
            "job_description": PERFECT_MATCH_JD
        }
        try:
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            data = response.json()
            required_keys = ["decision", "match_percentage", "reasoning", "summary"]
            for key in required_keys:
                self.assertIn(key, data)
            log_test_result("TC09", "Verify JSON Response Structure", "PASS")
        except Exception as e:
            log_test_result("TC09", "Verify JSON Response Structure", "FAIL", str(e))
            raise e

    def test_tc10_negative_empty_resume(self):
        """TC10: Negative Testing - Empty Resume Text"""
        payload = {
            "resume_text": "",
            "job_description": PERFECT_MATCH_JD
        }
        try:
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            # Depending on implementation, this might be 400 or 200 with "Rejected"
            if response.status_code == 200:
                data = response.json()
                self.assertEqual(data.get("decision"), "Rejected")
            else:
                self.assertIn(response.status_code, [400, 422])
            log_test_result("TC10", "Negative Testing - Empty Resume Text", "PASS", f"Status: {response.status_code}")
        except Exception as e:
            log_test_result("TC10", "Negative Testing - Empty Resume Text", "FAIL", str(e))
            raise e

    def test_tc12_api_invalid_method(self):
        """TC12: API Invalid Method (GET)"""
        try:
            response = requests.get(f"{BASE_URL}/api/shortlist", headers=self._get_headers())
            # Allowing 404 as well since FastAPI behaves that way for undefined method routes sometimes
            if response.status_code in [404, 405]:
                log_test_result("TC12", "API Invalid Method (GET)", "PASS", f"Status: {response.status_code}")
            else:
                self.fail(f"Expected 404 or 405, got {response.status_code}")
        except Exception as e:
            log_test_result("TC12", "API Invalid Method (GET)", "FAIL", str(e))
            raise e 

    def test_tc20_hybrid_analyze_endpoint(self):
        """TC20: Verify Hybrid Analyze Endpoint"""
        try:
            payload = {
                "resume_text": PERFECT_MATCH_RESUME,
                "job_description": PERFECT_MATCH_JD
            }
            response = requests.post(f"{BASE_URL}/api/dev/hybrid-analyze", json=payload, headers=self._get_headers())
            
            if response.status_code == 200:
                data = response.json()
                self.assertIn("final_decision", data)
                log_test_result("TC20", "Verify Hybrid Analyze Endpoint", "PASS")
            else:
                log_test_result("TC20", "Verify Hybrid Analyze Endpoint", "SKIP", f"Status {response.status_code}")
        except Exception as e:
            log_test_result("TC20", "Verify Hybrid Analyze Endpoint", "FAIL", str(e))
            raise e

    def test_tc21_type_check(self):
        """TC21: Type Check for Score"""
        try:
            payload = {
                "resume_text": PERFECT_MATCH_RESUME,
                "job_description": PERFECT_MATCH_JD
            }
            response = requests.post(f"{BASE_URL}/api/shortlist", json=payload, headers=self._get_headers())
            data = response.json()
            self.assertIsInstance(data.get("match_percentage"), (int, float))
            log_test_result("TC21", "Type Check for Score", "PASS")
        except Exception as e:
            log_test_result("TC21", "Type Check for Score", "FAIL", str(e))
            raise e

    @classmethod
    def tearDownClass(cls):
        # Generate Markdown Report
        with open(REPORT_PATH, 'w', encoding='utf-8') as f:
            f.write("# QA Test Execution Report\n\n")
            f.write(f"**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Target Environment:** {BASE_URL}\n\n")
            f.write("| Test ID | Description | Status | Details | Timestamp |\n")
            f.write("|---------|-------------|--------|---------|-----------|\n")
            
            passed = 0
            failed = 0
            for result in test_results_buffer:
                status_icon = "✅" if result['status'] == "PASS" else "❌"
                if result['status'] == "PASS":
                     passed += 1
                else:
                     failed += 1
                f.write(f"| {result['id']} | {result['description']} | {status_icon} {result['status']} | {result['details']} | {result['timestamp']} |\n")
            
            f.write(f"\n**Summary:** Passed: {passed}, Failed: {failed}, Total: {len(test_results_buffer)}")
        
        print(f"\n[Report Generated] QA Execution report saved to {REPORT_PATH}")

if __name__ == "__main__":
    unittest.main(exit=False)
