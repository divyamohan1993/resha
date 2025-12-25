#!/usr/bin/env python3
"""
============================================================================
TASK B: AI-Powered Resume Shortlisting Agent - Verification Test Script
============================================================================

This script verifies that the resume shortlisting system meets all Task B requirements:

REQUIREMENTS CHECKLIST:
1. ✓ Takes a Resume (plain text) as input
2. ✓ Takes a Job Description (JD) as input
3. ✓ Uses an AI model + prompt template to evaluate the match
4. ✓ Returns a final decision: Shortlisted (if resume matches JD) or Rejected (if not)
5. ✓ Returns a JSON summary explaining why it was shortlisted or rejected

AGENT REQUIREMENTS:
- Uses strict system instruction as specified
- Uses 70% threshold for shortlisting decisions
- Analyzes: skills, experience, projects, education, and role fit

ENDPOINTS TESTED:
- POST /api/shortlist - Main Task B endpoint (Gemini AI)
- POST /api/shortlist/reasoning-loop - Full reasoning loop JSON output
- POST /api/analyze - Original SBERT-based endpoint (for comparison)

Run with: python test_task_b_verification.py
"""

import os
import sys
import json
import time
import requests
from typing import Tuple, Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    # Load from project root .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    load_dotenv(env_path)
except ImportError:
    pass

# Configuration - reads from environment (loaded from .env)
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")  # Must be set in .env

# Test Data
SAMPLE_RESUME_MATCH = """
John Smith
Senior Software Engineer
Email: john.smith@example.com | Phone: (555) 123-4567

PROFESSIONAL SUMMARY
Highly skilled software engineer with 7+ years of experience in Python, AWS, Docker, 
and Kubernetes. Proven track record of designing and implementing scalable microservices 
architectures. Strong background in machine learning and data engineering.

SKILLS
- Programming: Python, JavaScript, Go, SQL
- Cloud: AWS (EC2, S3, Lambda, ECS, EKS), GCP
- DevOps: Docker, Kubernetes, Terraform, CI/CD, Jenkins
- Databases: PostgreSQL, MongoDB, Redis
- ML/AI: TensorFlow, PyTorch, Scikit-learn

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2020 - Present
- Led development of microservices handling 10M+ daily requests
- Designed and deployed ML pipelines on AWS SageMaker
- Mentored team of 5 junior developers

Software Engineer | DataSystems LLC | 2017 - 2020
- Built RESTful APIs using Python Flask and FastAPI
- Implemented containerization strategy using Docker and Kubernetes
- Reduced deployment time by 60% through CI/CD automation

EDUCATION
Master of Science in Computer Science | MIT | 2017
Bachelor of Science in Computer Science | Stanford University | 2015

PROJECTS
- Open source contributor to Kubernetes and Docker projects
- Personal ML project: Resume screening automation using NLP
"""

SAMPLE_RESUME_NO_MATCH = """
Sarah Johnson
Licensed Cosmetologist
Email: sarah.j@email.com | Phone: (555) 987-6543

PROFESSIONAL SUMMARY
Creative and passionate cosmetologist with 5 years of experience in hair styling, 
coloring, and customer service. Expert in the latest beauty trends and techniques.

SKILLS
- Hair cutting and styling
- Color treatments and highlights
- Customer service excellence
- Salon management
- Product knowledge

EXPERIENCE
Senior Stylist | Glamour Salon | 2019 - Present
- Manage 50+ regular clients
- Train new stylists on cutting techniques
- Increased salon revenue by 25%

Junior Stylist | Beauty Bar | 2017 - 2019
- Performed haircuts, coloring, and treatments
- Maintained clean and organized workstation

EDUCATION
Cosmetology License | State Beauty Academy | 2017
High School Diploma | Central High School | 2015
"""

SAMPLE_JD = """
Senior Python Developer - Cloud Infrastructure

About the Role:
We are seeking a Senior Python Developer to join our cloud infrastructure team. 
You will be responsible for building and maintaining scalable microservices.

Requirements:
- 5+ years of experience with Python
- Strong experience with AWS services (EC2, S3, Lambda, ECS)
- Proficiency in Docker and Kubernetes
- Experience with CI/CD pipelines
- Knowledge of RESTful API design
- Familiarity with SQL and NoSQL databases

Nice to Have:
- Experience with machine learning frameworks
- Contributions to open source projects
- Master's degree in Computer Science

What We Offer:
- Competitive salary
- Remote work options
- Professional development budget
"""


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with color coding."""
    status = "[PASS]" if passed else "[FAIL]"
    print(f"\n{status}: {test_name}")
    if details:
        print(f"   Details: {details}")


def test_health_check() -> bool:
    """Test if the API is running."""
    try:
        response = requests.get(f"{BASE_URL}/api/health/live", timeout=10)
        return response.status_code == 200
    except Exception as e:
        print(f"   Error: {e}")
        return False


def test_shortlist_endpoint_match() -> Tuple[bool, dict]:
    """
    Test the /api/shortlist endpoint with a matching resume.
    Expected: Shortlisted decision with high match percentage.
    """
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "resume_text": SAMPLE_RESUME_MATCH,
        "job_description": SAMPLE_JD
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/shortlist", 
            json=payload, 
            headers=headers,
            timeout=60  # LLM calls may take time
        )
        
        if response.status_code != 200:
            return False, {"error": f"Status {response.status_code}: {response.text}"}
        
        data = response.json()
        return True, data
        
    except Exception as e:
        return False, {"error": str(e)}


def test_shortlist_endpoint_no_match() -> Tuple[bool, dict]:
    """
    Test the /api/shortlist endpoint with a non-matching resume.
    Expected: Rejected decision with low match percentage.
    """
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "resume_text": SAMPLE_RESUME_NO_MATCH,
        "job_description": SAMPLE_JD
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/shortlist", 
            json=payload, 
            headers=headers,
            timeout=60
        )
        
        if response.status_code != 200:
            return False, {"error": f"Status {response.status_code}: {response.text}"}
        
        data = response.json()
        return True, data
        
    except Exception as e:
        return False, {"error": str(e)}


def test_reasoning_loop() -> Tuple[bool, dict]:
    """
    Test the /api/shortlist/reasoning-loop endpoint.
    Expected: Full reasoning loop JSON with all stages.
    """
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "resume_text": SAMPLE_RESUME_MATCH,
        "job_description": SAMPLE_JD
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/shortlist/reasoning-loop", 
            json=payload, 
            headers=headers,
            timeout=60
        )
        
        if response.status_code != 200:
            return False, {"error": f"Status {response.status_code}: {response.text}"}
        
        data = response.json()
        return True, data
        
    except Exception as e:
        return False, {"error": str(e)}


def test_original_analyze_endpoint() -> Tuple[bool, dict]:
    """
    Test the original /api/analyze endpoint (SBERT-based).
    For comparison with the new LLM-based endpoint.
    """
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "resume_text": SAMPLE_RESUME_MATCH,
        "job_description": SAMPLE_JD
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/analyze", 
            json=payload, 
            headers=headers,
            timeout=30
        )
        
        if response.status_code != 200:
            return False, {"error": f"Status {response.status_code}: {response.text}"}
        
        data = response.json()
        return True, data
        
    except Exception as e:
        return False, {"error": str(e)}


def validate_task_b_requirements(shortlist_response: dict) -> dict:
    """
    Validate that the response meets all Task B requirements.
    """
    requirements = {
        "has_decision": False,
        "decision_is_valid": False,
        "has_match_percentage": False,
        "has_reasoning": False,
        "has_summary": False,
        "uses_70_percent_threshold": False
    }
    
    # Check decision
    if "decision" in shortlist_response:
        requirements["has_decision"] = True
        if shortlist_response["decision"] in ["Shortlisted", "Rejected"]:
            requirements["decision_is_valid"] = True
    
    # Check match percentage
    if "match_percentage" in shortlist_response:
        requirements["has_match_percentage"] = True
        match_pct = shortlist_response["match_percentage"]
        decision = shortlist_response.get("decision")
        
        # Validate 70% threshold logic
        if (match_pct >= 70 and decision == "Shortlisted") or \
           (match_pct < 70 and decision == "Rejected"):
            requirements["uses_70_percent_threshold"] = True
    
    # Check reasoning
    if "reasoning" in shortlist_response and shortlist_response["reasoning"]:
        requirements["has_reasoning"] = True
    
    # Check summary
    if "summary" in shortlist_response and shortlist_response["summary"]:
        requirements["has_summary"] = True
    
    return requirements


def run_all_tests():
    """Run all verification tests."""
    print_header("TASK B: Resume Shortlisting Agent - Verification Tests")
    print(f"Target API: {BASE_URL}")
    print(f"Testing at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "task_b_compliant": False
    }
    
    # Test 1: Health Check
    print_header("Test 1: API Health Check")
    results["total_tests"] += 1
    if test_health_check():
        results["passed"] += 1
        print_result("API Health Check", True)
    else:
        results["failed"] += 1
        print_result("API Health Check", False, "API is not running or not accessible")
        print("\n[WARNING] Cannot proceed with further tests. Start the API first!")
        print("   Run: uvicorn src.main:app --reload")
        return results
    
    # Test 2: Shortlist Endpoint - Matching Resume
    print_header("Test 2: /api/shortlist - Matching Resume (Should be Shortlisted)")
    results["total_tests"] += 1
    success, data = test_shortlist_endpoint_match()
    
    if success:
        print_result("API Call Successful", True)
        print(f"\n   Response:")
        print(f"   - Decision: {data.get('decision')}")
        print(f"   - Match %: {data.get('match_percentage')}%")
        print(f"   - Summary: {data.get('summary', 'N/A')[:100]}...")
        
        if data.get("decision") == "Shortlisted":
            results["passed"] += 1
            print_result("Matching Resume Shortlisted", True)
        else:
            results["failed"] += 1
            print_result("Matching Resume Shortlisted", False, 
                        f"Got '{data.get('decision')}' instead of 'Shortlisted'")
    else:
        results["failed"] += 1
        print_result("Shortlist Endpoint (Match)", False, data.get("error", "Unknown error"))
    
    # Test 3: Shortlist Endpoint - Non-Matching Resume
    print_header("Test 3: /api/shortlist - Non-Matching Resume (Should be Rejected)")
    results["total_tests"] += 1
    success, data = test_shortlist_endpoint_no_match()
    
    if success:
        print_result("API Call Successful", True)
        print(f"\n   Response:")
        print(f"   - Decision: {data.get('decision')}")
        print(f"   - Match %: {data.get('match_percentage')}%")
        print(f"   - Summary: {data.get('summary', 'N/A')[:100]}...")
        
        if data.get("decision") == "Rejected":
            results["passed"] += 1
            print_result("Non-Matching Resume Rejected", True)
        else:
            results["failed"] += 1
            print_result("Non-Matching Resume Rejected", False,
                        f"Got '{data.get('decision')}' instead of 'Rejected'")
    else:
        results["failed"] += 1
        print_result("Shortlist Endpoint (No Match)", False, data.get("error", "Unknown error"))
    
    # Test 4: Validate Task B Requirements
    print_header("Test 4: Task B Requirements Validation")
    if success and data:
        requirements = validate_task_b_requirements(data)
        all_met = all(requirements.values())
        
        for req, met in requirements.items():
            results["total_tests"] += 1
            if met:
                results["passed"] += 1
                print_result(f"Requirement: {req}", True)
            else:
                results["failed"] += 1
                print_result(f"Requirement: {req}", False)
        
        results["task_b_compliant"] = all_met
    
    # Test 5: Reasoning Loop Endpoint
    print_header("Test 5: /api/shortlist/reasoning-loop - Agent Reasoning JSON")
    results["total_tests"] += 1
    success, data = test_reasoning_loop()
    
    if success:
        results["passed"] += 1
        print_result("Reasoning Loop Endpoint", True)
        print(f"\n   Reasoning Loop Stages:")
        for stage in ["stage_1_input_processing", "stage_2_prompt_construction", 
                      "stage_3_llm_inference", "stage_4_decision_output"]:
            if stage in data:
                print(f"   [OK] {stage}")
            else:
                print(f"   [MISSING] {stage}")
    else:
        results["failed"] += 1
        print_result("Reasoning Loop Endpoint", False, data.get("error", "Unknown error"))
    
    # Test 6: Compare with Original Endpoint
    print_header("Test 6: Comparison - Original /api/analyze vs New /api/shortlist")
    results["total_tests"] += 1
    success, original_data = test_original_analyze_endpoint()
    
    if success:
        results["passed"] += 1
        print_result("Original Analyze Endpoint", True)
        print(f"\n   Original Endpoint (SBERT-based):")
        print(f"   - Decision: {original_data.get('decision')}")
        print(f"   - Score: {original_data.get('score')}")
        print(f"   - Model: {original_data.get('meta', {}).get('model', 'N/A')}")
        print(f"\n   New Shortlist Endpoint (Gemini LLM):")
        print(f"   - Uses strict system instruction with 70% threshold")
        print(f"   - Provides detailed reasoning in JSON format")
        print(f"   - Follows exact Task B requirements")
    else:
        results["failed"] += 1
        print_result("Original Analyze Endpoint", False, original_data.get("error", "Unknown error"))
    
    # Final Summary
    print_header("FINAL RESULTS")
    print(f"\n   Total Tests: {results['total_tests']}")
    print(f"   Passed: {results['passed']}")
    print(f"   Failed: {results['failed']}")
    print(f"   Pass Rate: {(results['passed'] / results['total_tests'] * 100):.1f}%")
    print(f"\n   Task B Compliant: {'YES' if results['task_b_compliant'] else 'NO'}")
    
    if results['task_b_compliant']:
        print("\n   SUCCESS! The system meets all Task B requirements!")
        print("   The new /api/shortlist endpoint provides:")
        print("   - Resume (plain text) input")
        print("   - Job Description input")
        print("   - Gemini AI evaluation with strict prompt template")
        print("   - Shortlisted/Rejected decision (70% threshold)")
        print("   - JSON summary with reasoning")
    
    return results


def save_sample_output(filename: str = "sample_output.json"):
    """Save a sample output for documentation."""
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "resume_text": SAMPLE_RESUME_MATCH,
        "job_description": SAMPLE_JD
    }
    
    try:
        # Get shortlist response
        response = requests.post(
            f"{BASE_URL}/api/shortlist", 
            json=payload, 
            headers=headers,
            timeout=60
        )
        shortlist_data = response.json() if response.status_code == 200 else {"error": response.text}
        
        # Get reasoning loop
        response = requests.post(
            f"{BASE_URL}/api/shortlist/reasoning-loop", 
            json=payload, 
            headers=headers,
            timeout=60
        )
        reasoning_data = response.json() if response.status_code == 200 else {"error": response.text}
        
        output = {
            "shortlist_response": shortlist_data,
            "reasoning_loop": reasoning_data,
            "input": {
                "resume_text": SAMPLE_RESUME_MATCH[:200] + "...",
                "job_description": SAMPLE_JD[:200] + "..."
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n[OK] Sample output saved to: {filename}")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to save sample output: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Task B Verification Tests")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key (reads from .env if not provided)")
    parser.add_argument("--save-output", action="store_true", help="Save sample output to JSON")
    
    args = parser.parse_args()
    
    BASE_URL = args.url
    # Use provided API key or fall back to environment variable
    if args.api_key:
        API_KEY = args.api_key
    elif not API_KEY:
        print("[ERROR] API_KEY not set. Please set it in .env or provide --api-key argument.")
        sys.exit(1)
    
    results = run_all_tests()
    
    if args.save_output:
        save_sample_output()
    
    # Exit with appropriate code
    sys.exit(0 if results["failed"] == 0 else 1)

