# Test Cases - Resume Shortlisting AI Agent

## Overview
**Total Test Cases**: 22
**Types Covered**: Functional, API, Negative, AI Behavior, Security

| ID | Type | Test Case Description | Pre-conditions | Test Steps | Expected Result |
|----|------|-----------------------|----------------|------------|-----------------|
| **TC01** | Functional | Verify Shortlisting for Perfect Match | App running | Send Resume with 100% of JD skills to `/api/shortlist` | Decision: "Shortlisted", Match Score > 90% |
| **TC02** | Functional | Verify Rejection for Random Mismatch | App running | Send a Cooking Resume for a Software Engineering JD | Decision: "Rejected", Match Score < 20% |
| **TC03** | AI Behavior | Verify 70% Threshold Logic (Borderline Pass) | App running | Send Resume with approx 75% of JD skills | Decision: "Shortlisted", Match Score is 70-80% |
| **TC04** | AI Behavior | Verify 70% Threshold Logic (Borderline Fail) | App running | Send Resume with approx 60% of JD skills | Decision: "Rejected", Match Score < 70% |
| **TC05** | Functional | Verify Reasoning Output | App running | Send any valid request | JSON contains "Reason" field explaining decision |
| **TC06** | Functional | Verify Skills Matched List | App running | Send valid request | JSON contains "skills_matched" list |
| **TC07** | Functional | Verify Skills Missing List | App running | Send request with missing skills | JSON contains "skills_missing" list populated correctly |
| **TC08** | API | Verify HTTP 200 OK | App running | Send valid POST request to `/api/shortlist` | Response Status 200 OK |
| **TC09** | API | Verify JSON Response Structure | App running | Send valid request | Response is valid JSON with specific keys (decision, score, etc.) |
| **TC10** | Negative | Empty Resume Text | App running | Send request with empty `resume_text` | Error message / "Rejected" with reason stating empty input |
| **TC11** | Negative | Empty Job Description | App running | Send request with empty `job_description` | Error message / "Rejected" with reason stating empty input |
| **TC12** | API | Invalid HTTP Method | App running | Send GET request to `/api/shortlist` | Response Status 405 Method Not Allowed |
| **TC13** | Negative | Malformed JSON Body | App running | Send POST with broken JSON syntax | Response Status 422 Unprocessable Entity (or 400 Bad Request) |
| **TC14** | AI Behavior | Hallucination Check (Fake Skills) | App running | Send JD requiring "Flux Capacitor Maintenance" | AI identifies missing skill "Flux Capacitor Maintenance" without hallucinating experience |
| **TC15** | Security | Script Injection in Resume | App running | Send Resume containing `<script>alert('xss')</script>` | API handles it as text, does not execute script, sanitizes output if displayed |
| **TC16** | Security | SQL Injection in Input | App running | Send Resume containing `' OR '1'='1` | API treats it as literal text, no database errors |
| **TC17** | Performance | Response Time Check | App running | Send standard request | Response received within reasonable time (e.g., < 10s for LLM) |
| **TC18** | AI Behavior | Context Awareness (Seniority) | App running | Send Junior Resume for Senior Architect role | "Rejected" due to lack of experience years, even if skills match |
| **TC19** | AI Behavior | Context Awareness (Domain) | App running | Send "Java Developer" resume for "JavaScript Developer" JD | AI distinguishes Java vs JavaScript, potentially rejects or scores low |
| **TC20** | Endpoint | Verify Hybrid Analyze Endpoint | App running | Send request to `/api/dev/hybrid-analyze` | Response includes consensus detail from multiple models (if active) |
| **TC21** | Type Check | Boolean/Number Data Types | App running | Verify `match_score` is a number (0-100) and not a string | JSON types are correct (int/float for score) |
| **TC22** | Consistency | Idempotency Check | App running | Send exactly same request twice | Results (Decision/Score) are identical or extremely similar |
