# QA Test Plan - Resume Shortlisting AI Agent

## 1. Objective
The objective of this test plan is to validate the functionality, accuracy, and reliability of the AI-Powered Resume Shortlisting Agent. This ensures that the system correctly analyzes resumes against job descriptions, strictly follows the 70% match threshold, and provides accurate verification summaries without generating false information.

## 2. Scope
The scope of this testing includes:
- **Functional Testing**: Verifying the `/api/shortlist` and `/api/analyze` endpoints.
- **AI Behavior Testing**: confirming the AI adheres to the system instructions and the 70% threshold rule.
- **Negative Testing**: Handling invalid inputs (e.g., empty text, non-text files).
- **API Testing**: Validating request/response formats, status codes, and latency.
- **Security Testing**: Basic input validation and ensuring no PII leakage in logs (if applicable).

## 3. Out of Scope
- Performance/Load testing under extreme concurrency (beyond basic latency checks).
- Testing of third-party LLM provider uptime (Gemini/Ollama) beyond handling errors gracefully.
- UI/UX testing (as the primary deliverable is the backend agent).

## 4. Test Strategy
- **Manual API Testing**: Using Postman to send crafted requests and inspect JSON responses.
- **Automated Testing**: Using Python `pytest` scripts (`tests/test_task_b_verification.py`) to run regression suites.
- **Exploratory Testing**: Ad-hoc testing with edge-case resumes and tailored JDs to try and "trick" the AI.
- **Verification of AI Logic**: Manually reviewing a sample of AI decisions to ensure the "Reason" field aligns with the provided inputs.

## 5. Risks & Assumptions
### Risks
- **LLM Hallucinations**: The AI might invent skills or misinterpret vague resume points.
- **Latency**: External LLM calls might time out or be slow.
- **Rate Limits**: API keys for cloud providers might hit usage limits during testing.

### Assumptions
- The "Resume" and "Job Description" are provided as plain text strings.
- The external LLM service (Gemini or Local Ollama) is operational.
- A "match" is subjective, but the AI's reasoning should be logical and defensible based on the text.

## 6. Entry/Exit Criteria
### Entry Criteria
- The backend server is running and accessible (e.g., `http://localhost:8000`).
- API endpoints are documented.
- Test data (sample resumes and JDs) is available.

### Exit Criteria
- All Critical and High severity bugs are resolved.
- 95% of planned test cases pass.
- The AI consistently respects the "Shortlisted" vs "Rejected" output format.
- No critical security vulnerabilities are found.
