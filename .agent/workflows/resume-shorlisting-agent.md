---
description: resume-shortlisting-agent-creator
---

# Role: Principal Cloud Architect & DevSecOps Lead
# Target System: Ubuntu (WSL) / Linux Server
# Project: "TalentScout Pro" - Secure Resume Analysis Microservice

**OBJECTIVE:**
Engineer a "Military-Grade" deployment script (`deploy.sh`) for a Resume Shortlisting Microservice. The solution must be **self-contained**, **secure by design**, and compliant with modern enterprise standards (SBOM, Structured Logging, Health Checks).

**CONSTRAINTS:**
1.  **Single Artifact:** The output must be ONE bash script that orchestrates the entire lifecycle: Environment Check -> Security Scan -> Build -> Test -> Deploy.
2.  **Zero-Touch Deployment:** No manual intervention. It must run on a fresh Ubuntu install (assuming python3 is present).
3.  **Modern Tech Stack:**
    * **Runtime:** Python 3.10+ (via venv).
    * **API Framework:** FastAPI (Async, Pydantic V2 for strict validation).
    * **Server:** Gunicorn with Uvicorn Workers (Production standard, not just Uvicorn).
    * **Frontend:** Alpine.js + TailwindCSS (Modern, lightweight reactivity without node_modules bloat).
    * **Observability:** Structured JSON Logging (for ELK/Splunk compatibility).
    * **Security:** Input sanitization, basic rate-limiting middleware, and SBOM generation.

---

## 1. COMPLIANCE & SECURITY FEATURES (The "Enterprise" Difference)

### **A. Software Bill of Materials (SBOM)**
The script must automatically generate a `sbom.json` file in CycloneDX or SPDX format.
* *Implementation:* Use a Python script to scan the active `venv` and dump installed package versions, licenses, and hashes into a JSON file for security auditing.

### **B. Structured Logging**
Standard `print` statements are forbidden.
* *Implementation:* Configure a custom Python Logger that outputs strictly JSON Lines:
    `{"level": "INFO", "timestamp": "2024-...", "service": "talent-scout", "event": "resume_analyzed", "decision": "shortlisted"}`

### **C. Health & Readiness Probes**
Implement standard Kubernetes-style probes:
* `GET /health/live`: Returns 200 OK (Service is running).
* `GET /health/ready`: Returns 200 OK (Models loaded, DB connected).

---

## 2. FUNCTIONAL LOGIC (User Story 2 - Advanced)

**Architecture:** Use a **Service Layer Pattern**. Keep the Controller (API Routes) separate from the Business Logic (Resume Parsing).

**The Workflow:**
1.  **Ingest:** API receives Resume (Text) + Job Description (Text).
2.  **Validate:** Pydantic models ensure minimum character counts and reject injection attempts.
3.  **Analyze:**
    * **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency).
    * **Similarity:** Cosine Similarity calculation.
    * **Keyword Match:** Extract "hard skills" using set operations.
4.  **Decide:**
    * Score >= 0.70 (70%): **SHORTLIST**.
    * Score < 0.70: **REJECT**.
5.  **Audit:** Write transaction to `audit.db` (SQLite) and emit JSON log.

---

## 3. THE `deploy.sh` SCRIPT SPECIFICATION

Write a robust `bash` script with the following sequence:

### **Phase 1: Pre-Flight**
* `set -e` (Exit immediately on error).
* Check for Python 3.
* Create project structure: `/src`, `/logs`, `/config`, `/static`.

### **Phase 2: Supply Chain (Dependencies)**
* Create `venv`.
* Install: `fastapi`, `uvicorn`, `gunicorn`, `scikit-learn`, `python-multipart`, `pydantic-settings`.
* **Action:** Immediately generate `sbom.json` listing these versions.

### **Phase 3: Code Generation (Here-Docs)**
* **`conf.py`**: Centralized settings (Environment variables).
* **`logger.py`**: The JSON logging configuration.
* **`main.py`**: The FastAPI application.
* **`index.html`**: The Frontend.
    * *UI:* Use a "Glassmorphism" design with Tailwind.
    * *Logic:* Use Alpine.js `x-data` to handle state (loading spinners, error handling, result display) cleanly.

### **Phase 4: Smoke Test**
* The script must launch the server in the background, `curl` the `/health/live` endpoint.
    * If 200 OK: Print Green "DEPLOYMENT SUCCESSFUL".
    * If fail: Print Red Logs and exit.

### **Phase 5: Execution**
* Launch Gunicorn with 4 worker processes bound to 0.0.0.0:8000.

---

## FINAL INSTRUCTION TO AGENT

**Action:** Generate the comprehensive `deploy.sh` script.
**Detail:** Ensure the `main.py` includes the SBOM generation logic and the JSON logger. The frontend must look professional (Dark Mode).
**Output:** ONLY the bash script code.

**BEGIN GENERATION.**