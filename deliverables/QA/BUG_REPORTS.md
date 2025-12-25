# Bug Reports - Resume Shortlisting AI Agent

## Summary
This document tracks bugs identified during the QA phase.

---

### Bug #001: Content Security Policy (CSP) blocking API in Browser
**Status**: Resolved  
**Severity**: High  
**Date**: 2025-12-25  

**Steps to Reproduce**:
1. Open the web interface.
2. Attempt to trigger the AI analysis.
3. Check Browser Console.

**Expected vs Actual**:
- **Expected**: The request is sent successfully to the API.
- **Actual**: Browser blocks the request due to missing CSP headers.

**Resolution / Notes**:
Fixed by adding correct middleware to `run_server.py`.

---

### Bug #002: Analysis Error on Restricted Endpoints
**Status**: Resolved  
**Severity**: Medium  
**Date**: 2025-12-25  

**Steps to Reproduce**:
1. Call `/api/dev/hybrid-analyze` without an API key header.
2. `DEVELOPMENT_MODE` is set to `true`.

**Expected vs Actual**:
- **Expected**: In dev mode, the endpoint should be accessible without strict key validation or should accept a dev key.
- **Actual**: Returned "Analysis Error: Could not validate credentials".

**Resolution / Notes**:
Updated authentication middleware to bypass checks for `/api/dev/*` when `DEVELOPMENT_MODE=true`.

---

### Bug #003: LLM occasionally outputs Invalid JSON
**Status**: Monitoring  
**Severity**: Low  
**Date**: 2025-12-26  

**Steps to Reproduce**:
1. Send a very long or complex resume text to `/api/shortlist`.
2. Review the raw response from the LLM.

**Expected vs Actual**:
- **Expected**: A clean JSON string.
- **Actual**: Sometimes the LLM includes markdown backticks (```json ... ```) which breaks the parser if not handled.

**Resolution / Notes**:
Added a regex cleaner in the response handler to strip markdown formatting before parsing.

---

### Bug #004: 70% Threshold Borderline Case
**Status**: Open  
**Severity**: Low  

**Steps to Reproduce**:
1. Send a resume that is exactly a 69% match.
2. AI sometimes rounds up or hallucinates a minor skill to reach "Shortlisted".

**Expected vs Actual**:
- **Expected**: "Rejected" strictly.
- **Actual**: Occasionally returns "Shortlisted" with a score of 70/100.

**Resolution / Notes**:
Prompt engineering being refined to be more conservative.
