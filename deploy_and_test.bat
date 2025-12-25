@echo off
setlocal EnableDelayedExpansion
title Resha - Deployment and Test Suite

echo ========================================================
echo  Resha - One-Click Deployment and Test
echo ========================================================
echo.

:: 1. Environment Setup
echo [1/6] Setting up Environment...
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

echo [INFO] Installing/Updating Dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    rem pause
    exit /b 1
)

echo [INFO] checking Spacy model...
python -m spacy download en_core_web_sm >nul 2>&1

:: 2. Ollama Setup
echo.
echo [2/6] Checking AI Engine (Ollama)...

set OLLAMA_CMD=ollama
where ollama >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    :: Check default install location
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
        echo [INFO] Found Ollama at default location.
    ) else (
        echo [WARNING] Ollama not found in PATH.
        echo [INFO] Attempting to install via Winget...
        winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements --silent
        if %ERRORLEVEL% NEQ 0 (
            echo [ERROR] Failed to install Ollama. Please install manually.
        ) else (
            echo [INFO] Ollama installed.
            ping 127.0.0.1 -n 6 >nul
            if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
                set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
            )
        )
    )
)

echo [INFO] Using Ollama command: "%OLLAMA_CMD%"
echo [INFO] Ensuring Ollama service is running...
start /B "" "%OLLAMA_CMD%" serve >nul 2>&1
ping 127.0.0.1 -n 6 >nul

echo [INFO] Pulling recommended models...
echo        - Pulling phi3:mini...
"%OLLAMA_CMD%" pull phi3:mini
echo        - Pulling qwen2.5:3b...
"%OLLAMA_CMD%" pull qwen2.5:3b
echo [INFO] AI Models Ready.

:: 3. Server Startup
echo.
echo [3/6] Starting Application Server...
:: Clean up port 8000 first
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %%a >nul 2>&1

:: Start server in background
start /B python run_server.py --skip-install > server.log 2>&1
echo [INFO] Server launching in background...
echo [INFO] Waiting for server to become responsive...

:: Wait loop
set /a retries=0
:wait_loop
    ping 127.0.0.1 -n 3 >nul
curl -s http://localhost:8000/docs >nul
if %ERRORLEVEL% NEQ 0 (
    set /a retries+=1
    if !retries! LSS 15 (
        goto wait_loop
    ) else (
        echo [ERROR] Server failed to start. Check server.log.
        type server.log
        goto cleanup
    )
)
echo [OK] Server is UP!

:: 4. Run Verification Suite
echo.
echo [4/6] Running Verification Suite (Task B)...
echo ============================================
python tests/test_task_b_verification.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Verification Tests Failed.
    set TEST_FAIL=1
) else (
    echo [PASS] Verification Tests Passed.
)

:: 5. Run QA Automation Suite
echo.
echo [5/6] Running QA Automation Suite...
echo ============================================
python deliverables\QA\automation\qa_automation_suite.py
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] QA Automation Failed.
    set TEST_FAIL=1
) else (
    echo [PASS] QA Automation Passed.
)

:: 6. Run Validations and Pytest
echo.
echo [6/6] Running Unit ^& Integration Tests...
echo ============================================
python -m pytest tests/
if %ERRORLEVEL% NEQ 0 (
    echo [FAIL] Unit Tests Failed.
    set TEST_FAIL=1
) else (
    echo [PASS] Unit Tests Passed.
)

:cleanup
echo.
echo ========================================================
echo  Stopping Server...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %%a >nul 2>&1
echo ========================================================

if defined TEST_FAIL (
    echo [FATAL] Some tests failed. Please review the logs above.
    exit /b 1
) else (
    echo [SUCCESS] System Ready for Deployment! 
    echo           All tests passed. Server starts correctly.
    echo           run "start.bat" to launch the system for users.
    exit /b 0
)
