@echo off
SETLOCAL EnableExtensions EnableDelayedExpansion

:: Change directory to script location
cd /d "%~dp0"

echo ========================================================
echo  Resha - Enterprise Setup & Launcher
echo ========================================================
echo.

:: 1. Check for Python
echo [1/5] Checking Python installation...
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not found in PATH!
    echo         Please install Python 3.11+ and check "Add Python to PATH"
    pause
    exit /b 1
)
python --version

:: 2. Create/Check Virtual Environment
echo.
echo [2/5] Checking Virtual Environment (venv)...
IF NOT EXIST "venv" (
    echo        Creating new virtual environment...
    python -m venv venv
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create venv. Check permissions.
        pause
        exit /b 1
    )
    echo        Venv created successfully.
) ELSE (
    echo        Using existing venv.
)

:: 3. dependencies
echo.
echo [3/5] Installing/Updating Dependencies...
:: Activate venv for this session
call venv\Scripts\activate.bat

python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: 4. Configuration
echo.
echo [4/5] Checking Configuration...
IF NOT EXIST ".env" (
    echo        Creating .env file from template...
    IF EXIST ".env.example" (
        copy .env.example .env >nul
    ) ELSE (
        echo        Creating default .env...
        (
            echo APP_NAME="Resha"
            echo VERSION="3.0.0"
            echo DEBUG=false
            echo DEVELOPMENT_MODE=true
            echo API_KEY=dev_key_change_in_prod
            echo OLLAMA_BASE_URL=http://localhost:11434
            echo LOCAL_LLM_ENABLED=true
            echo GEMINI_API_KEY=
        ) > .env
    )
)

:: Auto-detect Ollama
where ollama >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo        [WARNING] Ollama not found. Local LLM features may not work.
    echo        Please install form https://ollama.ai if you want local AI.
) ELSE (
    echo        Ollama detected. Ensuring service is up...
    :: We can't easily check if it's running without blocking, 
    :: but the server handles connection errors gracefully.
)

:: 5. Run Server
echo.
echo [5/5] Starting Application...
echo ========================================================
echo  Server running at: http://localhost:8000
echo  Documentation at:  http://localhost:8000/docs
echo  Dev Dashboard at:  http://localhost:8000/dev.html
echo ========================================================
echo.

python run_server.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server crashed or stopped unexpectedly.
    pause
)
