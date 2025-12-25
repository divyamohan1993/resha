#!/bin/bash
set -e

# ==============================================================================
# AUTONOMOUS AGENT DEPLOYMENT ORCHESTRATOR
# ==============================================================================

# ANSI Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[INFO] $(date '+%H:%M:%S') - $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $(date '+%H:%M:%S') - $1${NC}"; }
error() { echo -e "${RED}[ERROR] $(date '+%H:%M:%S') - $1${NC}"; exit 1; }
divider() { echo -e "${CYAN}============================================================${NC}"; }

# Detect Environment
IS_WSL=false
if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null; then
    IS_WSL=true
fi

# ==============================================================================
# 1. SELF-HEALING SYSTEM PREP (Isolated Venv)
# ==============================================================================
divider
log "Initializing Isolated Orchestration Environment..."

# Ensure Python3 and Venv exist (Install if missing on Linux)
if ! command -v python3 &> /dev/null; then
    if [ "$IS_WSL" = true ]; then
        error "Python3 is missing in WSL. Please run 'sudo apt install python3 python3-venv' inside WSL."
    else
        warn "Python3 missing. Attempting auto-install..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update -qq && sudo apt-get install -y -qq python3 python3-venv python3-pip
        elif command -v yum &> /dev/null; then
            sudo yum install -y python3 python3-devel
        else
            error "Cannot auto-install Python3. Package manager not found."
        fi
    fi
fi

# Create Isolated Virtual Environment for Deployment Tools
# This ensures we don't mess with system python or require user interaction
DEPLOY_VENV=".agent/deploy_venv"
if [ ! -d "$DEPLOY_VENV" ]; then
    log "Creating isolated venv '$DEPLOY_VENV'..."
    mkdir -p .agent
    python3 -m venv "$DEPLOY_VENV"
fi

# Activate Venv for this script execution
source "$DEPLOY_VENV/bin/activate"

# Update tools in venv
pip install --disable-pip-version-check --quiet --upgrade pip

# ==============================================================================
# 2. DOCKER AUTO-CONFIGURATION
# ==============================================================================
divider
log "Checking Container Runtime..."

ensure_docker_linux() {
    if ! command -v docker &> /dev/null; then
        warn "Docker not found. Attempting AUTO-INSTALL (Production Mode)..."
        curl -fsSL https://get.docker.com | sh
        sudo usermod -aG docker $USER || true
        log "Docker installed. Starting service..."
        sudo systemctl start docker
        sudo systemctl enable docker
    fi
    
    # Wait for socket
    RETRIES=10
    while ! docker info &> /dev/null; do
        if [ $RETRIES -le 0 ]; then error "Docker daemon failed to start."; fi
        warn "Waiting for Docker Daemon..."
        sleep 2
        RETRIES=$((RETRIES-1))
    done
}

ensure_docker_wsl() {
    # In WSL, we rely on Docker Desktop from Windows being bridged
    if ! docker info &> /dev/null; then
        warn "Cannot connect to Docker Daemon in WSL."
        warn "Troubleshooting: Checking for docker.exe interop..."
        
        if command -v docker.exe &> /dev/null; then
            # We found the binary, but the socket might be missing
            warn "Docker Desktop found but WSL integration might be off."
            warn "Attempting to infer host..."
            export DOCKER_HOST=tcp://localhost:2375
            if ! docker info &> /dev/null; then
                error "Integration Failed. Ensure Docker Desktop is RUNNING on Windows."
            fi
        else
             error "Docker Desktop appears stopped or not installed. Please start it on Windows."
        fi
    fi
}

if [ "$IS_WSL" = true ]; then
    ensure_docker_wsl
else
    ensure_docker_linux
fi

log "Docker Runtime Verified."

# ==============================================================================
# 3. SECRETS & CONFIGURATION
# ==============================================================================
divider
log "Generating Secure Configuration..."
python3 scripts/generate_env.py
if [ -f ".env" ]; then
    set -a; source .env; set +a
fi

# ==============================================================================
# 4. UNIFIED DEPLOYMENT FLOW
# ==============================================================================

# A. CLEANUP (If requested or retrying)
docker compose down --remove-orphans &> /dev/null || true

# B. BUILD
divider
log "Step 1/4: Building Optimized Image..."
# Using standard docker build as it can be more stable in some WSL interop scenarios than compose build
docker build -t resha:latest . || error "Docker Build Failed."

# Update compose to use the local image we just built
export BUILD_TARGET=final

# C. STRICT CHECKS
divider
log "Step 2/4: Running Strict Validations (Ephemeral)..."
# We run checks in a throwaway container to ensure environment consistency
# -ll = only high/medium severity, -s B104 = skip "bind to all interfaces" (expected in Docker)
docker compose run --rm --no-deps --entrypoint /bin/sh app -c "
    echo 'Running Security Scan (Bandit)...' &&
    bandit -r src -ll -s B104 &&
    echo 'Running Reliability Tests (Pytest)...' &&
    pytest tests/ -v
" || error "Strict Checks Failed. Fix code before deploying."

# D. ARTIFACTS
divider
log "Step 3/4: Generating Compliance Artifacts (SBOM/RBOM)..."
mkdir -p artifacts
# We mount the artifacts dir to extract the files generated inside
docker compose run --rm --no-deps -v $(pwd)/artifacts:/app/artifacts app python scripts/generate_artifacts.py

# E. LAUNCH
divider
log "Step 4/4: Launching Production Stack..."
docker compose up -d

# F. HEALTH CHECK
log "Waiting for Service Health..."
RETRIES=60 # 2 minutes timeout
HEALTHY=false

for ((i=1; i<=RETRIES; i++)); do
    # Check docker inspect for health status
    STATUS=$(docker compose ps app --format json | grep -o '"Health":"[^"]*"' | cut -d'"' -f4)
    
    if [ "$STATUS" == "healthy" ]; then
        HEALTHY=true
        break
    fi
    
    # If container died, fail fast
    STATE=$(docker compose ps app --format json | grep -o '"State":"[^"]*"' | cut -d'"' -f4)
    if [ "$STATE" == "exited" ] || [ "$STATE" == "dead" ]; then
        error "Container crashed during startup. Check logs: docker compose logs app"
    fi
    
    echo -n "."
    sleep 2
done

if [ "$HEALTHY" = true ]; then
    divider
    log "Deployment SUCCESSFUL"
    log "App: http://localhost:$PORT"
    log "Metrics: http://localhost:$PORT/metrics"
    log "Docs: http://localhost:$PORT/docs"
else
    error "Service timed out waiting for health check."
fi
