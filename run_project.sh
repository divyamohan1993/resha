#!/bin/bash
# ==============================================================================
# RESHA - AI Resume Shortlisting Agent
# One-Click Deployment Script for reas.dmj.one/task2/
# ==============================================================================
# Usage: sudo bash run_project.sh
# ==============================================================================

set -e

# Configuration
APP_NAME="resha"
APP_PORT=22000
BASE_PATH="/task2"
NGINX_CONF="/etc/nginx/sites-available/reas.dmj.one"
INSTALL_DIR=$(dirname "$(readlink -f "$0")")
SERVICE_FILE="/etc/systemd/system/${APP_NAME}.service"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }
info() { echo -e "${CYAN}[i]${NC} $1"; }
divider() { echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"; }

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================
divider
echo -e "${CYAN}  RESHA - AI Resume Shortlisting Agent${NC}"
echo -e "${CYAN}  Deployment to: reas.dmj.one${BASE_PATH}/${NC}"
divider

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root. Use: sudo bash run_project.sh"
fi

cd "$INSTALL_DIR"
info "Working directory: $INSTALL_DIR"

# ==============================================================================
# STEP 1: SYSTEM DEPENDENCIES
# ==============================================================================
divider
info "Step 1/7: Installing system dependencies..."

apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip nginx curl lsof

log "System dependencies installed"

# ==============================================================================
# STEP 2: STOP EXISTING SERVICE (if running)
# ==============================================================================
divider
info "Step 2/7: Stopping existing service..."

# Stop systemd service if exists
if systemctl is-active --quiet ${APP_NAME}; then
    systemctl stop ${APP_NAME}
    log "Stopped existing ${APP_NAME} service"
else
    log "No existing service running"
fi

# Kill any process on the port
if lsof -i:${APP_PORT} -t > /dev/null 2>&1; then
    kill -9 $(lsof -i:${APP_PORT} -t) 2>/dev/null || true
    sleep 1
    log "Cleared port ${APP_PORT}"
fi

# ==============================================================================
# STEP 3: PYTHON VIRTUAL ENVIRONMENT
# ==============================================================================
divider
info "Step 3/7: Setting up Python virtual environment..."

# Remove old venv if exists (ensure clean install)
if [ -d "venv" ]; then
    warn "Removing existing venv for clean install..."
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

log "Virtual environment created"

# ==============================================================================
# STEP 4: INSTALL PYTHON DEPENDENCIES
# ==============================================================================
divider
info "Step 4/7: Installing Python dependencies (this may take a while)..."

# Install CPU-only PyTorch first to reduce size
pip install torch --index-url https://download.pytorch.org/whl/cpu -q

# Install all requirements
pip install -r requirements.txt -q

# Download spaCy model
python -m spacy download en_core_web_sm -q

log "Python dependencies installed"

# ==============================================================================
# STEP 5: CONFIGURE ENVIRONMENT
# ==============================================================================
divider
info "Step 5/7: Configuring environment..."

# Create .env if not exists
if [ ! -f ".env" ]; then
    cp .env.example .env
    
    # Generate secure random keys
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
    API_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(48))")
    
    # Update .env with secure values
    sed -i "s|SECRET_KEY=.*|SECRET_KEY=\"${SECRET_KEY}\"|" .env
    sed -i "s|API_KEY=.*|API_KEY=\"${API_KEY}\"|" .env
    sed -i "s|PORT=.*|PORT=${APP_PORT}|" .env
    sed -i "s|DEVELOPMENT_MODE=.*|DEVELOPMENT_MODE=true|" .env
    
    log "Created .env with secure credentials"
else
    # Just update the port
    sed -i "s|PORT=.*|PORT=${APP_PORT}|" .env
    log "Updated existing .env with port ${APP_PORT}"
fi

# ==============================================================================
# STEP 6: CREATE SYSTEMD SERVICE
# ==============================================================================
divider
info "Step 6/7: Creating systemd service..."

cat > ${SERVICE_FILE} << EOF
[Unit]
Description=Resha - AI Resume Shortlisting Agent
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
Environment="PATH=${INSTALL_DIR}/venv/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart=${INSTALL_DIR}/venv/bin/uvicorn src.main:app --host 127.0.0.1 --port ${APP_PORT}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${APP_NAME}
systemctl start ${APP_NAME}

# Wait for service to start
sleep 3

if systemctl is-active --quiet ${APP_NAME}; then
    log "Systemd service created and started"
else
    error "Service failed to start. Check: journalctl -u ${APP_NAME} -f"
fi

# ==============================================================================
# STEP 7: CONFIGURE NGINX
# ==============================================================================
divider
info "Step 7/7: Configuring Nginx for ${BASE_PATH}/..."

# Create backup if config exists
if [ -f "$NGINX_CONF" ]; then
    cp "$NGINX_CONF" "${NGINX_CONF}.backup.$(date +%s)"
    log "Created nginx config backup"
fi

# Remove conflicting Task1 nginx config if it exists
# Task1 creates its own config at /etc/nginx/sites-available/realtyassistant
# We need to consolidate into reas.dmj.one
TASK1_OLD_CONF="/etc/nginx/sites-available/realtyassistant"
TASK1_OLD_ENABLED="/etc/nginx/sites-enabled/realtyassistant"

if [ -f "$TASK1_OLD_ENABLED" ] || [ -L "$TASK1_OLD_ENABLED" ]; then
    # Read Task1's config before removing
    if [ -f "$TASK1_OLD_CONF" ]; then
        PRESERVE_TASK1=true
        warn "Found separate Task1 nginx config - will merge into unified config"
    fi
    rm -f "$TASK1_OLD_ENABLED"
fi

# Create the task2 block content
TASK2_BLOCK='
    # ════════════════════════════════════════════════════════════
    # Task 2: Resha - AI Resume Shortlisting Agent
    # Port: 22000
    # ════════════════════════════════════════════════════════════
    location /task2/ {
        proxy_pass http://127.0.0.1:22000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Script-Name /task2;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
'

# Use Python to safely modify the nginx config (avoids sed escaping issues)
python3 << PYEOF
import re
import os

nginx_conf = "$NGINX_CONF"
task2_block = '''$TASK2_BLOCK'''

# Task1 location block (in case we need to create a new unified config)
task1_block = '''
    # ════════════════════════════════════════════════════════════
    # Task 1: RealtyAssistant AI Agent (Chat/Voice)
    # Port: 20000
    # ════════════════════════════════════════════════════════════
    location /task1/ {
        proxy_pass http://127.0.0.1:20000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_set_header X-Forwarded-Prefix /task1;
        proxy_read_timeout 86400;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_redirect / /task1/;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    location /task1/static/ {
        proxy_pass http://127.0.0.1:20000/static/;
        proxy_set_header Host \$host;
        expires 7d;
    }

    location = /task1/widget.js {
        proxy_pass http://127.0.0.1:20000/widget.js;
        proxy_set_header Host \$host;
        expires 1d;
    }

    location = /task1 {
        return 301 /task1/demo;
    }
'''

# Check if config file exists
if os.path.exists(nginx_conf):
    with open(nginx_conf, 'r') as f:
        content = f.read()
    
    # Check if task1 location exists
    has_task1 = '/task1/' in content
    
    # Remove existing task2 location block if present
    # Pattern matches: location /task2/ { ... } with optional comments
    pattern = r'\s*#[^\n]*Task 2[^\n]*\n(?:\s*#[^\n]*\n)*\s*location\s+/task2/\s*\{[^}]*\}'
    content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # Also try simpler pattern for just the location block
    pattern2 = r'\s*location\s+/task2/\s*\{[^}]*\}'
    content = re.sub(pattern2, '', content, flags=re.DOTALL)
    
    # Also remove task2 redirect if present
    pattern3 = r'\s*location\s*=\s*/task2\s*\{[^}]*\}'
    content = re.sub(pattern3, '', content, flags=re.DOTALL)
    
    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    # If task1 is missing, add it too
    if not has_task1:
        # Find the first location block or after client_max_body_size
        insert_pos = content.find('location ')
        if insert_pos == -1:
            insert_pos = content.find('client_max_body_size')
            if insert_pos != -1:
                insert_pos = content.find('\n', insert_pos) + 1
        if insert_pos != -1:
            content = content[:insert_pos] + task1_block + '\n' + content[insert_pos:]
    
    # Insert task2 block before the final closing brace
    last_brace = content.rfind('}')
    if last_brace != -1:
        content = content[:last_brace] + task2_block + '\n' + content[last_brace:]
    
    with open(nginx_conf, 'w') as f:
        f.write(content)
    
    print("Updated existing nginx config with task2 block")
else:
    # Create new unified config with BOTH task1 and task2
    new_config = '''server {
    listen 80;
    server_name reas.dmj.one;
    
    client_max_body_size 50M;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
''' + task1_block + task2_block + '''

    # Error pages
    error_page 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
        internal;
    }
}
'''
    with open(nginx_conf, 'w') as f:
        f.write(new_config)
    
    print("Created new nginx config with both task1 and task2 blocks")
PYEOF

# Enable site if not already enabled
ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/reas.dmj.one 2>/dev/null || true

# Test nginx configuration
if nginx -t 2>/dev/null; then
    systemctl reload nginx
    log "Nginx configured and reloaded"
else
    error "Nginx configuration test failed. Check: nginx -t"
fi

# ==============================================================================
# DEPLOYMENT COMPLETE
# ==============================================================================
divider
echo -e "${GREEN}"
echo "  ╔═══════════════════════════════════════════════════════════╗"
echo "  ║         DEPLOYMENT SUCCESSFUL!                            ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║  Service:     ${APP_NAME}                                      ║"
echo "  ║  Internal:    http://127.0.0.1:${APP_PORT}                    ║"
echo "  ║  External:    https://reas.dmj.one${BASE_PATH}/                ║"
echo "  ╠═══════════════════════════════════════════════════════════╣"
echo "  ║  Commands:                                                ║"
echo "  ║  - Status:    systemctl status ${APP_NAME}                     ║"
echo "  ║  - Logs:      journalctl -u ${APP_NAME} -f                     ║"
echo "  ║  - Restart:   systemctl restart ${APP_NAME}                    ║"
echo "  ╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Test the service
info "Testing service health..."
sleep 2
if curl -s "http://127.0.0.1:${APP_PORT}/api/health" | grep -q "healthy\|ok"; then
    log "Service health check passed!"
else
    warn "Service may still be starting. Check: curl http://127.0.0.1:${APP_PORT}/api/health"
fi

divider
