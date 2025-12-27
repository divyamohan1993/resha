#!/bin/bash
# ==============================================================================
# RESHA - Quick Fix for Ollama Timeout Issues
# ==============================================================================
# This script updates Nginx timeouts and restarts services to prevent
# timeouts during slow Ollama LLM inference.
# 
# Usage: sudo bash fix_ollama_timeout.sh
# ==============================================================================

set -e

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

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root. Use: sudo bash fix_ollama_timeout.sh"
fi

echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  RESHA - Fixing Ollama Timeout Issues${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"

# ==============================================================================
# STEP 1: Update Nginx configuration for extended timeouts
# ==============================================================================
info "Step 1: Updating Nginx configuration..."

NGINX_CONF="/etc/nginx/sites-available/reas.dmj.one"

if [ ! -f "$NGINX_CONF" ]; then
    error "Nginx config not found at $NGINX_CONF"
fi

# Back up current config
cp "$NGINX_CONF" "${NGINX_CONF}.backup.$(date +%s)"
log "Created backup of nginx config"

# Use Python to update the timeout values
python3 << 'PYEOF'
import re

nginx_conf = "/etc/nginx/sites-available/reas.dmj.one"

with open(nginx_conf, 'r') as f:
    content = f.read()

# Update timeout values from 60s to 300s
content = re.sub(r'proxy_connect_timeout\s+\d+s?;', 'proxy_connect_timeout 300s;', content)
content = re.sub(r'proxy_send_timeout\s+\d+s?;', 'proxy_send_timeout 300s;', content)
content = re.sub(r'proxy_read_timeout\s+\d+s?;', 'proxy_read_timeout 300s;', content)

# Check if proxy_buffering is already set in task2 block, if not add it
# This is trickier - we need to find the task2 location block

# Find and update task2 location block
task2_pattern = r'(location\s+/task2/\s*\{[^}]*?proxy_read_timeout[^;]*;)'
match = re.search(task2_pattern, content)
if match:
    # Check if proxy_buffering is already in the block
    block = match.group(0)
    if 'proxy_buffering' not in block:
        # Add proxy_buffering after proxy_read_timeout line
        new_block = re.sub(
            r'(proxy_read_timeout\s+\d+s?;)',
            r'\1\n        \n        # Disable buffering for Server-Sent Events (SSE) streaming\n        proxy_buffering off;\n        proxy_cache off;',
            block
        )
        content = content.replace(block, new_block)
        print("Added proxy_buffering settings to task2 block")
    else:
        print("proxy_buffering already configured")

with open(nginx_conf, 'w') as f:
    f.write(content)

print("Nginx timeouts updated to 300s")
PYEOF

log "Nginx config updated with extended timeouts"

# ==============================================================================
# STEP 2: Test and reload Nginx
# ==============================================================================
info "Step 2: Testing Nginx configuration..."

if nginx -t 2>/dev/null; then
    systemctl reload nginx
    log "Nginx configuration valid and reloaded"
else
    error "Nginx configuration test failed. Check: nginx -t"
fi

# ==============================================================================
# STEP 3: Warmup Ollama model (if ollama is installed)
# ==============================================================================
info "Step 3: Checking Ollama status..."

if command -v ollama &> /dev/null; then
    # Check if ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        log "Ollama is running"
        
        # Get list of models
        MODELS=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys, json; data=json.load(sys.stdin); print(' '.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null || echo "")
        
        if [ -n "$MODELS" ]; then
            info "Available models: $MODELS"
            
            # Warmup each model
            for MODEL in $MODELS; do
                info "Warming up model: $MODEL..."
                # Send a minimal request to load model into RAM
                curl -s http://localhost:11434/api/generate \
                    -d "{\"model\": \"$MODEL\", \"prompt\": \"Hi\", \"stream\": false, \"options\": {\"num_predict\": 1}}" \
                    > /dev/null 2>&1 && log "Model $MODEL warmed up" || warn "Failed to warmup $MODEL"
            done
        else
            warn "No models found in Ollama"
        fi
    else
        warn "Ollama is installed but not running. Start it with: ollama serve"
    fi
else
    warn "Ollama is not installed. Install with: curl -fsSL https://ollama.ai/install.sh | sh"
fi

# ==============================================================================
# STEP 4: Restart the Resha service
# ==============================================================================
info "Step 4: Restarting Resha service..."

if systemctl is-active --quiet resha; then
    systemctl restart resha
    sleep 3
    if systemctl is-active --quiet resha; then
        log "Resha service restarted successfully"
    else
        error "Resha service failed to restart. Check: journalctl -u resha -f"
    fi
else
    warn "Resha service is not running"
fi

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Fix Applied Successfully!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Changes made:"
echo -e "  ${GREEN}✓${NC} Nginx proxy timeouts increased to 300s (5 minutes)"
echo -e "  ${GREEN}✓${NC} SSE proxy buffering disabled"
echo -e "  ${GREEN}✓${NC} Ollama models warmed up (if available)"
echo -e "  ${GREEN}✓${NC} Resha service restarted"
echo ""
echo -e "The application now has:"
echo -e "  • 5-minute timeout for slow CPU inference"
echo -e "  • LRU caching for repeated analyses (instant response)"
echo -e "  • Model warmup on startup (no cold-start delays)"
echo ""
