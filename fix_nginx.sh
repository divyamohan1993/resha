#!/bin/bash
# ==============================================================================
# RESHA - Fix Nginx Configuration for /task2/
# ==============================================================================
# Run this on the VM: sudo bash fix_nginx.sh
# ==============================================================================

set -e

APP_PORT=22000
NGINX_CONF="/etc/nginx/sites-available/reas.dmj.one"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root. Use: sudo bash fix_nginx.sh"
fi

info "Checking nginx configuration..."

# Check if the config file exists
if [ ! -f "$NGINX_CONF" ]; then
    error "Nginx config file not found: $NGINX_CONF"
fi

# Create backup
cp "$NGINX_CONF" "${NGINX_CONF}.backup.$(date +%s)"
log "Created backup of nginx config"

# Check if task2 location already exists
if grep -q "location /task2/" "$NGINX_CONF"; then
    info "task2 location already exists, removing old one first..."
    # Use perl for more reliable multi-line removal
    perl -i -0pe 's/\s*location \/task2\/\s*\{[^}]*\}//gs' "$NGINX_CONF"
fi

# Read the current config
CURRENT_CONFIG=$(cat "$NGINX_CONF")

# Find the last closing brace and insert task2 block before it
# We need to add the task2 location inside the server block

TASK2_BLOCK='
    # ════════════════════════════════════════════════════════════
    # Task 2: Resha - AI Resume Shortlisting Agent
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
        
        # Buffer settings for large responses
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }
'

# Create a temporary file with the task2 block
TEMP_FILE=$(mktemp)

# Use awk to insert the block before the last closing brace
awk -v block="$TASK2_BLOCK" '
    /^}/ && !inserted {
        print block
        inserted = 1
    }
    { print }
' "$NGINX_CONF" > "$TEMP_FILE"

# Move temp file to nginx config
mv "$TEMP_FILE" "$NGINX_CONF"
chmod 644 "$NGINX_CONF"

log "Added task2 location block"

# Display the config for verification
info "Current nginx config:"
echo "----------------------------------------"
cat "$NGINX_CONF"
echo "----------------------------------------"

# Test nginx configuration
info "Testing nginx configuration..."
if nginx -t; then
    log "Nginx configuration test passed"
    
    # Reload nginx
    systemctl reload nginx
    log "Nginx reloaded"
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Nginx configuration fixed!${NC}"
    echo -e "${GREEN}  Access: https://reas.dmj.one/task2/${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    
    # Test the endpoint
    echo ""
    info "Testing endpoint..."
    sleep 1
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:22000/")
    if [ "$HTTP_STATUS" = "200" ]; then
        log "Backend is responding (HTTP $HTTP_STATUS)"
    else
        info "Backend returned HTTP $HTTP_STATUS (may still be starting)"
    fi
else
    error "Nginx configuration test failed. Restoring backup..."
    cp "${NGINX_CONF}.backup."* "$NGINX_CONF" 2>/dev/null || true
    exit 1
fi
