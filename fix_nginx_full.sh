#!/bin/bash
# ==============================================================================
# RESHA - Complete Nginx Fix for both Task1 and Task2
# ==============================================================================
# This script consolidates both Task1 and Task2 into a SINGLE nginx config file.
# It removes conflicting configurations that were created separately.
#
# Problem: Task1 creates /etc/nginx/sites-available/realtyassistant
#          Task2 creates /etc/nginx/sites-available/reas.dmj.one
#          Both define the same server_name, causing conflicts!
#
# Solution: This script creates ONE unified config at reas.dmj.one
#           and removes the separate realtyassistant config.
#
# Run this on the VM: sudo bash fix_nginx_full.sh
# ==============================================================================

set -e

NGINX_CONF="/etc/nginx/sites-available/reas.dmj.one"
NGINX_ENABLED="/etc/nginx/sites-enabled/reas.dmj.one"
TASK1_OLD_CONF="/etc/nginx/sites-available/realtyassistant"
TASK1_OLD_ENABLED="/etc/nginx/sites-enabled/realtyassistant"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Complete Nginx Fix - Consolidating Task1 & Task2${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════${NC}"

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
    error "This script must be run as root. Use: sudo bash fix_nginx_full.sh"
fi

# ==============================================================================
# Step 1: Remove conflicting Task1 nginx config
# ==============================================================================
info "Checking for conflicting nginx configurations..."

if [ -f "$TASK1_OLD_ENABLED" ] || [ -L "$TASK1_OLD_ENABLED" ]; then
    rm -f "$TASK1_OLD_ENABLED"
    warn "Removed conflicting symlink: $TASK1_OLD_ENABLED"
fi

if [ -f "$TASK1_OLD_CONF" ]; then
    # Create backup
    cp "$TASK1_OLD_CONF" "${TASK1_OLD_CONF}.backup.$(date +%s)"
    rm -f "$TASK1_OLD_CONF"
    warn "Removed conflicting config: $TASK1_OLD_CONF (backup created)"
fi

# ==============================================================================
# Step 2: Create backup of existing reas.dmj.one config
# ==============================================================================
if [ -f "$NGINX_CONF" ]; then
    BACKUP_FILE="${NGINX_CONF}.backup.$(date +%s)"
    cp "$NGINX_CONF" "$BACKUP_FILE"
    info "Created backup: $BACKUP_FILE"
fi

# ==============================================================================
# Step 3: Write the unified nginx configuration
# ==============================================================================
info "Writing unified nginx configuration..."

cat > "$NGINX_CONF" << 'NGINX_EOF'
# ==============================================================================
# Unified Nginx Configuration for reas.dmj.one
# Contains both Task1 and Task2 location blocks
# ==============================================================================

server {
    listen 80;
    server_name reas.dmj.one;
    
    client_max_body_size 50M;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # ════════════════════════════════════════════════════════════
    # Task 1: RealtyAssistant AI Agent (Chat/Voice)
    # Backend: http://127.0.0.1:20000
    # ════════════════════════════════════════════════════════════
    location /task1/ {
        proxy_pass http://127.0.0.1:20000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Prefix /task1;
        proxy_read_timeout 86400;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        
        # Handle trailing slash
        proxy_redirect / /task1/;
        
        # Buffer settings
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Static files for Task 1
    location /task1/static/ {
        proxy_pass http://127.0.0.1:20000/static/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }

    # Widget JS for Task 1
    location = /task1/widget.js {
        proxy_pass http://127.0.0.1:20000/widget.js;
        proxy_set_header Host $host;
        add_header Content-Type "application/javascript";
        expires 1d;
    }

    # Task 1 Demo redirect
    location = /task1 {
        return 301 /task1/demo;
    }

    # Task 1 Health check
    location = /task1/health {
        proxy_pass http://127.0.0.1:20000/api/status;
        proxy_set_header Host $host;
    }

    # ════════════════════════════════════════════════════════════
    # Task 2: Resha - AI Resume Shortlisting Agent
    # Backend: http://127.0.0.1:22000
    # ════════════════════════════════════════════════════════════
    location /task2/ {
        proxy_pass http://127.0.0.1:22000/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Script-Name /task2;
        
        # Timeout settings
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer settings for large responses
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
    }

    # Task 2 redirect (no trailing slash)
    location = /task2 {
        return 301 /task2/;
    }

    # ════════════════════════════════════════════════════════════
    # Error pages
    # ════════════════════════════════════════════════════════════
    error_page 502 503 504 /50x.html;
    location = /50x.html {
        root /usr/share/nginx/html;
        internal;
    }
}
NGINX_EOF

log "Nginx configuration written"

# ==============================================================================
# Step 4: Enable the site
# ==============================================================================
ln -sf "$NGINX_CONF" "$NGINX_ENABLED" 2>/dev/null || true
log "Site enabled in sites-enabled"

# ==============================================================================
# Step 5: Display and test configuration
# ==============================================================================
echo ""
info "Current nginx configuration:"
echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
cat "$NGINX_CONF"
echo -e "${CYAN}────────────────────────────────────────────────────────────${NC}"
echo ""

info "Testing nginx configuration..."
if nginx -t 2>&1; then
    log "Nginx configuration test passed"
    
    # Reload nginx
    systemctl reload nginx
    log "Nginx reloaded"
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  SUCCESS! Nginx configuration fixed!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    # Test both endpoints
    info "Testing backend services..."
    sleep 1
    
    echo ""
    echo -e "${CYAN}Checking Task 1 (RealtyAssistant on Port 20000):${NC}"
    if systemctl is-active --quiet realtyassistant.service 2>/dev/null; then
        log "realtyassistant.service is running"
    else
        warn "realtyassistant.service is not running"
    fi
    HTTP_STATUS_1=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:20000/" 2>/dev/null || echo "failed")
    if [ "$HTTP_STATUS_1" = "200" ]; then
        log "Task1 backend responding (HTTP $HTTP_STATUS_1)"
    else
        warn "Task1 backend returned: $HTTP_STATUS_1"
    fi
    
    echo ""
    echo -e "${CYAN}Checking Task 2 (Resha on Port 22000):${NC}"
    if systemctl is-active --quiet resha.service 2>/dev/null; then
        log "resha.service is running"
    else
        warn "resha.service is not running"
    fi
    HTTP_STATUS_2=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:22000/" 2>/dev/null || echo "failed")
    if [ "$HTTP_STATUS_2" = "200" ]; then
        log "Task2 backend responding (HTTP $HTTP_STATUS_2)"
    else
        warn "Task2 backend returned: $HTTP_STATUS_2"
    fi
    
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Access Points:${NC}"
    echo -e "${GREEN}  • Task 1 Demo:  https://reas.dmj.one/task1/demo${NC}"
    echo -e "${GREEN}  • Task 1 Voice: https://reas.dmj.one/task1/voice${NC}"
    echo -e "${GREEN}  • Task 2:       https://reas.dmj.one/task2/${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    
    info "If any service is not running, start them with:"
    echo "   sudo systemctl start realtyassistant.service"
    echo "   sudo systemctl start resha.service"
    echo ""
    
else
    error "Nginx configuration test failed!"
fi
