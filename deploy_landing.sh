#!/bin/bash
# ==============================================================================
# Deploy Landing Page for reas.dmj.one
# Replaces the default Nginx welcome page with a custom landing page
# ==============================================================================
# Usage: sudo bash deploy_landing.sh
# ==============================================================================

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
LANDING_HTML="${SCRIPT_DIR}/landing/index.html"
NGINX_HTML_DIR="/var/www/html"

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Use: sudo bash deploy_landing.sh"
    exit 1
fi

info "Deploying landing page for reas.dmj.one..."

# Backup original if exists
if [ -f "${NGINX_HTML_DIR}/index.nginx-debian.html" ]; then
    mv "${NGINX_HTML_DIR}/index.nginx-debian.html" "${NGINX_HTML_DIR}/index.nginx-debian.html.bak" 2>/dev/null || true
fi

if [ -f "${NGINX_HTML_DIR}/index.html" ]; then
    cp "${NGINX_HTML_DIR}/index.html" "${NGINX_HTML_DIR}/index.html.bak.$(date +%s)"
    log "Backed up existing index.html"
fi

# Copy landing page
cp "${LANDING_HTML}" "${NGINX_HTML_DIR}/index.html"
chown www-data:www-data "${NGINX_HTML_DIR}/index.html"
chmod 644 "${NGINX_HTML_DIR}/index.html"

log "Landing page deployed to ${NGINX_HTML_DIR}/index.html"

# Test nginx and reload
if nginx -t 2>/dev/null; then
    systemctl reload nginx
    log "Nginx reloaded successfully"
else
    echo "Nginx config test failed"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Landing page deployed!${NC}"
echo -e "   Visit: ${CYAN}https://reas.dmj.one${NC}"
echo ""
