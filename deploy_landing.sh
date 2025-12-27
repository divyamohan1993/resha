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
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
LANDING_HTML="${SCRIPT_DIR}/landing/index.html"
LANDING_DIR="/var/www/reas-landing"
NGINX_CONF="/etc/nginx/sites-available/reas.dmj.one"

# Ensure running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root. Use: sudo bash deploy_landing.sh"
    exit 1
fi

info "Deploying landing page for reas.dmj.one..."

# ==============================================================================
# STEP 1: Create landing page directory and copy files
# ==============================================================================
mkdir -p "${LANDING_DIR}"
# Copy all files (index.html, favicon.svg, etc.)
cp -r "${SCRIPT_DIR}/landing/"* "${LANDING_DIR}/"
chown -R www-data:www-data "${LANDING_DIR}"
chmod 644 "${LANDING_DIR}"/*

log "Landing page files copied to ${LANDING_DIR}"

# ==============================================================================
# STEP 2: Update nginx config to serve landing page at root
# ==============================================================================
info "Updating nginx configuration..."

# Backup current config
if [ -f "$NGINX_CONF" ]; then
    cp "$NGINX_CONF" "${NGINX_CONF}.backup.$(date +%s)"
    log "Backed up nginx config"
fi

# Use Python to safely add the root location block
python3 <<PYEOF
import re
import os

nginx_conf = "$NGINX_CONF"
landing_dir = "$LANDING_DIR"

# Root location block to add
root_block = f'''
    # ════════════════════════════════════════════════════════════
    # Landing Page - Serves at root (/)
    # ════════════════════════════════════════════════════════════
    root {landing_dir};
    index index.html;
    
    location / {{
        # First try exact URI (for favicon, assets), then directory, then index.html
        try_files \$uri \$uri/ /index.html;
    }}
    
    # Cache settings for static assets
    location ~* \.(jpg|jpeg|png|gif|ico|css|js|svg)$ {{
        expires 7d;
        add_header Cache-Control "public, must-revalidate";
    }}
'''

if os.path.exists(nginx_conf):
    with open(nginx_conf, 'r') as f:
        content = f.read()
    
    # Check if root location already exists
    if 'Landing Page' in content or 'location = /' in content:
        print("Root location block already exists, updating...")
        # Remove existing root location blocks
        content = re.sub(r'\s*#[^\n]*Landing Page[^\n]*\n(?:[^\n]*\n)*?\s*location = /[^}]*}', '', content)
        content = re.sub(r'\s*root\s+/var/www/reas-landing;[^\n]*\n', '', content)
        content = re.sub(r'\s*index\s+index\.html;[^\n]*\n', '', content)
    
    # Find position after client_max_body_size or after server_name
    insert_match = re.search(r'(client_max_body_size[^;]*;)', content)
    if insert_match:
        insert_pos = insert_match.end()
    else:
        insert_match = re.search(r'(server_name[^;]*;)', content)
        if insert_match:
            insert_pos = insert_match.end()
        else:
            # Fallback: after first opening brace
            insert_pos = content.find('{') + 1
    
    # Insert the root block
    content = content[:insert_pos] + '\n' + root_block + content[insert_pos:]
    
    # Clean up multiple blank lines
    content = re.sub(r'\n{3,}', '\n\n', content)
    
    with open(nginx_conf, 'w') as f:
        f.write(content)
    
    print(f"Updated nginx config with landing page root location")
else:
    print(f"Error: nginx config not found at {nginx_conf}")
    exit(1)
PYEOF

log "Nginx configuration updated"

# ==============================================================================
# STEP 3: Test and reload nginx
# ==============================================================================
info "Testing nginx configuration..."

if nginx -t 2>&1; then
    systemctl reload nginx
    log "Nginx reloaded successfully"
else
    echo "Nginx config test failed. Restoring backup..."
    LATEST_BACKUP=$(ls -t ${NGINX_CONF}.backup.* 2>/dev/null | head -1)
    if [ -n "$LATEST_BACKUP" ]; then
        cp "$LATEST_BACKUP" "$NGINX_CONF"
        systemctl reload nginx
    fi
    exit 1
fi

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ Landing Page Deployed Successfully!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Landing:  ${CYAN}https://reas.dmj.one${NC}"
echo -e "  Task 1:   ${CYAN}https://reas.dmj.one/task1/${NC}"
echo -e "  Task 2:   ${CYAN}https://reas.dmj.one/task2/${NC}"
echo ""
