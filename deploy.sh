#!/bin/bash
# ==============================================================================
# RESHA - Legacy Docker Deployment Script
# ==============================================================================
# NOTE: For simpler deployment, use run_project.sh instead
# This script uses Docker Compose for containerized deployment
# ==============================================================================

set -e

DOMAIN="reas.dmj.one"
REPO_URL="https://github.com/divyamohan1993/resha.git"
INSTALL_DIR="/opt/resha"
APP_PORT=22000

# Ensure root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo bash deploy.sh)"
  exit 1
fi

echo "=== Starting Docker Deployment for $DOMAIN/task2/ ==="

# 1. Update System & Prerequisites
echo "[1/7] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq
apt-get install -y -qq curl git nginx certbot python3-certbot-nginx openssl

# 2. Install Docker
echo "[2/7] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
else
    echo "Docker already installed."
fi

# 3. Clone/Update Repo
echo "[3/7] Setting up application code..."
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing repository..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 4. Configure Environment
echo "[4/7] Configuring environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    # Generate random passwords for security
    SECRET_KEY=$(openssl rand -base64 32 | tr -dc 'a-zA-Z0-9')
    API_KEY=$(openssl rand -base64 48 | tr -dc 'a-zA-Z0-9')
    
    sed -i "s/SECRET_KEY=.*/SECRET_KEY=\"$SECRET_KEY\"/" .env
    sed -i "s/API_KEY=.*/API_KEY=\"$API_KEY\"/" .env
    sed -i "s/PORT=.*/PORT=$APP_PORT/" .env
    
    echo "Generated new .env with secure credentials."
else
    echo ".env already exists, updating port..."
    sed -i "s/PORT=.*/PORT=$APP_PORT/" .env
fi

# 5. Start Application
echo "[5/7] Starting Docker containers..."
docker compose up -d --build

# 6. Configure Nginx (add task2 location)
echo "[6/7] Configuring Nginx for /task2/..."

NGINX_CONF="/etc/nginx/sites-available/$DOMAIN"

if [ ! -f "$NGINX_CONF" ]; then
    # Create new config
    cat > "$NGINX_CONF" << EOF
server {
    listen 80;
    server_name $DOMAIN;
    
    client_max_body_size 50M;

    location /task2/ {
        proxy_pass http://127.0.0.1:$APP_PORT/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF
else
    # Add task2 location to existing config
    if ! grep -q "location /task2/" "$NGINX_CONF"; then
        # Insert task2 block before closing brace
        sed -i '/^}/i \
    location /task2/ {\
        proxy_pass http://127.0.0.1:'"$APP_PORT"'/;\
        proxy_set_header Host $host;\
        proxy_set_header X-Real-IP $remote_addr;\
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;\
        proxy_set_header X-Forwarded-Proto $scheme;\
        proxy_http_version 1.1;\
        proxy_set_header Upgrade $http_upgrade;\
        proxy_set_header Connection "upgrade";\
    }' "$NGINX_CONF"
    fi
fi

ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# 7. Summary
echo ""
echo "=== Deployment Complete ==="
echo "Public IP: $(curl -s ifconfig.me)"
echo "Access: https://$DOMAIN/task2/"
echo ""
echo "Commands:"
echo "  - Logs: docker compose logs -f"
echo "  - Restart: docker compose restart"
echo "  - Stop: docker compose down"
echo "==========================="
