#!/bin/bash
set -e

DOMAIN="resha.dmj.one"
REPO_URL="https://github.com/divyamohan1993/resha.git"
INSTALL_DIR="/opt/resha"

# Ensure root
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo bash deploy.sh)"
  exit 1
fi

echo "=== Starting Deployment for $DOMAIN ==="

# 1. Update System & Prerequisites
echo "[1/8] Updating system packages..."
apt-get update -qq && apt-get upgrade -y -qq
apt-get install -y -qq curl git nginx certbot python3-certbot-nginx openssl

# 2. Install Docker
echo "[2/8] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
else
    echo "Docker already installed."
fi

# 3. Install Ollama
echo "[3/8] Installing Ollama..."
if ! command -v ollama &> /dev/null; then
    curl -fsSL https://ollama.com/install.sh | sh
    systemctl enable --now ollama
else
    echo "Ollama already installed."
fi

# Pull Model (wait for ollama to be ready)
echo "Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags >/dev/null; do
    sleep 2
done
echo "Pulling phi3:mini model..."
ollama pull phi3:mini

# 4. Clone/Update Repo
echo "[4/8] Setting up application code..."
if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing repository..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# 5. Configure Environment
echo "[5/8] Configuring environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    # Generate random passwords for security
    DB_PASS=$(openssl rand -base64 12 | tr -dc 'a-zA-Z0-9')
    ROOT_PASS=$(openssl rand -base64 12 | tr -dc 'a-zA-Z0-9')
    
    sed -i "s/secure_db_password/$DB_PASS/g" .env
    sed -i "s/secure_root_password/$ROOT_PASS/g" .env
    sed -i "s/DOMAIN_NAME=.*/DOMAIN_NAME=\"$DOMAIN\"/" .env
    sed -i "s/ENVIRONMENT=.*/ENVIRONMENT=\"production\"/" .env
    
    # Ensure DATABASE_URL is updated
    # The sed above for secure_db_password should handle it if .env.example uses the same placeholder
    
    echo "Generated new .env with random DB passwords."
else
    echo ".env already exists, skipping generation."
fi

# 6. Start Application
echo "[6/8] Starting Docker containers..."
docker compose up -d --build

# 7. Configure Nginx
echo "[7/8] Configuring Nginx..."
cat > /etc/nginx/sites-available/$DOMAIN <<EOF
server {
    server_name $DOMAIN;
    
    client_max_body_size 50M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        
        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl restart nginx

# 8. Generate SSL (Optional attempt)
echo "[8/8] Attempting SSL setup..."
# We run certbot with --dry-run or check dns?
# User said they WILL map the IP. So it might fail now.
# We will leave a message.

echo "=== Deployment Complete ==="
echo "Public IP: $(curl -s ifconfig.me)"
echo "Domain: http://$DOMAIN"
echo "Instructions:"
echo "1. Map your DNS A record for $DOMAIN to the IP above."
echo "2. Once DNS propagates, run: certbot --nginx -d $DOMAIN --non-interactive --agree-tos --email admin@dmj.one"
echo "3. Edit $INSTALL_DIR/.env to add your GEMINI_API_KEY if needed, then run 'docker compose restart'."
echo "==========================="
