import secrets
import string
import os

def generate_password(length=32):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(length))

def load_existing_env(filepath=".env"):
    env_vars = {}
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Strip quotes if present
                    value = value.strip('\'"')
                    env_vars[key.strip()] = value
    return env_vars

existing = load_existing_env()

# Preserve DB Creds if they exist to prevent locking out of existing volumes
mysql_root_password = existing.get("MYSQL_ROOT_PASSWORD") or generate_password()
mysql_password = existing.get("MYSQL_PASSWORD") or generate_password()
mysql_user = existing.get("MYSQL_USER") or "resha_app"
mysql_db = existing.get("MYSQL_DATABASE") or "resha"

# Always allow changing host/port if needed, but defaults are strict
mysql_host = "db" 
mysql_port = 3306

# Rotate Secret Key (Optional - here we Preserve it to avoid session logout on re-deploy)
# Change 'or' to 'if True else' if you WANT rotation on every deploy
secret_key = existing.get("SECRET_KEY") or generate_password(50)

# Preserve other configs
app_name = existing.get("APP_NAME") or "Resha"
environment = existing.get("ENVIRONMENT") or "production"
debug = existing.get("DEBUG") or "false"
port = existing.get("PORT") or "8000"
domain = existing.get("DOMAIN_NAME") or "example.com"
email = existing.get("CONTACT_EMAIL") or "admin@example.com"
deploy_method = existing.get("DEPLOY_METHOD") or "docker"

env_content = f"""# ==============================================================================
# SECURITY CONFIGURATION
# GENERATED AUTOMATICALLY - DO NOT COMMIT TO VERSION CONTROL
# ==============================================================================

# Application Settings
APP_NAME="{app_name}"
ENVIRONMENT="{environment}"
DEBUG={debug}
PORT={port}

# Security (JWT / Encryption)
SECRET_KEY="{secret_key}"

# Deployment Settings
DOMAIN_NAME="{domain}"
CONTACT_EMAIL="{email}"
DEPLOY_METHOD="{deploy_method}"

# Database Configuration (MySQL)
# NOTE: Passwords are preserved to maintain access to persistent volumes
MYSQL_ROOT_PASSWORD="{mysql_root_password}"
MYSQL_DATABASE="{mysql_db}"
MYSQL_USER="{mysql_user}"
MYSQL_PASSWORD="{mysql_password}"
MYSQL_HOST="{mysql_host}"
MYSQL_PORT={mysql_port}

# Constructed Database URL for App
DATABASE_URL="mysql+pymysql://{mysql_user}:{mysql_password}@{mysql_host}:{mysql_port}/{mysql_db}"
"""

with open(".env", "w") as f:
    f.write(env_content)

print(".env file refreshed successfully (Existing credentials preserved).")
