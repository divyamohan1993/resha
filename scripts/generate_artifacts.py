import os
import sys
import json
import platform
import subprocess
from datetime import datetime
from typing import Dict, Any

def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except Exception:
        return "unknown"

def sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
    """
    Sanitize environment variables to remove secrets.
    """
    sanitized = {}
    secret_keywords = ['KEY', 'SECRET', 'PASSWORD', 'TOKEN', 'CREDENTIAL']
    for k, v in env.items():
        if any(keyword in k.upper() for keyword in secret_keywords):
            sanitized[k] = "[REDACTED]"
        else:
            sanitized[k] = v
    return sanitized

def generate_rbom():
    """
    Generates a Runtime Bill of Materials (RBOM).
    Captures the context in which the application is running.
    """
    rbom = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "type": "RBOM",
            "specVersion": "1.0"
        },
        "runtime": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "node": platform.node(),
            "architecture": platform.architecture()
        },
        "application": {
            "git_commit": get_git_revision_hash(),
            "cwd": os.getcwd(),
            "entry_point": sys.argv[0]
        },
        "environment": sanitize_env(dict(os.environ)),
        "dependencies_loaded": list(sys.modules.keys())
    }
    
    os.makedirs("artifacts", exist_ok=True)
    
    with open("artifacts/rbom.json", "w") as f:
        json.dump(rbom, f, indent=2)
    
    print("RBOM generated at artifacts/rbom.json")

def generate_sbom():
    """
    Generates SBOM using cyclonedx-bom.
    """
    print("Generating SBOM...")
    os.makedirs("artifacts", exist_ok=True)
    try:
        subprocess.run([
            sys.executable, "-m", "cyclonedx_py", 
            "requirements", 
            "requirements.txt",
            "--output", "artifacts/sbom.json",
            "--force"
        ], check=True)
        print("SBOM generated at artifacts/sbom.json")
    except subprocess.CalledProcessError as e:
        print(f"Failed to generate SBOM: {e}")

if __name__ == "__main__":
    generate_sbom()
    generate_rbom()
