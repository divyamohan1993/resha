#!/usr/bin/env python3
"""
=============================================================================
Resha - Server Startup Script
=============================================================================

This script ensures clean server startup by:
1. Terminating any existing process on the target port
2. Setting up the virtual environment if needed
3. Installing dependencies if needed
4. Starting the server

Usage:
    python run_server.py [--port 8000] [--host 0.0.0.0] [--reload]
"""

import os
import sys
import signal
import socket
import subprocess
import time
import argparse
from pathlib import Path

# Configuration
DEFAULT_PORT = 8000
DEFAULT_HOST = "0.0.0.0"
PROJECT_ROOT = Path(__file__).parent.absolute()
VENV_PATH = PROJECT_ROOT / "venv"
PID_FILE = PROJECT_ROOT / ".server.pid"


def is_port_in_use(port: int) -> bool:
    """Check if a port is currently in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def get_pid_using_port(port: int) -> list:
    """Get PIDs of processes using a specific port."""
    pids = []
    try:
        if sys.platform == "win32":
            # Windows: use netstat
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"],
                capture_output=True, text=True, shell=True
            )
            for line in result.stdout.split('\n'):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        try:
                            pid = int(parts[-1])
                            if pid not in pids:
                                pids.append(pid)
                        except ValueError:
                            pass
        else:
            # Unix: use lsof
            result = subprocess.run(
                ["lsof", "-t", "-i", f":{port}"],
                capture_output=True, text=True
            )
            for pid in result.stdout.strip().split('\n'):
                if pid:
                    try:
                        pids.append(int(pid))
                    except ValueError:
                        pass
    except Exception as e:
        print(f"[WARNING] Could not check port usage: {e}")
    
    return pids


def terminate_process(pid: int) -> bool:
    """Terminate a process by PID."""
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], 
                         capture_output=True, check=True)
        else:
            os.kill(pid, signal.SIGTERM)
            time.sleep(0.5)
            # Force kill if still running
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        print(f"[OK] Terminated process {pid}")
        return True
    except Exception as e:
        print(f"[WARNING] Could not terminate process {pid}: {e}")
        return False


def cleanup_port(port: int) -> bool:
    """Terminate any processes using the specified port."""
    if not is_port_in_use(port):
        return True
    
    print(f"[INFO] Port {port} is in use. Terminating existing processes...")
    
    pids = get_pid_using_port(port)
    if not pids:
        # Try reading from PID file
        if PID_FILE.exists():
            try:
                pid = int(PID_FILE.read_text().strip())
                pids.append(pid)
            except (ValueError, FileNotFoundError):
                pass
    
    for pid in pids:
        terminate_process(pid)
    
    # Wait for port to be released
    for _ in range(10):
        if not is_port_in_use(port):
            print(f"[OK] Port {port} is now available")
            return True
        time.sleep(0.5)
    
    print(f"[ERROR] Could not free port {port}")
    return False


def save_pid(pid: int):
    """Save the current process PID to file."""
    PID_FILE.write_text(str(pid))


def cleanup_pid_file():
    """Remove the PID file on exit."""
    if PID_FILE.exists():
        PID_FILE.unlink()


def check_venv() -> bool:
    """Check if virtual environment exists."""
    if sys.platform == "win32":
        return (VENV_PATH / "Scripts" / "python.exe").exists()
    else:
        return (VENV_PATH / "bin" / "python").exists()


def get_python_path() -> str:
    """Get the path to the Python interpreter in venv."""
    if sys.platform == "win32":
        return str(VENV_PATH / "Scripts" / "python.exe")
    else:
        return str(VENV_PATH / "bin" / "python")


def setup_venv():
    """Create virtual environment if it doesn't exist."""
    if check_venv():
        print("[OK] Virtual environment exists")
        return True
    
    print("[INFO] Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_PATH)], check=True)
        print("[OK] Virtual environment created")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to create virtual environment: {e}")
        return False


def install_dependencies():
    """Install dependencies from requirements.txt."""
    requirements_file = PROJECT_ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("[WARNING] requirements.txt not found")
        return True
    
    print("[INFO] Installing dependencies...")
    try:
        subprocess.run(
            [get_python_path(), "-m", "pip", "install", "-r", str(requirements_file), 
             "--prefer-binary", "-q"],
            check=True,
            cwd=str(PROJECT_ROOT)
        )
        print("[OK] Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        return False


def download_spacy_model():
    """Download spacy model if not present."""
    try:
        result = subprocess.run(
            [get_python_path(), "-c", "import spacy; spacy.load('en_core_web_sm')"],
            capture_output=True, cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            print("[INFO] Downloading spacy model...")
            subprocess.run(
                [get_python_path(), "-m", "spacy", "download", "en_core_web_sm"],
                check=True, cwd=str(PROJECT_ROOT)
            )
            print("[OK] Spacy model downloaded")
    except Exception as e:
        print(f"[WARNING] Could not verify spacy model: {e}")


def start_server(host: str, port: int, reload: bool = False):
    """Start the uvicorn server."""
    print(f"\n{'='*60}")
    print(f"  Starting Resha Server")
    print(f"{'='*60}")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  URL:  http://{host}:{port}")
    print(f"{'='*60}\n")
    
    cmd = [
        get_python_path(), "-m", "uvicorn",
        "src.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        process = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
        save_pid(process.pid)
        
        # Wait for server to be ready
        for _ in range(30):
            if is_port_in_use(port):
                print(f"[OK] Server is running on http://{host}:{port}")
                break
            time.sleep(0.5)
        
        # Keep running
        process.wait()
        
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down server...")
        process.terminate()
        process.wait(timeout=5)
    finally:
        cleanup_pid_file()


def main():
    parser = argparse.ArgumentParser(description="TalentScout Pro AI Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run on")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--skip-install", action="store_true", help="Skip dependency installation")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"  Resha - Startup Script")
    print(f"{'='*60}\n")
    
    # Step 1: Cleanup existing processes on the port
    if not cleanup_port(args.port):
        print("[ERROR] Failed to cleanup existing processes. Exiting.")
        sys.exit(1)
    
    # Step 2: Setup virtual environment
    if not setup_venv():
        sys.exit(1)
    
    # Step 3: Install dependencies
    if not args.skip_install:
        if not install_dependencies():
            sys.exit(1)
        download_spacy_model()
    
    # Step 4: Start the server
    start_server(args.host, args.port, args.reload)


if __name__ == "__main__":
    main()
