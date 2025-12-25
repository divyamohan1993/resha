#!/bin/bash
set -e

echo "Setting up WSL Test Environment..."

# Update and install python3-venv if needed
# sudo apt-get update && sudo apt-get install -y python3-venv

if [ ! -d "venv_wsl" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_wsl
fi

source venv_wsl/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
pip install pytest httpx --quiet

echo "Downloading Spacy Model..."
python -m spacy download en_core_web_sm

echo "Running Tests..."
export PYTHONPATH=$(pwd)
pytest tests/ -v
