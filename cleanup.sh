#!/bin/bash
# ==============================================================================
# RESHA - Repository Cleanup Script
# ==============================================================================
# This script removes unnecessary files before deployment
# Usage: bash cleanup.sh
# ==============================================================================

echo "🧹 Cleaning up repository..."

# Remove Windows-specific files
echo "Removing Windows-specific files..."
rm -f deploy_and_test.bat
rm -f start.bat
rm -f start.ps1
rm -f deliverables/QA/run_qa.bat

# Remove log files
echo "Removing log files..."
rm -f server.log
rm -f *.log
rm -rf logs/*

# Remove sample/test output files
echo "Removing sample output files..."
rm -f sample_output.json

# Remove virtual environment (will be recreated on deployment)
echo "Removing virtual environment..."
rm -rf venv/

# Remove Python cache files
echo "Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove pytest cache
echo "Removing pytest cache..."
rm -rf .pytest_cache/

# Remove database files (will be recreated)
echo "Removing database files..."
find . -type f -name "*.db" -delete 2>/dev/null || true

# Remove artifacts directory contents
echo "Cleaning artifacts..."
rm -rf artifacts/*

# Remove PID files
echo "Removing PID files..."
rm -f .server.pid

# Remove IDE-specific files
echo "Cleaning IDE files..."
rm -rf .idea/
rm -rf .vscode/

# Remove agent deploy venv
echo "Cleaning agent files..."
rm -rf .agent/deploy_venv/

echo ""
echo "✅ Repository cleaned!"
echo ""
echo "Files to commit:"
find . -maxdepth 2 -type f ! -path './.git/*' | head -30
