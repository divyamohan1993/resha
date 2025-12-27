#!/bin/bash
# ==============================================================================
# RESHA - Stop Service Script
# ==============================================================================
# Usage: sudo bash stop_service.sh
# ==============================================================================

APP_NAME="resha"
APP_PORT=22000

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}[i] Stopping Resha service...${NC}"

# Stop systemd service
if systemctl is-active --quiet ${APP_NAME}; then
    systemctl stop ${APP_NAME}
    echo -e "${GREEN}[✓] Service stopped${NC}"
else
    echo -e "${RED}[!] Service was not running${NC}"
fi

# Kill any remaining processes on the port
if lsof -i:${APP_PORT} -t > /dev/null 2>&1; then
    kill -9 $(lsof -i:${APP_PORT} -t) 2>/dev/null || true
    echo -e "${GREEN}[✓] Cleared port ${APP_PORT}${NC}"
fi

echo -e "${GREEN}[✓] Done${NC}"
