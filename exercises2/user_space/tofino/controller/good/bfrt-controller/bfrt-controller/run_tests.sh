#!/bin/bash
source bfrt/bin/activate
GREEN='\033[0;32m'
BLUE='\033[1;34m'
RESET='\033[0m'

echo -e "${BLUE}==> Testing BFRT connectivity...${RESET}"
python3 bfrt_connect.py

echo -e "${BLUE}==> Running Controller initialization...${RESET}"
python3 main.py