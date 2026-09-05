#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR" || exit 1

echo "=========================================================="
echo " 🌐 EXPOSING WILDLIFE AI WEBSITE (PORT 8051) TO PUBLIC WEB"
echo " Target Port: 8051"
echo "=========================================================="

if [ -f "/home/vaibhav/.local/bin/cloudflared" ]; then
    /home/vaibhav/.local/bin/cloudflared tunnel --url http://localhost:8051
else
    ssh -o StrictHostKeyChecking=no -R 80:localhost:8051 serveo.net
fi
