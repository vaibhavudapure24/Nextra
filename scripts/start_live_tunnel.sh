#!/bin/bash
echo "=========================================================="
echo " 🌐 EXPOSING WILDLIFE AI FRONTEND TO THE LIVE PUBLIC WEB"
echo "=========================================================="
echo "Connecting secure HTTPS tunnel..."
ssh -o StrictHostKeyChecking=no -R 80:localhost:8000 serveo.net
