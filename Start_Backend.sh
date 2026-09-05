#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

echo "=========================================================="
echo " ⚙️ STARTING WILDLIFE BACKEND (PORT 8000)"
echo " API Docs: http://localhost:8000/docs"
echo "=========================================================="
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
