#!/bin/bash
echo "=========================================================="
echo " ⚙️ STARTING WILDLIFE BACKEND (PORT 8000)"
echo " API Docs: http://localhost:8000/docs"
echo "=========================================================="
python3 -m uvicorn api.index:app --host 0.0.0.0 --port 8000
