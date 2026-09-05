#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || exit 1

echo "=========================================================="
echo " 🎨 STARTING WILDLIFE FRONTEND (PORT 3000)"
echo " Local URL: http://localhost:3000"
echo "=========================================================="
export API_URL="http://localhost:8000"
python3 -m streamlit run dashboard/app.py --server.port 3000 --server.headless true
