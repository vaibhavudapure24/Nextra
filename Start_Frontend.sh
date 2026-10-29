#!/bin/bash
echo "=========================================================="
echo " 🖥️ STARTING WILDLIFE FRONTEND (PORT 3000)"
echo " Dashboard: http://localhost:3000"
echo "=========================================================="
python3 -m streamlit run dashboard/app.py --server.port 3000 --server.headless true
