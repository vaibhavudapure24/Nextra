# 🐾 Autonomous AI Wildlife Monitoring & Sanctuary Surveillance System

[![Python 3.12+](https://img.shields.io/badge/python-3.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14%20CUDA%2013.0-EE4C2C.svg)](https://pytorch.org/)
[![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16.0-336791.svg)](https://www.postgresql.org/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live%20Demo-brightgreen.svg)](#live-demo-github-pages)

A production-grade, autonomous AI computer vision platform designed for real-time wildlife monitoring, multi-animal kinematics tracking, gait and vital health diagnostics, geo-fence escape detection, drone aerial surveillance, and automated conservation census reporting.

---

## 🌐 Live Demo (GitHub Pages)

The interactive **Tactical C2 Recon & Surveillance Web App** is ready to be hosted on **GitHub Pages**:
- **Live Interactive Demo**: `https://<YOUR-GITHUB-USERNAME>.github.io/<YOUR-REPO-NAME>/`
- Features in the GitHub Pages client:
  - 🎥 **Live WebRTC Camera Support**: Click *"TURN ON LIVE MONITORING"* to stream your local camera feed with real-time AI bounding box overlays, motion ribbons, and HUD metrics.
  - 🛰️ **Aviation Drone HUD**: Real-time artificial horizon ladder, ground speed tape, altitude ribbon, compass tape, and flight modes (Orbit, Hover, Auto-Patrol, Return-to-Home).
  - 📡 **360° Tactical Bio-Radar**: Continuous sweep scanner with real-time target blips and clickable animal dossiers.
  - 🚨 **Interactive Geo-Fence Zone Editor**: Visual perimeter boundary editing with immediate siren and flash alarms.
  - 🔊 **Web Audio Synthesizer**: Built-in tactical sound effects and alarm siren without external dependencies.

---

## 🏛️ System Architecture

```
                                  ┌───────────────────────────┐
                                  │   Video Ingestion Layer   │
                                  │ (RTSP / USB / Drone / IP) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │   Neural Detection Engine │
                                  │    (YOLO11 / YOLOv8 CUDA) │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │  Multi-Object Tracker     │
                                  │   (ByteTrack / DeepSORT)  │
                                  └─────────────┬─────────────┘
                                                │
                        ┌───────────────────────┼───────────────────────┐
                        ▼                       ▼                       ▼
              ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
              │ BiLSTM Behavior  │    │ Anomaly Detector │    │  Health & Gait   │
              │  Classification  │    │(Isolation Forest)│    │ Vitality Monitor │
              └─────────┬────────┘    └─────────┬────────┘    └─────────┬────────┘
                        │                       │                       │
                        └───────────────────────┼───────────────────────┘
                                                ▼
                                  ┌───────────────────────────┐
                                  │ PostgreSQL Storage & CRUD │
                                  │   (SQLite Auto-Fallback)  │
                                  └─────────────┬─────────────┘
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
              ┌──────────────────┐                            ┌──────────────────┐
              │ FastAPI Backend  │                            │ Streamlit 9-Page │
              │ REST & WebSocket │                            │ Executive Canvas │
              │  (Port 8000)     │                            │   (Port 8501)    │
              └──────────────────┘                            └──────────────────┘
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/<YOUR-USERNAME>/<YOUR-REPO>.git
cd <YOUR-REPO>

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Launching the Services

#### Option A: One-Click Desktop Launchers (Linux)
Double-click the desktop launcher icons generated on your Desktop:
- **`Start_Frontend.desktop`**: Starts Streamlit Dashboard (`http://localhost:8501`)
- **`Start_Backend.desktop`**: Starts FastAPI Backend & Tactical C2 Web App (`http://localhost:8000`)
- **`Start_Both_Frontend_and_Backend.desktop`**: Starts both simultaneously!

#### Option B: Terminal Commands
```bash
# 1. Initialize PostgreSQL / SQLite Database Schema
python3 main.py --init-db

# 2. Run Executive Streamlit Dashboard
python3 -m streamlit run dashboard/app.py --server.port 8501

# 3. Run FastAPI Backend & Tactical Web App
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚁 Wildlife Drone Integration (DJI / PX4 / ArduPilot)

1. **Video Downlink**: Point `configs/cameras.yaml` to your drone's RTSP or UDP feed:
   ```yaml
   cameras:
     - id: "drone_alpha"
       name: "Aerial Recon Drone 01"
       type: "drone"
       url: "rtsp://192.168.1.150:8554/live" # Or "udp://0.0.0.0:5600"
   ```
2. **MAVLink Telemetry Link**: Connect autopilot flight telemetry over UDP port `14550` or USB telemetry radio in `configs/config.yaml`:
   ```yaml
   integrations:
     drone:
       enabled: true
       connection_string: "udp:0.0.0.0:14550"
   ```
3. **Run Drone Surveillance**:
   ```bash
   python3 main.py --mode pipeline --source drone --camera-id drone_alpha
   ```

---

## 🧪 Testing Suite

Run the full automated test suite (29 tests):
```bash
python3 -m pytest tests/ -v
```

---

## 📄 License
This project is licensed under the MIT License.
