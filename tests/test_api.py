import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from starlette.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_docs():
    response = client.get("/docs")
    assert response.status_code == 200


def test_api_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "device" in data
    assert data["service"] == "AI Wildlife Animal Monitoring Backend"


def test_api_live():
    response = client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_api_history():
    response = client.get("/history")
    assert response.status_code == 200
    data = response.json()
    assert "detections" in data


def test_api_reports_list():
    response = client.get("/report")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_api_track_endpoint():
    payload = {
        "tracker_type": "bytetrack",
        "camera_id": "test_cam",
        "detections": [
            {
                "class_id": 0,
                "species": "elephant",
                "confidence": 0.88,
                "x1": 100,
                "y1": 100,
                "x2": 250,
                "y2": 250,
            }
        ],
    }
    response = client.post("/track", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tracker"] == "bytetrack"
    assert "tracks" in data


def test_api_anomaly_endpoint():
    payload = {
        "features": [1.2, 0.1, 0.2, 0.05, 0.0, 0.0, 0.8, 10.0],
        "animal_id": 1,
    }
    response = client.post("/anomaly", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
