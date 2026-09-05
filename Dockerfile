# ==============================================================================
# AI Wildlife Animal Monitoring System - Application Dockerfile
# Multi-purpose image: can run main.py (pipeline), api/main.py (FastAPI),
# or dashboard/app.py (Streamlit) depending on the CMD override in
# docker-compose.yml.
#
# For GPU/CUDA inference, build with a CUDA base image instead:
#   docker build -f Dockerfile.gpu -t wildlife-monitoring:gpu .
# (see Dockerfile.gpu below in this same directory)
# ==============================================================================

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System dependencies required by OpenCV, psycopg2, and video codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
        libpq-dev \
        gcc \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs outputs models datasets

EXPOSE 8000 8501

# Default command runs the FastAPI backend; override in docker-compose.yml
# to run the Streamlit dashboard or the standalone main.py pipeline instead.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
