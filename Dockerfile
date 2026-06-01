FROM python:3.11-slim

WORKDIR /app

# Only what the API needs — no OpenCV, no GPU libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Split requirements: API-only install (no ultralytics / opencv)
COPY requirements.api.txt .
RUN pip install --no-cache-dir -r requirements.api.txt

# App package
COPY app/ ./app/

# Only the pipeline modules the API actually imports at runtime:
#   emit.py      — event schema (imported by ingestion.py via models)
#   __init__.py  — makes pipeline a package
# tracker.py / staff_classifier.py / zone_classifier.py / yolov8n.pt
# are ONLY used by detect.py which runs on-premises, not in this image.
COPY pipeline/__init__.py  ./pipeline/__init__.py
COPY pipeline/emit.py      ./pipeline/emit.py

# Persistent directories
RUN mkdir -p /app/events /data

EXPOSE 8000

# 2 workers is fine for SQLite; bump to 4 if you switch to Postgres
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]