# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY static ./static

# Base OS deps (curl/ca-certificates handy; --no-install-recommends keeps it slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
  && rm -rf /var/lib/apt/lists/*

# 1) Install Python deps (playwright pinned in requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) Install Chromium + system deps that Playwright needs
#    --with-deps works on Debian-based images and Cloud Run
RUN python -m playwright install --with-deps chromium

# 3) Copy application code
COPY recruiting ./recruiting
COPY main.py .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
