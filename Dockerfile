# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # keep Playwright browser binaries inside the image (Cloud Run friendly)
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    # prevent chromium from trying to use tiny /dev/shm
    NODE_OPTIONS=--max-old-space-size=256

# --- System deps needed by Chromium/Playwright and TLS ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    # common Chromium deps
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libxkbcommon0 \
    libxdamage1 libxcomposite1 libxrandr2 libxfixes3 libdrm2 libgbm1 \
    libasound2 libpangocairo-1.0-0 libgtk-3-0 fonts-liberation \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Static assets (served by FastAPI at /static)
COPY static ./static

# --- Python deps ---
COPY requirements.txt .
# Install your deps, then Playwright + Chromium
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir playwright==1.47.2 \
 && python -m playwright install --with-deps chromium

# --- App code ---
COPY recruiting ./recruiting
COPY main.py .

# Cloud Run listens on $PORT; default to 8080
ENV PORT=8080

# Optional: make a non-root user (safer). Cloud Run supports this fine.
RUN useradd -m appuser
USER appuser

EXPOSE 8080

# Uvicorn entrypoint.
# NOTE: In your code, launch Chromium with args=["--no-sandbox","--disable-dev-shm-usage"]
# (You already wired Playwright in code; just ensure those flags are set when launching.)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
