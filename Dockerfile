# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # optional: keep browsers in a predictable path
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# OS deps (Playwright will pull most libs with --with-deps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    # helpful for font/rendering stability
    fonts-liberation fonts-noto-color-emoji \
 && rm -rf /var/lib/apt/lists/*

# Python deps (must include `playwright`)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Chromium + required system libs
RUN python -m playwright install --with-deps chromium

# App code
COPY . .

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
