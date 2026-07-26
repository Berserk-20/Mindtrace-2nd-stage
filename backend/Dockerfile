# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies (required by OpenCV-headless & MediaPipe on Linux) ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python packages (cached layer) ───────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY . .

# ── HF Spaces requires port 7860 ─────────────────────────────────────────────
EXPOSE 7860

# ── Start FastAPI ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
