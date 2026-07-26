---
title: MindTrace Backend
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# MindTrace Backend API

FastAPI backend for MindTrace — real-time emotion & engagement tracking.

## Endpoints
- `POST /analyze_frame` — Accepts a base64 webcam frame, returns detected emotion + bounding box
- `GET /start` · `GET /stop` · `GET /pause` · `GET /resume` — Session control
- `GET /metrics/live` — Live session metrics
- `POST /signup` · `POST /login` — Authentication
