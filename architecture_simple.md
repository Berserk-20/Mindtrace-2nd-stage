# MindTrace — Simplified Architecture

## System Layers

```mermaid
graph TD
    A["👤 User\n(Browser + Webcam)"]

    subgraph F ["🖥️ Frontend — Vercel"]
        F1["React + TypeScript App"]
        F2["Pages: Live Session, Dashboard,\nHistory, Reports, Alerts"]
    end

    subgraph B ["⚙️ Backend — Hugging Face Spaces"]
        B1["FastAPI\n(REST API)"]
        B2["ML Pipeline\n(ResNet18 + MediaPipe)"]
        B3["Auth\n(JWT Tokens)"]
    end

    subgraph D ["🗄️ Database — MongoDB Atlas"]
        D1["users"]
        D2["sessions"]
        D3["emotions"]
        D4["events"]
    end

    A -->|"uses"| F
    F -->|"HTTP requests\n+ JWT"| B
    B2 -->|"detects face\npredicts emotion"| B1
    B1 -->|"reads/writes"| D
```

---

## Live Session Flow (Step by Step)

```mermaid
sequenceDiagram
    participant U as 👤 User
    participant F as 🖥️ Frontend
    participant B as ⚙️ Backend
    participant DB as 🗄️ MongoDB

    U->>F: Clicks "Start Session"
    F->>B: GET /start
    B->>DB: Create session record
    B-->>F: session_id ✅

    loop Every 1 second
        F->>F: Capture webcam frame
        F->>B: POST /analyze_frame
        B->>B: Detect face → Predict emotion
        B-->>F: emotion + focus score
        F->>F: Show live overlay & stats
    end

    U->>F: Clicks "Stop"
    F->>B: GET /stop
    B->>DB: Save all emotions + end session
    B-->>F: Done ✅
```

---

## What Each Part Does

| Part | Role |
|------|------|
| 🖥️ **Frontend** (Vercel) | UI, webcam capture, shows charts & live stats |
| ⚙️ **Backend** (HF Spaces) | API server, processes frames, handles login |
| 🤖 **ML Pipeline** | Detects face → predicts emotion → calculates focus score |
| 🗄️ **MongoDB Atlas** | Stores users, sessions, emotion data |
