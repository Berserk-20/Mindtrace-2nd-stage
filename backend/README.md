# 🧠 MindTrace Backend API

Welcome to the **MindTrace Backend**! This repository houses the powerful FastAPI application that drives real-time emotion detection and engagement tracking for the MindTrace platform. 

MindTrace uses computer vision and deep learning to analyze facial expressions from video feeds (like webcams) in real-time, providing immediate feedback on user engagement and emotional states.

## 🚀 Features

- **Real-Time Emotion Detection:** Accepts base64 encoded webcam frames and processes them instantly.
- **Deep Learning Powered:** Utilizes state-of-the-art PyTorch models trained on facial expression datasets (like RAF-DB).
- **Engagement Metrics:** Calculates focus and engagement scores based on detected emotions over time.
- **Session Management:** Start, pause, resume, and stop tracking sessions with ease.
- **Authentication:** Built-in user signup and login routes.

## 🛠️ Technology Stack

- **Framework:** [FastAPI](https://fastapi.tiangolo.com/) - Lightning fast, modern Python web framework.
- **Machine Learning:** [PyTorch](https://pytorch.org/) - For running inference on emotion detection models.
- **Server:** [Uvicorn](https://www.uvicorn.org/) - ASGI server for Python.
- **Database:** SQLite (local dev) / PostgreSQL (production ready).

## 🚏 Core Endpoints

### Session & Analysis
- `POST /analyze_frame` - Submit a base64 frame, returns detected emotion & bounding box.
- `GET /metrics/live` - Retrieve live session metrics and engagement scores.
- `GET /start` - Initialize a new tracking session.
- `GET /pause` - Pause the current session.
- `GET /resume` - Resume a paused session.
- `GET /stop` - End the current session and save logs.

### Authentication
- `POST /signup` - Register a new user.
- `POST /login` - Authenticate and retrieve a session token.

## 💻 Local Development Setup

Follow these steps to get the backend running locally on your machine.

### Prerequisites
- Python 3.9+
- pip (Python package manager)

### Installation

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Copy the example environment file and fill in your details:
   ```bash
   cp .env.example .env
   ```

5. **Start the Development Server:**
   ```bash
   uvicorn api:app --reload --host 0.0.0.0 --port 8000
   ```
   The API will now be available at `http://localhost:8000`. You can view the interactive API documentation at `http://localhost:8000/docs`.

## 🐳 Docker Support

If you prefer using Docker, you can build and run the container easily:

```bash
docker build -t mindtrace-backend .
docker run -p 8000:8000 mindtrace-backend
```
