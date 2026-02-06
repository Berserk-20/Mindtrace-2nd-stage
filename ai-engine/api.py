from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import time

import shared_state
from background_agent import (
    start_agent_thread,
    stop_agent,
    pause_agent,
    resume_agent
)

app = FastAPI(title="MindTrace API")

# =====================================================
# START BACKGROUND AGENT ON SERVER START
# =====================================================
@app.on_event("startup")
def startup_event():
    start_agent_thread()

# =====================================================
# VIDEO STREAM (MJPEG)
# =====================================================
def generate_frames():
    last_sent = 0
    STREAM_FPS = 10
    INTERVAL = 1.0 / STREAM_FPS

    while True:
        now = time.time()
        if now - last_sent < INTERVAL:
            time.sleep(0.005)
            continue
        last_sent = now

        with shared_state.FRAME_LOCK:
            frame = (
                shared_state.LATEST_FRAME.copy()
                if shared_state.LATEST_FRAME is not None
                else None
            )

        if frame is None:
            time.sleep(0.05)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# =====================================================
# AGENT CONTROL ENDPOINTS
# =====================================================
@app.get("/start")
def start():
    start_agent_thread()
    return JSONResponse({
        "success": True,
        "agent_running": shared_state.AGENT_RUNNING
    })

@app.get("/pause")
def pause():
    pause_agent()
    return JSONResponse({
        "success": True,
        "paused": True
    })

@app.get("/resume")
def resume():
    resume_agent()
    return JSONResponse({
        "success": True,
        "paused": False
    })

@app.get("/stop")
def stop():
    stop_agent()
    return JSONResponse({
        "success": True,
        "agent_running": False
    })

# =====================================================
# HEALTH CHECK (VERY USEFUL)
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_running": shared_state.AGENT_RUNNING,
        "paused": shared_state.PAUSE_REQUESTED,
        "frame_available": shared_state.LATEST_FRAME is not None
    }
