from fastapi import FastAPI, HTTPException, Body, Depends
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 IMPORTANT (temporary fix)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel, EmailStr, validator
import json
import os
from dotenv import load_dotenv
load_dotenv()

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import re

import cv2
import time
import collections
import numpy as np
from bson import ObjectId

import shared_state
# background_agent imports removed to prevent blocking on fastAPI startup
from db import create_user, authenticate_user, user_exists, get_user_sessions, create_session, end_session, log_emotion, get_last_session_timeline_data, get_emotion_distribution, get_emotion_transitions, get_session_emotion_breakdown, get_recent_system_events, get_user_profile, get_user_summary, emotions_col, sessions_col
from auth import get_current_user

# =====================================================
# FASTAPI APP
# =====================================================
app = FastAPI(title="MindTrace API")

# =====================================================
# CORS (REQUIRED FOR FRONTEND)
# =====================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-mindtrace-frontend.vercel.app",  # Production URL
        "http://localhost:8080",                       # Local dev
        "http://localhost:5173",                       # Local dev
        "http://127.0.0.1:8080",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

# =====================================================
# STARTUP EVENT
# =====================================================
@app.on_event("startup")
def startup_event():
    # Application startup logic (background agent starts only via /start API)
    pass

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

# =====================================================
# STREAM ENDPOINT
# =====================================================
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
def start(current_user: dict = Depends(get_current_user)):
    # Store current user email
    shared_state.CURRENT_USER_EMAIL = current_user["email"]
    
    # Create session in database
    session_id = create_session(current_user["email"])
    shared_state.CURRENT_SESSION_ID = session_id
    
    # Agent disabled on cloud deployment to prevent worker timeout
    return JSONResponse({
        "success": True,
        "message": "Agent disabled on cloud deployment",
        "agent_running": False,
        "session_id": session_id
    })

@app.get("/pause")
def pause(current_user: dict = Depends(get_current_user)):
    try:
        from background_agent import pause_agent
        pause_agent()
    except Exception as e:
        print("Pause agent skipped:", e)
    
    return {
        "success": True,
        "paused": True
    }

@app.get("/resume")
def resume(current_user: dict = Depends(get_current_user)):
    try:
        from background_agent import resume_agent
        resume_agent()
    except Exception as e:
        print("Resume agent skipped:", e)
    
    return {
        "success": True,
        "paused": False
    }

@app.get("/stop")
def stop(current_user: dict = Depends(get_current_user)):
    try:
        from background_agent import stop_agent, get_engagement_score
    except Exception as e:
        print("Stop agent skipped:", e)
        stop_agent = None
        get_engagement_score = None

    # 👇 KEEP YOUR EXISTING DB LOGIC SAME
    if shared_state.CURRENT_SESSION_ID:
        with shared_state.DATA_LOCK:
            for emotion_data in shared_state.EMOTION_HISTORY:
                from datetime import datetime
                detection_time = datetime.fromtimestamp(emotion_data["timestamp"])

                if "focus_score" in emotion_data:
                    raw_score = emotion_data["focus_score"]
                elif get_engagement_score:
                    raw_score = get_engagement_score(emotion_data["emotion"])
                else:
                    raw_score = 50  # fallback

                focus_score = raw_score / 100.0

                log_emotion(
                    shared_state.CURRENT_SESSION_ID,
                    emotion_data["emotion"],
                    focus_score,
                    timestamp=detection_time
                )

        end_session(shared_state.CURRENT_SESSION_ID)
        shared_state.CURRENT_SESSION_ID = None

    if stop_agent:
        stop_agent()

    return {
        "success": True,
        "agent_running": False
    }

# =====================================================
# HEALTH CHECK (DEBUG / TEST)
# =====================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "agent_running": shared_state.AGENT_RUNNING,
        "paused": shared_state.PAUSE_REQUESTED,
        "frame_available": shared_state.LATEST_FRAME is not None
    }

# =====================================================
# METRICS ENDPOINT (REPLACES MOCK DATA)
# =====================================================
@app.get("/metrics")
def get_metrics(current_user: dict = Depends(get_current_user)):
    # Helper to safe get list
    with shared_state.DATA_LOCK:
        emotion_history = list(shared_state.EMOTION_HISTORY)
        engagement_history = list(shared_state.ENGAGEMENT_HISTORY)
        session_start = shared_state.SESSION_START_TIME

    # 1. KPI Data
    # Calculate averages from history or default
    if engagement_history:
        current_eng = engagement_history[-1]["score"]
        avg_eng = np.mean([e["score"] for e in engagement_history])
        eng_change = current_eng - avg_eng # Simplified
    else:
        current_eng = 0
        eng_change = 0

    kpiData = {
        "engagementScore": {"value": round(current_eng, 1), "change": round(eng_change, 1), "label": "Engagement Score"},
        "focusStability": {"value": 85.0, "change": 0.0, "label": "Focus Stability"}, # Placeholder
        "emotionVariance": {"value": 0.5, "change": 0.0, "label": "Emotion Variance"}, # Placeholder
        "distractionIndex": {"value": 10.0, "change": 0.0, "label": "Distraction Index"}, # Placeholder
    }

    # 2. Emotion Timeline Data
    # Get data from last completed session instead of live data
    user_email = current_user["email"]
    last_session_data = get_last_session_timeline_data(user_email)
    
    emotionTimelineData = []
    
    if last_session_data and last_session_data["timeline"]:
        # Aggregate emotions by time bins (every 5 seconds or so)
        timeline = last_session_data["timeline"]
        
        # Group by time windows
        time_bins = {}
        for entry in timeline:
            time_key = entry["time"]
            emotion = entry["emotion"].lower()
            
            # Normalize emotion names
            if emotion == "happy": emotion = "joy"
            if emotion == "angry": emotion = "frustration"
            if emotion == "sad": emotion = "frustration"
            
            if time_key not in time_bins:
                time_bins[time_key] = {
                    "joy": 0, "focus": 0, "surprise": 0,
                    "neutral": 0, "frustration": 0, "count": 0
                }
            
            # Increment the detected emotion
            if emotion in time_bins[time_key]:
                time_bins[time_key][emotion] += entry["focus_level"]
                time_bins[time_key]["count"] += 1
        
        # Convert to list format
        for time_key in sorted(time_bins.keys()):
            bin_data = time_bins[time_key]
            count = bin_data.pop("count")
            
            # Normalize values if needed
            if count > 0:
                for key in bin_data:
                    bin_data[key] = bin_data[key] / count if count > 0 else 0
            
            emotionTimelineData.append({
                "time": time_key,
                **bin_data
            })
    
    # If no last session, return sample data
    if not emotionTimelineData:
        emotionTimelineData = []
        for i in range(30):
            hours = i // 2
            mins = "00" if i % 2 == 0 else "30"
            emotionTimelineData.append({
                "time": f"{hours:02d}:{mins}",
                "joy": 40 + np.random.uniform(0, 30),
                "focus": 60 + np.random.uniform(0, 25),
                "surprise": 10 + np.random.uniform(0, 20),
                "neutral": 30 + np.random.uniform(0, 20),
                "frustration": 5 + np.random.uniform(0, 15)
            })


    # 3. Engagement Data (Engagement vs Attention from last session)
    engagementData = []
    
    if last_session_data and last_session_data["timeline"]:
        # Use the same timeline data
        for entry in last_session_data["timeline"]:
            engagement_score = entry["focus_level"] * 100  # Convert to percentage
            attention_score = engagement_score * 0.95  # Attention slightly lower
            
            engagementData.append({
                "hour": entry["time"],
                "engagement": round(engagement_score, 1),
                "attention": round(attention_score, 1)
            })
    
    # If no last session, return sample data
    if not engagementData:
        engagementData = []
        for i in range(24):
            engagementData.append({
                "hour": f"{i:02d}:00",
                "engagement": 50 + np.sin(i / 3) * 30 + np.random.uniform(0, 10),
                "attention": 60 + np.cos(i / 4) * 20 + np.random.uniform(0, 10)
            })

    # 4. Insights (Mocked/Rule-based)
    insights = [
        {
            "id": "1",
            "type": "info",
            "title": "Live Session Active",
            "description": "Real-time data is being streamed from the Python backend.",
            "timestamp": "Just now",
        }
    ]

    # 5. Sessions (Real data from MongoDB, filtered by user)
    user_email = current_user["email"]
    sessions = get_user_sessions(user_email)

    # 6. Emotion Distribution
    # Calculate from history (Real data from DB)
    emotionDistribution = get_emotion_distribution(user_id=current_user["email"])

    # 7. Emotion Transitions (Real data from DB)
    emotionTransitions = get_emotion_transitions(user_id=current_user["email"])

    # 8. Live Metrics
    # Get latest
    current_emotion = emotion_history[-1]["emotion"] if emotion_history else "Neutral"
    emotion_confidence = emotion_history[-1]["confidence"] if emotion_history else 0.0
    
    # Calculate session duration accounting for paused time
    duration_str = "00:00:00"
    if session_start:
        if shared_state.PAUSE_REQUESTED and shared_state.PAUSE_START_TIME:
            # Currently paused - duration should freeze at pause time
            elapsed = shared_state.PAUSE_START_TIME - session_start - shared_state.TOTAL_PAUSED_TIME
        elif shared_state.PAUSE_REQUESTED:
            # Paused but PAUSE_START_TIME not set yet (edge case)
            elapsed = 0
        else:
            # Running normally - calculate actual elapsed time minus all paused time
            elapsed = time.time() - session_start - shared_state.TOTAL_PAUSED_TIME
        
        # Ensure we don't show negative duration
        elapsed = max(0, elapsed)
        duration_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))

    liveMetrics = {
        "currentEmotion": current_emotion,
        "emotionConfidence": round(emotion_confidence, 2),
        "engagementLevel": round(current_eng, 1),
        "attentionScore": round(current_eng * 0.95, 1),
        "sessionDuration": duration_str,
        "sessionStartTime": session_start,  # Send session start time for client-side timer
    }

    # 9. Session Emotion Breakdown (Real data from DB)
    sessionEmotions = get_session_emotion_breakdown(user_id=current_user["email"], limit=5)

    # 10. System Events (Real data from DB)
    systemEvents = get_recent_system_events(user_id=current_user["email"], limit=5)

    return {
        "kpiData": kpiData,
        "emotionTimelineData": emotionTimelineData,
        "engagementData": engagementData,
        "insights": insights,
        "sessions": sessions,
        "emotionDistribution": emotionDistribution,
        "emotionTransitions": emotionTransitions,
        "sessionEmotions": sessionEmotions,
        "liveMetrics": liveMetrics,
        "systemEvents": systemEvents
    }

# =====================================================
# SESSION LIST ENDPOINT
# =====================================================
@app.get("/api/sessions")
def get_sessions_list(current_user: dict = Depends(get_current_user)):
    """Get all sessions for the authenticated user"""
    user_email = current_user["email"]
    sessions = get_user_sessions(user_email)
    return sessions

# =====================================================
# EMOTIONS FOR SESSION ENDPOINT
# =====================================================
@app.get("/api/emotions")
def get_emotions_for_session(session_id: str, current_user: dict = Depends(get_current_user)):
    """Get all emotion entries for a specific session"""
    try:
        # Try to get session by ObjectId first
        session = sessions_col.find_one({"_id": ObjectId(session_id)})
    except:
        # If that fails, session_id might already be a string
        session = None
    
    if not session:
        # Session not found, return empty
        print(f"Session not found: {session_id}")
        return []
    
    session_start = session["start_time"]
    
    # Get emotions from database - session_id is stored as string
    emotions_data = list(emotions_col.find({"session_id": session_id}).sort("timestamp", 1))
    
    print(f"Found {len(emotions_data)} emotions for session {session_id}")
    print(f"Session start time: {session_start}")
    
    # Format for frontend with session-relative timestamps
    result = []
    for i, entry in enumerate(emotions_data):
        # Calculate elapsed time from session start
        elapsed_seconds = (entry["timestamp"] - session_start).total_seconds()
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        relative_time = f"{minutes:02d}:{seconds:02d}"
        
        if i < 3:  # Log first 3 entries for debugging
            print(f"Entry {i}: timestamp={entry['timestamp']}, elapsed={elapsed_seconds}s, relative={relative_time}")
        
        result.append({
            "timestamp": relative_time,  # Session-relative time
            "emotion": entry["emotion"],
            "focus_level": entry["focus_level"]
        })
    
    print(f"Returning {len(result)} emotion entries")
    return result


# =====================================================
# AUTHENTICATION & SECURITY
# =====================================================

# JWT settings
# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("CRITICAL: SECRET_KEY environment variable is not set!")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440

class UserSignup(BaseModel):
    name: str
    email: EmailStr
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        if len(v.encode('utf-8')) > 72:
            raise ValueError('Password too long (max 72 bytes allowed)')
    
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
    
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        
        if not re.search(r'[0-9]', v):
            raise ValueError('Password must contain at least one digit')
        
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str


def create_access_token(data: dict, expires_delta: timedelta = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/signup")
def signup(user: UserSignup):
    """Register a new user with MongoDB and hashed password"""
    # Check if email already exists
    if user_exists(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create user in MongoDB (password is hashed in db.py)
    create_user(user.name, user.email, user.password)
    
    return {
        "message": "User created successfully", 
        "user": {"name": user.name, "email": user.email}
    }

@app.post("/login")
def login(user: UserLogin):
    """Authenticate user with MongoDB and return JWT token"""
    # Authenticate with MongoDB
    authenticated = authenticate_user(user.email, user.password)
    
    if not authenticated:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create JWT token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": authenticated["email"], 
            "name": authenticated["name"],
            "role": authenticated["role"]
        },
        expires_delta=access_token_expires
    )
    
    return {
        "message": "Login successful",
        "user": {
            "name": authenticated["name"], 
            "email": authenticated["email"],
            "role": authenticated["role"]
        },
        "token": access_token
    }

# =====================================================
# REPORTS AND ALERTS ENDPOINTS
# =====================================================
from db import get_engagement_stats, generate_alerts

@app.get("/api/reports")
def get_reports(current_user: dict = Depends(get_current_user)):
    """Get engagement reports and heatmap data"""
    # Force pause to allow for slight data processing delay if needed, or just run query
    return get_engagement_stats(current_user["email"])

@app.get("/api/alerts")
def get_alerts(current_user: dict = Depends(get_current_user)):
    """Get generated system alerts"""
    return generate_alerts(current_user["email"])

# =====================================================
# ADMIN ENDPOINTS
# =====================================================
from db import get_user_summary

@app.get("/api/admin/stats")
def get_admin_stats(current_user: dict = Depends(get_current_user)):
    """Get system-wide statistics for Admin Dashboard"""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=403, 
            detail="Access forbidden: Admins only"
        )
    
    # Get user summary from DB
    users_summary = get_user_summary()
    
    # Aggregate stats
    total_users = len(users_summary)
    total_sessions = sum(u["total_sessions"] for u in users_summary)
    total_emotions = sum(u["total_emotions"] for u in users_summary)
    active_sessions = 0 # TODO: Calculate real active sessions if needed
    
    return {
        "stats": {
            "totalUsers": total_users,
            "totalSessions": total_sessions,
            "totalEmotions": total_emotions,
            "activeSessions": active_sessions
        },
        "users": users_summary
    }

@app.get("/api/me")
def read_users_me(current_user: dict = Depends(get_current_user)):
    """Get current user profile"""
    # Fetch full profile from DB to get all fields, not just what's in token
    user_profile = get_user_profile(current_user["email"])
    if not user_profile:
        raise HTTPException(status_code=404, detail="User not found")
    return user_profile

@app.get("/")
def home():
    return {"message": "MindTrace API is running 🚀"}