from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from passlib.context import CryptContext
import shared_state

# ----------------------------------------------------------
# MONGODB CONNECTION
# ----------------------------------------------------------
import os
import sys
import time
import logging
from pymongo.errors import ConnectionFailure

# Set up basic logging for the database connection
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Use environment variable for production, fallback to localhost for dev
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "mindtrace_db"

def _connect_with_health_check(uri: str, max_retries: int = 3, timeout_ms: int = 5000):
    """
    Establish a MongoDB connection with health checks, retries, and fail-fast timeouts.
    Fails fast if Atlas (or local server) is unreachable within timeout_ms.
    Instantiates a fresh MongoClient on each retry attempt.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to MongoDB... (Attempt {attempt}/{max_retries})")
            # Fail fast if server selection timeout is exceeded
            client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
            
            # 'ping' command performs a health check to ensure the server is responsive
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB!")
            return client
        except ConnectionFailure as e:
            logger.warning(f"MongoDB connection failed on attempt {attempt}: {e}")
            if hasattr(locals(), 'client') and client is not None:
                client.close()  # Clean up failed client
            
            if attempt < max_retries:
                logger.info("Retrying in 2 seconds...")
                time.sleep(2)
            else:
                logger.error("Max retries reached. MongoDB is unreachable.")
                raise RuntimeError("Failed to connect to MongoDB Atlas after multiple retries. Check connection string and network access.")

client = _connect_with_health_check(MONGO_URI)
db = client[DB_NAME]

sessions_col = db["sessions"]
emotions_col = db["emotions"]
events_col = db["system_events"]
users_col = db["users"]

# ----------------------------------------------------------
# PASSWORD HANDLING
# ----------------------------------------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ----------------------------------------------------------
# SESSION FUNCTIONS
# ----------------------------------------------------------
def create_session(user_id: str) -> str:
    session = {
        "user_id": user_id,
        "start_time": datetime.now(),
        "end_time": None,
        "status": "running"
    }
    session_id = str(sessions_col.insert_one(session).inserted_id)
    
    # Log system event
    log_system_event(user_id, "session_start", "Session started", {"session_id": session_id})
    
    return session_id

def end_session(session_id: str):
    sessions_col.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"end_time": datetime.now(), "status": "stopped"}}
    )
    
    # Need to look up user_id from session to log event properly, or pass it in.
    # efficient update doesn't return document. Let's find, then update.
    session = sessions_col.find_one({"_id": ObjectId(session_id)})
    if session:
        log_system_event(session["user_id"], "session_end", "Session ended", {"session_id": session_id, "duration": "calculated_elsewhere"})

def get_user_sessions(user_id: str):
    """
    Retrieve all sessions for a specific user with calculated metrics.
    Returns a list of session objects with engagement and emotion data.
    """
    user_sessions = list(sessions_col.find({"user_id": user_id}).sort("start_time", -1))
    
    result = []
    for session in user_sessions:
        session_id = str(session["_id"])
        
        # Get emotions for this session
        session_emotions = list(emotions_col.find({"session_id": session_id}))
        
        # Calculate metrics
        if session_emotions:
            avg_focus = sum(e["focus_level"] for e in session_emotions) / len(session_emotions)
            # Find dominant emotion
            emotion_counts = {}
            for e in session_emotions:
                emotion = e["emotion"]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "Neutral"
        else:
            avg_focus = 0
            dominant_emotion = "Neutral"
        
        # Calculate duration
        start_time = session.get("start_time")
        end_time = session.get("end_time")
        
        # Handle active sessions
        if end_time is None:
             if session_id == shared_state.CURRENT_SESSION_ID:
                 end_time = datetime.now()
             else:
                 # It's a "zombie" session (crashed/stopped without updating DB)
                 # Find the last emotion timestamp to determine when it actually ended
                 last_entry = emotions_col.find_one({"session_id": session_id}, sort=[("timestamp", -1)])
                 if last_entry:
                     end_time = last_entry["timestamp"]
                 else:
                     end_time = start_time
        
        if start_time and end_time:
            # Ensure both are datetime objects
            if isinstance(start_time, datetime) and isinstance(end_time, datetime):
                duration_seconds = (end_time - start_time).total_seconds()
            else:
                duration_seconds = 0
        else:
            duration_seconds = 0
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        
        result.append({
            "id": session_id,
            "date": start_time.strftime("%Y-%m-%d") if start_time else "Unknown",
            "duration": duration_str,
            "engagement": round(avg_focus, 0),
            "dominantEmotion": dominant_emotion,
            "status": session.get("status", "unknown")
        })
    
    return result

def get_last_session_timeline_data(user_id: str):
    """
    Get emotion timeline data from the last completed session for a user.
    Returns data with session-relative timestamps (starting from 00:00).
    """
    # Get the most recent completed session
    last_session = sessions_col.find_one(
        {"user_id": user_id, "status": "stopped"},
        sort=[("start_time", -1)]
    )
    
    if not last_session:
        return None
    
    session_id = str(last_session["_id"])
    session_start = last_session.get("start_time")
    
    if not session_start:
        return None

    # Get all emotions for this session
    session_emotions = list(emotions_col.find({"session_id": session_id}).sort("timestamp", 1))
    
    if not session_emotions:
        return None
    
    # Build timeline data
    timeline_data = []
    for emotion_entry in session_emotions:
        entry_time = emotion_entry.get("timestamp")
        if not entry_time or not isinstance(entry_time, datetime):
            continue
            
        # Calculate elapsed time from session start
        elapsed_seconds = (entry_time - session_start).total_seconds()
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Map emotion to timeline format
        emotion = emotion_entry.get("emotion", "Neutral")
        focus_level = emotion_entry.get("focus_level", 0)
        
        timeline_data.append({
            "time": time_str,
            "emotion": emotion,
            "focus_level": focus_level
        })
    
    return {
        "session_id": session_id,
        "start_time": session_start,
        "end_time": last_session.get("end_time"),
        "timeline": timeline_data
    }



# ----------------------------------------------------------
# EMOTION + FOCUS LOGGING
# ----------------------------------------------------------
def log_emotion(session_id: str, emotion: str, focus_level: float, timestamp=None):
    """Log emotion with optional timestamp (defaults to now if not provided)"""
    emotions_col.insert_one({
        "session_id": session_id,
        "timestamp": timestamp if timestamp else datetime.now(),
        "emotion": emotion,
        "focus_level": round(float(focus_level), 3)
    })

# ----------------------------------------------------------
# SYSTEM EVENTS / NOTIFICATIONS
# ----------------------------------------------------------
def log_event(event_type: str, meta: dict | None = None):
    events_col.insert_one({
        "event": event_type,
        "meta": meta or {},
        "timestamp": datetime.now()
    })

# ----------------------------------------------------------
# USER MANAGEMENT
# ----------------------------------------------------------
def create_user(name: str, email: str, password: str, role: str = "user"):
    """Create a new user with email-based authentication"""
    users_col.insert_one({
        "name": name,
        "email": email,
        "password_hash": pwd_context.hash(password),
        "role": role,
        "created_at": datetime.now()
    })

def authenticate_user(email: str, password: str):
    """Authenticate user by email and password"""
    user = users_col.find_one({"email": email})
    if not user:
        return None

    if not pwd_context.verify(password, user["password_hash"]):
        return None

    return {
        "user_id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "role": user["role"]
    }

def user_exists(email: str) -> bool:
    """Check if user with email already exists"""
    return users_col.find_one({"email": email}) is not None

def change_password(user_id: str, new_password: str):
    users_col.update_one(
        {"_id": ObjectId(user_id)},
        {"$set": {"password_hash": pwd_context.hash(new_password)}}
    )

def delete_user(user_id: str):
    users_col.delete_one({"_id": ObjectId(user_id)})
    sessions_col.delete_many({"user_id": user_id})
    emotions_col.delete_many({"session_id": user_id})

# ----------------------------------------------------------
# ADMIN SUMMARY
# ----------------------------------------------------------
def get_user_summary():
    users = list(users_col.find({}, {"_id": 1, "name": 1, "email": 1, "role": 1}))
    summary = []

    for u in users:
        uid = str(u["_id"])
        user_sessions = list(sessions_col.find({"user_id": uid}, {"_id": 1}))
        session_ids = [str(s["_id"]) for s in user_sessions]
        emotion_count = emotions_col.count_documents({"session_id": {"$in": session_ids}}) if session_ids else 0
        summary.append({
            "user_id": uid,
            "name": u.get("name", "Unknown"),
            "email": u.get("email", ""),
            "role": u["role"],
            "total_sessions": len(session_ids),
            "total_emotions": emotion_count
        })

    return summary


# ----------------------------------------------------------
# EMOTION ANALYTICS
# ----------------------------------------------------------
def get_emotion_distribution(user_id: str):
    """Calculate emotion distribution across all user sessions"""
    from collections import defaultdict
    
    # Get all sessions for this user
    user_sessions = list(sessions_col.find({"user_id": user_id}))
    if not user_sessions:
        return []
    
    session_ids = [str(s["_id"]) for s in user_sessions]
    
    # Count emotions across all sessions
    emotion_counts = defaultdict(int)
    total = 0
    
    for session_id in session_ids:
        emotions = list(emotions_col.find({"session_id": session_id}))
        for e in emotions:
            emotion_counts[e["emotion"]] += 1
            total += 1
    
    if total == 0:
        return []
    
    # Color mapping for each emotion
    emotion_colors = {
        "Neutral": "hsl(215, 15%, 70%)",
        "Happy": "hsl(152, 60%, 45%)",
        "Joy": "hsl(152, 60%, 45%)",
        "Angry": "hsl(0, 72%, 60%)",
        "Sad": "hsl(221, 83%, 65%)",
        "Fear": "hsl(45, 93%, 60%)",
        "Surprise": "hsl(38, 92%, 60%)",
        "Disgust": "hsl(291, 64%, 60%)",
        "Focus": "hsl(187, 80%, 48%)",
        "Frustration": "hsl(0, 72%, 60%)"
    }
    
    # Calculate percentages
    distribution = []
    for emotion, count in emotion_counts.items():
        percentage = round((count / total) * 100, 1)
        distribution.append({
            "emotion": emotion,
            "value": percentage,
            "color": emotion_colors.get(emotion, "hsl(215, 15%, 50%)")
        })
    
    # Sort by value descending
    distribution.sort(key=lambda x: x["value"], reverse=True)
    return distribution


def get_emotion_transitions(user_id: str):
    """Track emotion transitions (sequential changes) across user sessions"""
    from collections import defaultdict
    
    # Get all sessions for this user
    user_sessions = list(sessions_col.find({"user_id": user_id}))
    if not user_sessions:
        return []
    
    session_ids = [str(s["_id"]) for s in user_sessions]
    
    # Track transitions
    transitions = defaultdict(int)
    
    for session_id in session_ids:
        # Get emotions in chronological order
        emotions = list(emotions_col.find({"session_id": session_id}).sort("timestamp", 1))
        
        # Track sequential pairs
        for i in range(len(emotions) - 1):
            current = emotions[i]["emotion"]
            next_emotion = emotions[i + 1]["emotion"]
            
            # Only count if emotion actually changed
            if current != next_emotion:
                transitions[(current, next_emotion)] += 1
    
    # Convert to list format
    transition_list = []
    for (from_emotion, to_emotion), count in transitions.items():
        transition_list.append({
            "from": from_emotion,
            "to": to_emotion,
            "count": count
        })
    
    # Sort by count descending
    transition_list.sort(key=lambda x: x["count"], reverse=True)
    
    # Return top 10 transitions
    return transition_list[:10]


def get_session_emotion_breakdown(user_id: str, limit: int = 5):
    """
    Get emotion distribution for the last N sessions.
    Returns format suitable for Recharts stacked bar chart:
    [
      { "session": "S-1024", "Joy": 20, "Focus": 50, ... },
      ...
    ]
    """
    from collections import defaultdict
    
    # Get last N sessions for this user, sorted by start_time descending
    user_sessions = list(sessions_col.find({"user_id": user_id}).sort("start_time", -1).limit(limit))
    
    if not user_sessions:
        return []
    
    result = []
    
    # Process mostly recent first, so reverse to show chronological left-to-right if desired, 
    # or keep as is. Usually charts show time axis left-to-right.
    # Let's sort explicitly by start_time ascending for the chart
    user_sessions.sort(key=lambda x: x["start_time"])
    
    for session in user_sessions:
        session_id = str(session["_id"])
        # Use a short identifier for the chart axis
        short_id = f"S-{session_id[-4:]}"
        
        # Get all emotions for this session
        emotions = list(emotions_col.find({"session_id": session_id}))
        total = len(emotions)
        
        if total == 0:
            continue
            
        # Count emotions
        counts = defaultdict(int)
        for e in emotions:
            counts[e["emotion"]] += 1
            
        # Build the data object
        session_data = {"session": short_id}
        
        # Calculate percentages for each emotion
        # We need to ensure all potential emotions are keys if we want consistent stacking,
        # but Recharts handles missing keys fine (just no bar segment).
        for emotion, count in counts.items():
            percentage = round((count / total) * 100, 1)
            session_data[emotion] = percentage
            
        result.append(session_data)
        
    return result

def get_user_profile(user_id: str):
    """Get user profile details excluding password"""
    user = users_col.find_one({"email": user_id}, {"password": 0})
    if user:
        user["id"] = str(user["_id"])
        del user["_id"]
        return user
    return None

# ----------------------------------------------------------
# SYSTEM EVENTS LOGGING
# ----------------------------------------------------------
def log_system_event(user_id: str, event_type: str, description: str, metadata: dict = None):
    """Log a system event (e.g., Session Start, Alert Triggered)"""
    event = {
        "user_id": user_id,
        "timestamp": datetime.now(),
        "type": event_type,
        "description": description,
        "metadata": metadata or {}
    }
    events_col.insert_one(event)

def get_recent_system_events(user_id: str, limit: int = 10):
    """Get recent system events for a user"""
    events = list(events_col.find({"user_id": user_id}).sort("timestamp", -1).limit(limit))
    
    result = []
    for event in events:
        result.append({
            "id": str(event["_id"]),
            "timestamp": event["timestamp"], # Keep as datetime for now, or format string
            "type": event["type"],
            "description": event["description"],
            "metadata": event.get("metadata", {})
        })
    return result

# ----------------------------------------------------------
# REPORTS & ALERTS HELPERS
# ----------------------------------------------------------

def get_engagement_stats(user_id: str):
    """
    Get aggregated engagement statistics for the user.
    """
    from datetime import timedelta
    
    # Check if user_id is an ObjectId (it might be passed as string email or ID)
    # The current system uses EMAIL as user_id for sessions?
    # No, create_session takes user_id, which is email from start() endpoint.
    # So user_id is email.
    
    now = datetime.now()
    seven_days_ago = now - timedelta(days=7)
    
    # 1. Weekly Trend (Last 7 days)
    daily_stats = {}
    for i in range(7):
        day_date = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_stats[day_date] = {"total_engagement": 0, "session_count": 0}
    
    # Fetch sessions 
    recent_sessions = list(sessions_col.find({
        "user_id": user_id,
        "start_time": {"$gte": seven_days_ago}
    }))
    
    total_engagement_sum = 0
    total_duration_seconds = 0
    
    # Session IDs for emotion aggregation
    recent_session_ids = []
    
    for session in recent_sessions:
        sid = str(session["_id"])
        recent_session_ids.append(sid)
        
        start_date = session["start_time"].strftime("%Y-%m-%d")
        
        # Get emotions for this session to calc average
        emotions = list(emotions_col.find({"session_id": sid}))
        if emotions:
            avg_val = sum(e["focus_level"] for e in emotions) / len(emotions)
        else:
            avg_val = 0
            
        if start_date in daily_stats:
            daily_stats[start_date]["total_engagement"] += (avg_val * 100)
            daily_stats[start_date]["session_count"] += 1
            
        total_engagement_sum += avg_val
        
        # Duration calculation
        start = session.get("start_time")
        end = session.get("end_time")
        if start and end: 
            total_duration_seconds += (end - start).total_seconds()

    # Format Chart Data
    weekly_data = []
    sorted_dates = sorted(daily_stats.keys())
    for d in sorted_dates:
        stats = daily_stats[d]
        avg = stats["total_engagement"] / stats["session_count"] if stats["session_count"] > 0 else 0
        day_name = datetime.strptime(d, "%Y-%m-%d").strftime("%a")
        weekly_data.append({"day": day_name, "avg": round(avg, 1)})

    # Summary Metrics
    total_sessions_count = len(recent_sessions)
    avg_engagement_overall = (total_engagement_sum / total_sessions_count * 100) if total_sessions_count > 0 else 0
    avg_duration = total_duration_seconds / total_sessions_count if total_sessions_count > 0 else 0
    
    hours = int(avg_duration // 3600)
    minutes = int((avg_duration % 3600) // 60)
    avg_duration_str = f"{hours}h {minutes}m"

    # 2. Hourly Heatmap (All Time)
    # Get ALL user session IDs first
    all_user_sessions = list(sessions_col.find({"user_id": user_id}, {"_id": 1}))
    all_session_ids = [str(s["_id"]) for s in all_user_sessions]
    
    hourly_pipeline = [
        {"$match": {"session_id": {"$in": all_session_ids}}},
        {"$project": {
            "hour": {"$hour": "$timestamp"},
            "focus_level": 1
        }},
        {"$group": {
            "_id": "$hour",
            "avg_focus": {"$avg": "$focus_level"}
        }},
        {"$sort": {"_id": 1}}
    ]
    
    hourly_engagement = []
    if all_session_ids:
        agg_res = list(emotions_col.aggregate(hourly_pipeline))
        res_map = {item["_id"]: item["avg_focus"] * 100 for item in agg_res}
        
        for h in range(24):
            val = res_map.get(h, 0)
            hourly_engagement.append({
                "hour": f"{h:02d}:00",
                "engagement": round(val, 1)
            })
    else:
        # Default empty
        for h in range(24):
            hourly_engagement.append({"hour": f"{h:02d}:00", "engagement": 0})

    return {
        "weeklyData": weekly_data,
        "summary": {
            "avgEngagement": f"{round(avg_engagement_overall, 1)}%",
            "trend": "+0.0%",  # Complex to calc real trend simply
            "totalSessions": str(total_sessions_count),
            "avgDuration": avg_duration_str
        },
        "hourlyData": hourly_engagement
    }

def generate_alerts(user_id: str):
    """
    Generate alerts from user sessions.
    """
    alerts = []
    # Get last 10 sessions
    sessions = list(sessions_col.find({"user_id": user_id}).sort("start_time", -1).limit(10))
    
    count = 1
    for session in sessions:
        sid = str(session["_id"])
        short_id = f"S-{sid[-4:]}"
        s_time = session["start_time"]
        
        # Calculate time ago
        now = datetime.now()
        diff = now - s_time
        if diff.days > 0:
            time_ago = f"{diff.days}d ago"
        elif diff.seconds > 3600:
            time_ago = f"{diff.seconds // 3600}h ago"
        else:
            time_ago = f"{diff.seconds // 60}m ago"
            
        emotions = list(emotions_col.find({"session_id": sid}))
        
        # Analysis
        if emotions:
            avg_focus = sum(e["focus_level"] for e in emotions) / len(emotions)
            frustration_count = sum(1 for e in emotions if e["emotion"] in ["Angry", "Disgust", "Frustration"])
            frustration_ratio = frustration_count / len(emotions)
            
            # Message Logic
            if frustration_ratio > 0.3:
                alerts.append({
                    "id": count,
                    "type": "error",
                    "title": "High Frustration detected",
                    "description": f"Frustration levels spiked >30% during session {short_id}.",
                    "time": time_ago,
                    "session": short_id
                })
                count += 1
            if avg_focus < 0.3:
                alerts.append({
                    "id": count,
                     "type": "warning",
                    "title": "Low Engagement",
                    "description": f"Engagement dropped significantly during {short_id}.",
                    "time": time_ago,
                    "session": short_id
                })
                count += 1
            if avg_focus > 0.7:
                alerts.append({
                    "id": count,
                    "type": "success",
                    "title": "High Focus Session",
                    "description": f"Great focus maintained (>70%) in {short_id}.",
                    "time": time_ago,
                    "session": short_id
                })
                count += 1
                
    # Add a welcome/system alert if empty
    if not alerts:
        alerts.append({
            "id": 999,
            "type": "info",
            "title": "System Ready",
            "description": "Welcome to MindTrace. Alerts will appear here as you complete sessions.",
            "time": "Just now",
            "session": "System"
        })
        
    return alerts
