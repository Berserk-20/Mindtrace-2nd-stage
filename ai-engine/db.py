from pymongo import MongoClient
from datetime import datetime
from bson import ObjectId
from passlib.context import CryptContext

# ----------------------------------------------------------
# MONGODB CONNECTION
# ----------------------------------------------------------
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "mindtrace_db"

client = MongoClient(MONGO_URI)
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
    return str(sessions_col.insert_one(session).inserted_id)

def end_session(session_id: str):
    sessions_col.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"end_time": datetime.now(), "status": "stopped"}}
    )

# ----------------------------------------------------------
# EMOTION + FOCUS LOGGING
# ----------------------------------------------------------
def log_emotion(session_id: str, emotion: str, focus_level: float):
    emotions_col.insert_one({
        "session_id": session_id,
        "timestamp": datetime.now(),
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
def create_user(username: str, password: str, role: str = "user"):
    users_col.insert_one({
        "username": username,
        "password_hash": pwd_context.hash(password),
        "role": role
    })

def authenticate_user(username: str, password: str):
    user = users_col.find_one({"username": username})
    if not user:
        return None

    if not pwd_context.verify(password, user["password_hash"]):
        return None

    return {
        "user_id": str(user["_id"]),
        "username": user["username"],
        "role": user["role"]
    }

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
    users = list(users_col.find({}, {"_id": 1, "username": 1, "role": 1}))
    summary = []

    for u in users:
        uid = str(u["_id"])
        user_sessions = list(sessions_col.find({"user_id": uid}, {"_id": 1}))
        session_ids = [str(s["_id"]) for s in user_sessions]
        emotion_count = emotions_col.count_documents({"session_id": {"$in": session_ids}}) if session_ids else 0
        summary.append({
            "user_id": uid,
            "username": u["username"],
            "role": u["role"],
            "total_sessions": len(session_ids),
            "total_emotions": emotion_count
        })

    return summary
