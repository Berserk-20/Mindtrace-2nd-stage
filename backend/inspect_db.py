
import sys
import os
from pprint import pprint

# Add current directory to path so we can import from db
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import sessions_col, emotions_col, users_col

EMAIL = "sankalpgaikwad10@gmail.com"

def inspect_data():
    print(f"Inspecting data for {EMAIL}...")
    
    # 1. Check User
    user = users_col.find_one({"email": EMAIL})
    if not user:
        print("User not found!")
        return
    else:
        print(f"User found: {user['_id']}")

    # 2. Check Sessions
    sessions = list(sessions_col.find({"user_id": EMAIL}))
    print(f"\nTotal Sessions found: {len(sessions)}")
    
    for s in sessions:
        sid = str(s["_id"])
        status = s.get("status", "unknown")
        start = s.get("start_time")
        end = s.get("end_time")
        print(f" - Session {sid}: {status} | Start: {start} | End: {end}")
        
        # 3. Check Emotions for this session
        emotion_count = emotions_col.count_documents({"session_id": sid})
        print(f"   -> Emotions logged: {emotion_count}")

if __name__ == "__main__":
    inspect_data()
