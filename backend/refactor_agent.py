import sys
import os

filepath = 'c:/Users/sanka/MindTrace/backend/background_agent.py'
with open(filepath, 'r') as f:
    lines = f.readlines()

new_lines = lines[:287]

new_code = """# ==============================
# STATELESS AGENT PROCESSOR
# ==============================

def process_single_frame(frame):
    \"\"\"
    Process a single frame coming from the frontend via POST.
    Updates the shared state history for metrics.
    \"\"\"
    import shared_state
    import time
    import cv2
    import numpy as np
    
    if not shared_state.AGENT_RUNNING or shared_state.PAUSE_REQUESTED:
        return {"emotion": "Neutral", "focus_score": 50, "face_found": False, "confidence": 1.0}

    global last_face, last_face_time, last_emotion
    global eye_closed_frames, total_blinks, blink_start_time
    global looking_away_frames, is_looking_away
    
    now = time.time()
    h, w = frame.shape[:2]
    
    # Run MediaPipe Face Mesh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mesh = get_face_mesh()
    results = mesh.process(rgb_frame)

    face_found = False
    
    if results.multi_face_landmarks:
        face_found = True
        landmarks = results.multi_face_landmarks[0].landmark
        
        # Disable blink computation because 1 FPS is too slow for blink tracking
        # But evaluate head pose for distraction tracking
        pitch, yaw, roll = get_head_pose(landmarks, w, h)
        if abs(yaw) > 30 or pitch > 30 or pitch < -30:
            looking_away_frames += 1
            if looking_away_frames >= 2:  # Adjusted for 1FPS
                is_looking_away = True
        else:
            looking_away_frames = 0
            is_looking_away = False
            
        # Bounding box for ResNet
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            if x < x_min: x_min = x
            if y < y_min: y_min = y
            if x > x_max: x_max = x
            if y > y_max: y_max = y
        
        pad = 20
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(w, x_max + pad)
        y_max = min(h, y_max + pad)
        
        last_face = (x_min, y_min, x_max-x_min, y_max-y_min)
        last_face_time = now

        face_roi = frame[y_min:y_max, x_min:x_max]
        if face_roi.size > 0:
            emotion, conf = predict_emotion(face_roi)

            if conf < CONFIDENCE_THRESHOLD:
                emotion = "Neutral"

            emotion_buffer.append(emotion)
            last_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

            # Update shared state history
            with shared_state.DATA_LOCK:
                bpm = 15 # Default/mocked since blinked tracker is disabled
                eng_score = get_engagement_score(last_emotion, is_looking_away, bpm)

                shared_state.EMOTION_HISTORY.append({
                    "timestamp": now,
                    "emotion": last_emotion,
                    "confidence": float(conf),
                    "focus_score": eng_score
                })
                
                shared_state.ENGAGEMENT_HISTORY.append({
                    "timestamp": now,
                    "score": eng_score,
                    "head_pose": "Away" if is_looking_away else "Center",
                    "blink_rate": bpm
                })
                
                if len(shared_state.EMOTION_HISTORY) > 1000:
                    shared_state.EMOTION_HISTORY.pop(0)
                if len(shared_state.ENGAGEMENT_HISTORY) > 1000:
                    shared_state.ENGAGEMENT_HISTORY.pop(0)
                    
            return {
                "emotion": last_emotion, 
                "focus_score": eng_score, 
                "face_found": True, 
                "confidence": float(conf)
            }
            
    # If no face found
    return {"emotion": "Neutral", "focus_score": 0, "face_found": False, "confidence": 1.0}

# ==============================
# SESSION CONTROLS
# ==============================
def start_agent_thread():
    import shared_state
    import time
    # Only marks state as running, no threads spawned
    if shared_state.AGENT_RUNNING:
        return
    shared_state.AGENT_RUNNING = True
    shared_state.PAUSE_REQUESTED = False
    
    with shared_state.DATA_LOCK:
        if shared_state.SESSION_START_TIME is None:
            shared_state.SESSION_START_TIME = time.time()

def stop_agent():
    import shared_state
    
    shared_state.AGENT_RUNNING = False
    shared_state.PAUSE_REQUESTED = False
    
    with shared_state.DATA_LOCK:
        shared_state.SESSION_START_TIME = None
        shared_state.PAUSE_START_TIME = None
        shared_state.TOTAL_PAUSED_TIME = 0
        shared_state.EMOTION_HISTORY = []
        shared_state.ENGAGEMENT_HISTORY = []
        shared_state.DISTRACTION_EVENTS = []
        shared_state.CURRENT_SESSION_ID = None
        
    with shared_state.FRAME_LOCK:
        shared_state.LATEST_FRAME = None

def pause_agent():
    import shared_state
    import time
    if not shared_state.PAUSE_REQUESTED:
        shared_state.PAUSE_REQUESTED = True
        with shared_state.DATA_LOCK:
            shared_state.PAUSE_START_TIME = time.time()

def resume_agent():
    import shared_state
    import time
    if shared_state.PAUSE_REQUESTED:
        shared_state.PAUSE_REQUESTED = False
        with shared_state.DATA_LOCK:
            if shared_state.PAUSE_START_TIME:
                paused_duration = time.time() - shared_state.PAUSE_START_TIME
                shared_state.TOTAL_PAUSED_TIME += paused_duration
                shared_state.PAUSE_START_TIME = None
"""

new_lines.append(new_code)
with open(filepath, 'w') as f:
    f.writelines(new_lines)
print('Rewrite successful')
