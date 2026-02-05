import cv2
import time
import threading
import numpy as np
from collections import deque
import os

import mediapipe as mp
import shared_state

# ==============================
# THREAD HANDLE
# ==============================
agent_thread = None

# ==============================
# CONFIG
# ==============================
CAMERA_INDEX = 0

TARGET_FPS = 15
FRAME_INTERVAL = 1.0 / TARGET_FPS

INFERENCE_INTERVAL = 0.4
CONFIDENCE_THRESHOLD = 0.6
FACE_PERSISTENCE = 0.6

SNAPSHOT_COOLDOWN = 3.0  # seconds between saves

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
emotion_buffer = deque(maxlen=6)

# ==============================
# SNAPSHOT STATE
# ==============================
last_snapshot_time = 0

# ==============================
# DIRECTORY SETUP
# ==============================
BASE_DIR = "snapshots"
ENGAGEMENT_DIR = os.path.join(BASE_DIR, "engagement")
EXPR_DIR = os.path.join(BASE_DIR, "expressions")

for d in [
    f"{ENGAGEMENT_DIR}/engaged",
    f"{ENGAGEMENT_DIR}/neutral",
    f"{ENGAGEMENT_DIR}/disengaged",
    *[f"{EXPR_DIR}/{e.lower()}" for e in EMOTIONS],
]:
    os.makedirs(d, exist_ok=True)

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# ==============================
# FACE STATE
# ==============================
last_face = None
last_face_time = 0
last_emotion = "Neutral"

# ==============================
# MODEL PLACEHOLDER
# ==============================
def predict_emotion(face_img):
    return np.random.choice(EMOTIONS), np.random.uniform(0.4, 0.95)

# ==============================
# ENGAGEMENT MAPPING
# ==============================
def get_engagement(emotion):
    if emotion in ["Happy", "Surprise"]:
        return "engaged"
    if emotion == "Neutral":
        return "neutral"
    if emotion in ["Sad", "Fear"]:
        return "neutral"
    return "disengaged"

# ==============================
# AGENT LOOP
# ==============================
def run_agent():
    global last_face, last_face_time, last_emotion, last_snapshot_time

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        shared_state.AGENT_RUNNING = False
        return

    last_frame_time = 0
    last_inference_time = 0

    while shared_state.AGENT_RUNNING:

        if shared_state.PAUSE_REQUESTED:
            time.sleep(0.05)
            continue

        now = time.time()
        if now - last_frame_time < FRAME_INTERVAL:
            continue
        last_frame_time = now

        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape

        with shared_state.FRAME_LOCK:
            shared_state.LATEST_FRAME = frame.copy()

        # ---------- Inference ----------
        if now - last_inference_time >= INFERENCE_INTERVAL:
            last_inference_time = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detector.process(rgb)

            if results.detections:
                det = results.detections[0]
                box = det.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                last_face = (x, y, bw, bh)
                last_face_time = now

                face_roi = frame[y:y+bh, x:x+bw]
                emotion, conf = predict_emotion(face_roi)

                if conf < CONFIDENCE_THRESHOLD:
                    emotion = "Neutral"

                emotion_buffer.append(emotion)
                last_emotion = max(set(emotion_buffer), key=emotion_buffer.count)

                # ---------- SNAPSHOT CAPTURE ----------
                if now - last_snapshot_time > SNAPSHOT_COOLDOWN:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")

                    # Expression snapshot
                    expr_path = f"{EXPR_DIR}/{last_emotion.lower()}/{timestamp}.jpg"
                    cv2.imwrite(expr_path, frame)

                    # Engagement snapshot
                    engagement = get_engagement(last_emotion)
                    eng_path = f"{ENGAGEMENT_DIR}/{engagement}/{timestamp}.jpg"
                    cv2.imwrite(eng_path, frame)

                    last_snapshot_time = now

        # ---------- Overlay ----------
        if last_face and (now - last_face_time) <= FACE_PERSISTENCE:
            x, y, bw, bh = last_face
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)
            cv2.putText(
                frame,
                last_emotion,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )

        with shared_state.FRAME_LOCK:
            shared_state.LATEST_FRAME = frame.copy()

    cap.release()

# ==============================
# THREAD CONTROLS
# ==============================
def start_agent_thread():
    global agent_thread
    if shared_state.AGENT_RUNNING:
        return
    shared_state.AGENT_RUNNING = True
    shared_state.PAUSE_REQUESTED = False
    agent_thread = threading.Thread(target=run_agent, daemon=True)
    agent_thread.start()

def stop_agent():
    shared_state.AGENT_RUNNING = False

def pause_agent():
    shared_state.PAUSE_REQUESTED = True

def resume_agent():
    shared_state.PAUSE_REQUESTED = False
