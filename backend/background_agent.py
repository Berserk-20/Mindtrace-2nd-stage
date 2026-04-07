import cv2
import time
import threading
import numpy as np
from collections import deque
import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==============================
# MODEL SETUP
# ==============================
class EmotionResNet18(nn.Module):
    """Same architecture as used in training"""
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = models.resnet18(weights=None)  # No pretrained weights, we'll load trained
        
        # Enhanced classification head matching training script
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove original fc layer
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.model(x)
        return self.classifier(features)

# Initialize model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotion_model = None

# Image preprocessing (must match training transforms)
IMG_SIZE = 96
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model lazily
def get_emotion_model():
    global emotion_model
    if emotion_model is not None:
        return emotion_model
    try:
        model = EmotionResNet18(num_classes=7)
        model.load_state_dict(torch.load("models/best_model_rafdb.pth", map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        emotion_model = model
        print(f"✓ Emotion model loaded successfully on {DEVICE}")
    except Exception as e:
        print(f"⚠ Warning: Could not load emotion model: {e}")
        print("  Using random predictions as fallback")
        emotion_model = None
    return emotion_model

# ==============================
# MEDIAPIPE AND OTHER IMPORTS
# ==============================
import mediapipe as mp
import shared_state

# ==============================
# THREAD HANDLE
# ==============================
# ==============================
# CONFIG
# ==============================
FACE_PERSISTENCE = 2.0  # Increased to prevent flickering
CONFIDENCE_THRESHOLD = 0.4

# RAF-DB emotion classes (7 emotions)
RAF_DB_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTIONS = RAF_DB_EMOTIONS  # Use RAF-DB classes
emotion_buffer = deque(maxlen=6)

# ==============================
# MEDIAPIPE SETUP
# ==============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh_instance = None

def get_face_mesh():
    global face_mesh_instance
    if face_mesh_instance is None:
        face_mesh_instance = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    return face_mesh_instance

# ==============================
# FACE STATE
# ==============================
last_face = None
last_face_time = 0
last_emotion = "Neutral"

# Blink State
BLINK_THRESHOLD = 0.22  # EAR threshold (Eye Aspect Ratio)
eye_closed_frames = 0
total_blinks = 0
blink_start_time = 0
blinks_in_window = deque(maxlen=20) # Track blinks for rate calc

# Head Pose State
looking_away_frames = 0
is_looking_away = False

# ==============================
# HELPER FUNCTIONS
# ==============================
def calculate_ear(eye_landmarks, w, h):
    """Calculate Eye Aspect Ratio"""
    # Vertical distances
    A = np.linalg.norm(np.array([eye_landmarks[1].x*w, eye_landmarks[1].y*h]) - 
                       np.array([eye_landmarks[5].x*w, eye_landmarks[5].y*h]))
    B = np.linalg.norm(np.array([eye_landmarks[2].x*w, eye_landmarks[2].y*h]) - 
                       np.array([eye_landmarks[4].x*w, eye_landmarks[4].y*h]))
    # Horizontal distance
    C = np.linalg.norm(np.array([eye_landmarks[0].x*w, eye_landmarks[0].y*h]) - 
                       np.array([eye_landmarks[3].x*w, eye_landmarks[3].y*h]))
    return (A + B) / (2.0 * C)

def get_head_pose(landmarks, w, h):
    """Estimate head pose (Pitch, Yaw, Roll) from landmarks"""
    # 3D model points
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # 2D image points from landmarks
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h), # Chin
        (landmarks[33].x * w, landmarks[33].y * h),   # Left eye left corner
        (landmarks[263].x * w, landmarks[263].y * h), # Right eye right corner
        (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth corner
        (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth corner
    ], dtype="double")

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))
    
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs
    )

    # Get rotational matrix
    rmat, jac = cv2.Rodrigues(rotation_vector)
    
    # Get angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360
    
    return pitch, yaw, roll

# ==============================
# MODEL INFERENCE
# ==============================
def predict_emotion(face_img):
    """Predict emotion from face ROI using trained model"""
    model = get_emotion_model()
    if model is None:
        # Fallback to random if model failed to load
        return np.random.choice(EMOTIONS), np.random.uniform(0.4, 0.95)
    
    try:
        # Convert BGR (OpenCV) to RGB (PIL)
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        # Preprocess
        img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # --- Heuristic Bias Correction ---
            # Indices: 0:Angry, 1:Disgust, 2:Fear, 3:Happy, 4:Neutral, 5:Sad, 6:Surprise
            # User reports Sad/Fear misclassified as Disgust
            
            # Suppress Disgust (1)
            probs[0, 1] *= 0.2  # Aggressive suppression
            
            # Boost Fear (2) and Sad (5) and Happy (3)
            probs[0, 2] *= 1.5
            probs[0, 5] *= 1.5
            probs[0, 3] *= 1.2 # Slight boost to Happy too
            
            # Re-normalize
            probs = probs / probs.sum()
            
            confidence, pred_idx = torch.max(probs, dim=1)
            
            # Get Top 3 for debug
            top3_conf, top3_idx = torch.topk(probs, 3, dim=1)
            top3_emotions = [EMOTIONS[idx.item()] for idx in top3_idx[0]]
            top3_confs = [f"{conf.item():.2f}" for conf in top3_conf[0]]
            print(f"Pred: {EMOTIONS[pred_idx.item()]} | Top3: {list(zip(top3_emotions, top3_confs))}")

        emotion = EMOTIONS[pred_idx.item()]
        conf = confidence.item()
        
        return emotion, conf
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Neutral", 0.5

# ==============================
# ENGAGEMENT MAPPING
# ==============================
def get_engagement(emotion):
    if emotion in ["Happy", "Surprise"]: # Removed "Focus"
        return "engaged"
    if emotion == "Neutral":
        return "neutral"
    if emotion in ["Sad", "Fear", "Disgust", "Angry"]:
        return "disengaged"
    return "neutral"

def get_engagement_score(emotion, is_looking_away=False, blink_rate=15):
    """
    Calculate Focus Score (0-100) based on:
    1. Emotion (Base score)
    2. Head Pose (Penalty if looking away)
    3. Blink Rate (Penalty if drowsy > 30bpm, Bonus if staring < 5bpm)
    """
    blink_rate = max(0, blink_rate) # Ensure non-negative
    
    # 1. Base Score from Emotion
    if emotion in ["Happy", "Surprise"]:
        score = np.random.uniform(85, 95)
    elif emotion == "Neutral":
        score = np.random.uniform(65, 80)
    else: # Negative emotions
        score = np.random.uniform(30, 50)
        
    # 2. Head Pose Penalty
    if is_looking_away:
        score -= 30 # Significant penalty for looking away
        
    # 3. Blink Rate Modifier
    if blink_rate > 35: # Drowsy/Fatigued
        score -= 15
    elif blink_rate < 5: # Intense Focus (Staring)
        score += 10
        
    # Clamp score
    return max(0, min(100, score))

# ==============================
# ==============================
# STATELESS AGENT PROCESSOR
# ==============================

def process_single_frame(frame):
    """
    Process a single frame coming from the frontend via POST.
    Updates the shared state history for metrics.
    """
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
