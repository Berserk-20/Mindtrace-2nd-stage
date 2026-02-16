import threading

LATEST_FRAME = None
FRAME_LOCK = threading.Lock()

AGENT_RUNNING = False
PAUSE_REQUESTED = False

# New shared state for metrics
DATA_LOCK = threading.Lock()
SESSION_START_TIME = None
PAUSE_START_TIME = None  # Track when session was paused
TOTAL_PAUSED_TIME = 0  # Accumulate total paused duration
EMOTION_HISTORY = []  # List of {"timestamp": float, "emotion": str, "confidence": float}
ENGAGEMENT_HISTORY = [] # List of {"timestamp": float, "score": float}
DISTRACTION_EVENTS = [] # List of {"timestamp": float, "duration": float}
CURRENT_USER_EMAIL = None  # Email of the user running the current session
CURRENT_USER_EMAIL = None  # Email of the user running the current session
CURRENT_SESSION_ID = None  # MongoDB session ID
SNAPSHOT_COUNTS = {} # Track number of snapshots taken per category in current session

