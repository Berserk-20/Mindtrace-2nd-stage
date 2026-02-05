import threading

LATEST_FRAME = None
FRAME_LOCK = threading.Lock()

AGENT_RUNNING = False
PAUSE_REQUESTED = False
