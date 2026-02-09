
from collections import deque
from datetime import datetime, timedelta
from db import log_event

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
LOW_FOCUS_THRESHOLD = 0.4
WINDOW_SIZE = 30          
COOLDOWN_SECONDS = 60     

# ----------------------------------------------------------
# STATE (per session)
# ----------------------------------------------------------
session_state = {}

# ----------------------------------------------------------
# NOTIFIER LOGIC
# ----------------------------------------------------------
def notify(session_id: str, emotion: str, focus_level: float):
    now = datetime.now()

    if session_id not in session_state:
        session_state[session_id] = {
            "focus_window": deque(maxlen=WINDOW_SIZE),
            "last_alert": None
        }

    state = session_state[session_id]
    state["focus_window"].append(focus_level)

    avg_focus = sum(state["focus_window"]) / len(state["focus_window"])

    # ------------------------------------------------------
    # LOW FOCUS ALERT
    # ------------------------------------------------------
    if avg_focus < LOW_FOCUS_THRESHOLD:
        if (
            state["last_alert"] is None or
            (now - state["last_alert"]).seconds > COOLDOWN_SECONDS
        ):
            log_event(
                "low_focus_detected",
                {
                    "session_id": session_id,
                    "avg_focus": round(avg_focus, 2),
                    "emotion": emotion
                }
            )
            state["last_alert"] = now
