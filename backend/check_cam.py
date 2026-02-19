import cv2

cap = cv2.VideoCapture(0)
if cap.isOpened():
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Default Resolution: {int(w)}x{int(h)}")
    cap.release()
else:
    print("Could not open webcam to check resolution. (Camera may be in use by backend)")
