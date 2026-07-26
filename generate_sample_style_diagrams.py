"""
generate_sample_style_diagrams.py
Generates Chapter 4 diagrams matching the exact visual styles from the MindTrace sample screenshots.
- DFDs: Light teal fills, no borders, grey arrows.
- Use Case: Black & white, thin borders, simple stick figures.
- State: Rounded boxes with black borders, red arrows.
- Class: Standard 3-tier UML boxes, black borders.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse
import os

OUT = r"c:\Users\sanka\MindTrace\report\images"
os.makedirs(OUT, exist_ok=True)
DPI = 220

TEAL = '#cde7e7'
ARR_GREY = '#666666'
RED = '#cc3333'

def arrow(ax, x1, y1, x2, y2, label="", color='black', lw=1.2, ls='-', label_y_offset=0.1, label_x_offset=0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=12))
    if label:
        ax.text((x1+x2)/2 + label_x_offset, (y1+y2)/2 + label_y_offset, label,
                ha='center', va='center', fontsize=8, color=color,
                bbox=dict(fc='white', ec='none', pad=1))

def rect(ax, x, y, w, h, label="", fc='white', ec='black', lw=1.0, fs=9):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, multialignment='center')

def rrect(ax, x, y, w, h, label="", fc='white', ec='black', lw=1.0, fs=9):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h, boxstyle="round,pad=0.1", fc=fc, ec=ec, lw=lw))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, multialignment='center')

def circle(ax, x, y, r, label="", fc='white', ec='black', lw=1.0, fs=9):
    ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, multialignment='center')

def oval(ax, x, y, w, h, label="", fc='white', ec='black', lw=1.0, fs=9):
    ax.add_patch(Ellipse((x, y), w, h, fc=fc, ec=ec, lw=lw))
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs, multialignment='center')

def stick_figure(ax, x, y, label="", fs=10):
    ax.add_patch(Circle((x, y+0.45), 0.15, fc='white', ec='black', lw=1.2))
    ax.plot([x, x], [y+0.3, y-0.15], 'k-', lw=1.2)
    ax.plot([x-0.25, x, x+0.25], [y+0.1, y+0.1, y+0.1], 'k-', lw=1.2)
    ax.plot([x-0.2, x, x+0.2], [y-0.5, y-0.15, y-0.5], 'k-', lw=1.2)
    if label:
        ax.text(x, y-0.7, label, ha='center', va='top', fontsize=fs)

def uml_class(ax, x, y, name, attrs, methods, w=3.0, rh=0.5):
    total_h = rh + rh*len(attrs) + rh*len(methods)
    top = y + total_h/2
    ax.add_patch(FancyBboxPatch((x-w/2, y-total_h/2), w, total_h, boxstyle="square,pad=0", fc='white', ec='black', lw=1.2))
    ax.text(x, top - rh/2, name, ha='center', va='center', fontsize=9, fontstyle='italic', fontweight='bold')
    ax.plot([x-w/2, x+w/2], [top-rh, top-rh], 'k-', lw=1.0)
    for i, attr in enumerate(attrs):
        ax.text(x-w/2+0.1, top - rh - rh*i - rh/2, attr, ha='left', va='center', fontsize=8)
    ax.plot([x-w/2, x+w/2], [top-rh-rh*len(attrs), top-rh-rh*len(attrs)], 'k-', lw=1.0)
    for i, m in enumerate(methods):
        ax.text(x-w/2+0.1, top - rh - rh*len(attrs) - rh*i - rh/2, m, ha='left', va='center', fontsize=8)

# ── 1. DFD Level 0 (Sample: Screenshot 132404 top) ───────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis('off'); fig.patch.set_facecolor('white')

rect(ax, 1.5, 1.75, 2.0, 0.8, "Video / Camera", fc=TEAL, ec='none')
circle(ax, 5.0, 1.75, 1.2, "MindTrace\nEmotion & Engagement\nRecognition System", fc=TEAL, ec='none')
rect(ax, 8.5, 1.75, 2.0, 0.8, "User / Admin", fc=TEAL, ec='none')

arrow(ax, 2.5, 1.75, 3.8, 1.75, "Input video", color=ARR_GREY)
arrow(ax, 6.2, 2.0, 7.5, 2.0, "Emotion Notification", color=ARR_GREY, label_y_offset=0.15)
arrow(ax, 6.2, 1.5, 7.5, 1.5, "Engagement Score", color=ARR_GREY, label_y_offset=0.15)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_dfd0.png'), dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()

# ── 2. DFD Level 1 (Sample: Screenshot 132404 bottom) ────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off'); fig.patch.set_facecolor('white')

# Outer light grey box like the sample
ax.add_patch(FancyBboxPatch((0.2, 0.2), 11.6, 7.6, boxstyle="square,pad=0", fc='#f5f5f5', ec='none'))

rect(ax, 2.5, 6.8, 1.8, 0.6, "Camera", fc=TEAL, ec='none')
rect(ax, 6.5, 6.8, 1.8, 0.6, "User", fc=TEAL, ec='none')
rect(ax, 9.5, 2.5, 2.0, 0.8, "Database", fc=TEAL, ec='none')
rect(ax, 9.5, 4.5, 2.0, 0.8, "Admin Dashboard", fc=TEAL, ec='none')

circle(ax, 2.0, 3.5, 1.0, "Face\nDetection", fc=TEAL, ec='none')
circle(ax, 5.5, 4.5, 1.1, "Emotion &\nEngagement\nEngine", fc=TEAL, ec='none')
circle(ax, 2.5, 1.5, 1.0, "Session\nScoring", fc=TEAL, ec='none')
circle(ax, 6.5, 2.0, 1.0, "Authentication", fc=TEAL, ec='none')

arrow(ax, 3.4, 6.8, 5.6, 6.8, "send video", color=ARR_GREY, label_y_offset=0.15)
ax.plot([5.6, 5.6], [6.8, 5.6], color=ARR_GREY, lw=1.2)
ax.annotate("", xy=(5.5, 5.6), xytext=(5.6, 5.6), arrowprops=dict(arrowstyle="-|>", color=ARR_GREY, lw=1.2))

arrow(ax, 2.5, 6.5, 2.5, 4.5, "send frame", color=ARR_GREY)
ax.annotate("", xy=(2.0, 4.5), xytext=(2.5, 4.5), arrowprops=dict(arrowstyle="-|>", color=ARR_GREY, lw=1.2))

arrow(ax, 2.9, 4.0, 4.5, 4.3, "send ROI", color=ARR_GREY)
arrow(ax, 4.4, 4.8, 2.8, 4.0, "returns result", color=ARR_GREY)

arrow(ax, 5.0, 3.5, 3.3, 1.9, "send frame", color=ARR_GREY)
arrow(ax, 3.5, 1.5, 4.8, 3.5, "returns score", color=ARR_GREY, label_y_offset=-0.15, label_x_offset=0.5)

arrow(ax, 6.6, 4.5, 8.5, 4.5, "Notify", color=ARR_GREY)
arrow(ax, 5.5, 5.6, 5.5, 6.8, "notify", color=ARR_GREY)

arrow(ax, 6.0, 3.5, 6.3, 2.9, "returns session", color=ARR_GREY)
arrow(ax, 6.7, 2.9, 6.3, 4.0, "user data", color=ARR_GREY, label_x_offset=0.5)

arrow(ax, 7.5, 2.2, 8.5, 2.5, "retrieve data", color=ARR_GREY)
arrow(ax, 8.5, 2.0, 7.5, 1.7, "store record", color=ARR_GREY)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_dfd1.png'), dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()

# ── 3. Use Case Diagram (Sample: Screenshot 132421) ──────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 10); ax.set_ylim(0, 8); ax.axis('off'); fig.patch.set_facecolor('white')

# Outer box
ax.add_patch(FancyBboxPatch((2.5, 0.5), 5.0, 7.0, boxstyle="square,pad=0", fc='white', ec='black', lw=1.2))
ax.text(5.0, 7.2, "MindTrace System", ha='center', fontsize=10)

stick_figure(ax, 1.0, 4.0, "Student")
stick_figure(ax, 9.0, 4.0, "Admin")

ucs = [
    (5.0, 6.5, "Register / Login"),
    (5.0, 5.5, "Input Webcam Video"),
    (5.0, 4.5, "Input Uploaded Video"),
    (5.0, 3.5, "View Live Dashboard"),
    (5.0, 2.5, "View Session History"),
    (5.0, 1.5, "Manage Users"),
    (5.0, 0.5, "Generate Admin Reports")
]

for y in [6.5, 5.5, 4.5, 3.5, 2.5]:
    ax.plot([1.3, 3.2], [4.0, y], 'k-', lw=1.0)
    
for y in [6.5, 3.5, 2.5, 1.5, 0.5]:
    ax.plot([8.7, 6.8], [4.0, y], 'k-', lw=1.0)

for x, y, lbl in ucs:
    oval(ax, x, y, 3.2, 0.7, lbl, ec='black', lw=1.0)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_usecase.png'), dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()

# ── 4. State Diagram (Sample: Screenshot 132427) ─────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis('off'); fig.patch.set_facecolor('white')

# Initial
ax.add_patch(Circle((0.5, 5.0), 0.15, fc='black'))
ax.text(0.5, 5.3, "Initial State", ha='center', fontsize=8)
arrow(ax, 0.65, 5.0, 1.5, 5.0, color=RED)

states = {
    "Idle": (2.2, 5.0),
    "Login": (4.2, 5.0),
    "Select Input": (6.5, 5.0),
    "Check User Auth": (9.0, 5.0),
    "Process Frame": (6.5, 3.5),
    "Face Detection": (4.2, 3.5),
    "Emotion Inference": (2.2, 3.5),
    "Notify Dashboard": (2.2, 2.0),
    "Engagement Scoring": (6.5, 2.0),
    "Store Session DB": (6.5, 0.8),
    "Final": (11.0, 2.0)
}

for name, (x, y) in states.items():
    if name == "Final":
        ax.add_patch(Circle((x, y), 0.15, fc='black'))
        ax.add_patch(Circle((x, y), 0.22, fc='none', ec=RED, lw=1.5))
        ax.text(x+0.4, y, "Final State\nSystem Terminates", ha='left', va='center', fontsize=8)
    else:
        rrect(ax, x, y, 1.6, 0.8, name, ec='black')

arrow(ax, 3.0, 5.0, 3.4, 5.0, color=RED) # Idle->Login
arrow(ax, 5.0, 5.0, 5.7, 5.0, color=RED) # Login->Select
arrow(ax, 7.3, 5.0, 8.2, 5.0, color=RED) # Select->Check
arrow(ax, 9.0, 4.6, 9.0, 2.2, color=RED)
ax.plot([9.0, 10.7], [2.2, 2.2], color=RED) # Check->Final
ax.annotate("", xy=(10.7, 2.0), xytext=(10.7, 2.2), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2))

arrow(ax, 6.5, 4.6, 6.5, 3.9, color=RED) # Select->Process
arrow(ax, 5.7, 3.5, 5.0, 3.5, color=RED) # Process->Face
arrow(ax, 3.4, 3.5, 3.0, 3.5, color=RED) # Face->Emotion
arrow(ax, 2.2, 3.1, 2.2, 2.4, color=RED) # Emotion->Notify
ax.plot([2.2, 2.2], [1.6, 0.3], color=RED, lw=1.2)
ax.plot([2.2, 11.0], [0.3, 0.3], color=RED, lw=1.2)
arrow(ax, 11.0, 0.3, 11.0, 1.7, color=RED) # Notify->Final

arrow(ax, 6.5, 3.1, 6.5, 2.4, color=RED) # Process->Engage
arrow(ax, 6.5, 1.6, 6.5, 1.2, color=RED) # Engage->Store
ax.plot([7.3, 11.0], [0.8, 0.8], color=RED, lw=1.2)
arrow(ax, 11.0, 0.8, 11.0, 1.7, color=RED) # Store->Final

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_state.png'), dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()

# ── 5. Class Diagram (Sample: Screenshot 132434) ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))
ax.set_xlim(0, 11); ax.set_ylim(0, 7); ax.axis('off'); fig.patch.set_facecolor('white')

uml_class(ax, 3.5, 5.5, "System", ["admin_id : string", "Password : string"], ["adminLogin()"], w=2.5)
uml_class(ax, 8.0, 5.5, "Admin", [], ["Registering User()", "Inputting video()", "Inputting camera()"], w=2.5)

uml_class(ax, 2.0, 2.0, "FaceDetection", ["user_email_id : string", "bbox_coords : list"], ["userFaceDetection()", "extractROI()", "notifyUser()"], w=2.4)
uml_class(ax, 5.0, 2.0, "EmotionEngine", ["model : string", "frame_count : int"], ["classifyEmotion()", "computeEngagement()", "checkHeadPose()"], w=2.6)
uml_class(ax, 8.0, 2.0, "Session", ["user_email_id : string", "start_time : date", "avg_score : float"], ["start()", "end()", "generateReport()"], w=2.6)

arrow(ax, 6.75, 5.5, 4.75, 5.5) # Admin -> System

ax.plot([2.0, 2.0], [3.5, 4.5], 'k-', lw=1.2)
arrow(ax, 2.0, 4.5, 3.0, 4.5) # Face -> System

ax.plot([5.0, 5.0], [3.5, 4.5], 'k-', lw=1.2)
arrow(ax, 5.0, 4.5, 3.5, 4.5) # Emotion -> System

ax.plot([8.0, 8.0], [3.5, 4.5], 'k-', lw=1.2)
arrow(ax, 8.0, 4.5, 4.0, 4.5) # Session -> System

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_class.png'), dpi=DPI, bbox_inches='tight', facecolor='white')
plt.close()

print("All sample-style diagrams successfully generated!")
