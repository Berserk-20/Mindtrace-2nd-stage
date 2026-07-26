"""
generate_diagrams.py
Generates all Chapter 4 diagrams as PNG images for the MindTrace LaTeX report.
Output: report/images/diag_*.png
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
import numpy as np
import os

OUT = r"c:\Users\sanka\MindTrace\report\images"
os.makedirs(OUT, exist_ok=True)
DPI = 200

def arrow(ax, x1, y1, x2, y2, label="", lw=1.5, color="black", fs=8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my, label, ha='center', va='bottom', fontsize=fs,
                bbox=dict(fc='white', ec='none', pad=1))

def rect(ax, x, y, w, h, label="", fc='#f0f0f0', ec='black', lw=1.5, fs=9, bold=False):
    r = FancyBboxPatch((x-w/2, y-h/2), w, h,
                       boxstyle="round,pad=0.02", fc=fc, ec=ec, lw=lw)
    ax.add_patch(r)
    if label:
        weight = 'bold' if bold else 'normal'
        ax.text(x, y, label, ha='center', va='center', fontsize=fs,
                fontweight=weight, wrap=True, multialignment='center')

def circle(ax, x, y, r, label="", fc='#ddeeff', ec='black', lw=1.5, fs=8):
    c = Circle((x, y), r, fc=fc, ec=ec, lw=lw)
    ax.add_patch(c)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs,
                multialignment='center')

def oval(ax, x, y, w, h, label="", fc='white', ec='black', lw=1.5, fs=9):
    e = Ellipse((x, y), w, h, fc=fc, ec=ec, lw=lw)
    ax.add_patch(e)
    if label:
        ax.text(x, y, label, ha='center', va='center', fontsize=fs,
                multialignment='center')

def stick_figure(ax, x, y, label="", fs=9):
    ax.add_patch(Circle((x, y+0.55), 0.18, fc='white', ec='black', lw=1.5))
    ax.plot([x, x], [y+0.37, y-0.1], 'k-', lw=1.5)
    ax.plot([x-0.3, x, x+0.3], [y+0.15, y+0.1, y+0.15], 'k-', lw=1.5)
    ax.plot([x-0.25, x, x+0.25], [y-0.55, y-0.1, y-0.55], 'k-', lw=1.5)
    if label:
        ax.text(x, y-0.75, label, ha='center', va='top', fontsize=fs)

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 1: Architecture
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')
ax.set_facecolor('white'); fig.patch.set_facecolor('white')

# Input UI
ax.add_patch(FancyBboxPatch((0.2,1.0), 2.4, 3.0, boxstyle="round,pad=0.05",
                             fc='#f9f9f9', ec='black', lw=1.5))
ax.text(1.4, 3.75, 'Input UI', ha='center', va='center', fontsize=10, fontweight='bold')
rect(ax, 1.4, 2.85, 2.0, 0.65, 'Webcam\n(Browser)', fc='white', ec='black', fs=8)
rect(ax, 1.4, 1.85, 2.0, 0.65, 'Upload\nVideo', fc='white', ec='black', fs=8)

# Arrow Input → System
arrow(ax, 2.6, 2.5, 3.2, 2.5)

# System boundary
ax.add_patch(FancyBboxPatch((3.2, 0.6), 7.8, 3.8, boxstyle="round,pad=0.05",
                             fc='#f0f4ff', ec='black', lw=1.5))
ax.text(7.1, 4.2, "System's Internal Structure", ha='center', va='center',
        fontsize=10, fontweight='bold')

# Face Detection sub-box
ax.add_patch(FancyBboxPatch((3.4, 0.8), 3.4, 3.2, boxstyle="round,pad=0.05",
                             fc='white', ec='black', lw=1.2))
ax.text(5.1, 3.75, 'Face Detection Module', ha='center', fontsize=8.5, fontweight='bold')
rect(ax, 4.3, 2.95, 1.5, 0.65, 'MediaPipe\nFaceMesh', fc='#eaf0ff', ec='black', fs=8)
rect(ax, 4.3, 2.0, 1.5, 0.65, 'Extract\nFace ROI', fc='#eaf0ff', ec='black', fs=8)
rect(ax, 6.2, 2.5, 1.3, 0.65, 'Detected\nFace', fc='#eaf0ff', ec='black', fs=8)
arrow(ax, 4.3, 2.62, 4.3, 2.35)
arrow(ax, 5.05, 2.0, 5.6, 2.5)

# Emotion & Engagement sub-box
ax.add_patch(FancyBboxPatch((6.9, 0.8), 3.9, 3.2, boxstyle="round,pad=0.05",
                             fc='white', ec='black', lw=1.2))
ax.text(8.85, 3.75, 'Emotion & Engagement Module', ha='center', fontsize=8.5, fontweight='bold')
rect(ax, 7.9, 2.95, 1.6, 0.65, 'ResNet-18\nInference', fc='#eaf0ff', ec='black', fs=8)
rect(ax, 7.9, 2.0, 1.6, 0.65, 'Head Pose\n(PnP Solver)', fc='#eaf0ff', ec='black', fs=8)
rect(ax, 10.2, 2.5, 1.5, 0.65, 'Engagement\nScore', fc='#eaf0ff', ec='black', fs=8)
arrow(ax, 8.7, 2.95, 9.45, 2.65)
arrow(ax, 8.7, 2.0, 9.45, 2.35)
arrow(ax, 9.45, 2.5, 9.5, 2.5)

# Internal arrow Face → Emotion
arrow(ax, 6.85, 2.5, 6.9, 2.5)

# Arrow System → Output
arrow(ax, 11.0, 2.5, 11.5, 2.5)

# Output UI
ax.add_patch(FancyBboxPatch((11.4, 1.0), 2.4, 3.0, boxstyle="round,pad=0.05",
                             fc='#f9f9f9', ec='black', lw=1.5))
ax.text(12.6, 3.75, 'Output UI', ha='center', va='center', fontsize=10, fontweight='bold')
rect(ax, 12.6, 2.85, 2.0, 0.65, 'Live\nDashboard', fc='white', ec='black', fs=8)
rect(ax, 12.6, 1.85, 2.0, 0.65, 'MongoDB\nSession', fc='white', ec='black', fs=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_architecture.png'), dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ Architecture diagram saved")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 2: DFD Level 0
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4))
ax.set_xlim(0, 11); ax.set_ylim(0, 4); ax.axis('off')
fig.patch.set_facecolor('white')

rect(ax, 1.6, 2.0, 2.4, 1.0, 'Browser /\nWebcam', fc='white', ec='black', lw=2, fs=10)
circle(ax, 5.5, 2.0, 1.2, 'MindTrace\nEmotion\nRecognition\nSystem', fc='#ddeeff', fs=9)
rect(ax, 9.4, 2.0, 2.4, 1.0, 'User /\nAdmin', fc='white', ec='black', lw=2, fs=10)

arrow(ax, 2.8, 2.0, 4.3, 2.0, 'Input frame', fs=9)
arrow(ax, 6.7, 2.2, 8.2, 2.2, 'Emotion label', fs=9)
arrow(ax, 6.7, 1.8, 8.2, 1.8, 'Engagement score', fs=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_dfd0.png'), dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ DFD Level 0 saved")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 3: DFD Level 1
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(-1, 12); ax.set_ylim(-1, 8); ax.axis('off')
fig.patch.set_facecolor('white')

# External entities
rect(ax, 0.5, 6.5, 2.2, 0.9, 'Browser', fc='white', ec='black', lw=2, fs=10)
rect(ax, 10.5, 6.5, 2.2, 0.9, 'Admin', fc='white', ec='black', lw=2, fs=10)
rect(ax, 10.5, 1.0, 2.2, 0.9, 'MongoDB', fc='white', ec='black', lw=2, fs=10)

# Processes
circle(ax, 2.0, 4.0, 0.9, 'Authen-\ntication', fc='#ddeeff', fs=8)
circle(ax, 5.5, 6.5, 0.9, 'Face\nDetection', fc='#ddeeff', fs=8)
circle(ax, 5.5, 4.0, 0.9, 'Emotion\nInference', fc='#ddeeff', fs=8)
circle(ax, 5.5, 1.5, 0.9, 'Engagement\nScoring', fc='#ddeeff', fs=8)
circle(ax, 9.0, 4.0, 0.9, 'Session\nStorage', fc='#ddeeff', fs=8)

# Flows
arrow(ax, 1.6, 6.5, 2.5, 4.8, 'credentials', fs=8)
arrow(ax, 1.6, 6.5, 4.6, 6.5, 'frame', fs=8)
arrow(ax, 5.5, 5.6, 5.5, 4.9, 'face ROI', fs=8)
arrow(ax, 5.5, 3.1, 5.5, 2.4, 'emotion', fs=8)
arrow(ax, 6.4, 4.0, 8.1, 4.0, 'results', fs=8)
arrow(ax, 6.4, 1.5, 8.1, 3.2, 'score', fs=8)
arrow(ax, 9.0, 4.9, 9.7, 6.1, 'report', fs=8)
arrow(ax, 9.9, 4.0, 11.4, 1.0, 'store', fs=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_dfd1.png'), dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ DFD Level 1 saved")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 4: Use Case Diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis('off')
fig.patch.set_facecolor('white')

# System boundary
ax.add_patch(FancyBboxPatch((2.5, 0.5), 7.0, 9.0, boxstyle="round,pad=0.1",
                             fc='#f9f9ff', ec='black', lw=2))
ax.text(6.0, 9.25, 'MindTrace: Emotion Recognition System',
        ha='center', va='center', fontsize=11, fontweight='bold')

# Use cases
ucs = [
    (6.0, 8.0, 'Register / Login'),
    (6.0, 6.8, 'Start Session'),
    (6.0, 5.6, 'View Live Dashboard'),
    (6.0, 4.4, 'View Session History'),
    (6.0, 3.2, 'View Admin Panel'),
    (6.0, 2.0, 'Manage Users'),
]
for x, y, lbl in ucs:
    oval(ax, x, y, 3.5, 0.8, lbl, fc='white', ec='black', lw=1.5, fs=10)

# Actors
stick_figure(ax, 1.2, 7.2, 'Student', fs=11)
stick_figure(ax, 10.8, 7.2, 'Admin', fs=11)

# Student lines (to first 4 use cases)
student_ucs = [8.0, 6.8, 5.6, 4.4]
for uy in student_ucs:
    ax.plot([1.2+0.05, 4.25], [6.7, uy], 'k-', lw=1.2)

# Admin lines (to register + last 2)
admin_ucs = [8.0, 3.2, 2.0]
for uy in admin_ucs:
    ax.plot([10.8-0.05, 7.75], [6.7, uy], 'k-', lw=1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_usecase.png'), dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ Use Case diagram saved")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 5: State Diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-0.5, 14); ax.set_ylim(-1, 5); ax.axis('off')
fig.patch.set_facecolor('white')

# Initial state dot
ax.add_patch(Circle((0.3, 3.5), 0.2, fc='black', ec='black'))
arrow(ax, 0.5, 3.5, 1.0, 3.5)

states = [
    (2.0, 3.5, 'Idle'),
    (4.5, 3.5, 'Login'),
    (7.0, 3.5, 'Session\nReady'),
    (9.5, 3.5, 'Analyzing\nFrame'),
    (9.5, 1.5, 'Computing\nScore'),
    (7.0, 1.5, 'Update\nDashboard'),
    (4.5, 1.5, 'Session\nEnd'),
]

for x, y, lbl in states:
    r = FancyBboxPatch((x-1.1, y-0.45), 2.2, 0.9,
                       boxstyle="round,pad=0.08", fc='white', ec='black', lw=1.5)
    ax.add_patch(r)
    ax.text(x, y, lbl, ha='center', va='center', fontsize=9, multialignment='center')

transitions = [
    (0.5, 3.5, 1.0-1.1, 3.5, 'login'),       # init → idle
    (3.1, 3.5, 3.4, 3.5, 'credentials'),      # idle → login
    (5.6, 3.5, 5.9, 3.5, 'authenticated'),    # login → ready
    (8.1, 3.5, 8.4, 3.5, 'start'),            # ready → analyzing
    (9.5, 3.05, 9.5, 1.95, 'detected'),       # analyzing → scoring
    (8.4, 1.5, 8.1, 1.5, 'score computed'),   # scoring → update
    (6.9-1.1, 1.5, 6.0, 1.5, 'next frame\n(loop)'),  # update → ???
]

arrow(ax, 3.1, 3.5, 3.4, 3.5, 'user login', fs=8)
arrow(ax, 5.6, 3.5, 5.9, 3.5, 'authenticated', fs=8)
arrow(ax, 8.1, 3.5, 8.4, 3.5, 'start session', fs=8)
arrow(ax, 9.5, 3.05, 9.5, 1.95, 'emotion\ndetected', fs=8)
arrow(ax, 8.4, 1.5, 8.1, 1.5, 'score\ncomputed', fs=8)

# Loop: update → analyzing (curved)
ax.annotate("", xy=(9.5, 2.5), xytext=(7.0, 2.5),
            arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5,
                            connectionstyle="arc3,rad=-0.4"))
ax.text(8.25, 1.8, 'next frame', ha='center', fontsize=8)

# Stop session: update → session end
arrow(ax, 5.9, 1.5, 5.6, 1.5, 'stop session', fs=8)

# Final state
ax.add_patch(Circle((3.4, 1.5), 0.22, fc='black', ec='black'))
ax.add_patch(Circle((3.4, 1.5), 0.32, fc='none', ec='black', lw=2))
arrow(ax, 3.4, 1.5, 3.4, 1.5+0.01)  # dummy to get arrow style; draw line instead
ax.plot([3.4, 3.4], [1.95, 1.32+0.32], 'k-', lw=1.5)
ax.annotate("", xy=(3.4, 1.82), xytext=(3.4, 1.95),
            arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
ax.text(3.4, 0.9, 'System\nTerminates', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_state.png'), dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ State diagram saved")

# ─────────────────────────────────────────────────────────────────────────────
# DIAGRAM 6: Class Diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')
fig.patch.set_facecolor('white')

def uml_class(ax, x, y, name, attrs, methods, w=3.2, rh=0.55):
    total_h = rh + rh*len(attrs) + rh*len(methods)
    top = y + total_h/2
    # outer box
    ax.add_patch(FancyBboxPatch((x-w/2, y-total_h/2), w, total_h,
                                boxstyle="square,pad=0", fc='white', ec='black', lw=1.5))
    # name section
    ax.text(x, top - rh/2, name, ha='center', va='center',
            fontsize=10, fontstyle='italic', fontweight='bold')
    ax.plot([x-w/2, x+w/2], [top-rh, top-rh], 'k-', lw=1.2)
    # attributes
    for i, attr in enumerate(attrs):
        ay = top - rh - rh*i - rh/2
        ax.text(x-w/2+0.1, ay, attr, ha='left', va='center', fontsize=8)
    ax.plot([x-w/2, x+w/2], [top-rh-rh*len(attrs), top-rh-rh*len(attrs)], 'k-', lw=1.2)
    # methods
    for i, m in enumerate(methods):
        my = top - rh - rh*len(attrs) - rh*i - rh/2
        ax.text(x-w/2+0.1, my, m, ha='left', va='center', fontsize=8)

uml_class(ax, 2.2, 3.0, 'User',
          ['username : string', 'email : string', 'hashed_pwd : string', 'role : string'],
          ['register()', 'login()', 'getHistory()'])

uml_class(ax, 7.0, 3.0, 'Session',
          ['user_id : ObjectId', 'start_time : DateTime', 'avg_engagement : float', 'dominant_emotion : str'],
          ['start()', 'end()', 'appendFrame()'])

uml_class(ax, 11.8, 3.0, 'EmotionDetector',
          ['model : ResNet18', 'device : string'],
          ['detectFace()', 'classifyEmotion()', 'preprocess()', 'computeEngagement()'])

# Relationships
ax.annotate("", xy=(5.35, 3.2), xytext=(3.8, 3.2),
            arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
ax.text(4.575, 3.35, '1', ha='center', fontsize=9)
ax.text(5.1, 3.35, 'N', ha='center', fontsize=9)

ax.annotate("", xy=(10.15, 3.2), xytext=(8.65, 3.2),
            arrowprops=dict(arrowstyle="-|>", color='black', lw=1.5))
ax.text(9.4, 3.35, 'uses', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'diag_class.png'), dpi=DPI, bbox_inches='tight',
            facecolor='white')
plt.close()
print("✓ Class diagram saved")

print("\nAll diagrams saved to:", OUT)
