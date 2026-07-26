"""
generate_arch_icons.py  (v2)
Architecture Diagram matching sample Black Book layout exactly:
  - Top-left: User actor icon → Input UI box (webcam + upload icons)
  - Center: System boundary with nested Face Detection + Emotion modules
  - Top-right: MongoDB cloud → FastAPI box
  - Bottom-right: Output UI (Dashboard + Session)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.path import Path
import numpy as np
import os, io, urllib.request

OUT = r"c:\Users\sanka\MindTrace\report\images"
os.makedirs(OUT, exist_ok=True)

ICONS = {
    "user":     "https://img.icons8.com/ios-filled/96/user.png",
    "webcam":   "https://img.icons8.com/color/96/webcam.png",
    "upload":   "https://img.icons8.com/color/96/upload.png",
    "mediapipe":"https://img.icons8.com/color/96/artificial-intelligence.png",
    "face":     "https://img.icons8.com/color/96/face-id.png",
    "brain":    "https://img.icons8.com/color/96/brain.png",
    "gear":     "https://img.icons8.com/color/96/settings.png",
    "chart":    "https://img.icons8.com/color/96/activity-feed.png",
    "react":    "https://img.icons8.com/color/96/react-native.png",
    "mongodb":  "https://img.icons8.com/color/96/mongodb.png",
    "api":      "https://img.icons8.com/color/96/api-settings.png",
    "dashboard":"https://img.icons8.com/color/96/dashboard.png",
}

def dl(url, size=(64,64)):
    try:
        with urllib.request.urlopen(url, timeout=8) as r:
            data = r.read()
        from PIL import Image
        img = Image.open(io.BytesIO(data)).convert("RGBA").resize(size)
        return np.array(img)
    except Exception as e:
        print(f"  ⚠ {url}: {e}")
        return None

def icon(ax, arr, x, y, zoom=0.30):
    if arr is None: return
    ab = AnnotationBbox(OffsetImage(arr, zoom=zoom), (x, y), frameon=False)
    ax.add_artist(ab)

def box(ax, x, y, w, h, label="", fc='white', ec='black', lw=1.5, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="square,pad=0", fc=fc, ec=ec, lw=lw,
                                clip_on=False))
    if label:
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=fs,
                fontweight='bold' if bold else 'normal', multialignment='center')

def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color='black',
                                lw=1.5, mutation_scale=14), clip_on=False)

def cloud(ax, cx, cy, w=2.4, h=1.4):
    """Draw a simple cloud shape centred at (cx,cy)."""
    from matplotlib.patches import Ellipse
    for dx, dy, rx, ry in [
        (0,   0,    w*0.38, h*0.55),
        (-w*0.26, -h*0.1, w*0.28, h*0.42),
        ( w*0.26, -h*0.1, w*0.28, h*0.42),
        (0,  -h*0.22, w*0.45, h*0.38),
    ]:
        ax.add_patch(Ellipse((cx+dx, cy+dy), rx*2, ry*2,
                              fc='#f0f0f0', ec='black', lw=1.5, zorder=2))

# ── Download icons ─────────────────────────────────────────────────────────────
print("Downloading icons...")
ic = {}
for k, u in ICONS.items():
    print(f"  {k}...", end=" ", flush=True)
    ic[k] = dl(u)
    print("ok" if ic[k] is not None else "FAILED")

# ── Canvas ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, 15); ax.set_ylim(0, 10)
ax.axis('off'); fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# ═══════════════════════════════════════════════════════════════
# TOP-LEFT: User actor
# ═══════════════════════════════════════════════════════════════
icon(ax, ic["user"], 1.5, 9.1, zoom=0.45)
ax.text(1.5, 8.65, "User / Student", ha='center', fontsize=9)
arr(ax, 1.5, 8.55, 1.5, 7.9)

# Input UI box
box(ax, 0.4, 6.5, 2.2, 1.3, fc='#f9f9f9', ec='black', lw=1.5)
ax.text(1.5, 7.65, "Input UI", ha='center', fontsize=9, fontweight='bold')

# Webcam sub-box
box(ax, 0.55, 6.6, 0.85, 0.85, fc='white', ec='black', lw=1.2)
icon(ax, ic["webcam"], 0.975, 7.02, zoom=0.28)

# Upload sub-box
box(ax, 1.6, 6.6, 0.85, 0.85, fc='white', ec='black', lw=1.2)
icon(ax, ic["upload"], 2.025, 7.02, zoom=0.28)

# Arrow Input UI → System
arr(ax, 1.5, 6.5, 1.5, 5.9)

# ═══════════════════════════════════════════════════════════════
# CENTER: System's Internal Structure
# ═══════════════════════════════════════════════════════════════
box(ax, 0.2, 1.2, 10.0, 4.6, fc='#f5f7ff', ec='black', lw=2)
ax.text(0.45, 5.65, "System's Internal Structure", ha='left',
        fontsize=9, fontweight='bold')

# ── Face Detection Module ─────────────────────────────────────
box(ax, 0.4, 2.8, 4.8, 2.8, fc='white', ec='black', lw=1.3)
ax.text(0.6, 5.45, "Face Detection Module", ha='left', fontsize=8.5, fontweight='bold')

# MediaPipe box
box(ax, 0.55, 3.8, 1.55, 1.6, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["mediapipe"], 1.325, 4.88, zoom=0.26)
ax.text(1.325, 3.97, "MediaPipe\nFaceMesh", ha='center', fontsize=7.5, multialignment='center')
arr(ax, 2.1, 4.6, 2.5, 4.6)

# Face ROI box
box(ax, 2.5, 3.8, 1.45, 1.6, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["face"], 3.225, 4.88, zoom=0.26)
ax.text(3.225, 3.97, "Extract\nFace ROI", ha='center', fontsize=7.5, multialignment='center')
arr(ax, 3.95, 4.6, 4.3, 4.6)

# Detected face result box
box(ax, 4.3, 4.15, 0.75, 0.85, fc='white', ec='black', lw=1.1)
ax.text(4.675, 4.73, "Detected\nFace", ha='center', fontsize=7, multialignment='center')

# Outcome labels (like Masked/NoMasked in sample)
ax.text(4.85, 5.25, "Face Found →", ha='center', fontsize=7, color='#007700')
ax.text(4.85, 4.0,  "No Face ×",   ha='center', fontsize=7, color='#cc0000')

# ── Emotion & Engagement Module ───────────────────────────────
box(ax, 0.4, 1.3, 4.8, 1.35, fc='white', ec='black', lw=1.3)
ax.text(0.6, 2.5, "Emotion & Engagement Module", ha='left', fontsize=8.5, fontweight='bold')

box(ax, 0.55, 1.4, 1.35, 0.9, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["brain"], 1.225, 2.05, zoom=0.22)
ax.text(1.225, 1.5, "ResNet-18\nInference", ha='center', fontsize=7, multialignment='center')
arr(ax, 1.9, 1.85, 2.3, 1.85)

box(ax, 2.3, 1.4, 1.35, 0.9, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["gear"], 2.975, 2.05, zoom=0.22)
ax.text(2.975, 1.5, "Head Pose\nEstimation", ha='center', fontsize=7, multialignment='center')
arr(ax, 3.65, 1.85, 4.05, 1.85)

box(ax, 4.05, 1.4, 1.0, 0.9, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["chart"], 4.55, 2.05, zoom=0.22)
ax.text(4.55, 1.5, "Score\n0–100", ha='center', fontsize=7, multialignment='center')

# ── MongoDB / Authentication sub-box (right side of system) ──
box(ax, 5.6, 2.8, 4.4, 2.8, fc='white', ec='black', lw=1.3)
ax.text(5.8, 5.45, "Authentication & Storage Module", ha='left',
        fontsize=8.5, fontweight='bold')

box(ax, 5.75, 4.0, 1.5, 1.4, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["api"], 6.5, 4.9, zoom=0.24)
ax.text(6.5, 4.1, "FastAPI\nAuth", ha='center', fontsize=7.5, multialignment='center')
arr(ax, 7.25, 4.7, 7.65, 4.7)

box(ax, 7.65, 4.0, 1.5, 1.4, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["mongodb"], 8.4, 4.9, zoom=0.24)
ax.text(8.4, 4.1, "MongoDB\nAtlas", ha='center', fontsize=7.5, multialignment='center')

box(ax, 5.75, 1.4, 4.1, 1.2, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["react"], 6.5, 1.95, zoom=0.22)
ax.text(6.5, 1.5, "React Dashboard", ha='center', fontsize=7.5)
icon(ax, ic["mongodb"], 8.5, 1.95, zoom=0.22)
ax.text(8.5, 1.5, "Session Store", ha='center', fontsize=7.5)

# Arrow face detection → emotion module
arr(ax, 2.9, 2.8, 2.9, 2.55)

# Arrow emotion module → auth/storage
arr(ax, 5.05, 1.85, 5.6, 1.85)
arr(ax, 5.05, 4.7,  5.6, 4.7)

# ═══════════════════════════════════════════════════════════════
# TOP-RIGHT: User Information Cloud → FastAPI → Output
# ═══════════════════════════════════════════════════════════════
ax.text(12.2, 9.8, "User Information", ha='center', fontsize=9, fontweight='bold')
cloud(ax, 12.2, 9.1, w=2.8, h=1.5)

# Bidirectional arrow cloud ↕ FastAPI
ax.annotate("", xy=(12.2, 8.1), xytext=(12.2, 8.5),
            arrowprops=dict(arrowstyle="<|-|>", color='black', lw=1.5, mutation_scale=14))

# FastAPI box
box(ax, 10.9, 7.3, 2.6, 0.75, fc='white', ec='black', lw=1.5)
ax.text(12.2, 7.675, "FastAPI Backend", ha='center', fontsize=9)

# Arrow FastAPI → Output UI
arr(ax, 12.2, 7.3, 12.2, 5.95)

# Output UI box
box(ax, 10.5, 4.3, 3.4, 1.5, fc='white', ec='black', lw=1.5)
ax.text(12.2, 5.65, "Output UI", ha='center', fontsize=9, fontweight='bold')

box(ax, 10.65, 4.45, 1.4, 0.85, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["dashboard"], 11.35, 4.87, zoom=0.22)
ax.text(11.35, 4.5, "Live\nDashboard", ha='center', fontsize=7.5, multialignment='center')

box(ax, 12.25, 4.45, 1.5, 0.85, fc='#eef2ff', ec='black', lw=1.1)
icon(ax, ic["react"], 13.0, 4.87, zoom=0.22)
ax.text(13.0, 4.5, "Session\nReport", ha='center', fontsize=7.5, multialignment='center')

# Arrow system → output
arr(ax, 10.2, 3.5, 11.5, 4.3)

plt.tight_layout(pad=0.2)
plt.savefig(os.path.join(OUT, 'diag_architecture.png'),
            dpi=220, bbox_inches='tight', facecolor='white')
plt.close()
print("\n✓ Architecture diagram saved.")
