# MindTrace: Real-Time Behavioral & Emotion Tracking System
**Presentation Content Outline**

---

### Slide 1: Introduction
* **MindTrace** is an advanced AI-driven system designed to monitor, analyze, and report human emotions and engagement in real-time.
* Utilizing webcam feeds, it processes facial expressions and head movements to provide actionable insights.
* **Key Technologies:** React (Frontend), FastAPI (Backend), PyTorch (ResNet18 Model), MediaPipe (Face Mesh), and MongoDB.
* **Applications:** E-learning focus tracking, remote work productivity, telehealth emotional assessment, and usability testing.

---

### Slide 2: Literature Review
* Traditional emotion recognition relies heavily on intrusive hardware (EEG, wearables) which are impractical for daily use.
* Recent advancements in CNNs (Convolutional Neural Networks) have made vision-based emotion detection highly accurate.
* Studies show that combining multiple inputs—such as facial micro-expressions (via ResNet) and head pose orientation (via MediaPipe)—yields a much more accurate "Engagement/Focus" metric than emotion alone.
* MindTrace builds upon these findings by offering a stateless, lightweight, browser-based solution without requiring specialized hardware.

---

### Slide 3: Need / Objective and Scope
* **Need:** The shift to remote environments (work, education) has created a disconnect in reading non-verbal cues and assessing audience engagement.
* **Objective:** To develop a highly responsive, non-intrusive web application that tracks human emotion and calculates a real-time "Focus Score."
* **Scope:** 
  * Live monitoring of 7 core emotions.
  * Real-time calculation of user engagement based on head pose and expressions.
  * Generation of historical analytical reports and live alerts for administrators.

---

### Slide 4: Problem Statement
* *“In remote and digital environments, it is difficult to gauge user engagement, frustration, and attention span without intrusive monitoring.”*
* Existing solutions are either too computationally heavy for browsers, require native desktop installations, or fail to combine emotion with physical attention (head pose).

---

### Slide 5: Architectural Diagram
*(Note for PPT: Insert the architecture block diagram here)*
* **Frontend (Vercel):** React + TypeScript. Captures webcam feed at 2 FPS and renders live bounding boxes & alerts.
* **Backend (Hugging Face Spaces):** FastAPI Python server. Exposes stateless `/analyze_frame` endpoint.
* **AI Engine:** MediaPipe extracts face landmarks; ResNet18 predicts emotions.
* **Database (MongoDB Atlas):** Stores user sessions, historical emotion logs, and role-based access data.

---

### Slide 6: Mathematical Model with Formula

**1. Eye Aspect Ratio (EAR) for Blink Detection:**
Used to calculate $M_b$ (Blink Modifier) for fatigue tracking.
* **Formula:** $EAR = \frac{||p_2 - p_6|| + ||p_3 - p_5||}{2 ||p_1 - p_4||}$
* *Where:* $p_1, p_4$ are horizontal eye landmarks; $p_2, p_3, p_5, p_6$ are vertical eye landmarks.

**2. Head Pose Estimation (Perspective-n-Point / PnP):**
Used to map 3D face model points to 2D image coordinates to find Pitch ($\theta$), Yaw ($\phi$), and Roll ($\psi$).
* **Formula:** $s \mathbf{p}_c = \mathbf{K} [\mathbf{R} | \mathbf{t}] \mathbf{P}_w$
* *Where:* 
  * $s$ = scale factor
  * $\mathbf{p}_c$ = 2D image coordinates
  * $\mathbf{K}$ = Camera Intrinsic Matrix
  * $[\mathbf{R} | \mathbf{t}]$ = Rotation and Translation matrix
  * $\mathbf{P}_w$ = 3D World coordinates

**3. Total Engagement / Focus Score ($E$):**
Calculated on a scale of 0 to 100.
* **Formula:** $E = \max(0, \min(100, B_e - P_h + M_b))$
* *Where:*
  * $B_e$ (Base Emotion Score): 85-95 for Happy/Surprise, 65-80 for Neutral, 30-50 for Negative.
  * $P_h$ (Head Pose Penalty): $-30$ IF $|\phi| > 30^\circ$ OR $|\theta| > 30^\circ$ (Looking away).
  * $M_b$ (Blink Modifier): $-15$ if blink rate $> 35$ (Fatigued), $+10$ if blink rate $< 5$ (Intense focus).

---

### Slide 7: Features (Functionality & Algorithm Details)
* **Functionality:** Live tracking dashboard, secure role-based login (Admin/User), historical session analytics, and instant desktop alerts for low focus.
* **Algorithm Pipeline:**
  1. **Capture:** Frontend captures base64 image -> sends to Backend.
  2. **Detect:** MediaPipe extracts facial bounding box and 468 landmarks.
  3. **Crop & Transform:** Face ROI is cropped, resized to 96x96, and normalized.
  4. **Inference:** PyTorch ResNet18 classifies into 1 of 7 RAF-DB emotions.
  5. **Feedback:** Bounding box, confidence score, and engagement metric returned to UI.

---

### Slide 8: Use Case Diagram
*(Note for PPT: Draw standard stick-figure Use Case Diagram)*
* **Standard User:** Login -> Start Live Session -> View Own Dashboard -> Receive Real-time Alerts -> Stop Session.
* **Administrator:** Login -> Access Admin Panel -> View Global Analytics -> Monitor Active Sessions -> View Total Emotion Stats.
* **System (Backend):** Process Frames -> Calculate Focus Score -> Write to MongoDB.

---

### Slide 9: Class Diagram & DFD
*(Note for PPT: Describe DFD flow)*
* **Level 0 DFD:** User (Video Stream) $\rightarrow$ MindTrace System $\rightarrow$ Insights & Alerts (Output).
* **Level 1 DFD:** 
  1. Frontend Client captures frame.
  2. API Router validates JWT Token.
  3. ML Module performs ResNet inference.
  4. Database Manager logs timestamped data.
* **Class Diagram Entities:** `User`, `Session`, `EmotionLog`, `AlertManager`, `ML_Pipeline`.

---

### Slide 10: System Specification & Data Sets
* **Hardware:** Standard Web Camera, Any modern CPU/GPU for the server.
* **Software:** Node.js (Vite), Python 3.10+, PyTorch, OpenCV, MongoDB.
* **Dataset Used for Training:** **RAF-DB** (Real-world Affective Faces Database).
  * Contains thousands of heavily augmented, real-world facial images.
  * Mapped to 7 basic emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.

---

### Slide 11: Expected Results
* **High Accuracy:** >80% accuracy in real-world lighting conditions due to RAF-DB fine-tuning.
* **Low Latency:** Inference and network round-trip complete in <500ms (processing at 2 FPS).
* **Actionable Reporting:** Administrators can easily identify exactly when and why a user lost focus during a session through visual graphs.

---

### Slide 12: Conclusion
* MindTrace successfully bridges the gap between physical human behavior and digital analytics.
* By combining deep learning (ResNet) with geometric computer vision (MediaPipe) in a stateless web architecture, the system is both highly scalable and highly accessible.
* It proves that complex psychological monitoring can be done in real-time within a standard web browser.

---

### Slide 13: Paper Published
*(Note: If you haven't actually published a paper yet, you can title this "Proposed Publication" or list the target journal)*
* **Title:** "Real-Time Emotion and Engagement Tracking via Stateless Web Architectures using ResNet18."
* **Journal/Conference:** [Insert Target Conference/Journal here, e.g., IEEE Conference on Computer Vision / Springer]
* **Status:** [Drafted / Under Review / Published]

---

### Slide 14: References
1. Li, S., Deng, W., & Du, J. (2017). Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild (RAF-DB).
2. Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building Perception Pipelines.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. (ResNet).
4. FastAPI Documentation - tiangolo.com.
5. React & Vite Official Documentation.
