# ===============================
# DROWSINESS & YAWN DETECTION (FINAL - LOGIC UPDATE)
# ===============================

from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os
import mediapipe as mp
import tensorflow as tf
import pyttsx3  # Professional Audio Library

# ===============================
# 1. Dynamic Path Configuration
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

EYE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'eye_cnn.h5')
YAWN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yawn_cnn.h5')

# Load models
print("📦 Loading models...")
try:
    eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
    yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ===============================
# 2. Audio/Alarm Setup (pyttsx3)
# ===============================
engine = pyttsx3.init()
engine.setProperty('rate', 150) # Speaking speed

def speak_thread(msg):
    """Runs audio in a background thread."""
    try:
        engine.say(msg)
        engine.runAndWait()
    except:
        pass

def speak(msg):
    print(f"🔊 ALARM: {msg}")
    t = Thread(target=speak_thread, args=(msg,))
    t.start()

# ===============================
# 3. Mediapipe Setup
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# INDICES
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [0, 17, 61, 291]  # Full mouth (Top, Bottom, Left, Right)

# ===============================
# 4. Helper Function
# ===============================
def crop_and_preprocess(frame, landmarks, indices, size=(96,96)):
    h, w, _ = frame.shape
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    
    x, y, w_box, h_box = cv2.boundingRect(pts)
    
    pad = 5
    x, y = max(x - pad, 0), max(y - pad, 0)
    w_box, h_box = w_box + pad*2, h_box + pad*2
    
    roi = frame[y:y+h_box, x:x+w_box]
    if roi.size == 0: return None

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, size)
    roi_norm = roi_resized.astype("float32") / 255.0
    return np.expand_dims(roi_norm, axis=(0, -1))

# ===============================
# 5. Main Loop
# ===============================
print("📷 Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

drowsy_counter = 0
yawn_counter = 0
label_eye = "Open"
label_yawn = "No Yawn"
eye_color = (0, 255, 0)
yawn_color = (0, 255, 0)

frame_count = 0
SKIP_FRAMES = 5

try:
    while True:
        frame = vs.read()
        if frame is None: continue

        frame = imutils.resize(frame, width=450)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # --- PREDICTION LOGIC (Every 5 frames) ---
        if frame_count % SKIP_FRAMES == 0 and results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # 1. Eye Detection
            l_in = crop_and_preprocess(frame, face_landmarks.landmark, LEFT_EYE)
            r_in = crop_and_preprocess(frame, face_landmarks.landmark, RIGHT_EYE)

            if l_in is not None and r_in is not None:
                l_pred = eye_model.predict(l_in, verbose=0)[0][0]
                r_pred = eye_model.predict(r_in, verbose=0)[0][0]
                
                if (l_pred + r_pred) / 2 > 0.5:
                    label_eye = "Open"
                    eye_color = (0, 255, 0)
                    drowsy_counter = 0
                else:
                    label_eye = "Closed"
                    eye_color = (0, 0, 255)
                    drowsy_counter += 1

            # 2. Yawn Detection
            m_in = crop_and_preprocess(frame, face_landmarks.landmark, MOUTH)
            if m_in is not None:
                yawn_pred = yawn_model.predict(m_in, verbose=0)[0][0]
                if yawn_pred > 0.5:
                    label_yawn = "Yawn"
                    yawn_color = (0, 0, 255)
                    yawn_counter += 1
                else:
                    label_yawn = "No Yawn"
                    yawn_color = (0, 255, 0)
                    yawn_counter = 0

        # --- ALERT LOGIC (Combined Condition) ---
        
        # Condition 1: Eyes Closed AND Mouth Closed (Sleeping silently)
        if drowsy_counter > 3 and label_yawn == "No Yawn":
            cv2.putText(frame, "DROWSINESS ALERT!", (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if drowsy_counter % 5 == 0:
                speak("Wake up sir!")  # <--- Triggers only when both are closed

        # Condition 2: Mouth Open (Yawning)
        if yawn_counter > 3:
            cv2.putText(frame, "YAWN ALERT!", (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if yawn_counter % 5 == 0:
                speak("Take some fresh air!")

        # --- DISPLAY ---
        cv2.putText(frame, f"Eyes: {label_eye}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"Mouth: {label_yawn}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, yawn_color, 2)

        cv2.imshow("Safety System", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cv2.destroyAllWindows()
    vs.stop()
    print("Program stopped.")