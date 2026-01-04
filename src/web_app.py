# ===============================
# DROWSINESS & YAWN DETECTION (WEB APP)
# ===============================

from flask import Flask, render_template, Response
from imutils.video import VideoStream
from threading import Thread
import numpy as np
import imutils
import time
import cv2
import os
import mediapipe as mp
import tensorflow as tf

app = Flask(__name__)

# ===============================
# CONFIGURATION (TWEAK THIS IF EYES ARE WRONG)
# ===============================
# Change this to True if the app says "Closed" when your eyes are Open
INVERT_EYE_LOGIC = False  

# ===============================
# Load Trained CNN Models
# ===============================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
EYE_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'eye_cnn.h5')
YAWN_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'yawn_cnn.h5')


print("📦 Loading models...")
try:
    eye_model = tf.keras.models.load_model(EYE_MODEL_PATH)
    yawn_model = tf.keras.models.load_model(YAWN_MODEL_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

# ===============================
# Alarm Function
# ===============================
def speak_worker(msg):
    os.system(f'powershell "Add-Type -AssemblyName System.Speech; '
              f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{msg}\');"')

def speak(msg):
    print("🔊", msg)
    t = Thread(target=speak_worker, args=(msg,))
    t.start()

# ===============================
# VISUALIZATION HELPERS
# ===============================
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
DARK_GRAY = (50, 50, 50)
WHITE = (255, 255, 255)

def draw_corner_rect(img, bbox, color=CYAN, line_length=20, thickness=2):
    x, y, w, h = bbox
    cv2.line(img, (x, y), (x + line_length, y), color, thickness)
    cv2.line(img, (x, y), (x, y + line_length), color, thickness)
    cv2.line(img, (x + w, y), (x + w - line_length, y), color, thickness)
    cv2.line(img, (x + w, y), (x + w, y + line_length), color, thickness)
    cv2.line(img, (x, y + h), (x + line_length, y + h), color, thickness)
    cv2.line(img, (x, y + h), (x, y + h - line_length), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w - line_length, y + h), color, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - line_length), color, thickness)
    return img

def draw_hud(frame, eye_status, yawn_status, drowsy_score, yawn_score):
    overlay = frame.copy()
    h, w, _ = frame.shape
    
    cv2.rectangle(overlay, (0, 0), (160, h), (0, 0, 0), -1)
    
    if drowsy_score > 10 or yawn_score > 8:
        cv2.rectangle(overlay, (0, 0), (w, h), RED, -1)
        alpha = 0.3 
    else:
        alpha = 0.4 

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    cv2.putText(frame, "SYSTEM STATUS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CYAN, 1, cv2.LINE_AA)
    
    color_e = GREEN if eye_status == "Open" else RED
    cv2.putText(frame, "EYES", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, eye_status, (70, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_e, 2, cv2.LINE_AA)
    
    bar_width = 100
    fill = int((drowsy_score / 15) * bar_width)
    fill = min(fill, bar_width)
    cv2.rectangle(frame, (10, 75), (10 + bar_width, 85), DARK_GRAY, -1)
    cv2.rectangle(frame, (10, 75), (10 + fill, 85), color_e, -1)

    color_y = GREEN if yawn_status == "No Yawn" else MAGENTA
    cv2.putText(frame, "MOUTH", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, "Active" if yawn_status=="Yawn" else "Normal", (70, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_y, 2, cv2.LINE_AA)

    fill_y = int((yawn_score / 10) * bar_width)
    fill_y = min(fill_y, bar_width)
    cv2.rectangle(frame, (10, 125), (10 + bar_width, 135), DARK_GRAY, -1)
    cv2.rectangle(frame, (10, 125), (10 + fill_y, 135), color_y, -1)

# ===============================
# Mediapipe Setup
# ===============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 61, 291]

# ===============================
# Helper Functions
# ===============================
def crop_and_preprocess(frame, landmarks, indices, size=(96,96)):
    h, w, _ = frame.shape
    pts = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices])
    
    x, y, w_box, h_box = cv2.boundingRect(pts)
    
    pad_x = int(w_box * 0.2) 
    pad_y = int(h_box * 0.2) 
    
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w_box = min(w - x, w_box + (2 * pad_x))
    h_box = min(h - y, h_box + (2 * pad_y))

    roi = frame[y:y+h_box, x:x+w_box]
    if roi.size == 0:
        return None, None
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, size)
    roi_norm = roi_resized.astype("float32") / 255.0
    roi_input = np.expand_dims(roi_norm, axis=(0, -1))
    return roi_input, (x, y, w_box, h_box)

# ===============================
# FLASK GENERATOR LOOP
# ===============================
def generate_frames():
    cap = cv2.VideoCapture(0)
    
    drowsy_counter = 0
    yawn_counter = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = imutils.resize(frame, width=400)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        label_eye = "Open"
        label_yawn = "No Yawn"

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # ---- Eye detection ----
            left_eye_input, l_box = crop_and_preprocess(frame, face_landmarks.landmark, LEFT_EYE)
            right_eye_input, r_box = crop_and_preprocess(frame, face_landmarks.landmark, RIGHT_EYE)

            if left_eye_input is not None and right_eye_input is not None:
                draw_corner_rect(frame, l_box, CYAN) 
                draw_corner_rect(frame, r_box, CYAN)

                left_pred = eye_model.predict(left_eye_input, verbose=0)[0][0]
                right_pred = eye_model.predict(right_eye_input, verbose=0)[0][0]
            
                eye_avg = (left_pred + right_pred) / 2
                
                # DEBUG PRINT: Watch your terminal to see the numbers!
                # print(f"DEBUG: Eye Prob: {eye_avg:.4f} (L: {left_pred:.2f}, R: {right_pred:.2f})")

                if INVERT_EYE_LOGIC:
                    # Logic B: 0=Open, 1=Closed
                    label_eye = "Closed" if eye_avg > 0.5 else "Open"
                else:
                    # Logic A: 0=Closed, 1=Open (Default)
                    label_eye = "Open" if eye_avg > 0.5 else "Closed"

                if label_eye == "Closed":
                    drowsy_counter += 1
                    if drowsy_counter > 15:
                        if drowsy_counter % 5 == 0:
                            speak("wake up ,sir!")
                else:
                    drowsy_counter = 0

            # ---- Yawn detection ----
            mouth_input, m_box = crop_and_preprocess(frame, face_landmarks.landmark, MOUTH)
            
            if mouth_input is not None:
                draw_corner_rect(frame, m_box, MAGENTA)

                yawn_pred = yawn_model.predict(mouth_input, verbose=0)[0][0]
                label_yawn = "Yawn" if yawn_pred > 0.3 else "No Yawn"

                if label_yawn == "Yawn":
                    yawn_counter += 1
                    if yawn_counter > 10:
                        if yawn_counter % 5 == 0:
                            speak("Take some fresh air sir!")
                else:
                    yawn_counter = 0
        
        draw_hud(frame, label_eye, label_yawn, drowsy_counter, yawn_counter)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ===============================
# FLASK ROUTES
# ===============================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)