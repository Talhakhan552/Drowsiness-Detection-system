# 😴 Real-Time Drowsiness & Yawn Detection System

This project is a machine learning-based safety system designed to detect driver drowsiness and yawning in real-time. It utilizes **MediaPipe** for facial landmark detection and custom **Convolutional Neural Networks (CNN)** for classifying eye and mouth states.

If signs of fatigue (closed eyes or yawning) persist beyond a specific time threshold, the system triggers audio-visual alerts to warn the driver.

---

## 🌟 Key Features

* **Real-Time Monitoring:** Uses a webcam feed to track facial landmarks efficiently.
* **Dual CNN Models:**
    * **Eye Model:** Detects if eyes are "Open" or "Closed".
    * **Yawn Model:** Detects "Yawn" vs. "No Yawn".
* **Intelligent Alerting:**
    * **Visual:** Displays status text and "DROWSINESS ALERT" or "YAWN ALERT" on the video feed.
    * **Audio:** Uses system text-to-speech to announce "Wake up sir!" or "Take some fresh air sir!".
* **Robust Evaluation:** Includes scripts to generate Confusion Matrices, ROC Curves, and Classification Reports for model validation.

---


# 🚗 Real-Time Driver Drowsiness & Yawn Detection System

An AI-powered safety system that monitors driver fatigue in real-time using Deep Learning and Computer Vision.

## 🌟 Features
* **Real-Time Monitoring:** Tracks eyes and mouth using MediaPipe Face Mesh.
* **Deep Learning Models:** Custom CNNs for Eye Closure (97% Acc) and Yawning (94% Acc).
* **Smart Alerts:** * "Microsleep" detection (Eyes Closed > 15 frames).
    * "Yawn" detection (Mouth Open > 10 frames).
* **Voice Warnings:** Text-to-Speech alerts ("Wake up sir!").
* **Web Dashboard:** Flask-based HUD with visual probability bars.

## 🛠️ Tech Stack
* Python 3.x
* TensorFlow / Keras
* OpenCV & MediaPipe
* Flask (Web Interface)

## 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `python src/web_app.py`


## 📂 Project Structure

Based on the repository organization:

```text
ML_Project/
├── models/                  # Stores trained .h5 models
│   ├── eye_cnn.h5
│   └── yawn_cnn.h5
│
├── result/                  # Evaluation metrics and plots
│   ├── Eye_Drowsiness_Model_confusion_matrix.png
│   ├── Yawn_Detection_Model_metrics.png
│   └── ...
│
├── src/                     # Source code scripts
│   ├── drowsiness_app.py    # MAIN APPLICATION
│   ├── train_eye_model.py   # Training script for eyes
│   ├── train_yawn_model.py  # Training script for yawns
│   ├── eye_test_result.py   # Evaluation script for eyes
│   └── yawn_test_result.py  # Evaluation script for yawns
│
├── train/                   # Dataset for Eye detection
│   ├── Closed_Eyes/
│   └── Open_Eyes/
│
└── train_yawn/              # Dataset for Yawn detection
    ├── Yawn/                # (Positive class)
    └── no yawn/             # (Negative class)




🛠️ Prerequisites & Installation
Ensure you have Python installed. You will need the following libraries:

==> pip install tensorflow opencv-python mediapipe imutils scikit-learn matplotlib seaborn os

TensorFlow/Keras: For loading and running the CNN models
MediaPipe: For extracting face mesh landmarks.
OpenCV & Imutils: For video processing and resizing.
Matplotlib/Seaborn: For plotting evaluation graphs.



🧠 Model Architecture
The system uses two separate CNN models with the following structure:
Input: 96x96 Grayscale Images.
Layers: 2x Convolutional Layers (ReLU) + Max Pooling, followed by a Flatten layer and Dense layers.
Output: Sigmoid activation (Binary Classification).


⚙️ Detection Logic
Drowsiness (Eyes): The alarm triggers if eyes remain "Closed" for more than 15 consecutive frames.
Yawning (Mouth): The alarm triggers if a "Yawn" is detected for more than 10 consecutive frames.