# GuardianEye: Embedded Real-Time Suspicious Activity Detection

GuardianEye is an intelligent surveillance system that leverages an ESP32-CAM edge device and a Deep Learning backend to detect suspicious human activities in real-time. By utilizing MediaPipe Pose Estimation, the system analyzes body language to distinguish between normal behavior and potential security threats, triggering automated email alerts with visual evidence.

---

##  Features

- **Edge Video Streaming**: Low-latency MJPEG streaming from ESP32-CAM over Wi-Fi.  
- **AI-Powered Detection**: Pose estimation using MediaPipe (33 landmarks) combined with a custom TensorFlow/Keras Neural Network.  
- **Real-Time Analytics**: Live dashboard built with Streamlit showing confidence scores, FPS, and activity classification.  
- **Automated Alert System**: Instant SMTP email notifications containing the captured frame of the suspicious event.  
- **Temporal Filtering**: Prediction smoothing logic to reduce false positives in dynamic environments.

---

##  System Architecture

### Edge (ESP32-CAM)
- Captures VGA video and streams it via HTTP.

### Backend (Flask / Python)
- Receives MJPEG stream.  
- Extracts 34 pose features (x, y coordinates of key joints).  
- Runs inference through the trained `.h5` model.  
- Manages the SMTP alert trigger.

### Frontend (Streamlit)
- Provides a high-level UI for security monitoring.

---

##  Hardware & Software Requirements

### Hardware
- ESP32-CAM Module (OV2640 Sensor)  
- FTDI Programmer (for uploading code to ESP32)  
- Local Wi-Fi Network  
- PC/Server with Python 3.10+

### Software
- **Python Libraries**:  
  `tensorflow`, `mediapipe`, `opencv-python`, `flask`, `streamlit`, `scikit-learn`, `pandas`
- **Embedded**:  
  Arduino IDE with ESP32 board support

---

##  Project Structure

- `backend_final.py` – Core processing engine (Flask + AI Inference)  
- `frontend_final.py` – Streamlit dashboard  
- `train_model.py` – Script to train the Keras classifier on collected CSV data  
- `datasetcreation.py` – Utility to record live pose data and label it for training  
- `suspicious_activity_model.h5` – Trained model weights for detection

---
## 🎥 Demo Video

https://github.com/santhoshici/Smart_Surveillance/video_demo.mp4
---

##  Getting Started

### 1. ESP32-CAM Setup
1. Open the Arduino IDE.  
2. Install the ESP32 board library.  
3. Upload the `CameraWebServer` example (configured for MJPEG streaming) or your custom streaming firmware.  
4. Note the IP Address assigned to the ESP32.

### 2. Backend Configuration
1. Update `ESP32_CAM_URL` in `backend_final.py` with your device IP.  
2. Set your Gmail/SMTP credentials in the `EMAIL_ADDRESS` and `EMAIL_PASSWORD` variables.  
3. Install dependencies:
   ```bash
   pip install mediapipe tensorflow flask opencv-python streamlit scikit-learn
