import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow import keras
import json
import time
import pickle
from collections import deque
from threading import Thread, Lock
from flask import Flask, jsonify, Response
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

ESP32_CAM_URL = "http://10.150.192.178:81/stream"
MODEL_PATH = "suspicious_activity_model.h5"
SCALER_PATH = "scaler.pkl"
LABEL_ENCODER_PATH = "label_encoder.json"
CONFIDENCE_THRESHOLD = 0.6
SMOOTHING_WINDOW = 5
DISPLAY_WIDTH, DISPLAY_HEIGHT = 640, 480
FEATURE_SIZE = 34

EMAIL_ADDRESS = "blendercandy050406@gmail.com"
EMAIL_PASSWORD = "jffc jcgc rrdc abhm"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
RECIPIENT_EMAIL = "santhosh05042006@gmail.com"

latest_info = {
    "label": "N/A",
    "confidence": 0.0,
    "fps": 0.0,
    "alert": False,
    "frame": None,
    "original_frame": None,
}
state_lock = Lock()


def load_model_and_artifacts():
    """Load model, scaler, and label encoder"""
    print("Loading model and artifacts...")
    model = keras.models.load_model(MODEL_PATH)

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    with open(LABEL_ENCODER_PATH, "r") as f:
        label_map = json.load(f)
    label_map = {int(k): v for k, v in label_map.items()}

    print(f"✓ Model loaded with input shape: {model.input_shape}")
    print(f"✓ Scaler loaded")
    print(f"✓ Label map: {label_map}")

    return model, scaler, label_map


def initialize_pose_detector():
    """Initialize MediaPipe Pose detector"""
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_pose, pose, mp_drawing, mp_drawing_styles


def extract_keypoints(pose_landmarks):
    """Extract 34 features (17 landmarks × 2 coordinates) from pose landmarks"""
    if pose_landmarks is None:
        return None

    keypoints = []
    for i in range(17):
        landmark = pose_landmarks.landmark[i]
        keypoints.extend([landmark.x, landmark.y])

    return np.array(keypoints, dtype=np.float32)


def send_email_alert(activity_label, confidence, image_bytes):
    """Send email alert with non-annotated image attachment"""
    try:
        subject = "🚨 Suspicious Activity Alert Detected"
        body = f"""Alert! Suspicious activity has been detected by the system.

Activity: {activity_label.upper()}
Confidence: {confidence * 100:.1f}%
Time: {time.strftime("%Y-%m-%d %H:%M:%S")}

Please check the attached image for details.
"""
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = RECIPIENT_EMAIL
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "plain"))

        image_attachment = MIMEBase("application", "octet-stream")
        image_attachment.set_payload(image_bytes)
        encoders.encode_base64(image_attachment)
        image_attachment.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"alert_{int(time.time())}.jpg",
        )
        msg.attach(image_attachment)

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("✉ Email alert with image sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")


class ActivityPredictor:
    """Predict activity from pose keypoints"""

    def __init__(self, model, scaler, label_map, smoothing_window=5):
        self.model = model
        self.scaler = scaler
        self.label_map = label_map
        self.smoothing_window = smoothing_window
        self.prediction_history = deque(maxlen=smoothing_window)
        self.last_alert_time = 0
        self.alert_cooldown = 10

    def predict(self, keypoints):
        """
        Predict activity from keypoints

        Args:
            keypoints: numpy array of shape (34,) containing 17 landmarks × 2 coordinates

        Returns:
            label: predicted activity label
            confidence: raw confidence score
            smoothed_confidence: smoothed confidence score over window
        """
        if keypoints is None:
            return None, 0.0, 0.0
        if len(keypoints) != FEATURE_SIZE:
            print(f"Warning: Expected {FEATURE_SIZE} features, got {len(keypoints)}")
            return None, 0.0, 0.0

        keypoints_scaled = self.scaler.transform(keypoints.reshape(1, -1))[0]
        keypoints_input = keypoints_scaled.reshape(1, -1)

        predictions = self.model.predict(keypoints_input, verbose=0)[0]
        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])

        self.prediction_history.append(predictions)

        if len(self.prediction_history) > 0:
            smoothed_predictions = np.mean(self.prediction_history, axis=0)
            smoothed_class = np.argmax(smoothed_predictions)
            smoothed_confidence = float(smoothed_predictions[smoothed_class])
            label = self.label_map[smoothed_class]
        else:
            smoothed_confidence = confidence
            label = self.label_map[predicted_class]

        return label, confidence, smoothed_confidence

    def should_send_alert(self):
        """Check if enough time has passed since last alert"""
        current_time = time.time()
        if current_time - self.last_alert_time >= self.alert_cooldown:
            self.last_alert_time = current_time
            return True
        return False


def draw_results(
    frame,
    pose_landmarks,
    label,
    confidence,
    fps,
    mp_pose,
    mp_drawing,
    mp_drawing_styles,
):
    """Draw pose landmarks and detection results on frame"""
    annotated_frame = frame.copy()

    if pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_frame,
            pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    if label == "suspicious":
        color = (0, 0, 255)
        bg_color = (0, 0, 200)
    else:
        color = (0, 255, 0)
        bg_color = (0, 200, 0)

    # Prepare text
    label_text = f"Activity: {label.upper()}"
    conf_text = f"Confidence: {confidence * 100:.1f}%"
    fps_text = f"FPS: {fps:.1f}"

    # Draw background rectangle and text
    cv2.rectangle(annotated_frame, (10, 10), (400, 110), (0, 0, 0), -1)
    cv2.rectangle(annotated_frame, (10, 10), (400, 110), bg_color, 2)
    cv2.putText(
        annotated_frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )
    cv2.putText(
        annotated_frame,
        conf_text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        annotated_frame,
        fps_text,
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Draw alert warning if suspicious activity detected
    if label == "suspicious" and confidence > CONFIDENCE_THRESHOLD:
        warning_text = "⚠ ALERT: SUSPICIOUS ACTIVITY DETECTED!"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (annotated_frame.shape[1] - text_size[0]) // 2

        # Blinking effect
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(
                annotated_frame,
                (text_x - 10, annotated_frame.shape[0] - 60),
                (text_x + text_size[0] + 10, annotated_frame.shape[0] - 20),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                annotated_frame,
                warning_text,
                (text_x, annotated_frame.shape[0] - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

    return annotated_frame


def attempt_reconnect(url):
    """Attempt to reconnect to camera/stream"""
    while True:
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            print("✓ Connected to ESP32-CAM (or webcam)")
            return cap
        print("Waiting for ESP32-CAM connection. Retrying in 5 seconds...")
        time.sleep(5)


def inference_loop():
    """Main inference loop"""
    try:
        # Load model and artifacts
        model, scaler, label_map = load_model_and_artifacts()
        mp_pose, pose, mp_drawing, mp_drawing_styles = initialize_pose_detector()
        predictor = ActivityPredictor(model, scaler, label_map, SMOOTHING_WINDOW)

        # Connect to camera
        cap = attempt_reconnect(ESP32_CAM_URL)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)

        fps, frame_count, start_time = 0, 0, time.time()

        print("✓ Starting inference loop...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Connection lost. Attempting to reconnect...")
                cap.release()
                cap = attempt_reconnect(ESP32_CAM_URL)
                continue

            # Store original frame for email attachment
            original_frame = frame.copy()

            # Process frame for pose detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            # Extract keypoints
            keypoints = extract_keypoints(results.pose_landmarks)

            # Predict activity
            if keypoints is not None:
                label, confidence, smoothed_confidence = predictor.predict(keypoints)
                person_detected = True
            else:
                label, confidence, smoothed_confidence = "No person detected", 0.0, 0.0
                person_detected = False

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)

            # Draw results on frame
            annotated_frame = draw_results(
                frame,
                results.pose_landmarks,
                label,
                smoothed_confidence,
                fps,
                mp_pose,
                mp_drawing,
                mp_drawing_styles,
            )

            # Check for suspicious activity and send email alert
            if (
                person_detected
                and label == "suspicious"
                and smoothed_confidence > CONFIDENCE_THRESHOLD
            ):
                if predictor.should_send_alert():
                    # Send email with non-annotated image
                    _, image_bytes = cv2.imencode(".jpg", original_frame)
                    email_thread = Thread(
                        target=send_email_alert,
                        args=(label, smoothed_confidence, image_bytes.tobytes()),
                        daemon=True,
                    )
                    email_thread.start()

            # Prepare JPG frame for dashboard
            _, jpg_frame = cv2.imencode(".jpg", annotated_frame)

            # Update global state
            with state_lock:
                latest_info.update(
                    label=label,
                    confidence=float(smoothed_confidence),
                    fps=float(fps),
                    alert=bool(
                        person_detected
                        and label == "suspicious"
                        and smoothed_confidence > CONFIDENCE_THRESHOLD
                    ),
                    frame=jpg_frame.tobytes(),
                    original_frame=original_frame,
                )

            # Small delay for CPU stability
            time.sleep(0.01)

    except Exception as e:
        print(f"ERROR - Fatal error in inference loop: {e}")
        import traceback

        traceback.print_exc()


# Flask App
flask_app = Flask(__name__)


@flask_app.route("/api/status")
def status():
    """Return current detection status"""
    with state_lock:
        data = latest_info.copy()

    # Remove frame data from JSON
    data.pop("frame", None)
    data.pop("original_frame", None)

    # Ensure all values are JSON serializable
    data["label"] = str(data["label"])
    data["confidence"] = float(data["confidence"])
    data["fps"] = float(data["fps"])
    data["alert"] = bool(data["alert"])

    return jsonify(data)


@flask_app.route("/api/frame")
def frame():
    """Return current video frame"""
    with state_lock:
        frame_bytes = latest_info.get("frame", None)

    if frame_bytes is None:
        return Response(status=204)

    return Response(frame_bytes, mimetype="image/jpeg")


if __name__ == "__main__":
    # Start inference thread
    infer_thread = Thread(target=inference_loop, daemon=True)
    infer_thread.start()

    # Start Flask server
    print("Starting Flask server on 0.0.0.0:5000...")
    flask_app.run("0.0.0.0", port=5000, debug=False)
