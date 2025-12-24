import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5)
cap = cv2.VideoCapture(0)  # Use 0 for webcam

data = []
frame_count = 0

print("Recording activity data...")
print("Press 'n' for NORMAL activity")
print("Press 's' for SUSPICIOUS activity")
print("Press 'q' to quit and save")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_pose.process(frame_rgb)

    # Draw pose on frame
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS
        )

        # Extract 34 features
        keypoints = []
        for i in range(17):
            landmark = results.pose_landmarks.landmark[i]
            keypoints.extend([landmark.x, landmark.y])

        # Get key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("n"):
            data.append(keypoints + ["normal"])
            print(f"✓ Recorded NORMAL activity ({len(data)} samples)")
        elif key == ord("s"):
            data.append(keypoints + ["suspicious"])
            print(f"✓ Recorded SUSPICIOUS activity ({len(data)} samples)")
        elif key == ord("q"):
            break

    cv2.imshow("Data Collection - Press n/s/q", frame)

cap.release()
cv2.destroyAllWindows()

if data:
    columns = [f"feature_{i}" for i in range(34)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("activity_data.csv", index=False)
    print(f"\n✓ Saved {len(data)} samples to activity_data.csv")
else:
    print("No data collected!")
