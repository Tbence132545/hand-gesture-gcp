import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
from scipy.spatial.distance import pdist
import pyvirtualcam
import os

#Paths to artifacts 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '../models')

MODEL_PATH = os.path.join(MODEL_DIR, 'hand_gesture_model_final.h5')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')


#  Feature Extraction Functions 
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_JOINTS = {
    'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]
}

def calculate_vector_angle(v1, v2):
    v1_u = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
    v2_u = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product) * (180/np.pi)

def get_all_features_from_landmarks(landmarks_np):
    wrist = landmarks_np[0]
    relative_landmarks = landmarks_np - wrist
    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    normalized_coords = (relative_landmarks / max_dist).flatten() if max_dist > 0 else relative_landmarks.flatten()
    fingertips = landmarks_np[FINGERTIP_INDICES]
    fingertip_distances = pdist(fingertips)
    v1 = landmarks_np[5] - landmarks_np[0]
    v2 = landmarks_np[17] - landmarks_np[0]
    hand_normal = np.cross(v1, v2)
    hand_normal /= np.linalg.norm(hand_normal) if np.linalg.norm(hand_normal) > 0 else 1.0
    finger_curl_angles = []
    for finger, joints in FINGER_JOINTS.items():
        p0, p1, p2 = landmarks_np[joints[0]], landmarks_np[joints[1]], landmarks_np[joints[2]]
        v_segment1, v_segment2 = p1 - p0, p2 - p1
        angle = calculate_vector_angle(v_segment1, v_segment2)
        finger_curl_angles.append(angle)
    return np.concatenate([
        normalized_coords, fingertip_distances, hand_normal, np.array(finger_curl_angles)
    ])

# Load Saved Artifacts
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Model, scaler, and label encoder loaded successfully.")
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

label_map = {i: label for i, label in enumerate(label_encoder.classes_)}

#  MediaPipe & Video Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

#  Variables for Heuristic Prediction 
last_known_features = None
change_threshold = 2.5
current_prediction = "Show Hand"

print("\nStarting heuristic prediction... Press ESC to quit.")

# Initialize Virtual Camera 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with pyvirtualcam.Camera(width=frame_width, height=frame_height, fps=30) as cam:
    print(f"Virtual camera running at {frame_width}x{frame_height} @30fps")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        display_frame = frame.copy()

        if results.multi_hand_landmarks:
            landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
            current_features = get_all_features_from_landmarks(landmarks_np)

            if last_known_features is None:
                features_scaled = scaler.transform(current_features.reshape(1, -1))
                prediction = model.predict(features_scaled, verbose=0)
                pred_index = np.argmax(prediction)
                confidence = np.max(prediction)
                current_prediction = f"{label_map.get(pred_index, '...')} ({confidence*100:.1f}%)"
                last_known_features = current_features
            else:
                distance = np.linalg.norm(current_features - last_known_features)
                if distance > change_threshold:
                    features_scaled = scaler.transform(current_features.reshape(1, -1))
                    prediction = model.predict(features_scaled, verbose=0)
                    pred_index = np.argmax(prediction)
                    confidence = np.max(prediction)
                    current_prediction = f"{label_map.get(pred_index, '...')} ({confidence*100:.1f}%)"
                    last_known_features = current_features
        else:
            last_known_features = None
            current_prediction = "Show Hand"

        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

        cv2.putText(display_frame, current_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        mirrored_frame = cv2.flip(display_frame, 1)

        # Send to virtual cam 
        cam.send(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        cam.sleep_until_next_frame()

        #  local preview
        cv2.imshow('Local Preview', display_frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
