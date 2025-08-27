import cv2
import mediapipe as mp
import numpy as np
import requests
from scipy.spatial.distance import pdist
import os
import json
from google.cloud import secretmanager
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from google.oauth2.service_account import IDTokenCredentials

#  Configuration 
BASE_URL = os.environ.get("CLOUD_RUN_URL")
SECRET_NAME = os.environ.get("GCP_CREDENTIALS_SECRET")
if not BASE_URL:
    raise ValueError("CLOUD_RUN_URL environment variable not set.")
if not SECRET_NAME:
    raise ValueError("GCP_CREDENTIALS_SECRET environment variable not set.")

CLOUD_RUN_URL = f"{BASE_URL}/predict"

# Global variable to hold our authentication token 
AUTH_TOKEN = None

def authenticate_once():
    """Authenticates ONCE at the start using Secret Manager and a service account."""
    global AUTH_TOKEN
    try:
        print("--- Authenticating with Google Cloud via Secret Manager SA... ---")
        
        # Access Secret Manager and fetch the JSON key
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": SECRET_NAME + "/versions/latest"})
        sa_info = json.loads(response.payload.data.decode("UTF-8"))

        # Create IDTokenCredentials for the Cloud Run endpoint
        creds = IDTokenCredentials.from_service_account_info(sa_info, target_audience=CLOUD_RUN_URL)
        auth_req = Request()
        creds.refresh(auth_req)
        AUTH_TOKEN = creds.token
        print("--- Authentication Successful. Token acquired. ---")
    except Exception as e:
        print(f"FATAL: Could not authenticate with Google Cloud. Error: {e}")
        AUTH_TOKEN = None

def get_prediction_from_gateway(features):
    """Sends features to the Cloud Run gateway and returns the prediction."""
    if not AUTH_TOKEN:
        return "Authentication Failed"
    try:
        headers = {"Authorization": f"Bearer {AUTH_TOKEN}"}
        payload = {"features": features.tolist()}
        response = requests.post(CLOUD_RUN_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        label = data.get('predicted_label', 'Unknown')
        confidence = data.get('confidence', 0.0)
        return f"{label} ({confidence*100:.1f}%)"
    except Exception as e:
        print(f"API Error: {e}")
        return "API Call Failed"

# Authenticate once at startup 
authenticate_once()

# Feature Extraction 
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_JOINTS = {
    'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12], 'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
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
    return np.concatenate([normalized_coords, fingertip_distances, hand_normal, np.array(finger_curl_angles)])

#  MediaPipe & Video Setup 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
cap = cv2.VideoCapture(0)

# Variables for Heuristic Prediction
last_known_features = None
change_threshold = 10
current_prediction = "Show Hand"

print("\nStarting CLOUD prediction... Press ESC to quit.")

# Main Loop 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting.")
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    display_frame = frame.copy()

    if results.multi_hand_landmarks:
        landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])
        current_features = get_all_features_from_landmarks(landmarks_np)

        distance = np.linalg.norm(current_features - last_known_features) if last_known_features is not None else float('inf')

        if distance > change_threshold:
            current_prediction = get_prediction_from_gateway(current_features)
            last_known_features = current_features
    else:
        last_known_features = None
        current_prediction = "Show Hand"

    if results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(display_frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

    cv2.putText(display_frame, current_prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow('Local Preview', display_frame)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Cleanup 
cap.release()
cv2.destroyAllWindows()
