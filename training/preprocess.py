import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from scipy.spatial.distance import pdist # <-- THIS IS THE FIX

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# --- Configuration ---
DATASET_DIR = './Rock-paper-Scissors'
CLASSES = ['rock', 'paper', 'scissors']
FINGERTIP_INDICES = [4, 8, 12, 16, 20]
FINGER_JOINTS = {
    'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8], 'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]
}

def calculate_vector_angle(v1, v2):
    """Calculates the angle in degrees between two vectors."""
    v1_u = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
    v2_u = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
    # Clip the dot product to handle floating point inaccuracies
    dot_product = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return np.arccos(dot_product) * (180/np.pi)

def extract_all_features(image_path):
    """Extracts the complete, robust feature set from a single image."""
    image = cv2.imread(image_path)
    if image is None: return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks: return None

    landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark])

    # 1. Normalized Coordinates
    wrist = landmarks_np[0]
    relative_landmarks = landmarks_np - wrist
    max_dist = np.max(np.linalg.norm(relative_landmarks, axis=1))
    normalized_coords = (relative_landmarks / max_dist).flatten() if max_dist > 0 else relative_landmarks.flatten()

    # 2. Fingertip Distances
    fingertip_distances = pdist(landmarks_np[FINGERTIP_INDICES])

    # 3. Hand Orientation (Normal Vector)
    v1 = landmarks_np[5] - landmarks_np[0]
    v2 = landmarks_np[17] - landmarks_np[0]
    hand_normal = np.cross(v1, v2)
    hand_normal /= np.linalg.norm(hand_normal) if np.linalg.norm(hand_normal) > 0 else 1.0

    # 4. Finger Curl Angles
    finger_curl_angles = []
    for finger, joints in FINGER_JOINTS.items():
        p0, p1, p2 = landmarks_np[joints[0]], landmarks_np[joints[1]], landmarks_np[joints[2]]
        v_segment1, v_segment2 = p1 - p0, p2 - p1
        angle = calculate_vector_angle(v_segment1, v_segment2)
        finger_curl_angles.append(angle)

    # Combine all features into a single vector
    return np.concatenate([
        normalized_coords, fingertip_distances, hand_normal, np.array(finger_curl_angles)
    ])

# --- Main Execution ---
print("Starting dataset creation from simple folder structure...")

all_features = []
all_labels = []

# Loop through each class folder ('rock', 'paper', 'scissors')
for label in CLASSES:
    class_folder = os.path.join(DATASET_DIR, label)
    if not os.path.exists(class_folder):
        print(f"Warning: Folder '{class_folder}' not found. Skipping.")
        continue

    image_files = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Processing {len(image_files)} images from '{label}' folder...")

    for img_file in image_files:
        img_path = os.path.join(class_folder, img_file)
        features = extract_all_features(img_path)
        if features is not None:
            all_features.append(features)
            all_labels.append(label)

if not all_features:
    print("\nFATAL ERROR: No images were processed. Check your DATASET_DIR and folder structure.")
    exit()

print(f"\nSuccessfully processed a total of {len(all_features)} images.")

# --- Label Encoding ---
le = LabelEncoder()
y_encoded = le.fit_transform(all_labels)
joblib.dump(le, 'label_encoder.joblib')
print(f"Label mapping saved. Classes: {list(le.classes_)}")

X = np.array(all_features)
y = y_encoded

# --- Create Train, Validation, and Test Splits ---
# First, split into training+validation (80%) and a final test set (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Then, split the training+validation set into final training and validation sets
# test_size=0.25 here because 0.25 * 0.8 = 0.2, giving a 60/20/20 split of the original data
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print("\nDataset split successfully:")
print(f"Training samples:   {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples:       {len(X_test)}")

# --- Scaling ---
scaler = StandardScaler()
print("\nFitting scaler on training data...")
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.joblib')
print("Scaler saved to scaler.joblib")

# --- Save Final Artifacts ---
np.save('X_features_train.npy', X_train_scaled)
np.save('y_labels_train.npy', y_train)
np.save('X_features_validation.npy', X_val_scaled)
np.save('y_labels_validation.npy', y_val)
np.save('X_features_test.npy', X_test_scaled)
np.save('y_labels_test.npy', y_test)

print("\nPreprocessing finished. All files saved.")
