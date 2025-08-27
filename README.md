# Hand Gesture Recognition (Rock-Paper-Scissors)

This repository contains a complete pipeline for real-time hand gesture recognition using webcam input, dataset preprocessing, a pre-trained MLP model, and Cloud Run + Vertex AI for cloud inference.

This project is mostly for fun and experimentation. By leveraging Medipipe for hand landmark detection and training an MLP on these features, it vastly outperformed any CNN I could have written (mainly due to a lack of diversity in the datasets online).  
To make the project a bit more “real-world,” the webcam client can send extracted features to a Cloud Run backend which forwards them to a Vertex AI endpoint for inference. This setup demonstrates cloud deployment and secure service-to-service communication without relying on local computation for predictions. It’s mainly for experimentation and to show how the pipeline could scale, rather than because the model actually needs the cloud to perform well.

<p align="center">
  <img src="https://github.com/user-attachments/assets/eccc9103-6791-4eef-9869-fbcd572155ce" alt="rps_demo_gif">
</p>

--------------------------------------------------------------------------------
## Table of Contents

1. [Project Overview](#project-overview)
2. [Model Explanation](#model-explanation)
3. [Data Preparation & Feature Extraction](#data-preparation--feature-extraction)
4. [Training the Model](#training-the-model)
5. [Saved Artifacts](#saved-artifacts)
6. [Cloud Infrastructure](#cloud-infrastructure)
7. [Terraform Folder & Files](#terraform-folder--files)
8. [Uploading Artifacts](#uploading-artifacts)
9. [Setting up Secret Manager](#setting-up-secret-manager)
10. [Running the Local Client](#running-the-local-client)
11. [Running the Cloud Client](#running-the-cloud-client)
12. [Requirements](#requirements)
13. [Notes](#notes)


--------------------------------------------------------------------------------
<a name="project-overview"></a>
## Project Overview

- Backend: Python FastAPI app (backend/main.py) deployed to Cloud Run. Sends features to a Vertex AI endpoint for predictions.
- Client:
  - video.py → sends webcam features to Cloud Run for inference.
  - localVideo.py → runs inference locally, using .h5 model.
- Model: Multi-layer perceptron (MLP) trained on hand landmark features extracted via MediaPipe.

--------------------------------------------------------------------------------
<a name="model-explanation"></a>
## Model Explanation

- Input: Features extracted from hand landmarks:
  1. Normalized coordinates relative to the wrist
  2. Distances between fingertips
  3. Hand orientation (normal vector)
  4. Finger curl angles
- Architecture: MLP with Dense → LeakyReLU → Dropout layers:

| Layer | Units | Activation / Notes            |
|-------|-------|-------------------------------|
| Dense | 256   | LeakyReLU + Dropout 0.4       |
| Dense | 128   | LeakyReLU + Dropout 0.3       |
| Dense | 64    | LeakyReLU + Dropout 0.2       |
| Dense | 3     | Softmax (rock/paper/scissors) |


- Training: categorical_crossentropy loss, Adam optimizer, early stopping, and learning rate reduction on plateau.
- Output: Probability for each class; predicted label = argmax.
<img width="600" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/5fb95099-5fbd-4db1-9fce-ed96585c2f80" />


--------------------------------------------------------------------------------
<a name="data-preparation--feature-extraction"></a>
## Data Preparation & Feature Extraction

1. Dataset folder structure:

Rock-paper-Scissors/  
rock/  
paper/  
scissors/

2. Preprocessing script: training/preprocess.py

- Reads images and extracts features via MediaPipe.
- Computes:
  - Normalized coordinates
  - Fingertip distances
  - Hand orientation
  - Finger curl angles
- Saves:
  - .npy features & labels
  - scaler.joblib (for standardization)
  - label_encoder.joblib (for encoding classes)

--------------------------------------------------------------------------------
<a name="training-the-model"></a>
## Training the Model

1. Script: training/rps_classifier.py
2. Loads preprocessed .npy files from features/
3. Builds and trains the MLP model
4. Saves:
  - hand_gesture_model.h5 → local testing
  - hand_gesture_model_savedmodel/ → upload to Vertex AI endpoint

--------------------------------------------------------------------------------
<a name="saved-artifacts"></a>
## Saved Artifacts

- models/hand_gesture_model_final.h5 → local testing
- models/hand_gesture_model_savedmodel/ → upload to Vertex AI endpoint
- models/scaler.joblib → used by backend to normalize features
- models/label_encoder.joblib → used to decode predictions

--------------------------------------------------------------------------------
<a name="cloud-infrastructure"></a>
## Cloud Infrastructure

Terraform in infra/ automates:

- Enabling required APIs (Cloud Run, Storage, IAM, Vertex AI)
- Creating GCS bucket for artifacts
- Creating Cloud Run service + service account
- Setting IAM roles:
  - Vertex AI endpoint invocation
  - Read artifacts from GCS
- Deploying backend container to Cloud Run

Important: Vertex AI model deployment is manual. User must upload SavedModel to Vertex AI and create an endpoint.  
From /infra you can run 
```
terraform init
terraform apply
```

--------------------------------------------------------------------------------
<a name="terraform-folder--files"></a>
## Terraform Folder & Files

- main.tf → GCP resources & Cloud Run service
- variables.tf → input variables
- terraform.tfvars → user-defined values (project ID, container URI, endpoint ID, email)

--------------------------------------------------------------------------------
<a name="uploading-artifacts"></a>
## Uploading Artifacts

1. Terraform creates a GCS bucket:
   ```
   <PROJECT_ID>-gesture-recognition-artifacts
   ```
3. Upload required artifacts using gsutil:
   ```
   gsutil cp models/scaler.joblib gs://<ARTIFACTS_BUCKET_NAME>/  
   gsutil cp models/label_encoder.joblib gs://<ARTIFACTS_BUCKET_NAME>/
   ```
5. Cloud Run service account has read access (roles/storage.objectViewer)

--------------------------------------------------------------------------------
<a name="setting-up-secret-manager"></a>
## Setting up Secret Manager

1. Create a secret in Secret Manager containing the service account JSON key used by Cloud Run.
2. Set environment variable:
   ```
   export GCP_CREDENTIALS_SECRET="projects/<PROJECT_ID>/secrets/<SECRET_NAME>"
   ```
Required for client/video.py authentication to Cloud Run.

--------------------------------------------------------------------------------
<a name="running-the-local-client"></a>
## Running the Local Client
```
python client/localVideo.py
```
- Uses local .h5 model and artifacts.
- Displays predictions directly from your machine.

--------------------------------------------------------------------------------
<a name="model-explanation"></a>
## Running the Cloud Client
```
export CLOUD_RUN_URL="https://<YOUR_CLOUD_RUN_URL>"  
export GCP_CREDENTIALS_SECRET="projects/<PROJECT_ID>/secrets/<SECRET_NAME>"
```
```
python client/video.py
```
- Authenticates via Secret Manager
- Captures webcam frames, extracts features
- Sends features to Cloud Run → Vertex AI endpoint
- Displays predicted label with confidence

--------------------------------------------------------------------------------
<a name="requirements"></a>
## Requirements
```
numpy  
pandas
tensorflow==2.15.0  
opencv-python
mediapipe  
scikit-learn
joblib  
scipy
requests  
pyvirtualcam
google-cloud-secret-manager  
google-auth
```
--------------------------------------------------------------------------------
<a name="notes"></a>
## Notes

- .tfvars and Terraform state files contain sensitive info → do not commit
- Secret Manager is mandatory for secure access to Cloud Run from the client
- Ensure correct IAM roles for Cloud Run service account
