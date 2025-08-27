import os
import io
import joblib
import numpy as np
import logging # New import
from flask import Flask, request, jsonify
from google.cloud import aiplatform, storage

# --- Configuration ---
PROJECT_ID = os.environ.get('GCP_PROJECT')
REGION = os.environ.get('GCP_REGION')
ENDPOINT_ID = os.environ.get('VERTEX_ENDPOINT_ID')
BUCKET_NAME = os.environ.get('ARTIFACTS_BUCKET')
print(PROJECT_ID)
print(REGION)
print(ENDPOINT_ID)
print(BUCKET_NAME)
# --- Initialize Flask App & Configure Logging ---
app = Flask(__name__)
# This makes Flask's logs show up in Cloud Run's logging system
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# --- Global variables to hold the loaded models ---
scaler = None
label_encoder = None

def load_artifacts():
    """Loads the scaler and label encoder from GCS into memory."""
    global scaler, label_encoder
    if scaler and label_encoder:
        return

    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        app.logger.info(f"Accessing GCS bucket: {BUCKET_NAME}")

        scaler_blob = bucket.blob('artifacts/scaler.joblib')
        scaler_bytes = scaler_blob.download_as_bytes()
        scaler = joblib.load(io.BytesIO(scaler_bytes))
        app.logger.info("Scaler loaded successfully.")

        le_blob = bucket.blob('artifacts/label_encoder.joblib')
        le_bytes = le_blob.download_as_bytes()
        label_encoder = joblib.load(io.BytesIO(le_bytes))
        app.logger.info("Label Encoder loaded successfully.")

    except Exception as e:
        app.logger.exception(f"FATAL: Could not load artifacts from GCS. Error: {e}")

# Load artifacts when the application instance starts
load_artifacts()

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info("Prediction request received.")

    if not scaler or not label_encoder:
        app.logger.error("Server is not configured correctly; models not loaded.")
        return jsonify({"error": "Server is not configured correctly; models not loaded."}), 500

    data = request.get_json()
    if not data or 'features' not in data:
        app.logger.error("Invalid input. 'features' key is required.")
        return jsonify({"error": "Invalid input. 'features' key is required."}), 400

    try:
        features_list = data['features']
        app.logger.info(f"Received {len(features_list)} features.")
        
        features_np = np.array(features_list).reshape(1, -1)
        app.logger.info("Features successfully converted to numpy array.")
        
        features_scaled = scaler.transform(features_np)
        app.logger.info("Features successfully scaled.")
        
        instances = features_scaled.tolist()

        endpoint = aiplatform.Endpoint(endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID}")
        app.logger.info("Calling Vertex AI endpoint...")
        
        prediction_response = endpoint.predict(instances=instances)
        app.logger.info("Received prediction from Vertex AI.")
        
        predictions = prediction_response.predictions[0]
        pred_index = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = label_encoder.classes_[pred_index]

        app.logger.info(f"Prediction successful: {predicted_label}")
        return jsonify({
            "predicted_label": predicted_label,
            "confidence": float(confidence)
        })

    except Exception as e:
        #log the message AND the full traceback.
        app.logger.exception(f"An exception occurred during prediction: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), debug=True)