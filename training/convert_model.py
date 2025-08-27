import tensorflow as tf
#I was using tensorflow 2.20 to train the model, but Vertex AI only accepts 2.15 and below. So with venv we can downgrade the version with this script
#to be used with vertex ai
# Load your existing H5 model
model = tf.keras.models.load_model("C:/Users/User/Desktop/hand-gesture-gcp/models/hand_gesture_model_final.h5")

# Save as TF SavedModel (compatible with Vertex AI TF2.15)
model.save("hand_gesture_model_tf2_15", save_format="tf")

print("Model successfully converted to TF2.15 SavedModel!")