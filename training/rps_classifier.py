import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = os.path.join(BASE_DIR, '../features')
MODEL_DIR = os.path.join(BASE_DIR, '../models')
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load Preprocessed Data ---
try:
    X_train = np.load(os.path.join(PREPROCESS_DIR, 'X_features_train.npy'))
    y_train = np.load(os.path.join(PREPROCESS_DIR, 'y_labels_train.npy'))
    X_val = np.load(os.path.join(PREPROCESS_DIR, 'X_features_validation.npy'))
    y_val = np.load(os.path.join(PREPROCESS_DIR, 'y_labels_validation.npy'))
    X_test = np.load(os.path.join(PREPROCESS_DIR, 'X_features_test.npy'))
    y_test = np.load(os.path.join(PREPROCESS_DIR, 'y_labels_test.npy'))
except FileNotFoundError as e:
    print(f"Error loading data file: {e}")
    exit()

# --- Data Sanity Checks ---
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Label distribution: \n{pd.Series(y_train).value_counts().sort_index()}")

# --- Prepare Data ---
num_classes = len(np.unique(y_train))
y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# --- Build MLP Model ---
model = Sequential([
    Dense(256, input_shape=(X_train.shape[1],)),
    LeakyReLU(alpha=0.1),
    Dropout(0.4),
    Dense(128),
    LeakyReLU(alpha=0.1),
    Dropout(0.3),
    Dense(64),
    LeakyReLU(alpha=0.1),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])

# --- Compile ---
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# --- Train ---
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train, y_train_cat,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val_cat),
    callbacks=[early_stopping, reduce_lr],
    verbose=2
)

# --- Evaluate ---
test_loss, test_acc = model.evaluate(X_test, y_test_cat, verbose=2)
print(f"Test Accuracy: {test_acc*100:.2f}%")

# --- Save Model ---
h5_path = os.path.join(MODEL_DIR, 'hand_gesture_model.h5')
model.save(h5_path)
print(f"H5 model saved to {h5_path}")

savedmodel_path = os.path.join(MODEL_DIR, 'hand_gesture_model_savedmodel')
tf.saved_model.save(model, savedmodel_path)
print(f"SavedModel saved to {savedmodel_path}")
