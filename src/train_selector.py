"""
Training script for BiLSTM Selector Network
"""

import os
import numpy as np
import tensorflow as tf

from config import (
    TRAIN_PATH,
    VALID_PATH,
    FEATURE_DIM,
    HIDDEN_DIM,
    SEQ_LENGTH,
    SELECTOR_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RECON_MODEL_PATH,
    SELECTOR_MODEL_PATH,
)

from reconstructor import build_reconstructor
from selector import build_selector


# =============================
# Load Features
# =============================

def load_features(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".npy")])
    all_features = []

    for file in files:
        feature_path = os.path.join(path, file)
        features = np.load(feature_path)
        all_features.append(features)

    return np.concatenate(all_features, axis=0)


# =============================
# Create Sequences
# =============================

def create_sequences(features, seq_length):
    total_frames = (features.shape[0] // seq_length) * seq_length
    features = features[:total_frames]
    return features.reshape(-1, seq_length, features.shape[-1])


# =============================
# Compute Reconstruction Error
# =============================

def compute_reconstruction_error(model, features):
    reconstructed = model.predict(features, verbose=0)
    errors = np.mean(np.square(features - reconstructed), axis=-1, keepdims=True)
    return errors


# =============================
# Normalize Labels
# =============================

def normalize_scores(scores):
    min_val = np.min(scores)
    max_val = np.max(scores)
    normalized = (scores - min_val) / (max_val - min_val + 1e-8)
    return normalized


# =============================
# Training Pipeline
# =============================

def train():
    print("📥 Loading training features...")
    train_features = load_features(TRAIN_PATH)

    print("📥 Loading validation features...")
    valid_features = load_features(VALID_PATH)

    train_features = create_sequences(train_features, SEQ_LENGTH)
    valid_features = create_sequences(valid_features, SEQ_LENGTH)

    # Load trained reconstructor
    print("📦 Loading trained reconstructor...")
    reconstructor = tf.keras.models.load_model(RECON_MODEL_PATH)

    print("📊 Computing reconstruction errors...")
    train_errors = compute_reconstruction_error(reconstructor, train_features)
    valid_errors = compute_reconstruction_error(reconstructor, valid_features)

    train_labels = normalize_scores(train_errors)
    valid_labels = normalize_scores(valid_errors)

    # Optional smoothing
    smooth_factor = 0.05
    train_labels = train_labels * (1 - smooth_factor) + (smooth_factor / 2)
    valid_labels = valid_labels * (1 - smooth_factor) + (smooth_factor / 2)

    selector = build_selector(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM
    )

    selector.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
    )

    print("🚀 Training Selector...")
    selector.fit(
        train_features,
        train_labels,
        validation_data=(valid_features, valid_labels),
        epochs=SELECTOR_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    os.makedirs(os.path.dirname(SELECTOR_MODEL_PATH), exist_ok=True)
    selector.save(SELECTOR_MODEL_PATH)

    print(f"✅ Selector saved at {SELECTOR_MODEL_PATH}")


if __name__ == "__main__":
    train()