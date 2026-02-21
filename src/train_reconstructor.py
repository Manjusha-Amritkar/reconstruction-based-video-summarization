"""
Training script for BiLSTM Reconstructor
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
    RECON_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RECON_MODEL_PATH,
)

from reconstructor import build_reconstructor


# =============================
# Utility: Load Features
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
# Utility: Create Sequences
# =============================

def create_sequences(features, seq_length):
    total_frames = (features.shape[0] // seq_length) * seq_length
    features = features[:total_frames]
    return features.reshape(-1, seq_length, features.shape[-1])


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

    print("Train shape:", train_features.shape)
    print("Valid shape:", valid_features.shape)

    model = build_reconstructor(
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="mse",
    )

    print("🚀 Training Reconstructor...")
    model.fit(
        train_features,
        train_features,
        validation_data=(valid_features, valid_features),
        epochs=RECON_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    os.makedirs(os.path.dirname(RECON_MODEL_PATH), exist_ok=True)
    model.save(RECON_MODEL_PATH)

    print(f"✅ Reconstructor saved at {RECON_MODEL_PATH}")


if __name__ == "__main__":
    train()