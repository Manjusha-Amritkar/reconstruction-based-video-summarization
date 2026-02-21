"""
Summary Generation Module
Generates frame-level summaries using trained selector model
"""

import numpy as np
import tensorflow as tf

from config import SELECTOR_MODEL_PATH, THRESHOLD


def load_selector():
    """
    Load trained selector model.
    """
    model = tf.keras.models.load_model(SELECTOR_MODEL_PATH)
    return model


def generate_summary(selector, features, threshold=THRESHOLD):
    """
    Generate summary indices based on importance threshold.

    Args:
        selector: trained selector model
        features: (num_frames, feature_dim)
        threshold: selection threshold

    Returns:
        selected_indices (np.ndarray)
        scores (np.ndarray)
    """

    features = np.expand_dims(features, axis=0)
    scores = selector.predict(features, verbose=0).flatten()

    selected_indices = np.where(scores > threshold)[0]

    return selected_indices, scores