"""
Feature Extraction using ResNet50
Extracts frame-level CNN features and saves as .npy
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input


def load_resnet():
    """
    Load pretrained ResNet50 without classification head.
    """
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    return model


def extract_features_from_folder(model, frame_folder):
    """
    Extract features from all frames in a folder.
    """
    features = []
    frame_list = sorted(os.listdir(frame_folder))

    for frame in frame_list:
        img_path = os.path.join(frame_folder, frame)

        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        feature_vector = model.predict(img_array, verbose=0)
        features.append(feature_vector.flatten())

    return np.array(features)


def save_features(frame_folder, output_path):
    """
    Extract and save features.
    """
    model = load_resnet()
    features = extract_features_from_folder(model, frame_folder)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, features)

    print(f"✅ Features saved at {output_path}")