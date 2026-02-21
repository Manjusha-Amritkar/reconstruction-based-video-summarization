"""
Evaluation script for SumMe dataset
"""

import os
import numpy as np
import pandas as pd
import scipy.io

from config import (
    BASE_FEATURE_PATH,
    SUMME_MAT_PATH,
    SUMME_RESULTS_PATH,
)

from summary_generator import load_selector, generate_summary


# =============================
# Compute Metrics
# =============================

def compute_metrics(pred_indices, gt_binary):
    total_frames = len(gt_binary)

    pred_binary = np.zeros(total_frames)
    pred_binary[pred_indices] = 1

    tp = np.sum((pred_binary == 1) & (gt_binary == 1))
    fp = np.sum((pred_binary == 1) & (gt_binary == 0))
    fn = np.sum((pred_binary == 0) & (gt_binary == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


# =============================
# Main Evaluation
# =============================

def evaluate():
    selector = load_selector()

    feature_files = [
        f for f in os.listdir(BASE_FEATURE_PATH)
        if f.endswith(".npy")
    ]

    results = []

    for file in feature_files:
        video_name = os.path.splitext(file)[0]
        print(f"🎥 Evaluating: {video_name}")

        feature_path = os.path.join(BASE_FEATURE_PATH, file)
        mat_path = os.path.join(SUMME_MAT_PATH, f"{video_name}.mat")

        if not os.path.exists(mat_path):
            print("⚠ Ground truth missing. Skipping.")
            continue

        features = np.load(feature_path)
        selected_frames, _ = generate_summary(selector, features)

        mat_data = scipy.io.loadmat(mat_path)
        user_scores = mat_data["user_score"]
        avg_scores = np.mean(user_scores, axis=1)
        gt_binary = np.where(avg_scores > 0, 1, 0)

        precision, recall, f1 = compute_metrics(selected_frames, gt_binary)

        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        results.append({
            "Video": video_name,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(SUMME_RESULTS_PATH), exist_ok=True)
    df.to_excel(SUMME_RESULTS_PATH, index=False)

    print(f"\n✅ Results saved to {SUMME_RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()