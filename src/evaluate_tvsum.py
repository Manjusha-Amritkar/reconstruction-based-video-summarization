"""
Evaluation script for TVSum dataset
"""

import os
import numpy as np
import pandas as pd
import h5py

from config import (
    BASE_FEATURE_PATH,
    TVSUM_MAT_PATH,
    TVSUM_RESULTS_PATH,
    SUMMARY_RATIO,
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
# Decode MATLAB strings
# =============================

def decode_str(mat_file, ref):
    return "".join(map(chr, mat_file[ref][()].flatten())).replace("\x00", "")


# =============================
# Main Evaluation
# =============================

def evaluate():
    selector = load_selector()

    results = []

    with h5py.File(TVSUM_MAT_PATH, "r") as mat_data:
        tvsum_group = mat_data["tvsum50"]

        video_refs = tvsum_group["video"][:, 0]
        user_refs = tvsum_group["user_anno"][:, 0]
        nframes_refs = tvsum_group["nframes"][:, 0]

        for idx, video_ref in enumerate(video_refs):
            video_name = decode_str(mat_data, video_ref)
            print(f"🎥 Evaluating: {video_name}")

            feature_path = os.path.join(BASE_FEATURE_PATH, f"{video_name}.npy")

            if not os.path.exists(feature_path):
                print("⚠ Feature file missing. Skipping.")
                continue

            features = np.load(feature_path)
            selected_frames, _ = generate_summary(selector, features)

            user_scores = mat_data[user_refs[idx]][:]
            num_frames = int(mat_data[nframes_refs[idx]][()][0][0])

            # Convert each user annotation to binary using top 15%
            f1_scores = []

            for user in range(user_scores.shape[0]):
                scores = user_scores[user]
                k = int(np.floor(num_frames * SUMMARY_RATIO))
                top_k_indices = np.argsort(scores)[-k:]
                gt_binary = np.zeros(num_frames)
                gt_binary[top_k_indices] = 1

                precision, recall, f1 = compute_metrics(selected_frames, gt_binary)
                f1_scores.append(f1)

            avg_f1 = np.mean(f1_scores)

            print(f"Average F1: {avg_f1:.4f}")

            results.append({
                "Video": video_name,
                "Avg_F1": avg_f1,
            })

    df = pd.DataFrame(results)

    os.makedirs(os.path.dirname(TVSUM_RESULTS_PATH), exist_ok=True)
    df.to_excel(TVSUM_RESULTS_PATH, index=False)

    print(f"\n✅ TVSum results saved to {TVSUM_RESULTS_PATH}")


if __name__ == "__main__":
    evaluate()