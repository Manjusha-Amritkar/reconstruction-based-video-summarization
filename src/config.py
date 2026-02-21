"""
Central configuration file for Reconstruction-Based Video Summarization
"""

import os

# =============================
# Dataset Paths
# =============================

# Base path where extracted CNN features (.npy) are stored
BASE_FEATURE_PATH = os.getenv("FEATURE_BASE_PATH", "data/features")

TRAIN_PATH = os.path.join(BASE_FEATURE_PATH, "train")
VALID_PATH = os.path.join(BASE_FEATURE_PATH, "valid")
TEST_PATH = os.path.join(BASE_FEATURE_PATH, "test")

# Ground-truth annotation directories
SUMME_MAT_PATH = os.getenv("SUMME_MAT_PATH", "data/annotations/summe")
TVSUM_MAT_PATH = os.getenv("TVSUM_MAT_PATH", "data/annotations/tvsum")

# =============================
# Model Parameters
# =============================

FEATURE_DIM = 2048
HIDDEN_DIM = 512
SEQ_LENGTH = 100

# =============================
# Training Parameters
# =============================

RECON_EPOCHS = 50
SELECTOR_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 1e-4

# =============================
# Evaluation Parameters
# =============================

FPS = 25
SUMMARY_RATIO = 0.15
THRESHOLD = 0.6

# =============================
# Output Directories
# =============================

MODEL_DIR = os.getenv("MODEL_DIR", "models")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")

RECON_MODEL_PATH = os.path.join(MODEL_DIR, "reconstructor.keras")
SELECTOR_MODEL_PATH = os.path.join(MODEL_DIR, "selector.keras")

SUMME_RESULTS_PATH = os.path.join(RESULTS_DIR, "summe_results.xlsx")
TVSUM_RESULTS_PATH = os.path.join(RESULTS_DIR, "tvsum_results.xlsx")