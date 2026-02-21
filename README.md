# Reconstruction-Based Video Summarization

### BiLSTM Encoder–Decoder with Selector Network for Unsupervised Video Summarization

---

## 📌 Overview

This repository implements a **two-stage unsupervised video summarization framework** based on reconstruction-driven learning.

The method consists of:

1. A **BiLSTM encoder–decoder (Reconstructor)** trained to reconstruct frame-level CNN features.
2. A **BiLSTM Selector network** trained using pseudo-labels derived from reconstruction error.
3. Threshold-based summary generation.
4. Evaluation on **SumMe** and **TVSum** benchmarks using Precision, Recall, and F1-score.

Unlike supervised approaches, this framework does **not require human-annotated labels during training**. Frame importance is inferred through reconstruction difficulty.

---

## 🧠 Methodology

### 1️⃣ Feature Extraction

Video frames are processed using a pretrained **ResNet50** backbone:

* Input resolution: 224×224
* Pooling: Global Average Pooling
* Output feature dimension: 2048

Frame features are stored as `.npy` files and used as input sequences.

---

### 2️⃣ Stage I — Reconstruction Model

A 4-layer BiLSTM encoder–decoder architecture is used:

**Encoder:**

* BiLSTM (hidden_dim)
* BiLSTM (hidden_dim / 2)

**Decoder:**

* BiLSTM (hidden_dim / 2)
* BiLSTM (hidden_dim)

The model is trained using Mean Squared Error (MSE):

```
L_recon = || X - X_hat ||^2
```

Where:

* `X` = original feature sequence
* `X_hat` = reconstructed features

Frames that are difficult to reconstruct tend to contain higher semantic variation.

---

### 3️⃣ Pseudo Label Generation

Frame-level reconstruction error is computed:

```
e_i = mean((x_i - x_hat_i)^2)
```

The errors are:

* Normalized to [0,1]
* Slightly smoothed to avoid extreme targets
* Used as pseudo-importance labels

---

### 4️⃣ Stage II — Selector Network

A BiLSTM-based selector predicts frame-level importance:

* BiLSTM (hidden_dim)
* Dense layer with sigmoid activation

Training loss:

```
L_selector = BinaryCrossEntropy(predicted_scores, pseudo_labels)
```

Where:

* Predicted scores are selector outputs
* Pseudo labels are normalized reconstruction errors

---

### 5️⃣ Summary Generation

At inference time:

* Importance scores are computed
* Frames with score > threshold τ are selected

Default threshold:

```
τ = 0.6
```

This produces a binary summary without requiring knapsack optimization.

---

## 📊 Datasets

### 🔹 SumMe

* 25 consumer videos
* Multiple human-annotated summaries per video
* Evaluation via Precision, Recall, F1-score

### 🔹 TVSum

* 50 videos across 10 categories
* 20 user annotations per video
* Evaluation via average F1-score over users

⚠ Datasets are **not included** due to licensing restrictions.

---

## 🏗 Repository Structure

```
reconstruction-based-video-summarization/
│
├── src/
│   ├── config.py
│   ├── feature_extraction.py
│   ├── reconstructor.py
│   ├── selector.py
│   ├── train_reconstructor.py
│   ├── train_selector.py
│   ├── summary_generator.py
│   ├── evaluate_summe.py
│   ├── evaluate_tvsum.py
│   └── utils.py
│
├── data/
├── models/
├── results/
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1 — Train Reconstructor

```
python src/train_reconstructor.py
```

---

### Step 2 — Train Selector

```
python src/train_selector.py
```

---

### Step 3 — Evaluate on SumMe

```
python src/evaluate_summe.py
```

---

### Step 4 — Evaluate on TVSum

```
python src/evaluate_tvsum.py
```

---

## 📈 Output

* Trained models saved in `models/`
* Evaluation results saved in `results/`
* Excel summaries generated for both datasets

---

## 🧩 Configuration

All parameters are centralized in:

```
src/config.py
```

Includes:

* Feature paths
* Annotation paths
* Model hyperparameters
* Training parameters
* Threshold
* Output paths

This design ensures portability and reproducibility.

---

## 🔬 Key Characteristics

✔ Fully unsupervised
✔ Two-stage training
✔ Reconstruction-driven importance learning
✔ No manual labeling required
✔ Modular, reproducible architecture
✔ Supports both SumMe and TVSum

---

## 📄 Citation

If you use this implementation, please cite:

```
Reconstruction-Based BiLSTM Selector for Unsupervised Video Summarization
```

(Replace with your formal citation once finalized.)

---

## 📜 License

This project is released under the MIT License.
>>>>>>> b82d44e (Add academic README with methodology, training, and evaluation details)
