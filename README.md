# 🫁 Synthesizing Chest X-Ray Images with Conditional AC-GAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

A deep learning project implementing an **Auxiliary Classifier Generative Adversarial Network (AC-GAN)** in **TensorFlow/Keras** to synthesize high-fidelity, conditioned $(64 \times 64 \times 3)$ Chest X-Ray images for **Normal** and **Pneumonia** pathology.

---

## 📌 Highlights & Key Contributions

- **Class-Conditioned Medical Image Generation**: Uses one-hot class vectors concatenated with 100-dim latent noise to generate targeted diagnostic states (`NORMAL` vs `PNEUMONIA`).
- **Stabilized GAN Training Dynamics**:
  - Employs **Least-Squares (MSE)** adversarial loss for gradient continuity and stability.
  - Features **Layer Normalization** in the generator and **LeakyReLU ($\alpha=0.2$)** + **$L_2$ Regularization** in the discriminator to prevent mode collapse.
  - Uses $5 \times 5$ kernel convolutions to capture broader anatomical thoracic spatial correlations.
- **Downstream Utility Evaluation (Synthetic-to-Real Transfer)**:
  - 30,000 synthetic images generated from the AC-GAN.
  - A **VGG16 deep classification network** was trained **solely on synthetic images** and tested on **unseen real clinical scans**.
  - Attained **93.90% Accuracy**, **95.76% F1-Score**, and **99.12% Recall** on real clinical scans, verifying the clinical feature validity of the synthetic distributions.

---

## 🏗️ AC-GAN Model Architecture

```
                       ┌───────────────────────────────┐
  Latent Vector (z)───►│ Concatenate                   │
   Shape: (100,)       │ ──────────► Dense(1024)       │
                       │ ──────────► Dense(8x8x256)    │
  Class Label (c) ────►│ ──────────► Reshape(8, 8, 256)│
   Shape: (2,)         │ ──────────► Conv2DTranspose   │
  (One-Hot)            │             (LayerNorm, ReLU) │
                       │ ──────────► Conv2DTranspose   │───► Generated Image (x_fake)
                       │             (Sigmoid Output)  │     Shape: (64, 64, 3)
                       └───────────────────────────────┘
                                   GENERATOR
                                       │
                                       ▼
                       ┌───────────────────────────────┐
  Input Image ────────►│ Conv2D (32, 64, 128, 256)     ├─► Real/Fake Score (MSE Loss)
  (Real / Fake)        │ LeakyReLU(0.2) + L2 Reg       │
  Shape: (64, 64, 3)   │ Flatten                       ├─► Class Logits: Softmax
                       │ Dense(256) + Dropout(0.3)     │   [Normal, Pneumonia]
                       └───────────────────────────────┘   (Categorical Crossentropy)
                                 DISCRIMINATOR
```

---

## 📊 Downstream Evaluation & Clinical Metrics

To empirically validate whether the generator synthesized true clinical features rather than memorized artifacts, a **VGG16 classifier** was trained on 30,000 purely synthetic images and tested on the original real clinical dataset:

| Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Accuracy** | **93.90%** | Robust generalization across patient distributions |
| **F1-Score** | **95.76%** | High balance between precision and sensitivity |
| **Recall (Sensitivity)** | **99.12%** | Near-zero false negatives (crucial for medical screening) |
| **Precision** | **92.62%** | High specificity minimizing false alarms |

---

## 📁 Repository Structure

```
Using-GAN-to-Generate-Chest-X-Ray-Images/
│
├── src/                                  # Modular Python source code
│   ├── __init__.py
│   ├── dataset.py                        # Dataset loading, resizing & normalization
│   ├── models.py                         # AC-GAN Generator & Discriminator network graphs
│   ├── train.py                          # Training loop with batch logging & checkpointing
│   ├── generate.py                       # Standalone inference & synthetic sampling
│   └── evaluate.py                       # VGG16 synthetic-to-real transfer evaluation
│
├── notebooks/
│   └── chest_xray_acgan_training.ipynb   # Interactive Jupyter / Colab training notebook
│
├── .gitignore                            # Standard git exclusions
├── LICENSE                               # MIT License
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Siddartharekanti/Using-GAN-to-Generate-Chest-X-Ray-Images.git
cd Using-GAN-to-Generate-Chest-X-Ray-Images
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset
The dataset can be obtained from [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):
```bash
python -c "import opendatasets as op; op.download('https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia')"
```

---

## 💻 Usage

### Train the AC-GAN
```bash
python -m src.train \
    --data_dir "chest-xray-pneumonia/chest_xray/train" \
    --epochs 32000 \
    --batch_size 32 \
    --lr 0.0001 \
    --output_dir "checkpoints"
```

### Generate Synthetic X-Ray Samples
```bash
# Generate 16 sample images (both classes)
python -m src.generate \
    --model_path "checkpoints/generator.h5" \
    --num_samples 16 \
    --class_label "both" \
    --output_image "synthetic_xrays.png"
```

### Run Synthetic-to-Real Evaluation
```bash
python -m src.evaluate \
    --generator_path "checkpoints/generator.h5" \
    --data_dir "chest-xray-pneumonia/chest_xray/train" \
    --num_synthetic 30000 \
    --epochs 60 \
    --output_dir "evaluation_results"
```

---

## 📜 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

## 👤 Author & Acknowledgments

- **Siddartha Arekanti** ([@Siddartharekanti](https://github.com/Siddartharekanti))
- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney.
