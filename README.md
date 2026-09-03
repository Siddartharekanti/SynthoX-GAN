# 🫁 SynthoX-GAN: Conditional Chest Radiograph Synthesis & Clinical Biomarker Validation

[![GitHub Repository](https://img.shields.io/badge/GitHub-SynthoX--GAN-181717?style=flat&logo=github)](https://github.com/Siddartharekanti/SynthoX-GAN)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)]()

> **SynthoX-GAN** is an advanced medical deep learning framework implementing an **Auxiliary Classifier Generative Adversarial Network (AC-GAN)** in **TensorFlow/Keras** to synthesize high-resolution, pathology-conditioned $(64 \times 64 \times 3)$ Chest X-Ray images across **Normal** (healthy lungs) and **Pneumonia** diagnostic classes.

---

## 🔬 Scientific Motivation & Core Contributions

Medical generative modeling often struggles with mode collapse, vanishing gradients, and the synthesis of clinically irrelevant visual artifacts. **SynthoX-GAN** tackles these challenges through an integrated architectural design and a rigorous downstream clinical transfer protocol:

1. **Pathology-Conditioned Synthesis**: Generates targeted radiological states (`NORMAL` vs. `PNEUMONIA`) by injecting one-hot categorical conditioning vectors into the generator latent space and utilizing auxiliary multi-task classification heads in the discriminator.
2. **Stable Adversarial Optimization**:
   - **Least-Squares GAN (LSGAN) Objective**: Uses Mean Squared Error (MSE) loss for discriminator score continuity and smoother gradient flow.
   - **Layer Normalization & Regularization**: Implements LayerNorm in generator transposed convolutions alongside $L_2$ weight decay ($\lambda = 0.001$) and LeakyReLU ($\alpha=0.2$) in the discriminator.
   - **$5 \times 5$ Thoracic Receptive Fields**: Uses larger convolution kernels to model complex anatomical rib, heart, and lung field dependencies.
3. **Synthetic-to-Real Clinical Transfer Evaluation**:
   - Generated **30,000 synthetic chest radiographs** from the trained generator.
   - Trained a deep **VGG16 classification network** exclusively on **synthetic images**.
   - Evaluated the classifier on **real, unseen clinical patient radiographs**.
   - Achieved **93.90% Accuracy**, **95.76% F1-Score**, and **99.12% Recall** on real scans—demonstrating that the synthetic distribution captured authentic pathological biomarkers.

---

## 🏗️ SynthoX-GAN Architecture

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

## 📊 Downstream Clinical Validation Results

| Diagnostic Metric | Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Accuracy** | **93.90%** | Exceptional generalization across clinical patient cohorts |
| **F1-Score** | **95.76%** | High harmonic balance between precision and sensitivity |
| **Recall (Sensitivity)** | **99.12%** | Critical reduction of false negatives in pneumonia screening |
| **Precision** | **92.62%** | High specificity minimizing false positives |

---

## 📁 Repository Structure

```
SynthoX-GAN/
│
├── src/                                  # Modular Python source code
│   ├── __init__.py                       # Package metadata & exports
│   ├── dataset.py                        # Dataset discovery, resizing, & normalization
│   ├── models.py                         # AC-GAN Generator & Discriminator network graphs
│   ├── train.py                          # Training loop with sample checkpointing
│   ├── generate.py                       # Inference CLI to sample synthetic X-rays
│   └── evaluate.py                       # VGG16 synthetic-to-real transfer evaluation
│
├── notebooks/
│   └── chest_xray_acgan_training.ipynb   # Interactive Jupyter / Google Colab notebook
│
├── .gitignore                            # Exclusions (datasets, weights, caches)
├── LICENSE                               # MIT Open-Source License
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation
```

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Siddartharekanti/SynthoX-GAN.git
cd SynthoX-GAN
```

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the Dataset
The dataset is available on Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia):
```bash
python -c "import opendatasets as op; op.download('https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia')"
```

---

## 💻 CLI Usage

### 🏋️ Train SynthoX-GAN
```bash
python -m src.train \
    --data_dir "chest-xray-pneumonia/chest_xray/train" \
    --epochs 32000 \
    --batch_size 32 \
    --lr 0.0001 \
    --output_dir "checkpoints"
```

### 🎨 Synthesize New Chest X-Rays
```bash
# Generate a grid of 16 synthetic chest radiographs (Normal & Pneumonia)
python -m src.generate \
    --model_path "checkpoints/generator.h5" \
    --num_samples 16 \
    --class_label "both" \
    --output_image "synthetic_xrays.png"
```

### 🧪 Run Synthetic-to-Real Evaluation Pipeline
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

This project is licensed under the **MIT License** - see the [`LICENSE`](LICENSE) file for details.

---

## 👨‍💻 Author & Contact

- **Siddartha Arekanti** - GitHub: [@Siddartharekanti](https://github.com/Siddartharekanti)
- Academic Affiliation: SRM University AP
