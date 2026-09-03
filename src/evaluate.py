"""
Evaluation script: Trains a VGG16 classifier on synthetic X-rays and tests on real clinical scans.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras import layers

from src.dataset import ChestXRayDataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AC-GAN Quality via Synthetic-to-Real Transfer")
    parser.add_argument("--generator_path", type=str, default="checkpoints/generator.h5",
                        help="Path to trained generator.h5")
    parser.add_argument("--data_dir", type=str, default="chest-xray-pneumonia/chest_xray/train",
                        help="Path to real dataset folder for evaluation")
    parser.add_argument("--num_synthetic", type=int, default=30000,
                        help="Number of synthetic images to generate for training VGG16")
    parser.add_argument("--epochs", type=int, default=60, help="Epochs to train VGG16")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for VGG16 training")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation metrics and plots")
    return parser.parse_args()


def build_vgg16_classifier(input_shape=(64, 64, 3)):
    base_model = tf.keras.applications.VGG16(
        weights=None,
        input_shape=input_shape,
        pooling="max",
        include_top=False,
    )
    x = layers.Dropout(0.4)(base_model.output)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)
    x = layers.Dropout(0.4)(x)
    output = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.models.Model(inputs=base_model.input, outputs=output, name="vgg16_classifier")
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=["accuracy"],
    )
    return model


def evaluate():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== [1/5] Loading Trained AC-GAN Generator ===")
    generator = tf.keras.models.load_model(args.generator_path)

    print(f"=== [2/5] Generating {args.num_synthetic} Synthetic Samples ===")
    noise = tf.random.uniform(shape=(args.num_synthetic, 100), minval=-1, maxval=1)
    synthetic_labels_int = np.random.choice(range(2), size=args.num_synthetic)
    synthetic_labels_onehot = tf.keras.utils.to_categorical(synthetic_labels_int, num_classes=2)

    synthetic_images = generator.predict([noise, synthetic_labels_onehot], batch_size=128, verbose=1)

    print("=== [3/5] Training VGG16 Classifier on Synthetic Images ===")
    classifier = build_vgg16_classifier(input_shape=(64, 64, 3))
    history = classifier.fit(
        synthetic_images,
        synthetic_labels_int,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=1,
    )

    # Plot training loss & accuracy curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("VGG16 Classifier Loss (Synthetic)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title("VGG16 Classifier Accuracy (Synthetic)")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, "vgg16_synthetic_training.png"))
    plt.close()

    print("=== [4/5] Loading Real Clinical Dataset ===")
    loader = ChestXRayDataLoader(dataset_path=args.data_dir, image_shape=(64, 64))
    real_images, real_labels = loader.load_data()

    print("=== [5/5] Evaluating on Real Clinical Images ===")
    raw_preds = tf.squeeze(classifier.predict(real_images, batch_size=64)).numpy()
    y_preds = (raw_preds >= 0.5).astype(int)

    acc = np.mean(y_preds == real_labels) * 100.0
    f1 = f1_score(real_labels, y_preds) * 100.0
    recall = recall_score(real_labels, y_preds) * 100.0
    precision = precision_score(real_labels, y_preds) * 100.0

    print("\n" + "=" * 50)
    print("      DOWNSTREAM GENERALIZATION EVALUATION")
    print("=" * 50)
    print(f"Accuracy:  {acc:.2f}%")
    print(f"F1-Score:  {f1:.2f}%")
    print(f"Recall:    {recall:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print("\nDetailed Classification Report:")
    report = classification_report(real_labels, y_preds, target_names=["NORMAL", "PNEUMONIA"])
    print(report)

    # Save report
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.2f}%\nF1-Score: {f1:.2f}%\nRecall: {recall:.2f}%\nPrecision: {precision:.2f}%\n\n")
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(real_labels, y_preds)
    cm_df = pd.DataFrame(cm, index=["NORMAL", "PNEUMONIA"], columns=["NORMAL", "PNEUMONIA"])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Synthetic-Trained Model on Real Images)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    plt.close()
    print(f"Evaluation artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    evaluate()
