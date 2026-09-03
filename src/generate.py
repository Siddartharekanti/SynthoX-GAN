"""
Inference script to sample synthetic Chest X-Ray images using a trained AC-GAN Generator.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Chest X-Ray Images")
    parser.add_argument("--model_path", type=str, default="checkpoints/generator.h5",
                        help="Path to trained generator.h5")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent noise vector dimension")
    parser.add_argument("--class_label", type=str, default="both", choices=["normal", "pneumonia", "both"],
                        help="Target condition to generate")
    parser.add_argument("--output_image", type=str, default="generated_samples.png",
                        help="Filepath to save generated image grid")
    return parser.parse_args()


def generate():
    args = parse_args()

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model file not found at: {args.model_path}")

    print(f"Loading Generator from {args.model_path}...")
    generator = tf.keras.models.load_model(args.model_path)

    # Sample random noise
    noise = tf.random.uniform(shape=(args.num_samples, args.latent_dim), minval=-1, maxval=1)

    # Assign class labels
    if args.class_label == "normal":
        labels_int = np.zeros(args.num_samples, dtype=int)
    elif args.class_label == "pneumonia":
        labels_int = np.ones(args.num_samples, dtype=int)
    else:  # balanced mixture
        half = args.num_samples // 2
        labels_int = np.array([0] * half + [1] * (args.num_samples - half))

    labels_onehot = tf.keras.utils.to_categorical(labels_int, num_classes=2)

    print(f"Synthesizing {args.num_samples} images...")
    synthetic_images = generator.predict([noise, labels_onehot])

    # Plot and save
    cols = min(8, args.num_samples)
    rows = (args.num_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    
    if args.num_samples == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx in range(args.num_samples):
        axes[idx].imshow(synthetic_images[idx])
        class_name = "NORMAL" if labels_int[idx] == 0 else "PNEUMONIA"
        axes[idx].set_title(class_name, fontsize=9)
        axes[idx].axis("off")

    for idx in range(args.num_samples, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(args.output_image, dpi=200)
    plt.close()
    print(f"Saved generated image grid to: {args.output_image}")


if __name__ == "__main__":
    generate()
