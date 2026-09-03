"""
Training script for Conditional AC-GAN on Chest X-Ray images.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.dataset import ChestXRayDataLoader
from src.models import ChestXRayACGAN


def parse_args():
    parser = argparse.ArgumentParser(description="Train AC-GAN on Chest X-Ray Images")
    parser.add_argument("--data_dir", type=str, default="chest-xray-pneumonia/chest_xray/train",
                        help="Path to training dataset folder")
    parser.add_argument("--epochs", type=int, default=32000, help="Total training iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per training step")
    parser.add_argument("--latent_dim", type=int, default=100, help="Latent noise vector dimension")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for RMSprop optimizer")
    parser.add_argument("--weight_decay", type=float, default=6e-9, help="Weight decay for optimizer")
    parser.add_argument("--kernel_size", type=int, default=5, help="Convolution kernel size")
    parser.add_argument("--sample_interval", type=int, default=5000, help="Interval to save sample outputs")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Directory to save model weights")
    return parser.parse_args()


def save_sample_grid(generator, latent_dim, epoch, output_dir="checkpoints"):
    os.makedirs(output_dir, exist_ok=True)
    num_samples = 16
    noise = tf.random.uniform(shape=(num_samples, latent_dim), minval=-1, maxval=1)
    
    # Half NORMAL (0), Half PNEUMONIA (1)
    labels_int = np.array([0] * 8 + [1] * 8)
    labels_onehot = tf.keras.utils.to_categorical(labels_int, num_classes=2)
    
    gen_images = generator.predict([noise, labels_onehot], verbose=0)
    
    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(gen_images[i])
        class_name = "NORMAL" if labels_int[i] == 0 else "PNEUMONIA"
        ax.set_title(class_name, fontsize=9)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sample_epoch_{epoch}.png"))
    plt.close()


def train():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== [1/4] Loading Dataset ===")
    loader = ChestXRayDataLoader(dataset_path=args.data_dir, image_shape=(64, 64))
    images, labels = loader.load_data()
    labels_onehot = tf.keras.utils.to_categorical(labels, num_classes=2)

    print("=== [2/4] Building and Compiling AC-GAN Models ===")
    acgan = ChestXRayACGAN(
        latent_dim=args.latent_dim,
        num_classes=2,
        image_shape=(64, 64, 3),
        kernel_size=args.kernel_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )
    generator, discriminator, gan = acgan.compile_models()

    print(f"=== [3/4] Starting AC-GAN Training for {args.epochs} Steps ===")
    num_train = len(images)

    for epoch in range(1, args.epochs + 1):
        # 1. Sample Real Batch
        indices = np.random.randint(0, num_train, size=args.batch_size)
        real_images = images[indices]
        real_labels = labels_onehot[indices]
        real_validity = tf.ones(shape=(args.batch_size,))

        # 2. Generate Fake Batch
        noise = tf.random.uniform(shape=(args.batch_size, args.latent_dim), minval=-1, maxval=1)
        fake_labels_int = np.random.choice(range(2), size=args.batch_size)
        fake_labels = tf.keras.utils.to_categorical(fake_labels_int, num_classes=2)
        fake_images = generator.predict([noise, fake_labels], verbose=0)
        fake_validity = tf.zeros(shape=(args.batch_size,))

        # 3. Train Discriminator
        all_images = np.vstack([real_images, fake_images])
        all_labels = np.vstack([real_labels, fake_labels])
        all_validity = np.hstack([real_validity, fake_validity])

        d_loss = discriminator.train_on_batch(all_images, [all_validity, all_labels])

        # 4. Train Generator (via combined GAN model)
        noise_gan = tf.random.uniform(shape=(args.batch_size, args.latent_dim), minval=-1, maxval=1)
        target_labels_int = np.random.choice(range(2), size=args.batch_size)
        target_labels = tf.keras.utils.to_categorical(target_labels_int, num_classes=2)
        trick_validity = tf.ones(shape=(args.batch_size,))

        g_loss = gan.train_on_batch([noise_gan, target_labels], [trick_validity, target_labels])

        if epoch % args.sample_interval == 0 or epoch == 1:
            print(
                f"[Epoch {epoch}/{args.epochs}] "
                f"D_Loss(Total: {d_loss[0]:.4f}, Adv: {d_loss[1]:.4f}, Cls: {d_loss[2]:.4f}) | "
                f"G_Loss(Total: {g_loss[0]:.4f}, Adv: {g_loss[1]:.4f}, Cls: {g_loss[2]:.4f})"
            )
            save_sample_grid(generator, args.latent_dim, epoch, args.output_dir)

    print("=== [4/4] Saving Final Generator Model ===")
    gen_save_path = os.path.join(args.output_dir, "generator.h5")
    generator.save(gen_save_path)
    print(f"Generator model saved successfully to: {gen_save_path}")


if __name__ == "__main__":
    train()
