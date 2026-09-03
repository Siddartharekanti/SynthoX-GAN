"""
Data loading and preprocessing module for Chest X-Ray images.
"""

import os
import pathlib
import cv2
import numpy as np
import tensorflow as tf


class ChestXRayDataLoader:
    """
    Loads, resizes, and preprocesses Chest X-Ray images from directory paths.
    """

    def __init__(self, dataset_path: str, labels=None, image_shape=(64, 64)):
        """
        Args:
            dataset_path (str): Path to folder containing class subdirectories (e.g., NORMAL, PNEUMONIA).
            labels (list): List of class directory names. Defaults to ['NORMAL', 'PNEUMONIA'].
            image_shape (tuple): Target (height, width) for image resizing.
        """
        if labels is None:
            labels = ['NORMAL', 'PNEUMONIA']
        self.dataset_path = dataset_path
        self.labels = labels
        self.image_shape = image_shape
        self.image_files = []

    def _discover_files(self):
        """Scans the directory for image files per class label."""
        self.image_files = []
        for label in self.labels:
            class_dir = os.path.join(self.dataset_path, label)
            if not os.path.exists(class_dir):
                raise FileNotFoundError(f"Class directory not found: {class_dir}")
            
            # Match common image formats (.jpeg, .jpg, .png)
            files = list(pathlib.Path(class_dir).glob('*.*'))
            valid_files = [
                f for f in files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            ]
            self.image_files.append(valid_files)

    def load_data(self):
        """
        Reads, normalizes, and returns images and corresponding class labels.

        Returns:
            images (np.ndarray): Array of shape (N, H, W, 3) normalized to [0, 1].
            labels (np.ndarray): Integer class labels of shape (N,).
        """
        self._discover_files()
        loaded_images = []
        loaded_labels = []

        for class_idx, file_list in enumerate(self.image_files):
            print(f"Loading {len(file_list)} images for class '{self.labels[class_idx]}'...")
            for img_path in file_list:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.resize(img, self.image_shape)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                loaded_images.append(img)
                loaded_labels.append(class_idx)

        images = np.array(loaded_images, dtype=np.float32)
        labels = np.array(loaded_labels, dtype=np.int32)
        
        print(f"Dataset successfully loaded: {images.shape[0]} samples with shape {images.shape[1:]}")
        return images, labels
