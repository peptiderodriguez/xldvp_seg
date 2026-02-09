"""
Simple Reinhard stain normalization for H&E images.

Based on: Reinhard et al., "Color Transfer between Images", IEEE CGA 2001
Matches mean and standard deviation in LAB color space.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
import pickle


class ReinhardNormalizer:
    """
    Reinhard color normalization in LAB space.

    Simple and effective method that matches mean and std of LAB channels
    to a target distribution.
    """

    def __init__(self):
        self.target_means = None
        self.target_stds = None

    def fit(self, target_image: np.ndarray):
        """
        Fit normalizer to a target image.

        Args:
            target_image: RGB image (H, W, 3) uint8
        """
        # Convert to LAB
        lab = cv2.cvtColor(target_image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Compute mean and std for each channel
        self.target_means = np.mean(lab, axis=(0, 1))
        self.target_stds = np.std(lab, axis=(0, 1))

    def fit_batch(self, images: list):
        """
        Fit normalizer to multiple images by averaging their stats.

        Args:
            images: List of RGB images
        """
        all_means = []
        all_stds = []

        for img in images:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
            all_means.append(np.mean(lab, axis=(0, 1)))
            all_stds.append(np.std(lab, axis=(0, 1)))

        # Average the statistics
        self.target_means = np.mean(all_means, axis=0)
        self.target_stds = np.mean(all_stds, axis=0)

    def transform(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize an image to match target statistics.

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            Normalized RGB image uint8
        """
        if self.target_means is None or self.target_stds is None:
            raise ValueError("Normalizer not fitted! Call fit() first.")

        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Normalize each channel: (x - mean) / std * target_std + target_mean
        for i in range(3):
            mean = np.mean(lab[:, :, i])
            std = np.std(lab[:, :, i])

            if std > 0:  # Avoid division by zero
                lab[:, :, i] = ((lab[:, :, i] - mean) / std) * self.target_stds[i] + self.target_means[i]

        # Clip to valid LAB range
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)  # L channel
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)  # A channel
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)  # B channel

        # Convert back to RGB
        lab = lab.astype(np.uint8)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return rgb

    def save(self, filepath: str):
        """Save normalizer parameters to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({'means': self.target_means, 'stds': self.target_stds}, f)

    @classmethod
    def load(cls, filepath: str):
        """Load normalizer parameters from file."""
        normalizer = cls()
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        normalizer.target_means = params['means']
        normalizer.target_stds = params['stds']
        return normalizer


def normalize_he_reinhard(image: np.ndarray, target_means: np.ndarray, target_stds: np.ndarray) -> np.ndarray:
    """
    Convenience function for single-shot Reinhard normalization.

    Args:
        image: RGB image uint8
        target_means: LAB means [L, A, B]
        target_stds: LAB stds [L, A, B]

    Returns:
        Normalized RGB image uint8
    """
    normalizer = ReinhardNormalizer()
    normalizer.target_means = target_means
    normalizer.target_stds = target_stds
    return normalizer.transform(image)
