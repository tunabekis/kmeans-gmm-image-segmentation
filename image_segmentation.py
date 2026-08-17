"""Image loading and clustering-based segmentation helpers."""

import os

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float


def load_image_as_pixels(image_path):
    """Load an image and flatten it into an (N, 3) array of RGB pixels.

    Returns the pixel array along with the original (height, width,
    channels) shape needed to reconstruct the segmented image.
    """
    img = img_as_float(io.imread(image_path))
    if img.shape[-1] == 4:  # Drop the alpha channel if present.
        img = img[..., :3]

    height, width, channels = img.shape
    pixels = img.reshape((-1, channels))
    return pixels, (height, width, channels)


def segment_image(model, pixels, shape):
    """Cluster `pixels` with `model` and rebuild the segmented image.

    Each pixel is replaced by the color of the cluster center it was
    assigned to, producing a flat-color segmentation map.
    """
    labels, centers = model.fit_predict(pixels)
    segmented = centers[labels].reshape(shape)
    return np.clip(segmented, 0, 1)


def save_image(image, output_dir, filename):
    """Save `image` as `filename` under `output_dir`, creating it if needed."""
    os.makedirs(output_dir, exist_ok=True)
    plt.imsave(os.path.join(output_dir, filename), image)
