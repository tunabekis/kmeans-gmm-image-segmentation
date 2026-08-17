"""Entry point: segments every image in ./images/ with K-Means and GMM.

Usage:
    python main.py

For each image and each K in K_VALUES, both algorithms are run from scratch
and the resulting segmentation maps are written to ./outputs/ as
"kmeans_k{K}_{image}" and "gmm_k{K}_{image}".
"""

import os

import numpy as np

from clustering import KMeansScratch, GMMScratch
from image_segmentation import load_image_as_pixels, segment_image, save_image

IMAGE_DIR = "./images/"
OUTPUT_DIR = "./outputs/"
K_VALUES = [2, 3, 5, 8]


def main():
    np.random.seed(42)  # Fixed seed for reproducible results across runs.

    image_files = [
        f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".jpeg", ".png"))
    ]

    for image_name in image_files:
        print(f"Processing {image_name}...")
        pixels, shape = load_image_as_pixels(os.path.join(IMAGE_DIR, image_name))

        for k in K_VALUES:
            print(f"  Running K={k}...")

            kmeans_result = segment_image(KMeansScratch(k=k), pixels, shape)
            save_image(kmeans_result, OUTPUT_DIR, f"kmeans_k{k}_{image_name}")

            gmm_result = segment_image(GMMScratch(k=k), pixels, shape)
            save_image(gmm_result, OUTPUT_DIR, f"gmm_k{k}_{image_name}")

    print("Segmentation complete! Check the 'outputs' folder for your 32 images.")


if __name__ == "__main__":
    main()
