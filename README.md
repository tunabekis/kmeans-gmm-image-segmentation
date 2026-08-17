# K-Means & GMM Image Segmentation

Color-based image segmentation using **K-Means** and a **Gaussian Mixture
Model (GMM)**, both implemented from scratch with NumPy/SciPy — no
scikit-learn or other ready-made clustering library.

Each pixel of an image is treated as a 3D point in RGB color space. The
clustering algorithms group similar colors together, and every pixel is then
repainted with the color of its cluster center, producing a segmentation map.

## How it works

- **K-Means** (`KMeansScratch`) — hard clustering via Lloyd's algorithm.
  Each pixel is assigned to its nearest centroid by Euclidean distance;
  centroids are updated to the mean of their assigned pixels until
  convergence. Multiple random restarts are run and the lowest-inertia
  result is kept, since K-Means is sensitive to initialization.
- **GMM** (`GMMScratch`) — soft clustering via Expectation-Maximization.
  Each pixel gets a probability of belonging to each cluster, and clusters
  are modeled as full Gaussians (mean + covariance), so they can capture
  elliptical, non-spherical color regions. Means are seeded with a quick
  K-Means run for stable convergence.

## Project structure

```
.
├── clustering.py          # KMeansScratch and GMMScratch implementations
├── image_segmentation.py  # Image loading, segmentation, and saving helpers
├── main.py                # Entry point: runs both algorithms over all images/K values
├── images/                # Input images
├── outputs/                # Generated segmentation maps
└── requirements.txt
```

## Setup & usage

```bash
pip install -r requirements.txt
python main.py
```

Place input images (`.jpg`, `.jpeg`, `.png`) in `images/`. Running `main.py`
segments every image with both algorithms for K = 2, 3, 5, and 8, and writes
each result to `outputs/` as `kmeans_k{K}_{image}` and `gmm_k{K}_{image}`.

## Results

- **Low K (2–3):** both algorithms produce coarse color blocks and lose
  fine detail (under-segmentation).
- **Higher K (5–8):** object boundaries and textures become noticeably
  sharper.
- **K-Means vs. GMM:** K-Means' hard, distance-based assignment produces
  sharp, blocky borders, especially where lighting or color gradients
  change smoothly. GMM's probabilistic, covariance-aware assignment
  segments shadows, reflections, and gradients more naturally.
- **Performance:** K-Means converges quickly since it only computes
  distances. GMM is considerably slower, since each EM iteration evaluates
  a multivariate normal density (and its covariance) for every pixel and
  every cluster.

## Tech stack

Python, NumPy, SciPy (`multivariate_normal`), Matplotlib, scikit-image.
