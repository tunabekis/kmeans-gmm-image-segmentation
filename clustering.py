"""From-scratch implementations of K-Means and Gaussian Mixture Model clustering.

Both algorithms operate on any (N, D) array of samples and are used here to
cluster pixel colors for image segmentation. Neither uses a ready-made
clustering implementation from a machine learning library.
"""

import numpy as np
from scipy.stats import multivariate_normal


class KMeansScratch:
    """K-Means clustering (Lloyd's algorithm) with multiple random restarts."""

    def __init__(self, k, max_iters=100, n_init=3, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.n_init = n_init
        self.tol = tol
        self.centroids = None

    def fit_predict(self, X):
        """Cluster `X` and return the (labels, centroids) of the best restart.

        Each restart is run independently from a random initialization; the
        restart with the lowest inertia (sum of squared distances to the
        assigned centroid) is kept, which reduces sensitivity to unlucky
        initial centroids.
        """
        best_centroids = None
        best_labels = None
        lowest_inertia = np.inf

        for _ in range(self.n_init):
            centroids = self._init_centroids(X)

            for _ in range(self.max_iters):
                labels = self._assign_labels(X, centroids)
                new_centroids = self._update_centroids(X, labels, centroids)
                converged = np.linalg.norm(new_centroids - centroids) < self.tol
                centroids = new_centroids
                if converged:
                    break

            # Recompute labels for the final centroids so the returned
            # assignment always matches the returned centroids exactly,
            # instead of lagging one iteration behind.
            labels = self._assign_labels(X, centroids)
            inertia = self._inertia(X, centroids, labels)

            if inertia < lowest_inertia:
                lowest_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        return best_labels, best_centroids

    def _init_centroids(self, X):
        """Pick k random data points as the initial centroids."""
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        return X[random_indices]

    def _assign_labels(self, X, centroids):
        """Assign each sample to its nearest centroid (Euclidean distance)."""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels, centroids):
        """Recompute each centroid as the mean of its assigned samples.

        A cluster that received no samples keeps its previous centroid
        instead of collapsing to NaN.
        """
        return np.array([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(self.k)
        ])

    def _inertia(self, X, centroids, labels):
        """Sum of squared distances from each sample to its assigned centroid."""
        diff = X - centroids[labels]
        return np.sum(np.einsum('ij,ij->i', diff, diff))


class GMMScratch:
    """Gaussian Mixture Model fit via Expectation-Maximization (EM).

    Means are initialized with a quick K-Means run, which is standard
    practice for stabilizing EM's convergence.
    """

    def __init__(self, k, max_iters=50, tol=1e-4, covariance_eps=1e-6):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.covariance_eps = covariance_eps
        self.means = None
        self.covariances = None
        self.pi = None

    def fit_predict(self, X):
        """Cluster `X` and return (hard labels, means) after EM convergence."""
        self._initialize_parameters(X)

        log_likelihood_old = -np.inf
        for _ in range(self.max_iters):
            responsibilities, log_likelihood_new = self._e_step(X)

            if np.abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break
            log_likelihood_old = log_likelihood_new

            self._m_step(X, responsibilities)

        # Final E-step so the returned assignment always matches the
        # parameters being returned (self.means), even when the loop exits
        # via max_iters right after an M-step update.
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1), self.means

    def _initialize_parameters(self, X):
        """Seed means/covariances/mixture weights from a K-Means partition."""
        n_samples, n_features = X.shape
        kmeans = KMeansScratch(k=self.k, n_init=1, max_iters=20)
        labels, means = kmeans.fit_predict(X)

        self.means = means
        self.covariances = np.zeros((self.k, n_features, n_features))
        self.pi = np.zeros(self.k)

        for i in range(self.k):
            cluster_data = X[labels == i]
            if len(cluster_data) > 1:
                self.covariances[i] = np.cov(cluster_data, rowvar=False) + \
                    np.eye(n_features) * self.covariance_eps
            else:
                self.covariances[i] = np.eye(n_features) * self.covariance_eps
            self.pi[i] = len(cluster_data) / n_samples

    def _e_step(self, X):
        """Compute normalized responsibilities and the data log-likelihood."""
        n_samples, n_features = X.shape
        weighted_densities = np.zeros((n_samples, self.k))

        for i in range(self.k):
            try:
                rv = multivariate_normal(self.means[i], self.covariances[i])
                weighted_densities[:, i] = self.pi[i] * rv.pdf(X)
            except np.linalg.LinAlgError:
                # Covariance became singular; regularize and retry.
                self.covariances[i] += np.eye(n_features) * 1e-4
                rv = multivariate_normal(self.means[i], self.covariances[i])
                weighted_densities[:, i] = self.pi[i] * rv.pdf(X)

        density_sum = weighted_densities.sum(axis=1)
        density_sum_safe = np.where(density_sum == 0, 1e-10, density_sum)  # Avoid log(0).

        log_likelihood = np.sum(np.log(density_sum_safe))
        responsibilities = weighted_densities / density_sum_safe[:, np.newaxis]
        return responsibilities, log_likelihood

    def _m_step(self, X, responsibilities):
        """Update means, covariances, and mixture weights (pi) in place."""
        n_samples, n_features = X.shape
        N_k = responsibilities.sum(axis=0)

        for i in range(self.k):
            if N_k[i] < 1e-10:
                continue  # Avoid division by zero for a dying cluster.

            weights = responsibilities[:, i][:, np.newaxis]
            self.means[i] = (weights * X).sum(axis=0) / N_k[i]

            diff = X - self.means[i]
            self.covariances[i] = (weights * diff).T @ diff / N_k[i] + \
                np.eye(n_features) * self.covariance_eps

            self.pi[i] = N_k[i] / n_samples
