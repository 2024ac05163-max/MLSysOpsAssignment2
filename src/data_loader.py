"""
data_loader.py
--------------
Load and preprocess datasets for training and evaluation.
Supports MNIST (via scikit-learn) and a synthetic dataset for quick testing.
"""

import numpy as np


def one_hot_encode(labels, num_classes=10):
    """Convert integer labels to one-hot encoded matrix."""
    n = labels.shape[0]
    one_hot = np.zeros((n, num_classes))
    one_hot[np.arange(n), labels.astype(int)] = 1.0
    return one_hot


def load_mnist(num_samples=None, test_size=0.15, seed=42):
    """
    Load MNIST dataset via scikit-learn's fetch_openml.

    Parameters
    ----------
    num_samples : int or None
        If given, subsample the dataset to this many total samples.
    test_size : float
        Fraction of data used for testing.
    seed : int
        Random seed for shuffling / subsampling.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
        y arrays are one-hot encoded.
    """
    from sklearn.datasets import fetch_openml

    print("Loading MNIST dataset (this may take a moment on first run)...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X, y = mnist.data.astype(np.float64), mnist.target.astype(np.int64)

    # Normalize pixel values to [0, 1]
    X = X / 255.0

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    X, y = X[indices], y[indices]

    if num_samples is not None:
        X, y = X[:num_samples], y[:num_samples]

    split = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train_raw, y_test_raw = y[:split], y[split:]

    y_train = one_hot_encode(y_train_raw, num_classes=10)
    y_test = one_hot_encode(y_test_raw, num_classes=10)

    print(
        f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples, "
        f"Features: {X_train.shape[1]}"
    )
    return X_train, y_train, X_test, y_test


def load_synthetic(num_samples=10000, num_features=784, num_classes=10,
                   test_size=0.15, seed=42):
    """
    Generate a synthetic classification dataset for quick testing.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    rng = np.random.RandomState(seed)

    # Random cluster centers
    centers = rng.randn(num_classes, num_features) * 2.0
    labels = rng.randint(0, num_classes, size=num_samples)
    X = centers[labels] + rng.randn(num_samples, num_features) * 0.5

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    indices = rng.permutation(num_samples)
    X, labels = X[indices], labels[indices]

    split = int(num_samples * (1 - test_size))
    X_train, X_test = X[:split], X[split:]
    y_train = one_hot_encode(labels[:split], num_classes)
    y_test = one_hot_encode(labels[split:], num_classes)

    print(
        f"  Synthetic dataset — Train: {X_train.shape[0]}, Test: {X_test.shape[0]}, "
        f"Features: {num_features}, Classes: {num_classes}"
    )
    return X_train, y_train, X_test, y_test
