"""
neural_network.py
-----------------
Multi-Layer Perceptron (MLP) implemented from scratch with NumPy.
Supports configurable architecture, forward/backward pass, and
gradient computation that can be called independently by parallel workers.
"""

import os
# Limit BLAS threads in worker processes (multiprocessing spawns new interpreters)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


class NeuralNetwork:
    """A fully-connected feed-forward neural network (MLP)."""

    def __init__(self, layer_sizes, seed=42):
        """
        Parameters
        ----------
        layer_sizes : list[int]
            Number of neurons in each layer, e.g. [784, 256, 128, 10].
        seed : int
            Random seed for reproducible weight initialization.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        rng = np.random.RandomState(seed)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers):
            # He initialization
            w = rng.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(
                2.0 / layer_sizes[i]
            )
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Parameter serialization (needed for multiprocessing)
    # ------------------------------------------------------------------
    def get_params(self):
        """Return a deep copy of all parameters as a dict of lists."""
        return {
            "weights": [w.copy() for w in self.weights],
            "biases": [b.copy() for b in self.biases],
        }

    def set_params(self, params):
        """Load parameters from a dict."""
        self.weights = [w.copy() for w in params["weights"]]
        self.biases = [b.copy() for b in params["biases"]]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, X):
        """
        Forward pass through all layers.

        Returns
        -------
        activations : list[np.ndarray]
            Activation at each layer (including input).
        pre_activations : list[np.ndarray]
            Pre-activation (z) at each layer.
        """
        activations = [X]
        pre_activations = []

        for i in range(self.num_layers):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(z)

            if i < self.num_layers - 1:
                # ReLU activation for hidden layers
                a = np.maximum(0, z)
            else:
                # Softmax activation for output layer
                z_shifted = z - np.max(z, axis=1, keepdims=True)
                exp_z = np.exp(z_shifted)
                a = exp_z / np.sum(exp_z, axis=1, keepdims=True)

            activations.append(a)

        return activations, pre_activations

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    def compute_loss(self, y_pred, y_true):
        """
        Cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray, shape (n, C)
            Predicted probabilities (softmax output).
        y_true : np.ndarray, shape (n, C)
            One-hot encoded true labels.
        """
        n = y_true.shape[0]
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / n
        return loss

    # ------------------------------------------------------------------
    # Backward pass — gradient computation
    # ------------------------------------------------------------------
    def compute_gradients(self, X, y):
        """
        Compute gradients of the loss w.r.t. all weights and biases.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Input data partition.
        y : np.ndarray, shape (n, C)
            One-hot encoded labels for the partition.

        Returns
        -------
        grads : dict with keys 'weights' and 'biases', each a list of
                np.ndarray matching the shapes of self.weights / self.biases.
        loss  : float
            Mean loss on this partition.
        """
        n = X.shape[0]
        activations, pre_activations = self.forward(X)

        # Loss for monitoring
        loss = self.compute_loss(activations[-1], y)

        weight_grads = [None] * self.num_layers
        bias_grads = [None] * self.num_layers

        # Derivative of softmax + cross-entropy combined
        delta = activations[-1] - y  # shape (n, C)

        for i in range(self.num_layers - 1, -1, -1):
            weight_grads[i] = (activations[i].T @ delta) / n
            bias_grads[i] = np.sum(delta, axis=0, keepdims=True) / n

            if i > 0:
                delta = delta @ self.weights[i].T
                # ReLU derivative
                delta = delta * (pre_activations[i - 1] > 0).astype(np.float64)

        grads = {"weights": weight_grads, "biases": bias_grads}
        return grads, loss

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------
    def predict(self, X):
        """Return predicted class indices."""
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    def accuracy(self, X, y_onehot):
        """Compute classification accuracy."""
        preds = self.predict(X)
        true_labels = np.argmax(y_onehot, axis=1)
        return np.mean(preds == true_labels)

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------
    def apply_gradients(self, grads, lr, momentum=0.0, velocity=None):
        """
        Update parameters using SGD (optionally with momentum).

        Parameters
        ----------
        grads : dict
            Gradient dict as returned by compute_gradients.
        lr : float
            Learning rate.
        momentum : float
            Momentum coefficient (0 = vanilla SGD).
        velocity : dict or None
            Previous velocity; created if None.

        Returns
        -------
        velocity : dict
            Updated velocity for next iteration.
        """
        if velocity is None:
            velocity = {
                "weights": [np.zeros_like(w) for w in self.weights],
                "biases": [np.zeros_like(b) for b in self.biases],
            }

        for i in range(self.num_layers):
            velocity["weights"][i] = (
                momentum * velocity["weights"][i] - lr * grads["weights"][i]
            )
            velocity["biases"][i] = (
                momentum * velocity["biases"][i] - lr * grads["biases"][i]
            )
            self.weights[i] += velocity["weights"][i]
            self.biases[i] += velocity["biases"][i]

        return velocity


# ======================================================================
# Standalone gradient function for multiprocessing workers
# ======================================================================
def compute_gradients_worker(args):
    """
    Standalone function that can be called by a worker process.

    Parameters
    ----------
    args : tuple
        (params_dict, X_partition, y_partition, layer_sizes)

    Returns
    -------
    grads : dict
    loss  : float
    n     : int  (number of samples in partition, for weighted averaging)
    """
    params, X_partition, y_partition, layer_sizes = args

    # Reconstruct the model in this worker process
    model = NeuralNetwork(layer_sizes, seed=0)
    model.set_params(params)

    grads, loss = model.compute_gradients(X_partition, y_partition)
    return grads, loss, X_partition.shape[0]
