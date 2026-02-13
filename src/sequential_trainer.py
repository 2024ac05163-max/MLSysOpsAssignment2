"""
sequential_trainer.py
---------------------
Sequential Mini-Batch SGD trainer.
Serves as the **baseline** for comparing against the parallel implementation.
"""

import time
import numpy as np
from .neural_network import NeuralNetwork


class SequentialTrainer:
    """Train a NeuralNetwork using sequential mini-batch SGD."""

    def __init__(self, model: NeuralNetwork, lr=0.1, momentum=0.9):
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def train(self, X_train, y_train, X_test, y_test,
              epochs=20, batch_size=256, verbose=True):
        """
        Run sequential mini-batch SGD training.

        Returns
        -------
        history : dict
            Keys: 'train_loss', 'test_loss', 'train_acc', 'test_acc',
                  'epoch_times', 'total_time'.
        """
        n = X_train.shape[0]
        history = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "epoch_times": [],
        }

        total_start = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Shuffle training data
            indices = np.random.permutation(n)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch loop
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute gradients
                grads, loss = self.model.compute_gradients(X_batch, y_batch)

                # Update parameters
                self.velocity = self.model.apply_gradients(
                    grads, self.lr, self.momentum, self.velocity
                )

                epoch_loss += loss
                num_batches += 1

            epoch_time = time.time() - epoch_start

            # Evaluation
            avg_loss = epoch_loss / num_batches
            train_acc = self.model.accuracy(X_train, y_train)
            test_acc = self.model.accuracy(X_test, y_test)

            # Compute test loss
            _, test_pre = self.model.forward(X_test)
            activations_test, _ = self.model.forward(X_test)
            test_loss = self.model.compute_loss(activations_test[-1], y_test)

            history["train_loss"].append(avg_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["epoch_times"].append(epoch_time)

            if verbose:
                print(
                    f"  [Sequential] Epoch {epoch:3d}/{epochs} | "
                    f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s"
                )

        total_time = time.time() - total_start
        history["total_time"] = total_time

        if verbose:
            print(f"  [Sequential] Total training time: {total_time:.2f}s")

        return history
