"""
parallel_trainer.py
-------------------
Parallel Mini-Batch SGD trainer with two strategies:

1. **Threading** (default) — uses `concurrent.futures.ThreadPoolExecutor`.
   True shared-memory: all threads access the same model parameters and data.
   NumPy releases the GIL during matrix operations so threads achieve genuine
   parallelism on the heavy compute (forward/backward pass).

2. **Multiprocessing** — uses `multiprocessing.Pool`.
   Each worker is a separate process; data is serialised via pickle.
   Higher communication overhead but avoids GIL entirely.

Data-parallel approach (both strategies):
  1. Each mini-batch is split into P partitions.
  2. Workers compute gradients on their partition in parallel.
  3. Gradients are aggregated (averaged) in the main thread/process.
  4. Parameters are updated once per mini-batch.
"""

import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from .neural_network import NeuralNetwork, compute_gradients_worker


# ======================================================================
# Thread-safe gradient computation (shares model memory directly)
# ======================================================================
def _thread_gradient_worker(model, X_part, y_part):
    """
    Compute gradients inside a thread using the SHARED model object.

    This is safe because compute_gradients() only READS model.weights /
    model.biases and creates new arrays for activations and gradients.
    Weights are not modified until all threads finish.
    """
    grads, loss = model.compute_gradients(X_part, y_part)
    return grads, loss, X_part.shape[0]


# ======================================================================
# Gradient aggregation
# ======================================================================
def _aggregate_gradients(results, num_layers):
    """
    Aggregate (average) partial gradients returned by workers.

    Parameters
    ----------
    results : list of (grads_dict, loss, n_samples)
    num_layers : int

    Returns
    -------
    agg_grads : dict
    avg_loss  : float
    """
    total_samples = sum(r[2] for r in results)

    agg_weight_grads = [None] * num_layers
    agg_bias_grads = [None] * num_layers
    weighted_loss = 0.0

    for grads, loss, n in results:
        weight = n / total_samples
        weighted_loss += loss * weight

        for i in range(num_layers):
            if agg_weight_grads[i] is None:
                agg_weight_grads[i] = grads["weights"][i] * weight
                agg_bias_grads[i] = grads["biases"][i] * weight
            else:
                agg_weight_grads[i] += grads["weights"][i] * weight
                agg_bias_grads[i] += grads["biases"][i] * weight

    agg_grads = {"weights": agg_weight_grads, "biases": agg_bias_grads}
    return agg_grads, weighted_loss


# ======================================================================
# Partitioning helper
# ======================================================================
def _partition_batch(X_batch, y_batch, P):
    """Split a mini-batch into P roughly-equal partitions."""
    n = X_batch.shape[0]
    partition_size = n // P
    partitions = []
    for w in range(P):
        p_start = w * partition_size
        p_end = n if w == P - 1 else (w + 1) * partition_size
        partitions.append((X_batch[p_start:p_end], y_batch[p_start:p_end]))
    return partitions


class ParallelTrainer:
    """Train a NeuralNetwork using parallel mini-batch SGD."""

    def __init__(self, model: NeuralNetwork, lr=0.1, momentum=0.9,
                 num_workers=4, strategy="threading"):
        """
        Parameters
        ----------
        strategy : str
            'threading'       — ThreadPoolExecutor (shared memory, low overhead)
            'multiprocessing' — multiprocessing.Pool (process isolation)
        """
        assert strategy in ("threading", "multiprocessing"), \
            f"Unknown strategy: {strategy}"
        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.num_workers = num_workers
        self.strategy = strategy
        self.velocity = None

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, X_train, y_train, X_test, y_test,
              epochs=20, batch_size=256, verbose=True):
        """
        Run parallel mini-batch SGD training.

        Returns
        -------
        history : dict
            Keys: 'train_loss', 'test_loss', 'train_acc', 'test_acc',
                  'epoch_times', 'total_time', 'compute_times',
                  'comm_times'.
        """
        n = X_train.shape[0]
        P = self.num_workers
        layer_sizes = self.model.layer_sizes
        label = f"{self.strategy[:3].upper()} P={P}"

        history = {
            "train_loss": [], "test_loss": [],
            "train_acc": [],  "test_acc": [],
            "epoch_times": [], "compute_times": [], "comm_times": [],
        }

        total_start = time.time()

        if self.strategy == "threading":
            self._train_threaded(
                X_train, y_train, X_test, y_test,
                epochs, batch_size, P, layer_sizes, label, history, verbose,
            )
        else:
            self._train_multiprocessing(
                X_train, y_train, X_test, y_test,
                epochs, batch_size, P, layer_sizes, label, history, verbose,
            )

        total_time = time.time() - total_start
        history["total_time"] = total_time

        if verbose:
            print(f"  [{label}] Total training time: {total_time:.2f}s")

        return history

    # ------------------------------------------------------------------
    # Threading strategy
    # ------------------------------------------------------------------
    def _train_threaded(self, X_train, y_train, X_test, y_test,
                        epochs, batch_size, P, layer_sizes, label,
                        history, verbose):
        n = X_train.shape[0]

        with ThreadPoolExecutor(max_workers=P) as executor:
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                epoch_compute_time = 0.0
                epoch_comm_time = 0.0

                indices = np.random.permutation(n)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                epoch_loss = 0.0
                num_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    # Partition the mini-batch (low cost — just slicing)
                    comm_start = time.time()
                    partitions = _partition_batch(X_batch, y_batch, P)
                    comm_time_send = time.time() - comm_start

                    # Submit gradient computation to thread pool.
                    # Threads share self.model directly (read-only access
                    # to weights during forward/backward — thread-safe).
                    compute_start = time.time()
                    futures = [
                        executor.submit(
                            _thread_gradient_worker,
                            self.model, X_p, y_p,
                        )
                        for X_p, y_p in partitions
                    ]
                    results = [f.result() for f in futures]
                    compute_time = time.time() - compute_start

                    # Aggregate
                    agg_start = time.time()
                    agg_grads, batch_loss = _aggregate_gradients(
                        results, self.model.num_layers
                    )
                    comm_time_agg = time.time() - agg_start

                    # Update (all threads are done, safe to write)
                    self.velocity = self.model.apply_gradients(
                        agg_grads, self.lr, self.momentum, self.velocity
                    )

                    epoch_loss += batch_loss
                    num_batches += 1
                    epoch_compute_time += compute_time
                    epoch_comm_time += comm_time_send + comm_time_agg

                self._record_epoch(
                    epoch, epochs, label, epoch_start,
                    epoch_loss, num_batches, epoch_compute_time,
                    epoch_comm_time, X_train, y_train, X_test, y_test,
                    history, verbose,
                )

    # ------------------------------------------------------------------
    # Multiprocessing strategy
    # ------------------------------------------------------------------
    def _train_multiprocessing(self, X_train, y_train, X_test, y_test,
                               epochs, batch_size, P, layer_sizes, label,
                               history, verbose):
        n = X_train.shape[0]

        with mp.Pool(processes=P) as pool:
            for epoch in range(1, epochs + 1):
                epoch_start = time.time()
                epoch_compute_time = 0.0
                epoch_comm_time = 0.0

                indices = np.random.permutation(n)
                X_shuffled = X_train[indices]
                y_shuffled = y_train[indices]

                epoch_loss = 0.0
                num_batches = 0

                for start in range(0, n, batch_size):
                    end = min(start + batch_size, n)
                    X_batch = X_shuffled[start:end]
                    y_batch = y_shuffled[start:end]

                    comm_start = time.time()
                    params = self.model.get_params()
                    partitions = _partition_batch(X_batch, y_batch, P)
                    worker_args = [
                        (params, X_p, y_p, layer_sizes)
                        for X_p, y_p in partitions
                    ]
                    comm_time_send = time.time() - comm_start

                    compute_start = time.time()
                    results = pool.map(compute_gradients_worker, worker_args)
                    compute_time = time.time() - compute_start

                    agg_start = time.time()
                    agg_grads, batch_loss = _aggregate_gradients(
                        results, self.model.num_layers
                    )
                    comm_time_agg = time.time() - agg_start

                    self.velocity = self.model.apply_gradients(
                        agg_grads, self.lr, self.momentum, self.velocity
                    )

                    epoch_loss += batch_loss
                    num_batches += 1
                    epoch_compute_time += compute_time
                    epoch_comm_time += comm_time_send + comm_time_agg

                self._record_epoch(
                    epoch, epochs, label, epoch_start,
                    epoch_loss, num_batches, epoch_compute_time,
                    epoch_comm_time, X_train, y_train, X_test, y_test,
                    history, verbose,
                )

    # ------------------------------------------------------------------
    # Shared epoch bookkeeping
    # ------------------------------------------------------------------
    def _record_epoch(self, epoch, epochs, label, epoch_start,
                      epoch_loss, num_batches, epoch_compute_time,
                      epoch_comm_time, X_train, y_train, X_test, y_test,
                      history, verbose):
        epoch_time = time.time() - epoch_start

        avg_loss = epoch_loss / num_batches
        train_acc = self.model.accuracy(X_train, y_train)
        test_acc = self.model.accuracy(X_test, y_test)

        activations_test, _ = self.model.forward(X_test)
        test_loss = self.model.compute_loss(activations_test[-1], y_test)

        history["train_loss"].append(avg_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["epoch_times"].append(epoch_time)
        history["compute_times"].append(epoch_compute_time)
        history["comm_times"].append(epoch_comm_time)

        if verbose:
            print(
                f"  [{label}] Epoch {epoch:3d}/{epochs} | "
                f"Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s "
                f"(compute: {epoch_compute_time:.2f}s, "
                f"comm: {epoch_comm_time:.3f}s)"
            )
