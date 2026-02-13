"""
main.py
-------
Main experiment runner for Parallel Mini-Batch SGD.

Runs sequential and parallel training, collects metrics, prints comparison,
and generates plots.

Usage:
    python main.py                                         # default: MNIST, 10 epochs
    python main.py --dataset synthetic --epochs 5           # quick test
    python main.py --workers 2 4 8 --strategy threading     # threading strategy
    python main.py --workers 2 4 --strategy multiprocessing # multiprocessing strategy
"""

# !! IMPORTANT: Limit NumPy's internal BLAS threading so our explicit
# parallelism (threads / processes) controls core utilisation.
# These MUST be set BEFORE importing numpy.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import sys
import time
import numpy as np

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.neural_network import NeuralNetwork
from src.data_loader import load_mnist, load_synthetic
from src.sequential_trainer import SequentialTrainer
from src.parallel_trainer import ParallelTrainer
from src.metrics import print_comparison_table, generate_report


def plot_results(seq_history, par_histories, save_dir="results"):
    """Generate and save comparison plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [Warning] matplotlib not installed — skipping plots.")
        return

    os.makedirs(save_dir, exist_ok=True)
    epochs_range = range(1, len(seq_history["train_loss"]) + 1)

    # ---- Plot 1: Training Loss Comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, seq_history["train_loss"], "k-o",
            label="Sequential", linewidth=2)
    for P in sorted(par_histories.keys()):
        ax.plot(epochs_range, par_histories[P]["train_loss"], "-s",
                label=f"Parallel P={P}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss: Sequential vs Parallel SGD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_loss.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/training_loss.png")

    # ---- Plot 2: Test Accuracy Comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, seq_history["test_acc"], "k-o",
            label="Sequential", linewidth=2)
    for P in sorted(par_histories.keys()):
        ax.plot(epochs_range, par_histories[P]["test_acc"], "-s",
                label=f"Parallel P={P}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy: Sequential vs Parallel SGD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "test_accuracy.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/test_accuracy.png")

    # ---- Plot 3: Epoch Time Comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_range, seq_history["epoch_times"], "k-o",
            label="Sequential", linewidth=2)
    for P in sorted(par_histories.keys()):
        ax.plot(epochs_range, par_histories[P]["epoch_times"], "-s",
                label=f"Parallel P={P}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time per Epoch (s)")
    ax.set_title("Epoch Time: Sequential vs Parallel SGD")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "epoch_times.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/epoch_times.png")

    # ---- Plot 4: Speedup Bar Chart ----
    worker_counts = sorted(par_histories.keys())
    speedups = []
    seq_total = seq_history["total_time"]
    for P in worker_counts:
        speedups.append(seq_total / par_histories[P]["total_time"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(worker_counts))
    bars = ax.bar(x_pos, speedups, color="steelblue", edgecolor="black")
    ax.plot(x_pos, worker_counts, "r--", marker="^", label="Ideal (linear)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"P={P}" for P in worker_counts])
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Number of Workers")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels on bars
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{s:.2f}x", ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "speedup.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/speedup.png")

    # ---- Plot 5: Communication Overhead ----
    fig, ax = plt.subplots(figsize=(10, 6))
    for P in worker_counts:
        h = par_histories[P]
        comm_pct = [
            (c / t * 100) if t > 0 else 0
            for c, t in zip(h["comm_times"], h["epoch_times"])
        ]
        ax.plot(epochs_range, comm_pct, "-s",
                label=f"P={P}", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Communication Overhead (%)")
    ax.set_title("Communication Overhead as % of Epoch Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "comm_overhead.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_dir}/comm_overhead.png")


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Mini-Batch SGD Experiment"
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "synthetic"],
        help="Dataset to use (default: mnist)"
    )
    parser.add_argument(
        "--num_samples", type=int, default=None,
        help="Subsample dataset to this many samples (None = full)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Mini-batch size (default: 256)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.1,
        help="Learning rate (default: 0.1)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9,
        help="Momentum coefficient (default: 0.9)"
    )
    parser.add_argument(
        "--workers", type=int, nargs="+", default=[2, 4],
        help="List of worker counts to test (default: 2 4)"
    )
    parser.add_argument(
        "--strategy", type=str, default="threading",
        choices=["threading", "multiprocessing"],
        help="Parallelism strategy (default: threading)"
    )
    parser.add_argument(
        "--hidden", type=int, nargs="+", default=[256, 128],
        help="Hidden layer sizes (default: 256 128)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ---- Load Data ----
    print("\n" + "=" * 60)
    print("PARALLEL MINI-BATCH SGD EXPERIMENT")
    print("=" * 60)

    if args.dataset == "mnist":
        X_train, y_train, X_test, y_test = load_mnist(
            num_samples=args.num_samples, seed=args.seed
        )
    else:
        X_train, y_train, X_test, y_test = load_synthetic(
            num_samples=args.num_samples or 20000, seed=args.seed
        )

    input_size = X_train.shape[1]
    num_classes = y_train.shape[1]
    layer_sizes = [input_size] + args.hidden + [num_classes]

    print(f"\nModel architecture: {layer_sizes}")
    total_params = sum(
        layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
        for i in range(len(layer_sizes) - 1)
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Batch size: {args.batch_size} | LR: {args.lr} | "
          f"Momentum: {args.momentum} | Epochs: {args.epochs}")
    print(f"Workers to test: {args.workers} | Strategy: {args.strategy}")

    # ---- Sequential Training ----
    print("\n" + "-" * 60)
    print("SEQUENTIAL TRAINING (Baseline)")
    print("-" * 60)

    model_seq = NeuralNetwork(layer_sizes, seed=args.seed)
    trainer_seq = SequentialTrainer(model_seq, lr=args.lr, momentum=args.momentum)
    seq_history = trainer_seq.train(
        X_train, y_train, X_test, y_test,
        epochs=args.epochs, batch_size=args.batch_size
    )

    # ---- Parallel Training ----
    par_histories = {}

    for P in args.workers:
        print(f"\n{'-' * 60}")
        print(f"PARALLEL TRAINING (P={P} workers, {args.strategy})")
        print("-" * 60)

        # Use the SAME initial weights for fair comparison
        model_par = NeuralNetwork(layer_sizes, seed=args.seed)
        trainer_par = ParallelTrainer(
            model_par, lr=args.lr, momentum=args.momentum,
            num_workers=P, strategy=args.strategy,
        )
        par_history = trainer_par.train(
            X_train, y_train, X_test, y_test,
            epochs=args.epochs, batch_size=args.batch_size
        )
        par_histories[P] = par_history

    # ---- Results ----
    print_comparison_table(seq_history, par_histories)

    # ---- Generate Report ----
    report = generate_report(seq_history, par_histories,
                             args.batch_size, args.epochs)
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "experiment_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\n  Report saved to: {report_path}")

    # ---- Generate Plots ----
    print("\nGenerating plots...")
    plot_results(seq_history, par_histories)

    print("\nDone!")


if __name__ == "__main__":
    main()
