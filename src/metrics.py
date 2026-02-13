"""
metrics.py
----------
Utilities for computing and reporting performance metrics:
  - Speedup
  - Efficiency
  - Communication overhead breakdown
  - Convergence comparison
"""

import numpy as np


def compute_speedup(seq_time, par_time):
    """Speedup = T_seq / T_par."""
    if par_time == 0:
        return float("inf")
    return seq_time / par_time


def compute_efficiency(speedup, num_workers):
    """Efficiency = Speedup / P."""
    return speedup / num_workers


def compute_comm_fraction(comm_times, epoch_times):
    """Fraction of each epoch spent on communication / sync."""
    return [c / t if t > 0 else 0.0 for c, t in zip(comm_times, epoch_times)]


def print_comparison_table(seq_history, par_histories):
    """
    Print a summary comparison table.

    Parameters
    ----------
    seq_history : dict
        History from the sequential trainer.
    par_histories : dict[int, dict]
        Map from num_workers → history from parallel trainer.
    """
    seq_total = seq_history["total_time"]
    seq_final_acc = seq_history["test_acc"][-1]
    seq_final_loss = seq_history["train_loss"][-1]

    print("\n" + "=" * 85)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 85)
    print(
        f"{'Config':<20} {'Total Time (s)':>14} {'Speedup':>10} "
        f"{'Efficiency':>12} {'Test Acc':>10} {'Train Loss':>12}"
    )
    print("-" * 85)
    print(
        f"{'Sequential':<20} {seq_total:>14.2f} {'1.00':>10} "
        f"{'1.00':>12} {seq_final_acc:>10.4f} {seq_final_loss:>12.4f}"
    )

    for P in sorted(par_histories.keys()):
        h = par_histories[P]
        par_total = h["total_time"]
        speedup = compute_speedup(seq_total, par_total)
        efficiency = compute_efficiency(speedup, P)
        final_acc = h["test_acc"][-1]
        final_loss = h["train_loss"][-1]

        print(
            f"{'Parallel P=' + str(P):<20} {par_total:>14.2f} {speedup:>10.2f} "
            f"{efficiency:>12.2f} {final_acc:>10.4f} {final_loss:>12.4f}"
        )

    print("=" * 85)

    # Communication overhead for each parallel config
    print("\nCOMMUNICATION OVERHEAD (average per epoch)")
    print("-" * 65)
    print(f"{'Config':<20} {'Avg Comm (s)':>14} {'Avg Epoch (s)':>14} {'% Overhead':>12}")
    print("-" * 65)

    for P in sorted(par_histories.keys()):
        h = par_histories[P]
        avg_comm = np.mean(h["comm_times"])
        avg_epoch = np.mean(h["epoch_times"])
        pct = (avg_comm / avg_epoch * 100) if avg_epoch > 0 else 0.0

        print(
            f"{'Parallel P=' + str(P):<20} {avg_comm:>14.4f} "
            f"{avg_epoch:>14.2f} {pct:>11.1f}%"
        )

    print("-" * 65)


def generate_report(seq_history, par_histories, batch_size, epochs):
    """
    Generate a text report summarizing the experiment.

    Returns
    -------
    report : str
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EXPERIMENT REPORT")
    lines.append(f"Batch Size: {batch_size} | Epochs: {epochs}")
    lines.append("=" * 70)

    seq_total = seq_history["total_time"]

    lines.append(f"\nSequential Training:")
    lines.append(f"  Total Time     : {seq_total:.2f}s")
    lines.append(f"  Final Train Loss: {seq_history['train_loss'][-1]:.4f}")
    lines.append(f"  Final Test Acc  : {seq_history['test_acc'][-1]:.4f}")

    for P in sorted(par_histories.keys()):
        h = par_histories[P]
        speedup = compute_speedup(seq_total, h["total_time"])
        efficiency = compute_efficiency(speedup, P)

        lines.append(f"\nParallel Training (P={P}):")
        lines.append(f"  Total Time     : {h['total_time']:.2f}s")
        lines.append(f"  Speedup        : {speedup:.2f}x")
        lines.append(f"  Efficiency     : {efficiency:.2f}")
        lines.append(f"  Final Train Loss: {h['train_loss'][-1]:.4f}")
        lines.append(f"  Final Test Acc  : {h['test_acc'][-1]:.4f}")
        lines.append(
            f"  Avg Comm Time  : {np.mean(h['comm_times']):.4f}s/epoch"
        )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)
