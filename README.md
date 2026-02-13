# Parallelization of Mini-Batch SGD — P2 Implementation

## Overview

This project implements **data-parallel Mini-Batch Stochastic Gradient Descent (SGD)** using
Python's `multiprocessing` module. A Multi-Layer Perceptron (MLP) is built from scratch with
NumPy, and gradient computation is parallelized across multiple CPU cores.

## Project Structure

```
Assignment2/
├── main.py                      # Main experiment runner
├── P1_revised_design.md         # Revised design document
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── src/
│   ├── __init__.py
│   ├── neural_network.py        # MLP: forward, backward, gradient computation
│   ├── data_loader.py           # MNIST / synthetic data loading
│   ├── sequential_trainer.py    # Sequential mini-batch SGD (baseline)
│   ├── parallel_trainer.py      # Parallel mini-batch SGD (multiprocessing)
│   └── metrics.py               # Performance metrics and reporting
└── results/                     # Generated plots and reports (after running)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

### Default (MNIST, 10 epochs, batch size 256, workers 2 & 4)

```bash
python main.py
```

### Quick test with synthetic data

```bash
python main.py --dataset synthetic --epochs 5 --batch_size 512 --workers 2 4
```

### Full experiment suite

```bash
python main.py --dataset mnist --epochs 20 --batch_size 256 --workers 2 4 8
```

### All options

```
--dataset       mnist | synthetic       Dataset choice (default: mnist)
--num_samples   N                       Subsample to N samples (default: all)
--epochs        N                       Training epochs (default: 10)
--batch_size    N                       Mini-batch size (default: 256)
--lr            F                       Learning rate (default: 0.1)
--momentum      F                       SGD momentum (default: 0.9)
--workers       P1 P2 ...              Worker counts to test (default: 2 4)
--hidden        H1 H2 ...              Hidden layer sizes (default: 256 128)
--seed          N                       Random seed (default: 42)
```

## Output

- **Console**: Per-epoch training logs, comparison table, communication overhead.
- **`results/experiment_report.txt`**: Text summary of all experiments.
- **`results/*.png`**: Plots for loss, accuracy, epoch time, speedup, and communication overhead.

## How It Works

1. **Sequential baseline**: Standard mini-batch SGD processes one batch at a time.
2. **Parallel version**: Each mini-batch is split into P partitions. Worker processes
   compute gradients on their partition simultaneously using `multiprocessing.Pool`.
   Gradients are aggregated (averaged) and a single parameter update is applied.
3. Both use the same initial weights, learning rate, and data order for fair comparison.
