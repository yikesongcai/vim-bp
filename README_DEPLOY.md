# VIM-BP Federated Unlearning Deployment Guide

This guide describes how to deploy and run the **Verifiable Incentive Mechanism with Budget Path (VIM-BP)** experiment on a new server, sharing existing datasets.

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd easyFL-FLGo
```

### 2. Prepare Environment
We recommend using Conda:
```bash
conda create -n vim_bp python=3.9
conda activate vim_bp
pip install -r requirements.txt
```

### 3. Share Existing Dataset
If you have the CIFAR-10 dataset stored in an external directory (e.g., `/opt/data/cifar/cifar-10-batches-py`), you can point the project to it using environment variables.

**Recommended for your server setup:**
```bash
# Directly point to the folder containing 'cifar-10-batches-py'
export FLGO_CIFAR10_PATH=/opt/data/cifar
```

**Alternative (Standard FLGo structure):**
If you have a root folder with multiple datasets:
```bash
export FLGO_DATA_ROOT=/opt/data
# This expects datasets at /opt/data/CIFAR10, /opt/data/MNIST, etc.
```

# On Windows (PowerShell)
$env:FLGO_DATA_ROOT="C:\path\to\data"
```

### 4. Run Experiment
Start a test run with 10 clients and 5 rounds:
```bash
python vim_bp/run_vim_experiment.py --num_rounds 5 --num_clients 10 --gpu 0
```

For full-scale experiments:
```bash
python vim_bp/run_vim_experiment.py --num_rounds 50 --num_clients 100 --loss_threshold 1.5
```

### 5. Visualize Results
Generate analysis plots:
```bash
python vim_bp/plot_results.py --results_dir ./vim_results
```

## 📂 Project Structure
- `vim_bp/`: VIM-BP implementation (Server, Client, MAB, Radioactive Data).
- `flgo/`: Base federated learning framework.
- `requirements.txt`: Python dependencies.
- `.gitignore`: Files excluded from the repository.
