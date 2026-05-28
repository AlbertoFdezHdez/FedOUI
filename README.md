# FedOUI for Federated Learning

This repository contains a PyTorch prototype for studying OUI-based client weighting in federated learning.

FedOUI modifies the server aggregation weights by combining local sample size with a structural OUI score:

```text
w_k^t proportional to n_k (epsilon + s_k^t)^gamma
```

The implementation includes the main federated baselines used in the experiments:

- FedAvg
- FedProx
- FedAlign
- FedOUI

The repository also includes scripts to reproduce the experimental summaries and paper-ready figures.

## Repository Layout

```text
code/                 Training, aggregation, datasets, models, and analysis scripts
display/code/         Legacy plotting and table-generation utilities
results/tables/       Sanitized aggregate result tables
results/figures/      Paper-ready figures in PDF and PNG
reports/              Text summaries of the reported experiments
requirements.txt      Python dependencies
```

Raw datasets, local run folders, logs, caches, and packaged zip artifacts are intentionally excluded.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Basic Usage

Run a single experiment:

```bash
python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedoui --seed 1
```

Run a short smoke test:

```bash
python code/main.py --config code/config/cifar10_dirichlet_0.1.yaml --method fedavg --rounds 1
```

Generate the paper-ready figures from the included aggregate tables:

```bash
python code/analysis/make_paper_figures.py
```

## Notes

- Dataset files are not included. The loaders use torchvision datasets where available.
- Large local run directories are excluded from the release folder.
- The included result tables have been sanitized to remove machine-specific absolute paths.
