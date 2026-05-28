# Reproducibility Notes

The main reported experiments compare FedAvg, FedProx, FedAlign, and FedOUI under two CIFAR-10 federated settings:

- Strong non-IID Dirichlet partitioning with alpha = 0.1
- Noisy-client setting

The aggregate tables in `results/tables/` summarize the runs used to generate the figures in `results/figures/`.

## Figure Generation

To regenerate the paper-ready figures:

```bash
python code/analysis/make_paper_figures.py
```

The script expects the aggregate CSV files under:

```text
results/tables/
```

If running directly from the original development layout, place or symlink the tables under `display/results/tables/`.

## Excluded Artifacts

The following are deliberately not included in this release folder:

- Raw datasets
- Per-run folders
- Local logs
- Python caches
- Packaged zip artifacts

These files are either too large for a lightweight repository or machine-specific.
