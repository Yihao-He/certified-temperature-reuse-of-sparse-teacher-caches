# Reproducibility files: Idea 2

These files regenerate the statistical plots and summary tables from the included compact row-level results. The rows come from 1,024 frozen-model logit records (512 per prompt domain). The original full-vocabulary logit array is not included; rerunning Qwen inference or the row-level extraction requires that separate 622 MB source array and model setup.

## Run

Python 3.10+, NumPy, and Matplotlib are required. From this directory:

```bash
python figures/make_tables.py
python figures/make_figures.py
```

The scripts write two PDF figures and compact summary/table files under `figures/` and `results/`. Both CSV input files contain the observations used for the plotted distributions; `certificate_and_baseline_summary.json` supplies the recorded certificate values used by the manuscript comparison table.

## Contents

- `figures/`: analysis and plotting scripts.
- `results/identifiability_rows.csv`, `results/local_cache_rows.csv`: compact per-row results.
- `results/certificate_and_baseline_summary.json`: certificate and baseline summaries.

The row tables contain generated-prompt identifiers and token positions, not prompt text. No model weights, credentials, or raw logits are included. The included data regenerate the reported analyses from those saved measurements; they do not rerun model inference.


