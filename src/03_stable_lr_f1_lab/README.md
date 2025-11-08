# 03 · Stable LR F1 Lab

This project freezes the winning recipe from `overall_ranking.csv`: basic
feature engineering, scaled numerics, and one-hot categoricals powering a
logistic regression core.  Instead of endlessly searching models, we only:

1. Run a lightweight hyper-parameter grid across a few regularization settings.
2. Compare the only lever that still seems promising—different up-samplers.

Everything else stays simple, transparent, and ready for hand-crafted tweaks.

## How it works

1. **Manual feature slot** – `add_manual_features` in `stable_lr_f1_lab.py`
   contains the small set of stable ratios/aggregations that performed best so
   far.  Extend this function as you try new ideas.
2. **Feature pipeline** – `build_preprocessor` mirrors the proven pipeline:
   log1p on skewed numerics → scaling → OneHotEncoder with `handle_unknown`.
3. **Sampler sweep** – `sampler_space` toggles between `none`, random over
   sampling, SMOTE variants, and ADASYN.  The model grid stays constant; only
   the sampler link in the pipeline changes.
4. **F1-first scoring** – every `GridSearchCV` uses F1 in 5-fold stratified CV.
   The best estimator (usually plain LR + SMOTE or none) is evaluated on
   `data/test.csv`, including a threshold sweep to squeeze the best possible F1.

## Running it

```bash
python src/03_stable_lr_f1_lab/stable_lr_f1_lab.py
```

Artifacts land under `src/03_stable_lr_f1_lab/artifacts/`:

| Subdir | Contents |
| --- | --- |
| `data/` | `sampler_summary.csv`, full grid results, holdout threshold sweep, predictions, compact JSON report |
| `figures/` | Sampler F1 bars, threshold-vs-F1 curve, confusion matrix (best threshold), PR curve |
| `models/` | `stable_lr_best_<sampler>.pkl` – ready for notebooks or deployment |

## Next manual steps

1. **Feature tweaks** – edit `add_manual_features`; rerun the script to
   immediately see the impact on CV and holdout F1.
2. **Sampler ideas** – drop new samplers into `sampler_space` if you want to
   try custom SMOTE settings or hybrids.
3. **Threshold policy** – the stored `threshold_sweep.csv` + figures make it
   easy to pick an operating point that balances recall/precision while keeping
   F1 in view.

That’s it—zero “热拔插” pipelines, just the dependable LR workflow surfaced by
the ranking file, now wrapped in a reproducible script with clean outputs.
