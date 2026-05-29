# AutoPrep-Uplift

> LLM-augmented pipeline for uplift modeling in marketing A/B tests.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

## What is this?

A research pipeline that solves a real problem: applying uplift modeling to any marketing A/B test dataset without manual preprocessing. 

**The core idea:** instead of writing custom preprocessing code for each new dataset, an LLM agent analyzes the data, generates `fit_preprocess()` + `apply_preprocess()` functions, self-validates them, and iteratively fixes errors — all without data leakage.

Then 4 meta-learners are benchmarked, anti-overfitting checks run automatically, and an LLM agent recommends the top-3 models with explanations.

## Why uplift modeling?

Classic response models predict *who will buy*. Uplift models predict *whose behavior will change because of the promotion*. This distinction matters:

| Segment | With promo | Without promo | Send promo? |
|---------|-----------|---------------|-------------|
| Persuadables | Buy | Don't buy | ✓ Yes |
| Sure things | Buy | Buy | ✗ Wasted budget |
| Lost causes | Don't buy | Don't buy | ✗ Wasted budget |
| Sleeping dogs | Don't buy | Buy | ✗ Harmful |

## Quick start

```powershell
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
copy .env.example .env  # add ANTHROPIC_API_KEY
jupyter notebook experiments/00_magnit_baseline.ipynb
```

## Pipeline

```
Raw A/B dataset
      ↓
Schema validation          # checks treatment_col, outcome_col, user_id
      ↓
LLM Agent: EDA             # analyzes dtypes, missingness, distributions
      ↓
LLM Agent: code generation # generates fit_preprocess() + apply_preprocess()
      ↓
Self-verification          # runs code on sample, fixes errors iteratively
      ↓
Data validation            # randomization check, leakage detection
      ↓
Train 4 meta-learners      # DR / T / Hurdle / T-Ridge
      ↓
CV + anti-overfitting      # k-fold, bootstrap CI, permutation test, repeated CV
      ↓
LLM Agent: top-3 selection # explains trade-offs, recommends ensemble strategy
      ↓
Ensemble (6 strategies)    # auto-selects best: equal/gap/ci/rank/pairwise/single
      ↓
predictions.csv
```

## Models

| Model | Method | Best for |
|-------|--------|----------|
| DR-Learner | Doubly robust pseudo-outcomes | General case, best empirical results |
| T-Learner LGB | Two separate LightGBM models | Fast strong baseline |
| T-Learner Ridge | Two separate Ridge models | Very fast linear baseline |
| X-Learner | Two-stage counterfactual imputation | Imbalanced groups (n_treat ≪ n_ctrl) |
| R-Learner | Robinson decomposition + R-loss | Theoretically optimal, quasi-oracle |
| Hurdle | P(Y>0) × E[Y\|Y>0] | Zero-inflated outcomes (~90% zeros) |

## Metrics

| Metric | Description |
|--------|-------------|
| `uplift@K` | Mean difference in top-K% by score. Lower 80% CI used as primary. |
| `auuc` | Area Under Uplift Curve — integral quality across full population. |
| `qini` | Qini coefficient — area between model curve and random baseline. |

All metrics support dynamic selection:
```python
evaluate(y, w, scores, metric='uplift@10')
evaluate(y, w, scores, metric='auuc')
evaluate_all(y, w, scores)  # all metrics at once
```

## Ensemble strategies

```python
# Auto-selects best strategy on validation data
ensemble = UpliftEnsemble(strategy='auto', metric='uplift@10')

# Or pick manually:
# 'equal_weights' | 'gap_weights' | 'ci_weights'
# 'rank_weights'  | 'best_single' | 'pairwise_best'
ensemble = UpliftEnsemble(strategy='ci_weights', metric='auuc')
```

## Validation & anti-overfitting

```python
from uplift import full_validation_report, permutation_test, repeated_cv

# Before training: check data quality
report = full_validation_report(train_df, 'treatment_flg', 'rec_spend')

# After training: check model significance
result = permutation_test(y_val, w_val, scores, n_permutations=200)

# More robust CV estimate
result = repeated_cv(X, y, w, model_fn, n_repeats=3, n_folds=3)
```

## Datasets

| Dataset | Size | Outcome | Source |
|---------|------|---------|--------|
| Hillstrom Email | 64K | continuous spend | MineThatData 2008 |
| Criteo Uplift v2 | 13.98M | binary visit | Diemert et al. 2018 |
| Lenta Retail | ~687K | binary response | Lenta / scikit-uplift |
| Starbucks | 84K | binary transaction | Udacity |
| Proprietary retail | 355K | continuous, 90% zeros | Private RCT |

## Branch strategy

```
main        ← stable releases only
dev         ← integration branch
feature/*   ← one branch per task, deleted after merge
```

## Project structure

```
uplift/
├── metrics.py       # uplift@K, AUUC, Qini, evaluate(), evaluate_all()
├── pipeline.py      # UpliftPipeline — main entry point
├── ensemble.py      # UpliftEnsemble — 6 strategies + auto
├── validation.py    # randomization check, leakage, permutation test
├── models/
│   ├── base.py      # BaseUpliftModel interface
│   ├── dr_learner.py
│   ├── t_learner.py
│   └── hurdle.py
└── agent/
    ├── eda_agent.py      # LLM agent for preprocessing
    └── model_selector.py # LLM agent for top-3 selection

experiments/
├── 00_magnit_baseline.ipynb
└── results/

docs/                # GitHub Pages site
```

## Status

Work in progress. Paper in preparation.

## Citation

If you use this work, please cite:
```bibtex
@software{autoprep_uplift_2026,
  author = {Desyatnikova, Alisa},
  title  = {AutoPrep-Uplift: LLM-Augmented Pipeline for Uplift Modeling},
  year   = {2026},
  url    = {https://github.com/allisksks/autoprep-uplift}
}
```

