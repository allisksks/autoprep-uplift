
# AutoPrep-Uplift

> LLM-augmented pipeline for uplift modeling in marketing A/B tests.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![Site](https://img.shields.io/badge/site-GitHub%20Pages-purple.svg)](https://allisksks.github.io/autoprep-uplift/)

## What is this?

A research pipeline that automates uplift modeling across any marketing A/B test dataset. An LLM agent analyzes data, generates preprocessing code, benchmarks 6 meta-learners, evaluates 108 ensemble combinations, and selects the optimal strategy — all without manual feature engineering.

**Live site:** https://allisksks.github.io/autoprep-uplift/

## Benchmark results

Results across 5 datasets. Metric: uplift@10, lower 80% bootstrap CI.

| Dataset | Size | Balance | Winner (CV) | Ensemble | Holdout CI | p-value |
|---------|------|---------|-------------|----------|------------|---------|
| **Magnit** (private) | 355K | 50/50 | DR-learner (16.37) | T-Ridge + T-LGB | 21.51 | 0.0000 |
| **Hillstrom** | 64K | 67/33 | T-Ridge (0.071) | T-Ridge + Hurdle | 0.100 | 0.0050 |
| **Lenta** | 550K | 75/25 | DR-learner (0.012) | DR + T-LGB + X | 0.025 | 0.0000 |
| **Megafon** | 600K | 50/50 | Hurdle (0.382) | T-LGB + T-Ridge + Hurdle | 0.452 | 0.0000 |
| **Synthetic** | 50K | 50/50 | R-learner (0.904) | T-Ridge + X + R | 0.895 | 0.0000 |

**Key findings:**
- DR-learner wins on balanced data (Magnit, Lenta) — doubly robust property matters
- T-Ridge wins on imbalanced data (Hillstrom 67/33) — propensity models suffer from imbalance
- Hurdle wins on telecom data (Megafon) — two-stage modeling rewards higher signal
- CV ranking ≠ ensemble contribution: X-learner last in CV on Lenta but in winning ensemble
- Ensemble consistently outperforms best single model by 10–38%

## Quick start

### Windows

```powershell
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
copy .env.example .env  # add ANTHROPIC_API_KEY
py run_pipeline.py --dataset hillstrom
```

### macOS / Linux

```bash
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # add ANTHROPIC_API_KEY
python run_pipeline.py --dataset hillstrom
```

## Run on any dataset

```powershell
py run_pipeline.py --dataset hillstrom
py run_pipeline.py --dataset lenta
py run_pipeline.py --dataset megafon
py run_pipeline.py --dataset synthetic
py run_pipeline.py --dataset magnit    # requires private data
```

Optional flags:
```powershell
py run_pipeline.py --dataset hillstrom --metric auuc --cv_folds 5 --no_fast
```

## Pipeline

```
Raw A/B dataset
      ↓
Schema validation          # checks treatment_col, outcome_col, user_id
      ↓
Data validation            # randomization check, leakage detection, balance
      ↓
LLM Agent: EDA             # analyzes dtypes, missingness, distributions
      ↓
LLM Agent: code generation # generates fit_preprocess() + apply_preprocess()
      ↓
Self-verification          # runs code on sample, fixes errors iteratively
      ↓
Train 6 meta-learners      # DR / T-LGB / T-Ridge / X / R / Hurdle
      ↓
CV + anti-overfitting      # k-fold, bootstrap CI, permutation test, repeated CV
      ↓
Evaluate 108 combinations  # single / pairs / triples / all-6, all strategies
      ↓
LLM Agent: ensemble select # sees all results, picks optimal with explanation
      ↓
predictions.csv
```

## Models

| Model | Method | Reference | Best for |
|-------|--------|-----------|----------|
| **DR-Learner** | Doubly robust pseudo-outcomes | Kennedy (2023) | Balanced data |
| T-Learner LGB | Two separate LightGBM | Künzel et al. (2019) | Fast strong baseline |
| T-Learner Ridge | Two separate Ridge | Künzel et al. (2019) | Imbalanced groups |
| X-Learner | Two-stage imputation | Künzel et al. (2019) | Ensemble diversity |
| R-Learner | Robinson decomposition | Nie & Wager (2021) | Theoretically optimal |
| **Hurdle** | P(Y>0) × E[Y\|Y>0] | Devriendt et al. (2022) | Zero-inflated outcomes |

## Metrics

```python
from uplift.metrics import evaluate, evaluate_all

evaluate(y, w, scores, metric='uplift@10')  # lower 80% CI
evaluate(y, w, scores, metric='uplift@5')
evaluate(y, w, scores, metric='auuc')
evaluate(y, w, scores, metric='qini')
evaluate_all(y, w, scores)  # all metrics at once
```

## Validation & anti-overfitting

```python
from uplift import full_validation_report, permutation_test

# before training
report = full_validation_report(train_df, 'treatment_flg', 'rec_spend')

# after training
perm = permutation_test(y_val, w_val, scores, n_permutations=200)
```

Five checks: randomization (t-test/chi-square), leakage detection, group balance, permutation test, repeated CV.

## Project structure

```
uplift/
├── metrics.py       # uplift@K, AUUC, Qini, evaluate(), evaluate_all()
├── pipeline.py      # UpliftPipeline — CV across all models
├── ensemble.py      # UpliftEnsemble — 6 weighting strategies
├── validation.py    # randomization, leakage, permutation test
├── models/
│   ├── base.py
│   ├── dr_learner.py
│   ├── t_learner.py
│   ├── x_learner.py
│   ├── r_learner.py
│   └── hurdle.py
└── agent/
    ├── eda_agent.py       # LLM agent: EDA + preprocessing
    └── model_selector.py  # LLM agent: ensemble selection

experiments/results/
├── magnit/     # figures, tables, predictions
├── hillstrom/
├── lenta/
├── megafon/
└── synthetic/

run_pipeline.py      # universal runner — all datasets
run_full_pipeline.py # full run with LLM agents on Magnit
docs/                # GitHub Pages site
```

## Datasets

| Dataset | Rows | Outcome | Source |
|---------|------|---------|--------|
| Hillstrom Email | 64K | continuous spend | `sklift.datasets.fetch_hillstrom()` |
| Lenta Retail | ~687K | binary response | `sklift.datasets.fetch_lenta()` |
| Megafon Telecom | 600K | binary response | `sklift.datasets.fetch_megafon()` |
| Synthetic | 50K | continuous (known CATE) | numpy generator |
| Magnit Retail | 355K | continuous, 90% zeros | private RCT |

## Branch strategy

```
main        ← stable releases only
dev         ← integration branch
feature/*   ← one branch per task, deleted after merge
```

## Status

Work in progress. Paper in preparation.

## Contact

Interested in running AutoPrep-Uplift on your data or collaborating?

Telegram: [@alli1ice](https://t.me/alli1ice)

## Citation

```bibtex
@software{autoprep_uplift_2026,
  author = {Desyatnikova, Alisa},
  title  = {AutoPrep-Uplift: LLM-Augmented Pipeline for Uplift Modeling},
  year   = {2026},
  url    = {https://github.com/allisksks/autoprep-uplift}
}
```
