
# AutoPrep-Uplift

> LLM-augmented pipeline for uplift modeling in marketing A/B tests.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()
[![Site](https://img.shields.io/badge/site-GitHub%20Pages-purple.svg)](https://allisksks.github.io/autoprep-uplift/)

## What is this?

A research pipeline that solves a real problem: applying uplift modeling to any marketing A/B test dataset without manual preprocessing.

**The core idea:** instead of writing custom preprocessing code for each new dataset, an LLM agent analyzes the data, generates `fit_preprocess()` + `apply_preprocess()` functions, self-validates them, and iteratively fixes errors — all without data leakage.

Then 6 meta-learners are benchmarked, anti-overfitting checks run automatically, and an LLM agent recommends the top-3 models with explanations.

## Benchmark results

Results on proprietary retail dataset (355K rows, 90% zero-inflated outcome):

| Model | CV point | CV lower CI | Notes |
|-------|----------|-------------|-------|
| **DR-learner** | 21.97 | **16.37** | Best — doubly robust |
| R-learner | 18.78 | 13.15 | Quasi-oracle property |
| T-learner Ridge | 18.67 | 12.98 | Linear, stable |
| Hurdle | 17.32 | 12.87 | Best for zero-inflated |
| X-learner | 18.73 | 12.64 | No gain on 50/50 data |
| T-learner LGB | 17.69 | 11.59 | High variance |

Metric: uplift@10, lower bound of 80% bootstrap CI. Fast mode: 100K sample, 3-fold CV.

## Quick start

### Windows

```powershell
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift

# создать окружение
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# установить зависимости
py -m pip install -r requirements.txt

# добавить API ключ
copy .env.example .env
notepad .env  # вставь ANTHROPIC_API_KEY=sk-ant-...

# запустить валидацию данных
py run_validation.py

# запустить сравнение моделей
py run_compare.py

# запустить ноутбук
jupyter notebook experiments\00_magnit_baseline.ipynb
```

### macOS / Linux

```bash
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift

# создать окружение
python3 -m venv .venv
source .venv/bin/activate

# установить зависимости
pip install -r requirements.txt

# добавить API ключ
cp .env.example .env
nano .env  # вставь ANTHROPIC_API_KEY=sk-ant-...

# запустить валидацию данных
python run_validation.py

# запустить сравнение моделей
python run_compare.py

# запустить ноутбук
jupyter notebook experiments/00_magnit_baseline.ipynb
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
LLM Agent: top-3 selection # explains trade-offs, recommends ensemble strategy
      ↓
Ensemble (6 strategies)    # auto-selects best: equal/gap/ci/rank/pairwise/single
      ↓
predictions.csv
```

## Models

| Model | Method | Reference | Best for |
|-------|--------|-----------|----------|
| **DR-Learner** | Doubly robust pseudo-outcomes | Kennedy (2023) | General case |
| T-Learner LGB | Two separate LightGBM | Künzel et al. (2019) | Fast baseline |
| T-Learner Ridge | Two separate Ridge | Künzel et al. (2019) | Linear baseline |
| X-Learner | Two-stage imputation | Künzel et al. (2019) | Imbalanced groups |
| R-Learner | Robinson decomposition | Nie & Wager (2021) | Theoretically optimal |
| Hurdle | P(Y>0) × E[Y\|Y>0] | Devriendt et al. (2022) | Zero-inflated outcomes |

## Metrics

```python
from uplift.metrics import evaluate, evaluate_all

# одна метрика
evaluate(y, w, scores, metric='uplift@10')  # lower 80% CI
evaluate(y, w, scores, metric='uplift@5')
evaluate(y, w, scores, metric='auuc')
evaluate(y, w, scores, metric='qini')

# все сразу
evaluate_all(y, w, scores)
# → {'uplift@5': ..., 'uplift@10': ..., 'uplift@20': ..., 'auuc': ..., 'qini': ...}
```

## Ensemble strategies

```python
from uplift.ensemble import UpliftEnsemble

# auto-selects best strategy on validation data
ensemble = UpliftEnsemble(strategy='auto', metric='uplift@10')

# available strategies:
# 'equal_weights' | 'gap_weights' | 'ci_weights'
# 'rank_weights'  | 'best_single' | 'pairwise_best'
```

## Validation & anti-overfitting

```python
from uplift import full_validation_report, permutation_test, repeated_cv

# перед обучением: проверка данных
report = full_validation_report(train_df, 'treatment_flg', 'rec_spend')
# → randomization check, leakage detection, balance, ATE

# после обучения: проверка значимости
result = permutation_test(y_val, w_val, scores, n_permutations=200)
# → p-value, null distribution

# надёжная CV оценка
result = repeated_cv(X, y, w, model_fn, n_repeats=3, n_folds=3)
# → mean ± std по нескольким разбиениям
```

## Datasets

| Dataset | Size | Outcome | Source |
|---------|------|---------|--------|
| Hillstrom Email | 64K | continuous spend | MineThatData 2008 |
| Criteo Uplift v2 | 13.98M | binary visit | Diemert et al. 2018 |
| Lenta Retail | ~687K | binary response | scikit-uplift |
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
│   ├── base.py
│   ├── dr_learner.py
│   ├── t_learner.py
│   ├── x_learner.py
│   ├── r_learner.py
│   └── hurdle.py
└── agent/
    ├── eda_agent.py       # LLM agent for preprocessing
    └── model_selector.py  # LLM agent for top-3 selection

experiments/
├── 00_magnit_baseline.ipynb
└── results/
    ├── tables/    # cv_results.csv
    └── figures/   # model_comparison.png, stability_check.png

docs/              # GitHub Pages site
run_validation.py  # быстрый запуск валидации
run_compare.py     # быстрый запуск сравнения моделей
run_plots.py       # генерация графиков
```

## Status

Work in progress. Paper in preparation.

## Citation

```bibtex
@software{autoprep_uplift_2026,
  author = {Desyatnikova, Alisa},
  title  = {AutoPrep-Uplift: LLM-Augmented Pipeline for Uplift Modeling},
  year   = {2026},
  url    = {https://github.com/allisksks/autoprep-uplift}
}
```


