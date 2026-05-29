"""
run_pipeline.py

Универсальный запуск пайплайна для любого датасета.

Использование:
  py run_pipeline.py --dataset hillstrom
  py run_pipeline.py --dataset magnit
  py run_pipeline.py --dataset lenta
  py run_pipeline.py --dataset criteo
  py run_pipeline.py --dataset starbucks

Параметры:
  --dataset    : название датасета (обязательно)
  --metric     : uplift@10 | uplift@5 | uplift@20 | auuc | qini (default: uplift@10)
  --cv_folds   : число фолдов CV (default: 3)
  --n_boot     : число bootstrap итераций (default: 100)
  --fast       : использовать 100K сэмпл для CV (default: True)
  --no_agent   : пропустить LLM агентов (default: False)
  --holdout    : доля holdout (default: 0.2)
"""

import sys, os, json, argparse
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Парсинг аргументов ────────────────────────────────────────
parser = argparse.ArgumentParser(description='AutoPrep-Uplift Pipeline')
parser.add_argument('--dataset',  type=str, required=True,
                    choices=['magnit','hillstrom','lenta','criteo','starbucks','synthetic'],
                    help='Датасет для запуска')
parser.add_argument('--metric',   type=str, default='uplift@10',
                    help='Метрика: uplift@10, uplift@5, uplift@20, auuc, qini')
parser.add_argument('--cv_folds', type=int, default=3)
parser.add_argument('--n_boot',   type=int, default=100)
parser.add_argument('--fast',     action='store_true', default=True,
                    help='CV на 100K сэмпле')
parser.add_argument('--no_fast',  action='store_true', default=False,
                    help='CV на полных данных')
parser.add_argument('--no_agent', action='store_true', default=False,
                    help='Пропустить LLM агентов')
parser.add_argument('--holdout',  type=float, default=0.2)
args = parser.parse_args()

if args.no_fast:
    args.fast = False

DATASET   = args.dataset
METRIC    = args.metric
CV_FOLDS  = args.cv_folds
N_BOOT    = args.n_boot
FAST      = args.fast
USE_AGENT = not args.no_agent
HOLDOUT   = args.holdout
SEED      = 42

print('=' * 60)
print(f'AUTOPREP-UPLIFT — {DATASET.upper()}')
print('=' * 60)
print(f'  metric={METRIC}  cv_folds={CV_FOLDS}  n_boot={N_BOOT}')
print(f'  fast={FAST}  use_agent={USE_AGENT}  holdout={HOLDOUT}')

# ── Конфиги датасетов ────────────────────────────────────────
DATASET_CONFIGS = {
    'magnit': {
        'treatment_col': 'treatment_flg',
        'outcome_col':   'rec_spend',
        'cat_cols':      ['communication_type'],
        'out_dir':       'experiments/results/magnit',
    },
    'hillstrom': {
        'treatment_col': 'treatment',
        'outcome_col':   'spend',
        'cat_cols':      ['history_segment', 'zip_code', 'channel'],
        'out_dir':       'experiments/results/hillstrom',
    },
    'lenta': {
        'treatment_col': 'treatment_flg',
        'outcome_col':   'response_att',
        'cat_cols':      [],
        'out_dir':       'experiments/results/lenta',
    },
    'criteo': {
        'treatment_col': 'treatment',
        'outcome_col':   'conversion',
        'cat_cols':      [],
        'out_dir':       'experiments/results/criteo',
    },
    'starbucks': {
        'treatment_col': 'treatment',
        'outcome_col':   'purchase',
        'cat_cols':      [],
        'out_dir':       'experiments/results/starbucks',
    },
    'synthetic': {
        'treatment_col': 'treatment',
        'outcome_col':   'outcome',
        'cat_cols':      [],
        'out_dir':       'experiments/results/synthetic',
    },
}

cfg = DATASET_CONFIGS[DATASET]
os.makedirs(cfg['out_dir'] + '/tables',  exist_ok=True)
os.makedirs(cfg['out_dir'] + '/figures', exist_ok=True)

# ── Загрузчики датасетов ──────────────────────────────────────
def load_dataset(name):
    print(f'\n[1] Загрузка {name}...')

    if name == 'magnit':
        train = pd.read_parquet('data/train.parquet')
        test  = pd.read_parquet('data/test.parquet')
        return train, test

    if name == 'hillstrom':
        from sklift.datasets import fetch_hillstrom
        data = fetch_hillstrom()
        df   = data.data.copy()
        df['spend']     = data.target
        df['treatment'] = (data.treatment != 'No E-Mail').astype(int)
        df['user_id']   = np.arange(len(df))
        n = int(len(df) * 0.8)
        train, test = df.iloc[:n].copy(), df.iloc[n:].copy()
        test_out = test.copy()
        test_out['spend'] = np.nan
        return train, test_out

    if name == 'lenta':
        from sklift.datasets import fetch_lenta
        data  = fetch_lenta()
        df    = data.data.copy()
        df['response_att']  = data.target
        df['treatment_flg'] = data.treatment
        df['user_id']       = np.arange(len(df))
        # Энкодим строковые колонки
        for col in df.select_dtypes(include=['object', 'string']).columns:
            df[col] = df[col].astype('category').cat.codes
        n = int(len(df) * 0.8)
        train, test = df.iloc[:n].copy(), df.iloc[n:].copy()
        test_out = test.copy()
        test_out['response_att'] = np.nan
        return train, test_out

    if name == 'starbucks':
        path = 'data/starbucks_train.csv'
        if not os.path.exists(path):
            raise FileNotFoundError(
                'Скачай датасет с Kaggle и положи в data/starbucks_train.csv\n'
                'https://www.kaggle.com/datasets/ihormuliar/starbucks-customer-data'
            )
        df = pd.read_csv(path)
        df['user_id'] = np.arange(len(df))
        n = int(len(df) * 0.8)
        train, test = df.iloc[:n].copy(), df.iloc[n:].copy()
        test_out = test.copy()
        test_out['purchase'] = np.nan
        return train, test_out

    if name == 'criteo':
        path = 'data/criteo-uplift-v2.1.csv'
        if not os.path.exists(path):
            raise FileNotFoundError(
                'Скачай датасет и положи в data/criteo-uplift-v2.1.csv\n'
                'https://ailab.criteo.com/criteo-uplift-prediction-dataset/'
            )
        df = pd.read_csv(path, nrows=500_000)  # берём 500K для скорости
        df['user_id'] = np.arange(len(df))
        n = int(len(df) * 0.8)
        train, test = df.iloc[:n].copy(), df.iloc[n:].copy()
        test_out = test.copy()
        test_out['conversion'] = np.nan
        return train, test_out

    if name == 'synthetic':
        from causalml.dataset import make_uplift_classification
        df, _ = make_uplift_classification(
            n_samples=50_000, treatment_name=['treatment'],
            n_classification_features=20, random_seed=SEED
        )
        df = df.rename(columns={'treatment_group_key': 'treatment'})
        df['treatment'] = (df['treatment'] == 'treatment').astype(int)
        df['outcome']   = df['conversion']
        df['user_id']   = np.arange(len(df))
        n = int(len(df) * 0.8)
        train, test = df.iloc[:n].copy(), df.iloc[n:].copy()
        test_out = test.copy()
        test_out['outcome'] = np.nan
        return train, test_out

    raise ValueError(f'Неизвестный датасет: {name}')


# ── Запуск ───────────────────────────────────────────────────
train, test = load_dataset(DATASET)
print(f'  train: {train.shape}  test: {test.shape}')

TREATMENT_COL = cfg['treatment_col']
OUTCOME_COL   = cfg['outcome_col']
CAT_COLS      = cfg['cat_cols']
OUT_DIR       = cfg['out_dir']

# Базовая статистика
ate = train[train[TREATMENT_COL]==1][OUTCOME_COL].mean() - \
      train[train[TREATMENT_COL]==0][OUTCOME_COL].mean()
zero_rate = (train[OUTCOME_COL] == 0).mean()
print(f'  ATE: {ate:.4f}  Zero rate: {zero_rate:.1%}')

# ── Валидация ─────────────────────────────────────────────────
print(f'\n[2] Валидация данных...')
from uplift import full_validation_report
report = full_validation_report(
    train, TREATMENT_COL, OUTCOME_COL, verbose=False
)
print(f'  Рандомизация: {report["randomization"]["imbalanced"].sum()} имбалансных')
print(f'  Leakage: {"не обнаружен" if report["leakage"]["leakage_risk"].sum()==0 else "обнаружен!"}')
print(f'  Баланс: {"OK" if report["balance_ok"] else "дисбаланс!"}')
print(f'  ATE: {report["ate"]:.4f}')

# ── Препроцессинг ─────────────────────────────────────────────
if USE_AGENT:
    print(f'\n[3] LLM агент — EDA и препроцессинг...')
    from uplift import generate_preprocess
    code, preproc_stats = generate_preprocess(
        train_df=train,
        treatment_col=TREATMENT_COL,
        outcome_col=OUTCOME_COL,
        verbose=True,
    )
    namespace = {'pd': pd, 'np': np}
    exec(code, namespace)
    train_proc = namespace['apply_preprocess'](train.copy(), preproc_stats)
    test_proc  = namespace['apply_preprocess'](test.copy(),  preproc_stats)
else:
    print(f'\n[3] Стандартный препроцессинг (без агента)...')
    from uplift.pipeline import fit_preprocess, apply_preprocess
    preproc_stats = fit_preprocess(
        train, TREATMENT_COL, OUTCOME_COL, cat_cols=CAT_COLS
    )
    train_proc = apply_preprocess(train.copy(), preproc_stats)
    test_proc  = apply_preprocess(test.copy(),  preproc_stats)

# Feature cols
service_cols = ['user_id', TREATMENT_COL, OUTCOME_COL]
feature_cols = [
    c for c in train_proc.columns
    if c not in service_cols
    and train_proc[c].dtype in ['float64','float32','int64','int32','int8']
]
X_full = train_proc[feature_cols].astype(float)
y_full = train[OUTCOME_COL].values
w_full = train[TREATMENT_COL].values
X_test = test_proc[[c for c in feature_cols if c in test_proc.columns]].astype(float)
print(f'  X_full: {X_full.shape}  NaN: {X_full.isnull().sum().sum()}')

# ── CV ────────────────────────────────────────────────────────
print(f'\n[4] CV по всем 6 моделям...')
from uplift import UpliftPipeline
pipe = UpliftPipeline(random_state=SEED)
cv_df = pipe.compare_all(
    train,
    treatment_col=TREATMENT_COL,
    outcome_col=OUTCOME_COL,
    cat_cols=CAT_COLS,
    cv_folds=CV_FOLDS,
    fast=FAST,
    n_boot=N_BOOT,
)
print('\nРезультаты CV:')
print(cv_df.to_string(index=False))
cv_df.to_csv(f'{OUT_DIR}/tables/cv_results.csv', index=False)
cv_dict = {row['model']: (row['cv_point'], row['cv_lower_ci'])
           for _, row in cv_df.iterrows()}

# ── Обучаем все модели на holdout ─────────────────────────────
print(f'\n[5] Обучаем все 6 моделей на holdout...')
from uplift.models import (
    DRLearner, TLearnerLGB, TLearnerRidge,
    XLearner, RLearner, HurdleLearner
)
MODEL_MAP = {
    'dr_learner':      DRLearner,
    't_learner_lgb':   TLearnerLGB,
    't_learner_ridge': TLearnerRidge,
    'x_learner':       XLearner,
    'r_learner':       RLearner,
    'hurdle':          HurdleLearner,
}

X_inner, X_hold, y_inner, y_hold, w_inner, w_hold = train_test_split(
    X_full, y_full, w_full, test_size=HOLDOUT, random_state=SEED
)
X_inner = X_inner.reset_index(drop=True)
X_hold  = X_hold.reset_index(drop=True)

from uplift.metrics import uplift_at_k
preds_hold, preds_test = {}, {}

for name, ModelClass in MODEL_MAP.items():
    print(f'  Обучаем {name}...')
    try:
        model = ModelClass(random_state=SEED)
        model.fit(X_inner, y_inner, w_inner)
        preds_hold[name] = model.predict_uplift(X_hold)
        preds_test[name] = model.predict_uplift(X_test)
        pt, lo, _ = uplift_at_k(y_hold, w_hold, preds_hold[name], n_boot=50)
        print(f'    holdout: point={pt:.4f} lower={lo:.4f}')
    except Exception as e:
        print(f'    ✗ {e}')

# ── Агент: выбор ансамбля ─────────────────────────────────────
print(f'\n[6] LLM агент — перебор всех комбинаций...')
from uplift import select_ensemble
selection = select_ensemble(
    predictions=preds_hold,
    y_val=y_hold,
    w_val=w_hold,
    cv_results=cv_dict,
    metric=METRIC,
    n_boot=N_BOOT,
    verbose=True,
)
combos_df = selection.pop('all_combinations')
combos_df.to_csv(f'{OUT_DIR}/tables/all_combinations.csv', index=False)

# ── Финальный предикт ─────────────────────────────────────────
print(f'\n[7] Финальный предикт...')
weights = selection['weights']
final_scores   = sum(w * preds_test[m] for m, w in weights.items() if m in preds_test)
holdout_scores = sum(w * preds_hold[m] for m, w in weights.items() if m in preds_hold)

# ── Метрики ───────────────────────────────────────────────────
print(f'\n[8] Финальные метрики на holdout...')
from uplift import evaluate_all
all_metrics = evaluate_all(y_hold, w_hold, holdout_scores, n_boot=200)
print(f'\n{"Метрика":15s}  {"Значение":>10}')
print('-' * 30)
for m, v in all_metrics.items():
    print(f'{m:15s}  {v:10.4f}')
pd.DataFrame([all_metrics]).to_csv(f'{OUT_DIR}/tables/metrics.csv', index=False)

# ── Permutation test ──────────────────────────────────────────
print(f'\n[9] Permutation test (n=200)...')
from uplift import permutation_test
perm = permutation_test(y_hold, w_hold, holdout_scores,
                        n_permutations=200, verbose=True)

# ── Графики ───────────────────────────────────────────────────
print(f'\n[10] Сохранение графиков...')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#0a0a0f')

for ax in axes:
    ax.set_facecolor('#12121a')
    ax.spines[['top','right','left','bottom']].set_color('#1a1a26')
    ax.tick_params(colors='#7a7890')
    ax.yaxis.set_tick_params(labelcolor='#7a7890')

x = np.arange(len(cv_df))
w = 0.35
colors_pt = ['#8b78ff' if i==0 else '#3a3a52' for i in range(len(cv_df))]
colors_lo = ['#4ecdc4' if i==0 else '#2a4a48' for i in range(len(cv_df))]

axes[0].bar(x-w/2, cv_df['cv_point'],    w, color=colors_pt, alpha=0.9, label='CV point')
axes[0].bar(x+w/2, cv_df['cv_lower_ci'], w, color=colors_lo, alpha=0.9, label='CV lower CI')
axes[0].set_xticks(x)
axes[0].set_xticklabels(cv_df['model'], rotation=20, ha='right', color='#7a7890', fontsize=10)
axes[0].set_ylabel(f'{METRIC}', color='#e8e6f0')
axes[0].set_title(f'Model comparison — {DATASET}', color='#e8e6f0', pad=12)
axes[0].legend(facecolor='#1a1a26', edgecolor='#3a3a52', labelcolor='#e8e6f0')

cv_df['gap'] = cv_df['cv_point'] - cv_df['cv_lower_ci']
colors_gap = ['#ff6b6b' if g > 6 else '#4ecdc4' for g in cv_df['gap']]
axes[1].bar(cv_df['model'], cv_df['gap'], color=colors_gap, alpha=0.9)
axes[1].set_xticklabels(cv_df['model'], rotation=20, ha='right', color='#7a7890', fontsize=10)
axes[1].set_ylabel('Gap', color='#e8e6f0')
axes[1].set_title('Stability check', color='#e8e6f0', pad=12)
axes[1].axhline(6, color='#ffd93d', linewidth=1, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/figures/model_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
plt.close()
print(f'  Сохранено: {OUT_DIR}/figures/model_comparison.png')

# ── Сохранение predictions ────────────────────────────────────
print(f'\n[11] Сохранение...')
test_ids = test['user_id'].values if 'user_id' in test.columns else np.arange(len(test))
sub = pd.DataFrame({'user_id': test_ids, 'UPLIFT_SCORE': final_scores})
sub.to_csv(f'{OUT_DIR}/predictions.csv', index=False)

with open(f'{OUT_DIR}/tables/ensemble_selection.json', 'w', encoding='utf-8') as f:
    json.dump(selection, f, ensure_ascii=False, indent=2)

print(f'\n{"="*60}')
print(f'ГОТОВО — {DATASET.upper()}')
print(f'{"="*60}')
print(f'Ансамбль:    {selection["selected_models"]}')
print(f'Стратегия:   {selection["strategy"]}')
print(f'Holdout CI:  {selection["holdout_lower_ci"]:.4f}')
print(f'Результаты:  {OUT_DIR}/')
print(f'{"="*60}')