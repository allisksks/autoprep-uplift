"""
run_full_pipeline.py
Полный прогон пайплайна на реальных данных.
Запускай: py run_full_pipeline.py
"""

import sys, os
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

print('=' * 60)
print('ПОЛНЫЙ ПРОГОН ПАЙПЛАЙНА')
print('=' * 60)

# ── 1. Загрузка ───────────────────────────────────────────────
print('\n[1] Загрузка данных...')
train = pd.read_parquet('data/train.parquet')
test  = pd.read_parquet('data/test.parquet')
print(f'  train: {train.shape}  test: {test.shape}')
print(f'  ATE: {train[train.treatment_flg==1].rec_spend.mean() - train[train.treatment_flg==0].rec_spend.mean():.4f}')
print(f'  Zero rate: {(train.rec_spend==0).mean():.1%}')

# ── 2. Валидация данных ───────────────────────────────────────
print('\n[2] Валидация данных...')
from uplift import full_validation_report
report = full_validation_report(train, 'treatment_flg', 'rec_spend', verbose=False)
print(f'  Рандомизация: {report["randomization"]["imbalanced"].sum()} имбалансных признаков')
print(f'  Leakage: {"не обнаружен" if report["leakage"]["leakage_risk"].sum() == 0 else "обнаружен!"}')
print(f'  Баланс: {"OK" if report["balance_ok"] else "дисбаланс!"}')
print(f'  ATE: {report["ate"]:.4f}')

# ── 3. LLM агент: препроцессинг ───────────────────────────────
print('\n[3] LLM агент — EDA и препроцессинг...')
from uplift import generate_preprocess
code, preproc_stats = generate_preprocess(
    train_df=train,
    treatment_col='treatment_flg',
    outcome_col='rec_spend',
    verbose=True,
)

# ── 4. Применяем препроцессинг ────────────────────────────────
print('\n[4] Применяем препроцессинг...')
namespace = {'pd': pd, 'np': np}
exec(code, namespace)

# Сохраняем таргет ДО препроцессинга
y_full    = train['rec_spend'].values
w_full    = train['treatment_flg'].values
test_ids  = test['user_id'].values

train_proc = namespace['apply_preprocess'](train.copy(), preproc_stats)
test_proc  = namespace['apply_preprocess'](test.copy(),  preproc_stats)

service_cols = ['user_id', 'treatment_flg', 'rec_spend']
feature_cols = [
    c for c in train_proc.columns
    if c not in service_cols
    and train_proc[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int8']
]

X_full = train_proc[feature_cols].astype(float)
X_test = test_proc[[c for c in feature_cols if c in test_proc.columns]].astype(float)

print(f'  X_full: {X_full.shape}  NaN: {X_full.isnull().sum().sum()}')
print(f'  X_test: {X_test.shape}')

# ── 5. CV по всем моделям ─────────────────────────────────────
print('\n[5] CV по всем 6 моделям (fast=True)...')
from uplift import UpliftPipeline
pipe = UpliftPipeline(random_state=42)
cv_df = pipe.compare_all(
    train,
    treatment_col='treatment_flg',
    outcome_col='rec_spend',
    cat_cols=['communication_type'],
    cv_folds=3,
    fast=True,
    n_boot=100,
)
print('\nРезультаты CV:')
print(cv_df.to_string(index=False))
os.makedirs('experiments/results/tables', exist_ok=True)
cv_df.to_csv('experiments/results/tables/cv_results.csv', index=False)

cv_dict = {row['model']: (row['cv_point'], row['cv_lower_ci'])
           for _, row in cv_df.iterrows()}

# ── 6. Обучаем все модели на train/holdout ────────────────────
print('\n[6] Обучаем все 6 моделей на holdout...')
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

# Сплит train/holdout
X_inner, X_hold, y_inner, y_hold, w_inner, w_hold = train_test_split(
    X_full, y_full, w_full, test_size=0.2, random_state=42
)
X_inner = X_inner.reset_index(drop=True)
X_hold  = X_hold.reset_index(drop=True)

preds_hold = {}
preds_test = {}

for name, ModelClass in MODEL_MAP.items():
    print(f'  Обучаем {name}...')
    try:
        model = ModelClass(random_state=42)
        model.fit(X_inner, y_inner, w_inner)
        preds_hold[name] = model.predict_uplift(X_hold)
        preds_test[name] = model.predict_uplift(X_test)
        from uplift.metrics import uplift_at_k
        pt, lo, _ = uplift_at_k(y_hold, w_hold, preds_hold[name], n_boot=50)
        print(f'    holdout: point={pt:.4f} lower={lo:.4f}')
    except Exception as e:
        print(f'    ✗ Ошибка: {e}')

# ── 7. LLM агент: выбор оптимального ансамбля ────────────────
print('\n[7] LLM агент — перебор всех комбинаций и выбор ансамбля...')
from uplift import select_ensemble

selection = select_ensemble(
    predictions=preds_hold,
    y_val=y_hold,
    w_val=w_hold,
    cv_results=cv_dict,
    metric='uplift@10',
    n_boot=100,
    verbose=True,
)

# Сохраняем таблицу комбинаций
combos_df = selection.pop('all_combinations')
combos_df.to_csv('experiments/results/tables/all_combinations.csv', index=False)
print(f'\n  Таблица комбинаций сохранена: {len(combos_df)} комбинаций')

# ── 8. Финальный предикт ──────────────────────────────────────
print('\n[8] Финальный предикт...')
selected_models = selection['selected_models']
weights         = selection['weights']

print(f'  Выбрано моделей: {selected_models}')
print(f'  Веса: {weights}')

final_scores = np.zeros(len(X_test))
for model_name, weight in weights.items():
    if model_name in preds_test:
        final_scores += weight * preds_test[model_name]

# ── 9. Все метрики на holdout ─────────────────────────────────
print('\n[9] Метрики на holdout...')
from uplift import evaluate_all

holdout_scores = np.zeros(len(X_hold))
for model_name, weight in weights.items():
    if model_name in preds_hold:
        holdout_scores += weight * preds_hold[model_name]

all_metrics = evaluate_all(y_hold, w_hold, holdout_scores, n_boot=200)
print(f'\n{"Метрика":15s}  {"Значение":>10}')
print('-' * 30)
for m, v in all_metrics.items():
    print(f'{m:15s}  {v:10.4f}')

# ── 10. Permutation test ──────────────────────────────────────
print('\n[10] Permutation test (n=200)...')
from uplift import permutation_test
perm = permutation_test(
    y_hold, w_hold, holdout_scores,
    n_permutations=200, verbose=True
)

# ── 11. Сохранение ───────────────────────────────────────────
print('\n[11] Сохранение результатов...')
sub = pd.DataFrame({
    'user_id':      test_ids,
    'UPLIFT_SCORE': final_scores,
})
sub.to_csv('experiments/results/predictions.csv', index=False)

# Сохраняем выбор агента
with open('experiments/results/tables/ensemble_selection.json', 'w', encoding='utf-8') as f:
    import json
    json.dump(selection, f, ensure_ascii=False, indent=2)

print(f'  predictions.csv: {len(sub)} строк')
print(f'  mean={final_scores.mean():.4f}  std={final_scores.std():.4f}')
print(f'  p10={np.percentile(final_scores,10):.4f}  p90={np.percentile(final_scores,90):.4f}')

print('\n' + '=' * 60)
print('ГОТОВО.')
print(f'Лучший ансамбль: {selected_models}')
print(f'Стратегия: {selection["strategy"]}')
print(f'Holdout lower CI: {selection["holdout_lower_ci"]:.4f}')
print('=' * 60)