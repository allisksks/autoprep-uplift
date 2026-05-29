"""
run_from_step9.py
Продолжение с шага 9 — метрики, permutation test, сохранение.
"""
import sys, os, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd

# Восстанавливаем данные
train = pd.read_parquet('data/train.parquet')
test  = pd.read_parquet('data/test.parquet')

# Препроцессинг (повторяем быстро)
from uplift.pipeline import fit_preprocess, apply_preprocess
preproc_stats = fit_preprocess(train, 'treatment_flg', 'rec_spend',
                               cat_cols=['communication_type'])
train_proc = apply_preprocess(train.copy(), preproc_stats)
test_proc  = apply_preprocess(test.copy(),  preproc_stats)

service_cols = ['user_id', 'treatment_flg', 'rec_spend']
feature_cols = [c for c in train_proc.columns
                if c not in service_cols
                and train_proc[c].dtype in ['float64','float32','int64','int32','int8']]

from sklearn.model_selection import train_test_split
X_full = train_proc[feature_cols].astype(float)
y_full = train['rec_spend'].values
w_full = train['treatment_flg'].values

X_inner, X_hold, y_inner, y_hold, w_inner, w_hold = train_test_split(
    X_full, y_full, w_full, test_size=0.2, random_state=42
)
X_inner = X_inner.reset_index(drop=True)
X_hold  = X_hold.reset_index(drop=True)
X_test  = test_proc[[c for c in feature_cols if c in test_proc.columns]].astype(float)

# Обучаем только выбранные модели
from uplift.models import TLearnerRidge, TLearnerLGB
weights = {'t_learner_ridge': 0.667, 't_learner_lgb': 0.333}

print('Обучаем t_learner_ridge...')
m1 = TLearnerRidge(random_state=42)
m1.fit(X_inner, y_inner, w_inner)
p1_hold = m1.predict_uplift(X_hold)
p1_test = m1.predict_uplift(X_test)

print('Обучаем t_learner_lgb...')
m2 = TLearnerLGB(random_state=42)
m2.fit(X_inner, y_inner, w_inner)
p2_hold = m2.predict_uplift(X_hold)
p2_test = m2.predict_uplift(X_test)

holdout_scores = weights['t_learner_ridge']*p1_hold + weights['t_learner_lgb']*p2_hold
final_scores   = weights['t_learner_ridge']*p1_test + weights['t_learner_lgb']*p2_test

# [9] Метрики
print('\n[9] Метрики на holdout...')
from uplift import evaluate_all
all_metrics = evaluate_all(y_hold, w_hold, holdout_scores, n_boot=200)
print(f'\n{"Метрика":15s}  {"Значение":>10}')
print('-' * 30)
for m, v in all_metrics.items():
    print(f'{m:15s}  {v:10.4f}')

# [10] Permutation test
print('\n[10] Permutation test (n=200)...')
from uplift import permutation_test
perm = permutation_test(y_hold, w_hold, holdout_scores,
                        n_permutations=200, verbose=True)

# [11] Сохранение
print('\n[11] Сохранение...')
sub = pd.DataFrame({
    'user_id':      test['user_id'].values,
    'UPLIFT_SCORE': final_scores,
})
sub.to_csv('experiments/results/predictions.csv', index=False)
print(f'  predictions.csv: {len(sub)} строк')
print(f'  mean={final_scores.mean():.4f}  std={final_scores.std():.4f}')
print(f'  p10={np.percentile(final_scores,10):.4f}  p90={np.percentile(final_scores,90):.4f}')

print('\n✓ Готово!')