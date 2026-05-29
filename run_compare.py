import sys
sys.path.insert(0, '.')
from uplift import UpliftPipeline
import pandas as pd

train = pd.read_parquet('data/train.parquet')

pipe = UpliftPipeline(random_state=42)
results = pipe.compare_all(
    train,
    treatment_col='treatment_flg',
    outcome_col='rec_spend',
    cat_cols=['communication_type'],
    cv_folds=3,
    fast=True,
    n_boot=100,
)

print('\n=== Результаты CV ===')
print(results.to_string(index=False))
results.to_csv('experiments/results/tables/cv_results.csv', index=False)
print('\nСохранено: experiments/results/tables/cv_results.csv')