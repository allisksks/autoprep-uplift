import sys
sys.path.insert(0, '.')
from uplift import UpliftPipeline, full_validation_report
import pandas as pd

train = pd.read_parquet('data/train.parquet')
test  = pd.read_parquet('data/test.parquet')

print('=== Валидация данных ===')
report = full_validation_report(
    train,
    treatment_col='treatment_flg',
    outcome_col='rec_spend',
)