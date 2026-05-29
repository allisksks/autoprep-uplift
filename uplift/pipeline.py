"""
uplift/pipeline.py

Главный класс пайплайна. Объединяет препроцессинг,
обучение моделей, CV-оценку и выбор топ-3.

Использование:
    from uplift.pipeline import UpliftPipeline

    pipe = UpliftPipeline(random_state=42)
    pipe.fit(train_df, treatment_col='treatment_flg', outcome_col='rec_spend')
    results = pipe.compare_all(train_df, cv_folds=3)
    pipe.predict(test_df, output_path='predictions.csv')
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List

from .metrics import uplift_at_k, auuc, qini_coefficient, train_val_gap, summary_table
from .models import DRLearner, TLearnerLGB, TLearnerRidge, HurdleLearner
from sklearn.model_selection import KFold, train_test_split

SEED = 42


# ── Препроцессинг ─────────────────────────────────────────────────────────────

def fit_preprocess(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    drop_cols: Optional[List[str]] = None,
    cat_cols: Optional[List[str]] = None,
    flag_cols: Optional[List[str]] = None,
    high_miss_threshold: float = 0.5,
) -> dict:
    """
    Обучаем препроцессинг только на train.
    Возвращаем stats для apply_preprocess.

    Гарантирует отсутствие data leakage:
    все параметры (медианы, маппинги) считаются только по train.
    """
    if drop_cols is None:
        drop_cols = []
    if cat_cols is None:
        cat_cols = []
    if flag_cols is None:
        flag_cols = []

    service_cols = ['user_id', treatment_col, outcome_col] + drop_cols
    stats = {
        'treatment_col':      treatment_col,
        'outcome_col':        outcome_col,
        'service_cols':       service_cols,
        'cat_cols':           cat_cols,
        'flag_cols':          flag_cols,
    }

    # Числовые признаки
    num_cols = [
        c for c in df.columns
        if c not in service_cols + cat_cols + flag_cols
        and df[c].dtype in ['float64', 'float32', 'int64', 'int32']
    ]
    stats['num_cols']    = num_cols
    stats['num_medians'] = df[num_cols].median().to_dict()

    # Label encoding для категориальных
    stats['cat_maps'] = {}
    for col in cat_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique().tolist()
            stats['cat_maps'][col] = {v: i for i, v in enumerate(unique_vals)}

    # Признаки с высокой долей пропусков
    miss_rate = df.isnull().mean()
    stats['high_miss_cols'] = [
        c for c in miss_rate[miss_rate > high_miss_threshold].index.tolist()
        if c not in service_cols
    ]

    return stats


def apply_preprocess(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """
    Применяем препроцессинг к любому датасету используя stats от fit.
    Работает одинаково для train и test — никакого leakage.
    """
    df = df.copy()

    # Числовые: заполняем медианой из train
    for col, median in stats['num_medians'].items():
        if col in df.columns:
            df[col] = df[col].fillna(median)

    # Флаги: заполняем нулём
    for col in stats['flag_cols']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Категориальные: label encoding
    for col, mapping in stats['cat_maps'].items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1).astype(int)

    # Дропаем признаки с высокой долей пропусков
    cols_to_drop = [c for c in stats['high_miss_cols'] if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df


def get_feature_cols(df: pd.DataFrame, stats: dict) -> List[str]:
    """Возвращает список признаков после препроцессинга."""
    return [c for c in df.columns if c not in stats['service_cols']]


# ── Пайплайн ──────────────────────────────────────────────────────────────────

class UpliftPipeline:
    """
    Универсальный uplift-пайплайн.

    Parameters
    ----------
    random_state : seed для воспроизводимости
    """

    # Все доступные модели
    MODELS = {
        'dr_learner':    DRLearner,
        't_learner_lgb': TLearnerLGB,
        't_learner_ridge': TLearnerRidge,
        'hurdle':        HurdleLearner,
    }

    def __init__(self, random_state: int = SEED):
        self.random_state   = random_state
        self._preproc_stats = None
        self._feature_cols  = None
        self._fitted_models: Dict[str, object] = {}
        self._cv_results:    Dict[str, Tuple[float, float]] = {}
        self._treatment_col = None
        self._outcome_col   = None

    def fit(
        self,
        train_df: pd.DataFrame,
        treatment_col: str = 'treatment_flg',
        outcome_col: str   = 'rec_spend',
        cat_cols: Optional[List[str]] = None,
        flag_cols: Optional[List[str]] = None,
        model_name: str = 'dr_learner',
    ) -> 'UpliftPipeline':
        """
        Препроцессинг + обучение одной модели на полном train.

        Parameters
        ----------
        model_name : одна из MODELS.keys()
        """
        self._treatment_col = treatment_col
        self._outcome_col   = outcome_col

        # Препроцессинг
        self._preproc_stats = fit_preprocess(
            train_df, treatment_col, outcome_col,
            cat_cols=cat_cols or [], flag_cols=flag_cols or []
        )
        train_proc = apply_preprocess(train_df, self._preproc_stats)
        self._feature_cols = get_feature_cols(train_proc, self._preproc_stats)

        X = train_proc[self._feature_cols].astype(float)
        y = train_proc[outcome_col].values
        w = train_proc[treatment_col].values

        # Обучаем модель
        ModelClass = self.MODELS[model_name]
        model = ModelClass(random_state=self.random_state)
        model.fit(X, y, w)
        self._fitted_models[model_name] = model

        print(f'Модель {model_name} обучена на {len(X)} строках.')
        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        model_name: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> pd.Series:
        """
        Предсказать uplift score для тестовой выборки.

        Parameters
        ----------
        model_name  : если None — берёт первую обученную модель
        output_path : если указан — сохраняет predictions.csv
        """
        if not self._fitted_models:
            raise RuntimeError('Сначала вызови .fit()')

        if model_name is None:
            model_name = list(self._fitted_models.keys())[0]

        if model_name not in self._fitted_models:
            raise ValueError(f'Модель {model_name} не обучена. '
                             f'Доступны: {list(self._fitted_models.keys())}')

        test_proc = apply_preprocess(test_df, self._preproc_stats)
        X_test    = test_proc[self._feature_cols].astype(float)

        scores = self._fitted_models[model_name].predict_uplift(X_test)

        result = pd.Series(scores, name='UPLIFT_SCORE')

        if output_path:
            sub = pd.DataFrame({
                'user_id':     test_df['user_id'].values,
                'UPLIFT_SCORE': scores
            })
            sub.to_csv(output_path, index=False)
            print(f'Сохранено: {output_path}  ({len(sub)} строк)')

        return result

    def compare_all(
        self,
        train_df: pd.DataFrame,
        treatment_col: str = 'treatment_flg',
        outcome_col: str   = 'rec_spend',
        cat_cols: Optional[List[str]] = None,
        flag_cols: Optional[List[str]] = None,
        cv_folds: int   = 3,
        fast: bool      = True,
        n_boot: int     = 100,
    ) -> pd.DataFrame:
        """
        Сравнить все модели через кросс-валидацию.

        Parameters
        ----------
        fast   : если True — CV на 100K сэмпле (быстро)
        n_boot : число bootstrap итераций для CV

        Returns
        -------
        DataFrame с cv_point и cv_lower_ci для каждой модели
        """
        self._treatment_col = treatment_col
        self._outcome_col   = outcome_col

        # Препроцессинг
        self._preproc_stats = fit_preprocess(
            train_df, treatment_col, outcome_col,
            cat_cols=cat_cols or [], flag_cols=flag_cols or []
        )
        train_proc = apply_preprocess(train_df, self._preproc_stats)
        self._feature_cols = get_feature_cols(train_proc, self._preproc_stats)

        X_full = train_proc[self._feature_cols].astype(float)
        y_full = train_proc[outcome_col].values
        w_full = train_proc[treatment_col].values

        # Сэмплируем для быстрого CV
        if fast:
            idx    = pd.Series(range(len(X_full)))\
                       .sample(n=min(100_000, len(X_full)), random_state=SEED)
            X_cv   = X_full.iloc[idx].reset_index(drop=True)
            y_cv   = y_full[idx]
            w_cv   = w_full[idx]
        else:
            X_cv, y_cv, w_cv = X_full, y_full, w_full

        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=SEED)

        for model_name, ModelClass in self.MODELS.items():
            print(f'\n── {model_name} ──')
            pts, los = [], []

            for fold, (tr_idx, va_idx) in enumerate(kf.split(X_cv), 1):
                Xtr, Xva = X_cv.iloc[tr_idx], X_cv.iloc[va_idx]
                ytr, yva = y_cv[tr_idx], y_cv[va_idx]
                wtr, wva = w_cv[tr_idx], w_cv[va_idx]

                model = ModelClass(random_state=self.random_state)
                model.fit(Xtr, ytr, wtr)
                scores = model.predict_uplift(Xva)

                pt, lo, _ = uplift_at_k(yva, wva, scores, n_boot=n_boot)
                pts.append(pt); los.append(lo)
                print(f'  fold {fold}: point={pt:.4f}  lower={lo:.4f}')

            pt_m = round(float(np.mean(pts)), 4)
            lo_m = round(float(np.mean(los)), 4)
            print(f'  CV point={pt_m}  CV lower={lo_m}')
            self._cv_results[model_name] = (pt_m, lo_m)

        return summary_table(self._cv_results)

    def recommend_top3(self) -> pd.DataFrame:
        """
        Вернуть топ-3 модели по cv_lower_ci с объяснением.
        Вызывай после compare_all().
        """
        if not self._cv_results:
            raise RuntimeError('Сначала вызови .compare_all()')

        df = summary_table(self._cv_results)
        top3 = df.head(3).copy()
        top3['explanation'] = top3.apply(
            lambda r: (
                f"CV lower CI = {r['cv_lower_ci']:.4f} — "
                f"{'лучшая стабильность' if r.name == 0 else 'хорошая стабильность'}"
            ),
            axis=1
        )
        return top3

    def save_preproc_stats(self, path: str):
        """Сохранить параметры препроцессинга для инференса."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._preproc_stats, f, ensure_ascii=False, indent=2)
        print(f'Препроцессинг сохранён: {path}')

    def load_preproc_stats(self, path: str):
        """Загрузить параметры препроцессинга."""
        with open(path, 'r', encoding='utf-8') as f:
            self._preproc_stats = json.load(f)
        print(f'Препроцессинг загружен: {path}')