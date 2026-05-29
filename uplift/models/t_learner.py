"""
uplift/models/t_learner.py

T-Learner (Two-Model approach) — Künzel et al. (2019).

Обучает две отдельные модели:
  mu1(x) = E[Y | X=x, W=1]  — на treated
  mu0(x) = E[Y | X=x, W=0]  — на control

uplift(x) = mu1(x) - mu0(x)

Простой и интерпретируемый baseline.
Проблема: при дисбалансе групп разность может быть смещена.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm import early_stopping as lgb_es, log_evaluation as lgb_log
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from .base import BaseUpliftModel

SEED = 42


class TLearnerLGB(BaseUpliftModel):
    """
    T-Learner с LightGBM.

    Parameters
    ----------
    random_state : seed для воспроизводимости
    """

    def __init__(self, random_state: int = SEED):
        super().__init__(random_state=random_state)
        self._model_t = None
        self._model_c = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        w = np.asarray(treatment)

        Xt = X[w == 1].reset_index(drop=True); yt = y[w == 1]
        Xc = X[w == 0].reset_index(drop=True); yc = y[w == 0]

        self._model_t = self._fit_lgb(Xt, yt)
        self._model_c = self._fit_lgb(Xc, yc)
        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._model_t.predict(X) - self._model_c.predict(X)

    def _fit_lgb(self, X_tr: pd.DataFrame, y_tr: np.ndarray) -> LGBMRegressor:
        n   = len(X_tr)
        cut = max(int(n * 0.15), 200)
        params = dict(
            n_estimators=1000, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_state, n_jobs=4, verbose=-1
        )
        model = LGBMRegressor(**params)
        model.fit(
            X_tr.iloc[:-cut], y_tr[:-cut],
            eval_set=[(X_tr.iloc[-cut:], y_tr[-cut:])],
            callbacks=[lgb_es(50), lgb_log(-1)]
        )
        return model

    def get_params(self) -> dict:
        return {'random_state': self.random_state}


class TLearnerRidge(BaseUpliftModel):
    """
    T-Learner с Ridge регрессией.
    Быстрый baseline, не требует GPU/много памяти.
    """

    def __init__(self, alpha: float = 10.0, random_state: int = SEED):
        super().__init__(random_state=random_state)
        self.alpha    = alpha
        self._model_t = None
        self._model_c = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        w = np.asarray(treatment)

        Xt = X[w == 1].reset_index(drop=True); yt = y[w == 1]
        Xc = X[w == 0].reset_index(drop=True); yc = y[w == 0]

        self._model_t = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0),
            StandardScaler(), Ridge(alpha=self.alpha)
        )
        self._model_c = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0),
            StandardScaler(), Ridge(alpha=self.alpha)
        )
        self._model_t.fit(Xt, yt)
        self._model_c.fit(Xc, yc)
        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._model_t.predict(X) - self._model_c.predict(X)

    def get_params(self) -> dict:
        return {'alpha': self.alpha, 'random_state': self.random_state}