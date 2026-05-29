"""
uplift/models/dr_learner.py

DR-Learner (Doubly Robust Learner) — Kennedy (2023).

Теория: состоятелен если хотя бы одна из двух nuisance функций
(outcome model ИЛИ propensity score) оценена корректно.

DR pseudo-outcome:
  psi = (mu1 - mu0) + W*(Y - mu1)/e - (1-W)*(Y - mu0)/(1-e)

Финальный tau(x) = регрессия psi на X.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor

from .base import BaseUpliftModel

SEED = 42


class DRLearner(BaseUpliftModel):
    """
    DR-Learner с CatBoost для outcome models и LightGBM для propensity.

    Parameters
    ----------
    n_folds      : число фолдов для cross-fitting nuisance моделей
    random_state : seed для воспроизводимости
    """

    def __init__(self, n_folds: int = 3, random_state: int = SEED):
        super().__init__(random_state=random_state)
        self.n_folds = n_folds
        self._final_model = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        w = np.asarray(treatment)

        psi = np.zeros(len(X))

        kf = KFold(n_splits=self.n_folds, shuffle=True,
                   random_state=self.random_state)

        for tr_idx, val_idx in kf.split(X):
            Xtr, Xval = X.iloc[tr_idx], X.iloc[val_idx]
            ytr, yval = y[tr_idx], y[val_idx]
            wtr, wval = w[tr_idx], w[val_idx]

            # Propensity score e(x) = P(W=1|X)
            ps = LGBMClassifier(
                n_estimators=200, learning_rate=0.05, num_leaves=31,
                min_child_samples=100, random_state=self.random_state,
                n_jobs=4, verbose=-1
            )
            ps.fit(Xtr, wtr)
            e = ps.predict_proba(Xval)[:, 1].clip(0.05, 0.95)

            # Outcome models mu1(x) и mu0(x)
            mu1 = self._fit_cb(
                Xtr[wtr == 1].reset_index(drop=True), ytr[wtr == 1]
            )
            mu0 = self._fit_cb(
                Xtr[wtr == 0].reset_index(drop=True), ytr[wtr == 0]
            )

            # DR pseudo-outcome
            psi[val_idx] = (
                mu1.predict(Xval) - mu0.predict(Xval)
                + wval * (yval - mu1.predict(Xval)) / e
                - (1 - wval) * (yval - mu0.predict(Xval)) / (1 - e)
            )

        # Финальная регрессия psi на X
        self._final_model = self._fit_cb(X, psi)
        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._final_model.predict(X)

    def _fit_cb(self, X_tr: pd.DataFrame, y_tr: np.ndarray) -> CatBoostRegressor:
        """CatBoost с early stopping."""
        n   = len(X_tr)
        cut = max(int(n * 0.15), 200)
        params = dict(
            iterations=1000, learning_rate=0.05, depth=5,
            l2_leaf_reg=5.0, early_stopping_rounds=50,
            random_seed=self.random_state, verbose=0, thread_count=4
        )
        model = CatBoostRegressor(**params)
        model.fit(
            X_tr.iloc[:-cut], y_tr[:-cut],
            eval_set=(X_tr.iloc[-cut:], y_tr[-cut:]),
            verbose=0
        )
        return model

    def get_params(self) -> dict:
        return {
            'n_folds':      self.n_folds,
            'random_state': self.random_state,
        }