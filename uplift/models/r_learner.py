"""
uplift/models/r_learner.py

R-Learner — Nie & Wager (2021), Biometrika.

Robinson decomposition:
  Y - m(X) = (W - e(X)) * tau(X) + epsilon

R-loss:
  L(tau) = E[(Y - m(X) - (W - e(X)) * tau(X))^2]

Оптимизируется как взвешенная регрессия:
  pseudo_outcome = (Y - m(X)) / (W - e(X))
  sample_weight  = (W - e(X))^2
  tau(X) = регрессия pseudo_outcome на X с весами

Quasi-oracle property:
  Скорость сходимости зависит только от сложности tau*(x),
  не от сложности m*(x) и e*(x).
  Если m* и e* оценены с точностью o(n^{-1/4}),
  tau^ достигает оракульной скорости.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from catboost import CatBoostRegressor

from .base import BaseUpliftModel

SEED = 42


class RLearner(BaseUpliftModel):
    """
    R-Learner с CatBoost для финальной tau(x)
    и LightGBM для nuisance функций m(x) и e(x).

    Parameters
    ----------
    random_state : seed для воспроизводимости
    """

    def __init__(self, random_state: int = SEED):
        super().__init__(random_state=random_state)
        self._tau_model = None  # финальная модель CATE

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> 'RLearner':
        X = X.reset_index(drop=True)
        y = np.asarray(y, dtype=float)
        w = np.asarray(treatment, dtype=float)

        # Шаг 1: оцениваем m(x) = E[Y|X]
        m_model = self._fit_lgb_reg(X, y)
        m_hat   = m_model.predict(X)

        # Шаг 2: оцениваем e(x) = E[W|X] = P(W=1|X)
        e_model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, random_state=self.random_state,
            n_jobs=4, verbose=-1
        )
        e_model.fit(X, w)
        e_hat = e_model.predict_proba(X)[:, 1].clip(0.05, 0.95)

        # Шаг 3: вычисляем Robinson residuals
        Y_res = y - m_hat        # Y - m(X)
        W_res = w - e_hat        # W - e(X)

        # Pseudo-outcome и sample weights
        pseudo = Y_res / (W_res + 1e-8)  # (Y-m) / (W-e)
        sw     = W_res ** 2              # (W-e)^2

        # Шаг 4: регрессия pseudo на X с весами sw
        self._tau_model = self._fit_cb_weighted(X, pseudo, sw)

        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return self._tau_model.predict(X)

    def _fit_lgb_reg(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
    ):
        """LightGBM регрессор для nuisance m(x)."""
        from lightgbm import LGBMRegressor
        from lightgbm import early_stopping as lgb_es, log_evaluation as lgb_log

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

    def _fit_cb_weighted(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        sample_weight: np.ndarray,
    ) -> CatBoostRegressor:
        """CatBoost с sample_weight для R-loss."""
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
            sample_weight=sample_weight[:-cut],
            verbose=0
        )
        return model

    def get_params(self) -> dict:
        return {'random_state': self.random_state}