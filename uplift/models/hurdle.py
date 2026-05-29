"""
uplift/models/hurdle.py

Hurdle Model — специально для zero-inflated continuous outcome.

Разбивает задачу на два шага:
  1. P(Y > 0 | X, W)  — вероятность покупки (классификатор)
  2. E[Y | Y > 0, X, W] — средний чек при покупке (регрессор)

uplift(x) = P_t(Y>0) * E_t[Y|Y>0] - P_c(Y>0) * E_c[Y|Y>0]

Актуально когда ~90% значений outcome — нули (как rec_spend в Magnit).
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from .base import BaseUpliftModel

SEED = 42


class HurdleLearner(BaseUpliftModel):
    """
    Hurdle Model с CatBoost.

    Parameters
    ----------
    random_state : seed для воспроизводимости
    """

    def __init__(self, random_state: int = SEED):
        super().__init__(random_state=random_state)
        self._clf_t = None
        self._clf_c = None
        self._reg_t = None
        self._reg_c = None

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray):
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        w = np.asarray(treatment)

        Xt = X[w == 1].reset_index(drop=True); yt = y[w == 1]
        Xc = X[w == 0].reset_index(drop=True); yc = y[w == 0]

        # Классификаторы: купит / не купит
        self._clf_t = self._fit_clf(Xt, (yt > 0).astype(int))
        self._clf_c = self._fit_clf(Xc, (yc > 0).astype(int))

        # Регрессоры: сколько потратит если купит
        # Обучаем только на ненулевых
        self._reg_t = self._fit_reg(
            Xt[yt > 0].reset_index(drop=True), yt[yt > 0]
        )
        self._reg_c = self._fit_reg(
            Xc[yc > 0].reset_index(drop=True), yc[yc > 0]
        )

        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()

        p_t = self._clf_t.predict_proba(X)[:, 1]
        p_c = self._clf_c.predict_proba(X)[:, 1]

        spend_t = self._reg_t.predict(X)
        spend_c = self._reg_c.predict(X)

        return p_t * spend_t - p_c * spend_c

    def _fit_clf(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray
    ) -> CatBoostClassifier:
        """CatBoost классификатор с early stopping и весами классов."""
        n   = len(X_tr)
        cut = max(int(n * 0.15), 200)
        params = dict(
            iterations=1000, learning_rate=0.05, depth=4,
            l2_leaf_reg=5.0, early_stopping_rounds=50,
            class_weights={0: 1, 1: 9},
            random_seed=self.random_state, verbose=0, thread_count=4
        )
        model = CatBoostClassifier(**params)
        model.fit(
            X_tr.iloc[:-cut], y_tr[:-cut],
            eval_set=(X_tr.iloc[-cut:], y_tr[-cut:]),
            verbose=0
        )
        return model

    def _fit_reg(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray
    ) -> CatBoostRegressor:
        """CatBoost регрессор с early stopping."""
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
        return {'random_state': self.random_state}