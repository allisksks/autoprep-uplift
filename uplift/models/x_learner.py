"""
uplift/models/x_learner.py

X-Learner — Künzel et al. (2019), PNAS.

Двухступенчатая импутация treatment effect.

Шаг 1: T-learner — обучаем mu1(x) и mu0(x)
Шаг 2: Импутируем контрфактические исходы:
  D_t = Y_t - mu0(X_t)  # для treated
  D_c = mu1(X_c) - Y_c  # для control
Шаг 3: Обучаем tau_t на (X_t, D_t) и tau_c на (X_c, D_c)
Шаг 4: tau(x) = p(x)*tau_t(x) + (1-p(x))*tau_c(x)
  где p(x) — propensity score

Преимущество над T-learner:
  Эффективен когда n_treatment != n_control.
  Импутация использует информацию из обеих групп.

На сбалансированных данных (50/50) преимущество минимально.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, LGBMClassifier
from lightgbm import early_stopping as lgb_es, log_evaluation as lgb_log

from .base import BaseUpliftModel

SEED = 42


class XLearner(BaseUpliftModel):
    """
    X-Learner с LightGBM.

    Parameters
    ----------
    random_state : seed для воспроизводимости
    """

    def __init__(self, random_state: int = SEED):
        super().__init__(random_state=random_state)
        self._mu1    = None  # outcome model для treated
        self._mu0    = None  # outcome model для control
        self._tau_t  = None  # CATE model на treated
        self._tau_c  = None  # CATE model на control
        self._ps     = None  # propensity score model

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> 'XLearner':
        X = X.reset_index(drop=True)
        y = np.asarray(y)
        w = np.asarray(treatment)

        Xt = X[w == 1].reset_index(drop=True); yt = y[w == 1]
        Xc = X[w == 0].reset_index(drop=True); yc = y[w == 0]

        # Шаг 1: T-learner — обучаем mu1 и mu0
        self._mu1 = self._fit_lgb(Xt, yt)
        self._mu0 = self._fit_lgb(Xc, yc)

        # Шаг 2: импутируем контрфактические исходы
        D_t = yt - self._mu0.predict(Xt)  # treated: Y - mu0(X)
        D_c = self._mu1.predict(Xc) - yc  # control: mu1(X) - Y

        # Шаг 3: обучаем CATE модели на псевдо-исходах
        self._tau_t = self._fit_lgb(Xt, D_t)
        self._tau_c = self._fit_lgb(Xc, D_c)

        # Шаг 4: propensity score для взвешивания
        self._ps = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            min_child_samples=100, random_state=self.random_state,
            n_jobs=4, verbose=-1
        )
        self._ps.fit(X, w)

        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()

        # Propensity score p(x) = P(W=1|X)
        p = self._ps.predict_proba(X)[:, 1]

        # Взвешенная сумма двух CATE моделей
        tau = p * self._tau_t.predict(X) + (1 - p) * self._tau_c.predict(X)
        return tau

    def _fit_lgb(
        self,
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
    ) -> LGBMRegressor:
        """LightGBM с early stopping."""
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