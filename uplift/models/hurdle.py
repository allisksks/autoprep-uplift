"""
uplift/models/hurdle.py

Hurdle Model — специально для zero-inflated continuous outcome.

Разбивает задачу на два шага:
  1. P(Y > 0 | X, W)  — вероятность покупки (классификатор)
  2. E[Y | Y > 0, X, W] — средний чек при покупке (регрессор)

uplift(x) = P_t(Y>0) * E_t[Y|Y>0] - P_c(Y>0) * E_c[Y|Y>0]

Актуально когда ~90% значений outcome — нули (как rec_spend в Magnit).
При малом числе ненулевых или константном outcome — fallback на полные данные.
"""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, CatBoostClassifier

from .base import BaseUpliftModel

SEED = 42
MIN_NONZERO = 50  # минимум ненулевых для регрессора на ненулевых


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
        y = np.asarray(y, dtype=float)
        w = np.asarray(treatment)

        Xt = X[w == 1].reset_index(drop=True); yt = y[w == 1]
        Xc = X[w == 0].reset_index(drop=True); yc = y[w == 0]

        # Классификаторы: купит / не купит
        # Если все значения одного класса — используем все данные как регрессор
        yt_bin = (yt > 0).astype(int)
        yc_bin = (yc > 0).astype(int)

        if yt_bin.sum() > 10 and (1 - yt_bin).sum() > 10:
            self._clf_t = self._fit_clf(Xt, yt_bin)
        else:
            self._clf_t = None  # нет вариации — не обучаем

        if yc_bin.sum() > 10 and (1 - yc_bin).sum() > 10:
            self._clf_c = self._fit_clf(Xc, yc_bin)
        else:
            self._clf_c = None

        # Регрессоры: сколько потратит
        yt_nonzero = yt[yt > 0]
        yc_nonzero = yc[yc > 0]

        # Fallback на полные данные если ненулевых мало или все одинаковые
        use_nonzero_t = (
            len(yt_nonzero) >= MIN_NONZERO and
            yt_nonzero.std() > 1e-6
        )
        use_nonzero_c = (
            len(yc_nonzero) >= MIN_NONZERO and
            yc_nonzero.std() > 1e-6
        )

        if use_nonzero_t:
            self._reg_t = self._fit_reg(
                Xt[yt > 0].reset_index(drop=True), yt_nonzero
            )
        else:
            self._reg_t = self._fit_reg(Xt, yt)

        if use_nonzero_c:
            self._reg_c = self._fit_reg(
                Xc[yc > 0].reset_index(drop=True), yc_nonzero
            )
        else:
            self._reg_c = self._fit_reg(Xc, yc)

        self.is_fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()

        spend_t = self._reg_t.predict(X)
        spend_c = self._reg_c.predict(X)

        # Если классификаторы обучены — используем вероятности
        if self._clf_t is not None and self._clf_c is not None:
            p_t = self._clf_t.predict_proba(X)[:, 1]
            p_c = self._clf_c.predict_proba(X)[:, 1]
            return p_t * spend_t - p_c * spend_c

        # Иначе просто разность регрессоров
        return spend_t - spend_c

    def _fit_clf(
        self, X_tr: pd.DataFrame, y_tr: np.ndarray
    ) -> CatBoostClassifier:
        """CatBoost классификатор с early stopping."""
        n   = len(X_tr)
        cut = max(int(n * 0.15), 200)
        # class_weights только если есть оба класса
        pos_rate = y_tr.mean()
        cw = {0: 1, 1: max(1, int((1 - pos_rate) / (pos_rate + 1e-8)))}
        params = dict(
            iterations=1000, learning_rate=0.05, depth=4,
            l2_leaf_reg=5.0, early_stopping_rounds=50,
            class_weights=cw,
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

        # Если все значения одинаковые — не нужен eval_set
        if y_tr.std() < 1e-6:
            params = dict(
                iterations=10, learning_rate=0.05, depth=3,
                random_seed=self.random_state, verbose=0, thread_count=4
            )
            model = CatBoostRegressor(**params)
            model.fit(X_tr, y_tr, verbose=0)
            return model

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
