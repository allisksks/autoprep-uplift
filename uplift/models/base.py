"""
uplift/models/base.py
Базовый класс для всех uplift-моделей.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class BaseUpliftModel(ABC):
    """
    Базовый класс. Все модели наследуют его и реализуют
    fit() и predict_uplift().
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.is_fitted    = False

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
    ) -> "BaseUpliftModel":
        """
        Обучить модель.

        Parameters
        ----------
        X         : признаки (без treatment и outcome)
        y         : outcome (rec_spend или другой)
        treatment : бинарный флаг (1 = treatment, 0 = control)
        """
        ...

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Предсказать uplift score для каждого клиента.

        Returns
        -------
        np.ndarray shape (n_samples,)
        """
        ...

    def fit_predict(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        treatment_train: np.ndarray,
        X_test: pd.DataFrame,
    ) -> np.ndarray:
        """Удобный метод: fit + predict за один вызов."""
        self.fit(X_train, y_train, treatment_train)
        return self.predict_uplift(X_test)

    def _check_is_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} не обучена. "
                "Сначала вызови .fit()"
            )

    def get_params(self) -> dict:
        """Вернуть параметры модели для логирования."""
        return {'random_state': self.random_state}

    def __repr__(self) -> str:
        params = ', '.join(f'{k}={v}' for k, v in self.get_params().items())
        return f"{self.__class__.__name__}({params})"