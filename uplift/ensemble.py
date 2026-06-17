"""
uplift/ensemble.py

Стратегии ансамблирования uplift-моделей.

Поддерживаемые стратегии:
  - equal_weights    : равные веса для всех моделей
  - gap_weights      : веса обратно пропорционально train/val gap
  - ci_weights       : веса пропорционально cv_lower_ci
  - rank_weights     : веса по рангу (1/rank)
  - stacking         : мета-модель поверх предсказаний
  - best_single      : лучшая одиночная модель без ансамбля
  - pairwise_best    : лучшая пара моделей из всех комбинаций
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import Ridge

from .metrics import evaluate

SEED = 42


# ── Базовые стратегии взвешивания ─────────────────────────────────────────────

def equal_weights(
    model_names: List[str],
    **kwargs,
) -> Dict[str, float]:
    """Равные веса для всех моделей."""
    w = 1.0 / len(model_names)
    return {name: w for name in model_names}


def gap_weights(
    model_names: List[str],
    gaps: Dict[str, float],
    **kwargs,
) -> Dict[str, float]:
    """
    Веса обратно пропорционально train/val gap.
    Модели с меньшим переобучением получают больший вес.
    """
    inv = {m: 1.0 / (gaps.get(m, 0.0) + 0.1) for m in model_names}
    total = sum(inv.values())
    return {m: v / total for m, v in inv.items()}


def ci_weights(
    model_names: List[str],
    cv_lower_cis: Dict[str, float],
    **kwargs,
) -> Dict[str, float]:
    """
    Веса пропорционально cv_lower_ci.
    Лучшие по стабильности модели получают больший вес.
    Отрицательные CI заменяются нулём.
    """
    vals = {m: max(cv_lower_cis.get(m, 0.0), 0.0) for m in model_names}
    total = sum(vals.values()) + 1e-8
    return {m: v / total for m, v in vals.items()}


def rank_weights(
    model_names: List[str],
    cv_lower_cis: Dict[str, float],
    **kwargs,
) -> Dict[str, float]:
    """
    Веса по рангу: 1/rank.
    Первое место получает 1/1, второе 1/2, и т.д.
    """
    sorted_models = sorted(
        model_names,
        key=lambda m: cv_lower_cis.get(m, 0.0),
        reverse=True
    )
    raw = {m: 1.0 / (i + 1) for i, m in enumerate(sorted_models)}
    total = sum(raw.values())
    return {m: v / total for m, v in raw.items()}


# ── Стратегии выбора моделей ──────────────────────────────────────────────────

def best_single(
    predictions: Dict[str, np.ndarray],
    cv_lower_cis: Dict[str, float],
    **kwargs,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Возвращает предсказания лучшей одиночной модели.
    Базовый бейзлайн для сравнения с ансамблем.
    """
    best_model = max(cv_lower_cis, key=cv_lower_cis.get)
    weights = {m: 1.0 if m == best_model else 0.0
               for m in predictions}
    scores = predictions[best_model].copy()
    return scores, weights


def pairwise_best(
    predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
    w_val: np.ndarray,
    metric: str = 'uplift@10',
    n_boot: int = 50,
    seed: int = SEED,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Перебирает все пары моделей и выбирает лучшую комбинацию.
    Полезно когда топ-3 избыточны или коррелированы.

    Returns
    -------
    (best_scores, weights_dict)
    """
    model_names = list(predictions.keys())
    best_score  = -np.inf
    best_pair   = None
    best_scores = None

    for m1, m2 in combinations(model_names, 2):
        combined = (predictions[m1] + predictions[m2]) / 2
        score = evaluate(y_val, w_val, combined,
                        metric=metric, n_boot=n_boot, seed=seed)
        if score > best_score:
            best_score  = score
            best_pair   = (m1, m2)
            best_scores = combined

    weights = {m: 0.5 if m in best_pair else 0.0
               for m in model_names}

    return best_scores, weights


def stacking(
    train_predictions: Dict[str, np.ndarray],
    test_predictions: Dict[str, np.ndarray],
    y_train: np.ndarray,
    treatment_train: np.ndarray,
    k: float = 0.10,
    seed: int = SEED,
) -> Tuple[np.ndarray, object]:
    """
    Стекинг: Ridge мета-модель поверх предсказаний базовых моделей.

    Обучает Ridge на train предсказаниях используя uplift score как таргет.
    Таргет для мета-модели: 1 если клиент в treatment и купил, иначе 0.

    Parameters
    ----------
    train_predictions : {model_name: scores на train}
    test_predictions  : {model_name: scores на test}

    Returns
    -------
    (stacked_test_scores, meta_model)
    """
    # Матрица предсказаний базовых моделей
    X_train_meta = np.column_stack([
        train_predictions[m] for m in sorted(train_predictions.keys())
    ])
    X_test_meta = np.column_stack([
        test_predictions[m] for m in sorted(test_predictions.keys())
    ])

    # Псевдо-таргет для мета-модели:
    # treated купившие = +1, control купившие = -1, остальные = 0
    y_meta = np.zeros(len(y_train))
    y_meta[(treatment_train == 1) & (y_train > 0)] = 1.0
    y_meta[(treatment_train == 0) & (y_train > 0)] = -1.0

    meta_model = Ridge(alpha=1.0, random_state=seed)
    meta_model.fit(X_train_meta, y_meta)

    stacked_scores = meta_model.predict(X_test_meta)
    return stacked_scores, meta_model


# ── Главный класс ансамбля ────────────────────────────────────────────────────

class UpliftEnsemble:
    """
    Универсальный ансамбль uplift-моделей.

    Поддерживает несколько стратегий и выбирает лучшую
    на основе валидационной метрики.

    Parameters
    ----------
    strategy  : стратегия взвешивания или 'auto' для автовыбора
    metric    : метрика для оценки ('uplift@10', 'auuc', 'qini')
    top_n     : сколько моделей включать в ансамбль
    """

    STRATEGIES = [
        'equal_weights',
        'gap_weights',
        'ci_weights',
        'rank_weights',
        'best_single',
        'pairwise_best',
    ]

    def __init__(
        self,
        strategy: str = 'auto',
        metric: str   = 'uplift@10',
        top_n: int    = 3,
        seed: int     = SEED,
    ):
        self.strategy = strategy
        self.metric   = metric
        self.top_n    = top_n
        self.seed     = seed

        self._weights:       Dict[str, float] = {}
        self._best_strategy: Optional[str]    = None
        self._strategy_scores: Dict[str, float] = {}

    def fit(
        self,
        predictions_val: Dict[str, np.ndarray],
        y_val: np.ndarray,
        w_val: np.ndarray,
        cv_results: Dict[str, Tuple[float, float]],
        gaps: Optional[Dict[str, float]] = None,
    ) -> 'UpliftEnsemble':
        """
        Выбирает веса на валидационной выборке.

        Parameters
        ----------
        predictions_val : {model_name: uplift scores на val}
        y_val           : outcome на val
        w_val           : treatment на val
        cv_results      : {model_name: (cv_point, cv_lower_ci)}
        gaps            : {model_name: train_val_gap}
        """
        # Выбираем топ-N моделей по cv_lower_ci
        sorted_models = sorted(
            cv_results.keys(),
            key=lambda m: cv_results[m][1],
            reverse=True
        )
        top_models = sorted_models[:self.top_n]

        cv_lower_cis = {m: cv_results[m][1] for m in top_models}
        preds_top    = {m: predictions_val[m] for m in top_models}

        if gaps is None:
            gaps = {m: cv_results[m][0] - cv_results[m][1]
                   for m in top_models}

        if self.strategy == 'auto':
            self._best_strategy, self._weights = self._auto_select(
                preds_top, y_val, w_val, cv_lower_cis, gaps
            )
        else:
            self._weights = self._compute_weights(
                self.strategy, top_models, cv_lower_cis, gaps,
                preds_top, y_val, w_val
            )
            self._best_strategy = self.strategy

        return self

    def predict(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Применяет выбранные веса к предсказаниям.
        """
        if not self._weights:
            raise RuntimeError('Сначала вызови .fit()')

        scores = np.zeros(len(next(iter(predictions.values()))))
        for model_name, weight in self._weights.items():
            if model_name in predictions and weight > 0:
                scores += weight * predictions[model_name]
        return scores

    def summary(self) -> pd.DataFrame:
        """Возвращает таблицу весов и стратегий."""
        rows = []
        for strategy, score in self._strategy_scores.items():
            rows.append({
                'strategy': strategy,
                'val_score': score,
                'selected': strategy == self._best_strategy,
            })
        df = pd.DataFrame(rows).sort_values('val_score', ascending=False)

        print(f'\nЛучшая стратегия: {self._best_strategy}')
        print('\nВеса моделей:')
        for m, w in sorted(self._weights.items(),
                           key=lambda x: x[1], reverse=True):
            if w > 0:
                print(f'  {m}: {w:.3f}')
        return df

    def _auto_select(
        self,
        predictions: Dict[str, np.ndarray],
        y_val: np.ndarray,
        w_val: np.ndarray,
        cv_lower_cis: Dict[str, float],
        gaps: Dict[str, float],
    ) -> Tuple[str, Dict[str, float]]:
        """Перебирает все стратегии и выбирает лучшую по метрике."""
        model_names  = list(predictions.keys())
        best_strategy = None
        best_score    = -np.inf
        best_weights  = {}

        for strategy in self.STRATEGIES:
            try:
                weights = self._compute_weights(
                    strategy, model_names, cv_lower_cis, gaps,
                    predictions, y_val, w_val
                )
                scores = sum(
                    w * predictions[m]
                    for m, w in weights.items()
                    if m in predictions and w > 0
                )
                val_score = evaluate(
                    y_val, w_val, scores,
                    metric=self.metric, n_boot=50, seed=self.seed
                )
                self._strategy_scores[strategy] = round(float(val_score), 4)

                if val_score > best_score:
                    best_score    = val_score
                    best_strategy = strategy
                    best_weights  = weights

            except Exception:
                self._strategy_scores[strategy] = None
                continue

        return best_strategy, best_weights

    def _compute_weights(
        self,
        strategy: str,
        model_names: List[str],
        cv_lower_cis: Dict[str, float],
        gaps: Dict[str, float],
        predictions: Dict[str, np.ndarray],
        y_val: np.ndarray,
        w_val: np.ndarray,
    ) -> Dict[str, float]:
        """Вычисляет веса для заданной стратегии."""
        if strategy == 'equal_weights':
            return equal_weights(model_names)

        if strategy == 'gap_weights':
            return gap_weights(model_names, gaps=gaps)

        if strategy == 'ci_weights':
            return ci_weights(model_names, cv_lower_cis=cv_lower_cis)

        if strategy == 'rank_weights':
            return rank_weights(model_names, cv_lower_cis=cv_lower_cis)

        if strategy == 'best_single':
            _, weights = best_single(predictions, cv_lower_cis)
            return weights

        if strategy == 'pairwise_best':
            _, weights = pairwise_best(
                predictions, y_val, w_val,
                metric=self.metric, seed=self.seed
            )
            return weights

        raise ValueError(f'Неизвестная стратегия: {strategy}')
