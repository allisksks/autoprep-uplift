"""
uplift/metrics.py

Метрики для оценки uplift-моделей.

Поддерживает динамический выбор метрики:
  - uplift@K  : основная метрика с bootstrap CI
  - auuc      : Area Under Uplift Curve
  - qini      : Qini coefficient

Использование:
  from uplift.metrics import evaluate, uplift_at_k, auuc, qini_coefficient
  score = evaluate(y, treatment, scores, metric='uplift@10')
  score = evaluate(y, treatment, scores, metric='auuc')
  score = evaluate(y, treatment, scores, metric='qini')
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Union

SEED = 42

# Поддерживаемые метрики
SUPPORTED_METRICS = ['uplift@k', 'auuc', 'qini']


# ── Парсинг метрики ───────────────────────────────────────────────────────────

def parse_metric(metric: str) -> Tuple[str, float]:
    """
    Парсит строку метрики в (тип, параметр).

    Examples
    --------
    'uplift@10' -> ('uplift@k', 0.10)
    'uplift@5'  -> ('uplift@k', 0.05)
    'uplift@20' -> ('uplift@k', 0.20)
    'auuc'      -> ('auuc', None)
    'qini'      -> ('qini', None)
    """
    metric = metric.lower().strip()

    if metric.startswith('uplift@'):
        k_str = metric.split('@')[1]
        k = float(k_str) / 100.0
        if not 0 < k <= 1:
            raise ValueError(f'K должно быть от 1 до 100, получено: {k_str}')
        return 'uplift@k', k

    if metric == 'auuc':
        return 'auuc', None

    if metric in ('qini', 'qini_coefficient'):
        return 'qini', None

    raise ValueError(
        f'Неизвестная метрика: {metric}. '
        f'Поддерживаются: uplift@K (например uplift@10), auuc, qini'
    )


# ── Основные метрики ──────────────────────────────────────────────────────────

def uplift_at_k(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    k: float = 0.10,
    n_boot: int = 300,
    ci: float = 0.80,
    seed: int = SEED,
) -> Tuple[float, float, float]:
    """
    uplift@K с bootstrap CI.

    Returns (point, lower_ci, upper_ci)
    """
    df = pd.DataFrame({'y': y, 'w': treatment, 's': scores})\
           .sort_values('s', ascending=False)\
           .reset_index(drop=True)

    top   = df.iloc[:int(len(df) * k)]
    n     = len(top)
    point = top[top.w == 1]['y'].mean() - top[top.w == 0]['y'].mean()

    rng   = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        samp = top.iloc[rng.choice(n, n, replace=True)]
        boots.append(
            samp[samp.w == 1]['y'].mean() - samp[samp.w == 0]['y'].mean()
        )

    alpha = (1 - ci) / 2
    return (
        round(float(point), 4),
        round(float(np.percentile(boots, alpha * 100)), 4),
        round(float(np.percentile(boots, (1 - alpha) * 100)), 4),
    )


def uplift_at_multiple_k(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    ks: List[float] = None,
    n_boot: int = 300,
    ci: float = 0.80,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Считает uplift@K для нескольких значений K одновременно.

    Parameters
    ----------
    ks : список долей, например [0.05, 0.10, 0.20, 0.30]

    Returns
    -------
    DataFrame с колонками: k, point, lower_ci, upper_ci
    """
    if ks is None:
        ks = [0.05, 0.10, 0.20, 0.30, 0.50]

    rows = []
    for k in ks:
        pt, lo, hi = uplift_at_k(y, treatment, scores, k=k,
                                  n_boot=n_boot, ci=ci, seed=seed)
        rows.append({
            'k':        f'{int(k*100)}%',
            'k_value':  k,
            'point':    pt,
            'lower_ci': lo,
            'upper_ci': hi,
        })
    return pd.DataFrame(rows)


def auuc(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Area Under Uplift Curve.
    normalize=True -> AUUC / AUUC_random
    """
    df = pd.DataFrame({'y': y, 'w': treatment, 's': scores})\
           .sort_values('s', ascending=False)\
           .reset_index(drop=True)

    n   = len(df)
    ate = df[df.w == 1]['y'].mean() - df[df.w == 0]['y'].mean()

    cumulative = []
    for k in range(1, n + 1):
        top = df.iloc[:k]
        if top['w'].sum() > 0 and (1 - top['w']).sum() > 0:
            u = top[top.w == 1]['y'].mean() - top[top.w == 0]['y'].mean()
        else:
            u = 0.0
        cumulative.append(u)

    area = float(np.trapezoid(cumulative)) / n

    if normalize:
        return round(area / (abs(ate) + 1e-8), 4)
    return round(area, 4)


def qini_coefficient(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Qini coefficient (Radcliffe, 2007).
    Площадь между uplift curve и случайным таргетингом.
    """
    df = pd.DataFrame({'y': y, 'w': treatment, 's': scores})\
           .sort_values('s', ascending=False)\
           .reset_index(drop=True)

    n         = len(df)
    n_treat   = df['w'].sum()
    n_control = n - n_treat

    gains_model  = []
    gains_random = []
    cum_t = 0.0
    cum_c = 0.0

    for i, row in df.iterrows():
        if row['w'] == 1:
            cum_t += row['y']
        else:
            cum_c += row['y']

        if n_treat > 0 and n_control > 0:
            gain = cum_t / n_treat - cum_c / n_control
        else:
            gain = 0.0

        gains_model.append(gain)
        gains_random.append((i + 1) / n * gain)

    qini = float(np.trapezoid(gains_model) - np.trapezoid(gains_random)) / n
    return round(qini, 6)


# ── Универсальная функция оценки ──────────────────────────────────────────────

def evaluate(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    metric: str = 'uplift@10',
    n_boot: int = 300,
    ci: float = 0.80,
    seed: int = SEED,
) -> float:
    """
    Универсальная функция оценки с динамическим выбором метрики.

    Parameters
    ----------
    metric : строка метрики — 'uplift@10', 'uplift@5', 'auuc', 'qini'

    Returns
    -------
    float — значение метрики (для uplift@K возвращает lower_ci)
    """
    metric_type, param = parse_metric(metric)

    if metric_type == 'uplift@k':
        _, lower_ci, _ = uplift_at_k(
            y, treatment, scores, k=param,
            n_boot=n_boot, ci=ci, seed=seed
        )
        return lower_ci

    if metric_type == 'auuc':
        return auuc(y, treatment, scores)

    if metric_type == 'qini':
        return qini_coefficient(y, treatment, scores)

    raise ValueError(f'Неизвестный тип метрики: {metric_type}')


def evaluate_all(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    metrics: List[str] = None,
    n_boot: int = 300,
    seed: int = SEED,
) -> Dict[str, float]:
    """
    Считает все метрики сразу.

    Parameters
    ----------
    metrics : список метрик, например
              ['uplift@5', 'uplift@10', 'uplift@20', 'auuc', 'qini']

    Returns
    -------
    dict {metric_name: value}
    """
    if metrics is None:
        metrics = ['uplift@5', 'uplift@10', 'uplift@20', 'auuc', 'qini']

    return {
        m: evaluate(y, treatment, scores, metric=m,
                    n_boot=n_boot, seed=seed)
        for m in metrics
    }


# ── Сводная таблица CV ────────────────────────────────────────────────────────

def train_val_gap(train_score: float, val_score: float) -> float:
    """Разность train/val. Чем меньше — тем лучше."""
    return round(float(train_score - val_score), 4)


def summary_table(
    results: Dict[str, Tuple[float, float]]
) -> pd.DataFrame:
    """
    Сводная таблица результатов CV.

    Parameters
    ----------
    results : {model_name: (cv_point, cv_lower_ci)}
    """
    rows = [
        {'model': name, 'cv_point': pt, 'cv_lower_ci': lo}
        for name, (pt, lo) in results.items()
    ]
    return pd.DataFrame(rows)\
             .sort_values('cv_lower_ci', ascending=False)\
             .reset_index(drop=True)