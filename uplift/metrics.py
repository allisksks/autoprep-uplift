"""
uplift/metrics.py
Метрики для оценки uplift-моделей.

Основные метрики:
  - uplift_at_k        : uplift@K с bootstrap CI (основная метрика)
  - auuc               : Area Under Uplift Curve
  - qini_coefficient   : Qini coefficient (нормированный AUUC)
  - train_val_gap      : разность train/val для проверки переобучения
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

SEED = 42


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

    Сортирует клиентов по убыванию uplift score, берёт топ-K%,
    считает разность средних outcome между treatment и control.

    Parameters
    ----------
    y         : outcome (rec_spend или другой непрерывный/бинарный)
    treatment : бинарный флаг (1 = treatment, 0 = control)
    scores    : uplift scores от модели
    k         : доля топ клиентов (0.10 = топ-10%)
    n_boot    : число bootstrap итераций
    ci        : уровень доверительного интервала
    seed      : random seed для воспроизводимости

    Returns
    -------
    (point_estimate, lower_ci, upper_ci)
    """
    df = pd.DataFrame({'y': y, 'w': treatment, 's': scores})\
           .sort_values('s', ascending=False)\
           .reset_index(drop=True)

    top = df.iloc[:int(len(df) * k)]
    n   = len(top)

    t_mean = top[top.w == 1]['y'].mean()
    c_mean = top[top.w == 0]['y'].mean()
    point  = t_mean - c_mean

    rng   = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        samp = top.iloc[rng.choice(n, n, replace=True)]
        t = samp[samp.w == 1]['y'].mean()
        c = samp[samp.w == 0]['y'].mean()
        boots.append(t - c)

    alpha = (1 - ci) / 2
    return (
        round(float(point), 4),
        round(float(np.percentile(boots, alpha * 100)), 4),
        round(float(np.percentile(boots, (1 - alpha) * 100)), 4),
    )


def auuc(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Area Under Uplift Curve (AUUC).

    Интегральная метрика качества ранжирования по всей популяции.
    normalize=True возвращает AUUC / AUUC_random (> 1 = лучше случайного).

    Parameters
    ----------
    normalize : если True — нормирует на случайный baseline
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

    area = float(np.trapz(cumulative)) / n

    if normalize:
        random_area = ate  # случайный бейзлайн = ATE по всей популяции
        return round(area / (abs(random_area) + 1e-8), 4)

    return round(area, 4)


def qini_coefficient(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
) -> float:
    """
    Qini coefficient (Radcliffe, 2007).

    Площадь между uplift curve модели и диагональю
    (случайным таргетингом). Чем выше — тем лучше.
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

        frac = (i + 1) / n
        if n_treat > 0 and n_control > 0:
            gain = cum_t / n_treat - cum_c / n_control
        else:
            gain = 0.0
        gains_model.append(gain)
        gains_random.append(frac * gains_model[-1])

    qini = float(np.trapz(gains_model) - np.trapz(gains_random)) / n
    return round(qini, 6)


def train_val_gap(
    train_score: float,
    val_score: float,
) -> float:
    """
    Разность между train и val метрикой.
    Используется для детекции переобучения.
    Чем меньше — тем лучше.
    """
    return round(float(train_score - val_score), 4)


def summary_table(results: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
    """
    Сводная таблица результатов CV.

    Parameters
    ----------
    results : {model_name: (cv_point, cv_lower_ci)}

    Returns
    -------
    DataFrame отсортированный по cv_lower_ci убыванию
    """
    rows = [
        {'model': name, 'cv_point': pt, 'cv_lower_ci': lo}
        for name, (pt, lo) in results.items()
    ]
    df = pd.DataFrame(rows).sort_values('cv_lower_ci', ascending=False)
    df = df.reset_index(drop=True)
    return df