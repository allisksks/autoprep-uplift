"""
uplift/validation.py

Проверки качества данных и антиовёрфиттинг.

Включает:
  1. Проверка рандомизации A/B теста
  2. Детекция data leakage
  3. Permutation test
  4. Learning curves
  5. Repeated CV
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Optional
from scipy import stats
from sklearn.model_selection import KFold

from .metrics import uplift_at_k

SEED = 42


# ── 1. Проверка рандомизации ──────────────────────────────────────────────────

def check_randomization(
    df: pd.DataFrame,
    treatment_col: str,
    feature_cols: Optional[List[str]] = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Проверяет баланс признаков между treatment и control.
    Использует t-test для числовых и chi-square для категориальных.

    Если A/B тест чистый — ни один признак не должен значимо
    отличаться между группами (p > alpha).

    Returns
    -------
    DataFrame с p-value и флагом имбаланса для каждого признака.
    """
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != treatment_col]

    results = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        try:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # t-test для числовых
                t_vals = treated[col].dropna()
                c_vals = control[col].dropna()
                _, p = stats.ttest_ind(t_vals, c_vals, equal_var=False)
                test = 'ttest'
                mean_t = t_vals.mean()
                mean_c = c_vals.mean()
                smd = abs(mean_t - mean_c) / (
                    np.sqrt((t_vals.std()**2 + c_vals.std()**2) / 2) + 1e-8
                )
            else:
                # chi-square для категориальных
                ct = pd.crosstab(df[col], df[treatment_col])
                _, p, _, _ = stats.chi2_contingency(ct)
                test  = 'chi2'
                mean_t = treated[col].mode()[0] if not treated[col].empty else None
                mean_c = control[col].mode()[0] if not control[col].empty else None
                smd = None

            results.append({
                'feature':    col,
                'test':       test,
                'p_value':    round(float(p), 4),
                'imbalanced': p < alpha,
                'mean_treat': mean_t,
                'mean_ctrl':  mean_c,
                'smd':        round(float(smd), 4) if smd is not None else None,
            })
        except Exception:
            continue

    result_df = pd.DataFrame(results).sort_values('p_value')

    if verbose:
        n_imbalanced = result_df['imbalanced'].sum()
        print(f'Проверка рандомизации: {len(result_df)} признаков')
        print(f'Имбалансных (p < {alpha}): {n_imbalanced}')
        if n_imbalanced > 0:
            print('⚠️  Имбалансные признаки:')
            print(result_df[result_df['imbalanced']][
                ['feature', 'p_value', 'mean_treat', 'mean_ctrl']
            ].to_string(index=False))
        else:
            print('✓ Рандомизация корректна')

    return result_df


# ── 2. Детекция data leakage ──────────────────────────────────────────────────

def check_leakage(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: Optional[List[str]] = None,
    threshold: float = 0.7,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Проверяет корреляцию признаков с treatment и outcome.

    Высокая корреляция признака с treatment (> threshold) может
    указывать на leakage — признак мог быть создан после назначения.

    Высокая корреляция с outcome — потенциальный post-treatment bias.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in [treatment_col, outcome_col, 'user_id']
        ]

    results = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        try:
            corr_treatment = abs(df[[col, treatment_col]].corr().iloc[0, 1])
            corr_outcome   = abs(df[[col, outcome_col]].corr().iloc[0, 1])
            results.append({
                'feature':         col,
                'corr_treatment':  round(float(corr_treatment), 4),
                'corr_outcome':    round(float(corr_outcome), 4),
                'leakage_risk':    corr_treatment > threshold,
            })
        except Exception:
            continue

    result_df = pd.DataFrame(results).sort_values(
        'corr_treatment', ascending=False
    )

    if verbose:
        n_risk = result_df['leakage_risk'].sum()
        print(f'Детекция leakage: {len(result_df)} признаков')
        print(f'Высокая корреляция с treatment (> {threshold}): {n_risk}')
        if n_risk > 0:
            print('⚠️  Риск leakage:')
            print(result_df[result_df['leakage_risk']][
                ['feature', 'corr_treatment', 'corr_outcome']
            ].head(10).to_string(index=False))
        else:
            print('✓ Leakage не обнаружен')

    return result_df


# ── 3. Permutation test ───────────────────────────────────────────────────────

def permutation_test(
    y: np.ndarray,
    treatment: np.ndarray,
    scores: np.ndarray,
    k: float = 0.10,
    n_permutations: int = 200,
    seed: int = SEED,
    verbose: bool = True,
) -> dict:
    """
    Permutation test для uplift@K.

    Проверяет H0: модель не лучше случайного ранжирования.
    Перемешивает treatment случайно и считает null distribution.

    Returns
    -------
    dict с observed, p_value, null_mean, null_std
    """
    rng = np.random.RandomState(seed)

    # Наблюдаемая метрика
    observed, _, _ = uplift_at_k(y, treatment, scores, k=k, n_boot=1)

    # Null distribution
    null_scores = []
    for _ in range(n_permutations):
        shuffled = rng.permutation(scores)
        val, _, _ = uplift_at_k(y, treatment, shuffled, k=k, n_boot=1)
        null_scores.append(val)

    null_scores = np.array(null_scores)
    p_value = (null_scores >= observed).mean()

    result = {
        'observed':   round(float(observed), 4),
        'p_value':    round(float(p_value), 4),
        'null_mean':  round(float(null_scores.mean()), 4),
        'null_std':   round(float(null_scores.std()), 4),
        'significant': p_value < 0.05,
    }

    if verbose:
        print(f'Permutation test (n={n_permutations}):')
        print(f'  Observed uplift@{int(k*100)}%: {result["observed"]:.4f}')
        print(f'  Null mean ± std: {result["null_mean"]:.4f} ± {result["null_std"]:.4f}')
        print(f'  p-value: {result["p_value"]:.4f}')
        flag = '✓ Значимо (p < 0.05)' if result['significant'] else '⚠️  Незначимо'
        print(f'  {flag}')

    return result


# ── 4. Learning curves ────────────────────────────────────────────────────────

def learning_curves(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    model_fn: Callable,
    fractions: List[float] = None,
    k: float = 0.10,
    n_boot: int = 50,
    seed: int = SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Строит learning curves: как uplift@K меняется с размером выборки.

    Помогает понять:
    - Достаточно ли данных?
    - Есть ли bias (модель не улучшается) или variance (нестабильна)?

    Parameters
    ----------
    model_fn : функция (Xt, yt, Xc, yc, Xv) -> scores
    fractions: доли от полного датасета [0.2, 0.4, 0.6, 0.8, 1.0]
    """
    if fractions is None:
        fractions = [0.2, 0.4, 0.6, 0.8, 1.0]

    rng = np.random.RandomState(seed)
    n   = len(X)
    results = []

    # Фиксированный holdout 20%
    hold_idx  = rng.choice(n, int(n * 0.2), replace=False)
    train_idx = np.setdiff1d(np.arange(n), hold_idx)

    X_hold = X.iloc[hold_idx].reset_index(drop=True)
    y_hold = y[hold_idx]
    w_hold = treatment[hold_idx]

    for frac in fractions:
        n_train = int(len(train_idx) * frac)
        idx     = rng.choice(train_idx, n_train, replace=False)

        X_tr = X.iloc[idx].reset_index(drop=True)
        y_tr = y[idx]
        w_tr = treatment[idx]

        Xt = X_tr[w_tr == 1].reset_index(drop=True); yt = y_tr[w_tr == 1]
        Xc = X_tr[w_tr == 0].reset_index(drop=True); yc = y_tr[w_tr == 0]

        try:
            scores = model_fn(Xt, yt, Xc, yc, X_hold)
            pt, lo, hi = uplift_at_k(y_hold, w_hold, scores, k=k, n_boot=n_boot)
        except Exception as e:
            pt, lo, hi = 0.0, 0.0, 0.0

        results.append({
            'fraction':    frac,
            'n_train':     n_train,
            'point':       pt,
            'lower_ci':    lo,
            'upper_ci':    hi,
        })

        if verbose:
            print(f'  frac={frac:.1f} n={n_train:6d}: '
                  f'point={pt:.4f} lower={lo:.4f}')

    return pd.DataFrame(results)


# ── 5. Repeated CV ────────────────────────────────────────────────────────────

def repeated_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    treatment: np.ndarray,
    model_fn: Callable,
    n_repeats: int = 3,
    n_folds: int = 3,
    k: float = 0.10,
    n_boot: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Repeated k-fold CV с разными random seeds.

    Более надёжная оценка чем одиночный CV —
    усредняет по нескольким разбиениям данных.

    Returns
    -------
    dict с mean_point, mean_lower, std_point, std_lower
    """
    all_pts, all_los = [], []

    for repeat in range(n_repeats):
        seed = SEED + repeat * 100
        kf   = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        pts, los = [], []

        for tr_idx, va_idx in kf.split(X):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            wtr, wva = treatment[tr_idx], treatment[va_idx]

            Xt = Xtr[wtr == 1].reset_index(drop=True); yt = ytr[wtr == 1]
            Xc = Xtr[wtr == 0].reset_index(drop=True); yc = ytr[wtr == 0]

            try:
                scores = model_fn(Xt, yt, Xc, yc, Xva)
                pt, lo, _ = uplift_at_k(yva, wva, scores, k=k, n_boot=n_boot)
                pts.append(pt); los.append(lo)
            except Exception:
                continue

        all_pts.extend(pts)
        all_los.extend(los)

        if verbose:
            print(f'  repeat {repeat+1}/{n_repeats}: '
                  f'point={np.mean(pts):.4f} lower={np.mean(los):.4f}')

    result = {
        'mean_point': round(float(np.mean(all_pts)), 4),
        'mean_lower': round(float(np.mean(all_los)), 4),
        'std_point':  round(float(np.std(all_pts)), 4),
        'std_lower':  round(float(np.std(all_los)), 4),
        'n_folds_total': len(all_pts),
    }

    if verbose:
        print(f'\nRepeated CV ({n_repeats}×{n_folds}-fold):')
        print(f'  point: {result["mean_point"]:.4f} ± {result["std_point"]:.4f}')
        print(f'  lower: {result["mean_lower"]:.4f} ± {result["std_lower"]:.4f}')

    return result


# ── Сводный отчёт ─────────────────────────────────────────────────────────────

def full_validation_report(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> dict:
    """
    Запускает все проверки данных и возвращает сводный отчёт.

    Используй перед обучением моделей.
    """
    print('=' * 55)
    print('ВАЛИДАЦИЯ ДАННЫХ')
    print('=' * 55)

    print('\n1. Проверка рандомизации')
    print('-' * 40)
    rand_df = check_randomization(
        df, treatment_col,
        feature_cols=feature_cols,
        verbose=verbose,
    )

    print('\n2. Детекция leakage')
    print('-' * 40)
    leak_df = check_leakage(
        df, treatment_col, outcome_col,
        feature_cols=feature_cols,
        verbose=verbose,
    )

    print('\n3. Базовая статистика')
    print('-' * 40)
    n_treat  = (df[treatment_col] == 1).sum()
    n_ctrl   = (df[treatment_col] == 0).sum()
    zero_rate = (df[outcome_col] == 0).mean()
    ate = (df[df[treatment_col]==1][outcome_col].mean() -
           df[df[treatment_col]==0][outcome_col].mean())

    print(f'  Treatment: {n_treat:,} | Control: {n_ctrl:,}')
    print(f'  Balance: {n_treat/(n_treat+n_ctrl):.1%} / {n_ctrl/(n_treat+n_ctrl):.1%}')
    print(f'  Zero rate в outcome: {zero_rate:.1%}')
    print(f'  ATE: {ate:.4f}')

    balance_ok = abs(n_treat/(n_treat+n_ctrl) - 0.5) < 0.05
    print(f'  {"✓ Баланс OK" if balance_ok else "⚠️  Дисбаланс групп"}')

    return {
        'randomization': rand_df,
        'leakage':       leak_df,
        'n_treat':       int(n_treat),
        'n_ctrl':        int(n_ctrl),
        'zero_rate':     float(zero_rate),
        'ate':           float(ate),
        'balance_ok':    balance_ok,
    }