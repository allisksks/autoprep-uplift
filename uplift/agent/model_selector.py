"""
uplift/agent/model_selector.py

LLM-агент для выбора оптимального ансамбля.

Workflow:
  1. Получает CV результаты всех моделей
  2. Получает результаты всех комбинаций на holdout
  3. Выбирает оптимальную комбинацию с объяснением
  4. Возвращает финальные веса для предикта
"""

import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from typing import Dict, Tuple
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    _MODEL  = 'claude-sonnet-4-5'
except Exception:
    _CLIENT = None
    _MODEL  = None


_SYSTEM_PROMPT = """You are an expert in causal inference and uplift modeling.

Your task: analyze ALL possible model combinations and select the optimal ensemble.

You receive:
1. CV results for each model (cv_point, cv_lower_ci, gap)
2. Holdout results for all combinations (single models, pairs, larger ensembles)
   with different weighting strategies

Your goal: find the combination that maximizes holdout uplift@10 lower CI
while minimizing overfitting risk.

Consider:
- Holdout performance is ground truth (not CV)
- Prefer combinations with low correlation between models (diversity)
- A single model can be better than an ensemble if others add noise
- Weighting strategy matters — equal weights often beats complex strategies

Return a JSON object with this exact structure:
{
  "selected_models": ["model1", "model2"],
  "weights": {"model1": 0.6, "model2": 0.4},
  "strategy": "strategy_name",
  "holdout_lower_ci": 0.0,
  "explanation": "Detailed explanation of why this combination was chosen",
  "alternatives_considered": ["What else was tried and why rejected"],
  "overfitting_assessment": "Assessment of overfitting risk"
}

Weights must sum to 1.0. Return ONLY valid JSON."""


def _compute_all_combinations(
    predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
    w_val: np.ndarray,
    cv_results: Dict[str, Tuple[float, float]],
    metric: str = 'uplift@10',
    n_boot: int = 100,
) -> pd.DataFrame:
    """
    Перебирает все возможные комбинации моделей и стратегий.
    Возвращает таблицу с результатами.
    """
    from ..metrics import evaluate

    model_names = list(predictions.keys())
    rows = []

    # Стратегии взвешивания
    def get_weights(names, strategy, cv_res):
        if strategy == 'equal':
            return {m: 1/len(names) for m in names}
        if strategy == 'ci_weights':
            vals = {m: max(cv_res[m][1], 0) for m in names}
            total = sum(vals.values()) + 1e-8
            return {m: v/total for m, v in vals.items()}
        if strategy == 'gap_weights':
            gaps = {m: cv_res[m][0] - cv_res[m][1] for m in names}
            inv  = {m: 1/(g+0.1) for m, g in gaps.items()}
            total = sum(inv.values())
            return {m: v/total for m, v in inv.items()}
        if strategy == 'rank_weights':
            sorted_m = sorted(names, key=lambda m: cv_res[m][1], reverse=True)
            raw = {m: 1/(i+1) for i, m in enumerate(sorted_m)}
            total = sum(raw.values())
            return {m: v/total for m, v in raw.items()}
        return {m: 1/len(names) for m in names}

    strategies = ['equal', 'ci_weights', 'gap_weights', 'rank_weights']

    # Одиночные модели
    for name in model_names:
        scores = predictions[name]
        score  = evaluate(y_val, w_val, scores, metric=metric, n_boot=n_boot)
        rows.append({
            'combination': name,
            'n_models':    1,
            'strategy':    'single',
            'models':      [name],
            'weights':     {name: 1.0},
            'holdout_lower_ci': score,
            'cv_lower_ci': cv_results.get(name, (0, 0))[1],
            'gap':         cv_results.get(name, (0, 0))[0] - cv_results.get(name, (0, 0))[1],
        })

    # Все пары
    for m1, m2 in combinations(model_names, 2):
        for strategy in strategies:
            w = get_weights([m1, m2], strategy, cv_results)
            scores = w[m1]*predictions[m1] + w[m2]*predictions[m2]
            score  = evaluate(y_val, w_val, scores, metric=metric, n_boot=n_boot)
            rows.append({
                'combination': f'{m1}+{m2}',
                'n_models':    2,
                'strategy':    strategy,
                'models':      [m1, m2],
                'weights':     w,
                'holdout_lower_ci': score,
                'cv_lower_ci': None,
                'gap':         None,
            })

    # Тройки
    for combo in combinations(model_names, 3):
        for strategy in ['equal', 'ci_weights']:
            names = list(combo)
            w = get_weights(names, strategy, cv_results)
            scores = sum(w[m]*predictions[m] for m in names)
            score  = evaluate(y_val, w_val, scores, metric=metric, n_boot=n_boot)
            rows.append({
                'combination': '+'.join(names),
                'n_models':    3,
                'strategy':    strategy,
                'models':      names,
                'weights':     w,
                'holdout_lower_ci': score,
                'cv_lower_ci': None,
                'gap':         None,
            })

    # Все модели
    for strategy in ['equal', 'ci_weights']:
        w = get_weights(model_names, strategy, cv_results)
        scores = sum(w[m]*predictions[m] for m in model_names)
        score  = evaluate(y_val, w_val, scores, metric=metric, n_boot=n_boot)
        rows.append({
            'combination': 'all_models',
            'n_models':    len(model_names),
            'strategy':    strategy,
            'models':      model_names,
            'weights':     w,
            'holdout_lower_ci': score,
            'cv_lower_ci': None,
            'gap':         None,
        })

    df = pd.DataFrame(rows).sort_values('holdout_lower_ci', ascending=False)
    return df


def select_ensemble(
    predictions: Dict[str, np.ndarray],
    y_val: np.ndarray,
    w_val: np.ndarray,
    cv_results: Dict[str, Tuple[float, float]],
    metric: str = 'uplift@10',
    n_boot: int = 100,
    verbose: bool = True,
) -> dict:
    """
    Полный перебор всех комбинаций + LLM агент выбирает лучшую.

    Parameters
    ----------
    predictions : {model_name: uplift scores на holdout}
    y_val       : outcome на holdout
    w_val       : treatment на holdout
    cv_results  : {model_name: (cv_point, cv_lower_ci)}
    metric      : метрика для оценки
    n_boot      : bootstrap итерации

    Returns
    -------
    dict с selected_models, weights, explanation
    """
    if _CLIENT is None:
        raise RuntimeError('ANTHROPIC_API_KEY не найден в .env')

    if verbose:
        print('Ensemble Agent: перебираем все комбинации...')

    # Перебираем все комбинации
    combos_df = _compute_all_combinations(
        predictions, y_val, w_val, cv_results,
        metric=metric, n_boot=n_boot
    )

    if verbose:
        print(f'  Проверено комбинаций: {len(combos_df)}')
        print('\n  Топ-10 комбинаций:')
        top10 = combos_df.head(10)[['combination','strategy','holdout_lower_ci','n_models']]
        print(top10.to_string(index=False))

    # Готовим данные для агента
    cv_table = []
    for name, (pt, lo) in cv_results.items():
        cv_table.append({
            'model': name,
            'cv_point': pt,
            'cv_lower_ci': lo,
            'gap': round(pt - lo, 4),
        })

    top20_combos = combos_df.head(20).copy()
    top20_combos['weights'] = top20_combos['weights'].apply(
        lambda w: {k: round(v, 3) for k, v in w.items()}
    )

    user_prompt = f"""CV results for all models:
{json.dumps(cv_table, indent=2)}

Top-20 combinations by holdout uplift@10 lower CI:
{top20_combos[['combination','strategy','holdout_lower_ci','n_models','weights']].to_json(orient='records', indent=2)}

Total combinations evaluated: {len(combos_df)}
Best single model holdout lower CI: {combos_df[combos_df.n_models==1]['holdout_lower_ci'].max():.4f}
Best pair holdout lower CI: {combos_df[combos_df.n_models==2]['holdout_lower_ci'].max():.4f}
Best triple holdout lower CI: {combos_df[combos_df.n_models==3]['holdout_lower_ci'].max():.4f}

Select the optimal ensemble. Consider diversity, stability, and overfitting risk."""

    if verbose:
        print('\nEnsemble Agent: запрашиваю решение у LLM...')

    response = _CLIENT.messages.create(
        model=_MODEL,
        max_tokens=1500,
        system=_SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_prompt}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith('```json'):
        raw = raw[7:]
    if raw.startswith('```'):
        raw = raw[3:]
    if raw.endswith('```'):
        raw = raw[:-3]
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise RuntimeError(f'Агент вернул невалидный JSON: {e}\n{raw}')

    # Добавляем таблицу комбинаций для логирования
    result['all_combinations'] = combos_df

    if verbose:
        print('\n── Решение агента ───────────────────────────────')
        print(f"Выбрано: {result['selected_models']}")
        print(f"Стратегия: {result['strategy']}")
        print(f"Веса: {result['weights']}")
        print(f"Holdout lower CI: {result['holdout_lower_ci']:.4f}")
        print(f"\nОбъяснение:\n{result['explanation']}")
        if result.get('overfitting_assessment'):
            print(f"\nОценка оверфиттинга:\n{result['overfitting_assessment']}")

    return result


def select_top3(
    cv_results: Dict[str, Tuple[float, float]],
    verbose: bool = True,
) -> dict:
    """
    Упрощённый выбор топ-3 по CV (без holdout данных).
    Используется когда holdout предсказания недоступны.
    """
    if _CLIENT is None:
        raise RuntimeError('ANTHROPIC_API_KEY не найден в .env')

    rows = []
    for name, (pt, lo) in cv_results.items():
        rows.append({
            'model':       name,
            'cv_point':    pt,
            'cv_lower_ci': lo,
            'gap':         round(pt - lo, 4),
        })

    df = pd.DataFrame(rows).sort_values('cv_lower_ci', ascending=False)

    if verbose:
        print('\nModel Selector: результаты CV:')
        print(df.to_string(index=False))
        print('\nModel Selector: запрашиваю рекомендацию...')

    system = """You are an expert in uplift modeling.
Select top-3 models and recommend ensemble strategy.
Return JSON: {"top3": [{"rank":1,"model":"name","cv_lower_ci":0.0,"cv_point":0.0,"explanation":"..."},...],
"ensemble_recommendation":"...","warnings":[]}
Return ONLY valid JSON."""

    response = _CLIENT.messages.create(
        model=_MODEL,
        max_tokens=1000,
        system=system,
        messages=[{'role': 'user', 'content':
            f'CV results:\n{df.to_json(orient="records", indent=2)}\nSelect top-3.'}],
    )

    raw = response.content[0].text.strip()
    for tag in ['```json', '```']:
        if raw.startswith(tag):
            raw = raw[len(tag):]
    if raw.endswith('```'):
        raw = raw[:-3]

    result = json.loads(raw.strip())

    if verbose:
        print('\n── Рекомендация ─────────────────────────────────')
        for item in result.get('top3', []):
            print(f"\n#{item['rank']} {item['model']}")
            print(f"   CV lower CI: {item['cv_lower_ci']:.4f}")
            print(f"   {item['explanation']}")
        print(f"\nАнсамбль: {result.get('ensemble_recommendation','')}")

    return result


def format_top3_table(selection: dict) -> pd.DataFrame:
    rows = []
    for item in selection.get('top3', []):
        rows.append({
            'rank':        item['rank'],
            'model':       item['model'],
            'cv_lower_ci': item['cv_lower_ci'],
            'cv_point':    item['cv_point'],
            'explanation': item['explanation'],
        })
    return pd.DataFrame(rows)
