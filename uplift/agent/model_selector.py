"""
uplift/agent/model_selector.py

LLM-агент для выбора топ-3 моделей с объяснением.

Получает таблицу результатов CV и возвращает:
  - топ-3 модели отсортированные по cv_lower_ci
  - объяснение на естественном языке для каждой
  - рекомендацию по ансамблю
"""

import os
import json
import pandas as pd
from typing import Dict, Tuple, List
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    _MODEL  = 'claude-sonnet-4-20250514'
except Exception:
    _CLIENT = None
    _MODEL  = None


_SYSTEM_PROMPT = """You are an expert in causal inference and uplift modeling.

Your task: analyze cross-validation results for uplift models and recommend top-3 models.

Consider:
- cv_lower_ci: lower bound of 80% bootstrap CI — primary stability metric
- cv_point: point estimate of uplift@10
- gap between point and lower_ci: indicates overfitting risk
- model characteristics (DR-learner is doubly robust, Hurdle handles zero-inflation, etc.)

Return a JSON object with this exact structure:
{
  "top3": [
    {
      "rank": 1,
      "model": "model_name",
      "cv_lower_ci": 0.0,
      "cv_point": 0.0,
      "explanation": "Why this model performs well and is stable"
    },
    ...
  ],
  "ensemble_recommendation": "Brief recommendation on whether to ensemble and how to weight",
  "warnings": ["Any concerns about overfitting or instability"]
}

Return ONLY valid JSON, no markdown, no extra text."""


def select_top3(
    cv_results: Dict[str, Tuple[float, float]],
    verbose: bool = True,
) -> dict:
    """
    Выбирает топ-3 модели через LLM-агента.

    Parameters
    ----------
    cv_results : {model_name: (cv_point, cv_lower_ci)}
    verbose    : печатать прогресс

    Returns
    -------
    dict с top3, ensemble_recommendation, warnings
    """
    if _CLIENT is None:
        raise RuntimeError(
            'Anthropic client не инициализирован. '
            'Проверь ANTHROPIC_API_KEY в .env'
        )

    # Формируем таблицу результатов
    rows = []
    for name, (pt, lo) in cv_results.items():
        rows.append({
            'model':        name,
            'cv_point':     pt,
            'cv_lower_ci':  lo,
            'gap':          round(pt - lo, 4),
        })

    df = pd.DataFrame(rows).sort_values('cv_lower_ci', ascending=False)

    if verbose:
        print('\nModel Selector Agent: результаты CV:')
        print(df.to_string(index=False))
        print('\nModel Selector Agent: запрашиваю рекомендацию...')

    user_prompt = f"""Cross-validation results for uplift models:

{df.to_json(orient='records', indent=2)}

Metric: uplift@10 with 80% bootstrap CI on held-out validation folds.
cv_lower_ci = lower bound of CI (primary metric — penalizes instability).
gap = cv_point - cv_lower_ci (overfitting indicator).

Select top-3 models and provide recommendations."""

    response = _CLIENT.messages.create(
        model=_MODEL,
        max_tokens=1000,
        system=_SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_prompt}],
    )

    raw = response.content[0].text.strip()

    # Убираем markdown если есть
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
        raise RuntimeError(f'Агент вернул невалидный JSON: {e}\n\nОтвет: {raw}')

    if verbose:
        print('\n── Рекомендация агента ──────────────────────────')
        for item in result.get('top3', []):
            print(f"\n#{item['rank']} {item['model']}")
            print(f"   CV lower CI: {item['cv_lower_ci']:.4f}  "
                  f"point: {item['cv_point']:.4f}")
            print(f"   {item['explanation']}")

        print(f"\nАнсамбль: {result.get('ensemble_recommendation', '')}")

        warnings = result.get('warnings', [])
        if warnings:
            print('\nПредупреждения:')
            for w in warnings:
                print(f'  ⚠️  {w}')

    return result


def format_top3_table(selection: dict) -> pd.DataFrame:
    """
    Форматирует результат select_top3() в DataFrame для отображения.
    """
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