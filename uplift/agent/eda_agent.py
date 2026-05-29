"""
uplift/agent/eda_agent.py

LLM-агент для автоматического EDA и генерации препроцессинга.

Workflow:
  1. Получает агрегированную статистику по train (не полные данные)
  2. Генерирует код fit_preprocess() + apply_preprocess()
  3. Верифицирует код на маленьком срезе данных
  4. Итеративно правит если верификация не прошла
  5. Возвращает финальные функции готовые к применению
"""

import os
import json
import textwrap
import traceback
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    _MODEL  = 'claude-sonnet-4-20250514'
except Exception:
    _CLIENT = None
    _MODEL  = None


# ── Сбор статистики для агента ────────────────────────────────────────────────

def _collect_stats(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    n_sample: int = 20,
) -> dict:
    """
    Собирает агрегированную статистику по датасету.
    Передаём агенту только статистику, не полные данные —
    это экономит токены и защищает данные.
    """
    service_cols = ['user_id', treatment_col, outcome_col]

    stats = {
        'shape':           list(df.shape),
        'treatment_col':   treatment_col,
        'outcome_col':     outcome_col,
        'service_cols':    service_cols,
        'columns':         df.columns.tolist(),
        'dtypes':          df.dtypes.astype(str).to_dict(),
        'missing_rate':    df.isnull().mean().round(4).to_dict(),
        'nunique':         df.nunique().to_dict(),
        'sample_head':     df.head(n_sample).to_dict(orient='records'),
    }

    # Числовые: describe
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        stats['numeric_describe'] = num_df.describe().round(4).to_dict()

    # Категориальные: топ-5 значений
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    stats['categorical_cols'] = cat_cols
    stats['cat_top5'] = {}
    for col in cat_cols:
        stats['cat_top5'][col] = df[col].value_counts().head(5).to_dict()

    # Доля нулей в outcome
    stats['outcome_zero_rate'] = float((df[outcome_col] == 0).mean())
    stats['outcome_describe']  = df[outcome_col].describe().round(4).to_dict()

    return stats


# ── Промпт ────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert ML engineer specializing in causal inference and uplift modeling.

Your task: analyze dataset statistics and generate Python preprocessing code for uplift modeling.

CRITICAL RULES:
1. Generated code must have NO data leakage: fit_preprocess() learns parameters only from train data.
   apply_preprocess() uses those parameters for both train and test.
2. Never use outcome_col or treatment_col as features.
3. Handle NaN values explicitly.
4. Return ONLY valid Python code, no markdown, no explanations outside comments.
5. The code must be importable and executable.

Generate exactly two functions:
- fit_preprocess(df, treatment_col, outcome_col) -> dict (stats)
- apply_preprocess(df, stats) -> pd.DataFrame

The stats dict must be JSON-serializable (no numpy types, no pandas objects).
"""

_USER_PROMPT_TEMPLATE = """Dataset statistics:

{stats_json}

Generate fit_preprocess() and apply_preprocess() for this dataset.
Consider:
- Missing rates per column (high missing > 0.5 → drop)
- Categorical columns (need encoding)
- Flag/binary columns (fill NaN with 0)
- Numeric columns (fill NaN with median from train)
- Service columns to exclude: {service_cols}

Return only Python code."""


# ── Верификация ───────────────────────────────────────────────────────────────

def _verify_code(
    code: str,
    df_sample: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> Tuple[bool, str]:
    """
    Запускает сгенерированный код на маленьком срезе данных.
    Проверяет:
      - код выполняется без ошибок
      - нет NaN в результате (кроме service cols)
      - типы корректные
    """
    namespace = {
        'pd': pd,
        'np': np,
    }

    try:
        exec(code, namespace)
    except Exception as e:
        return False, f'Syntax/runtime error: {e}\n{traceback.format_exc()}'

    if 'fit_preprocess' not in namespace:
        return False, 'fit_preprocess function not found in generated code'
    if 'apply_preprocess' not in namespace:
        return False, 'apply_preprocess function not found in generated code'

    try:
        stats = namespace['fit_preprocess'](
            df_sample.copy(), treatment_col, outcome_col
        )
    except Exception as e:
        return False, f'fit_preprocess() failed: {e}\n{traceback.format_exc()}'

    try:
        result = namespace['apply_preprocess'](df_sample.copy(), stats)
    except Exception as e:
        return False, f'apply_preprocess() failed: {e}\n{traceback.format_exc()}'

    # Проверяем что нет NaN в признаках
    service = ['user_id', treatment_col, outcome_col]
    feature_cols = [c for c in result.columns if c not in service]
    nan_count = result[feature_cols].isnull().sum().sum()
    if nan_count > 0:
        bad_cols = result[feature_cols].columns[
            result[feature_cols].isnull().any()
        ].tolist()
        return False, f'NaN remains in columns after preprocessing: {bad_cols}'

    # Проверяем JSON-сериализуемость stats
    try:
        json.dumps(stats)
    except Exception as e:
        return False, f'stats is not JSON-serializable: {e}'

    return True, 'OK'


# ── Основная функция агента ───────────────────────────────────────────────────

def generate_preprocess(
    train_df: pd.DataFrame,
    treatment_col: str = 'treatment_flg',
    outcome_col: str   = 'rec_spend',
    max_iterations: int = 3,
    verbose: bool = True,
) -> Tuple[str, dict]:
    """
    Генерирует код препроцессинга через LLM-агента.

    Parameters
    ----------
    train_df       : обучающая выборка
    treatment_col  : название колонки treatment
    outcome_col    : название колонки outcome
    max_iterations : максимум итераций верификации
    verbose        : печатать прогресс

    Returns
    -------
    (code_str, preproc_stats)
      code_str      : строка с Python кодом
      preproc_stats : словарь параметров препроцессинга
    """
    if _CLIENT is None:
        raise RuntimeError(
            'Anthropic client не инициализирован. '
            'Проверь ANTHROPIC_API_KEY в .env'
        )

    if verbose:
        print('EDA Agent: сбор статистики...')

    stats = _collect_stats(train_df, treatment_col, outcome_col)
    df_sample = train_df.sample(
        n=min(500, len(train_df)), random_state=42
    ).reset_index(drop=True)

    # Компактный JSON для промпта
    stats_for_prompt = {
        'shape':            stats['shape'],
        'missing_rate':     {k: v for k, v in stats['missing_rate'].items() if v > 0},
        'dtypes':           stats['dtypes'],
        'categorical_cols': stats['categorical_cols'],
        'cat_top5':         stats['cat_top5'],
        'outcome_zero_rate': stats['outcome_zero_rate'],
        'service_cols':     stats['service_cols'],
        'high_missing_cols': [k for k, v in stats['missing_rate'].items() if v > 0.5],
    }

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        stats_json=json.dumps(stats_for_prompt, indent=2, ensure_ascii=False),
        service_cols=stats['service_cols'],
    )

    messages = [{'role': 'user', 'content': user_prompt}]
    code = None

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f'EDA Agent: генерация кода (итерация {iteration}/{max_iterations})...')

        response = _CLIENT.messages.create(
            model=_MODEL,
            max_tokens=2000,
            system=_SYSTEM_PROMPT,
            messages=messages,
        )
        code = response.content[0].text.strip()

        # Убираем markdown если агент добавил
        if code.startswith('```python'):
            code = code[9:]
        if code.startswith('```'):
            code = code[3:]
        if code.endswith('```'):
            code = code[:-3]
        code = code.strip()

        if verbose:
            print(f'EDA Agent: верификация кода...')

        ok, error_msg = _verify_code(
            code, df_sample, treatment_col, outcome_col
        )

        if ok:
            if verbose:
                print(f'EDA Agent: код верифицирован ✓')
            break
        else:
            if verbose:
                print(f'EDA Agent: ошибка верификации — {error_msg[:200]}')
                print(f'EDA Agent: отправляю ошибку обратно агенту...')

            # Добавляем ошибку в контекст для следующей итерации
            messages.append({'role': 'assistant', 'content': code})
            messages.append({
                'role': 'user',
                'content': (
                    f'The code failed verification with this error:\n\n{error_msg}\n\n'
                    'Please fix the code and return the corrected version only.'
                )
            })

    if code is None:
        raise RuntimeError('Агент не сгенерировал код')

    # Выполняем финальный код и получаем stats
    namespace = {'pd': pd, 'np': np}
    exec(code, namespace)
    preproc_stats = namespace['fit_preprocess'](
        train_df.copy(), treatment_col, outcome_col
    )

    if verbose:
        print(f'EDA Agent: готово. Признаков после препроцессинга: '
              f'{len([k for k in preproc_stats.get("num_cols", [])])} числовых.')

    return code, preproc_stats