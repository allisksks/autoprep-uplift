"""
uplift/agent/eda_agent.py

LLM-агент для EDA, препроцессинга и feature engineering.

Workflow:
  1. generate_preprocess() — базовая очистка данных
  2. generate_features()   — feature engineering на очищенных данных
  3. generate_eda_report() — EDA отчёт для графиков

Разделение на два шага позволяет:
  - Агент видит уже чистые данные при генерации признаков
  - FE можно запускать отдельно или пропускать
  - Чёткое разделение ответственности
"""

import os
import json
import traceback
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from typing import Tuple, Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

try:
    import anthropic
    _CLIENT = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    _MODEL  = 'claude-sonnet-4-5'
except Exception:
    _CLIENT = None
    _MODEL  = None


# ══════════════════════════════════════════════════════════════════════════════
# СБОР СТАТИСТИКИ
# ══════════════════════════════════════════════════════════════════════════════

def _collect_stats(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    n_sample: int = 20,
) -> dict:
    """
    Собирает расширенную статистику для агента.
    Включает skewness, корреляции, баланс групп.
    """
    service_cols = ['user_id', treatment_col, outcome_col]

    stats = {
        'shape':         list(df.shape),
        'treatment_col': treatment_col,
        'outcome_col':   outcome_col,
        'service_cols':  service_cols,
        'columns':       df.columns.tolist(),
        'dtypes':        df.dtypes.astype(str).to_dict(),
        'missing_rate':  df.isnull().mean().round(4).to_dict(),
        'nunique':       df.nunique().to_dict(),
        'sample_head':   df.head(n_sample).to_dict(orient='records'),
    }

    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        desc = num_df.describe()
        stats['numeric_describe'] = {
            col: {
                'mean': float(desc[col]['mean']),
                'std':  float(desc[col]['std']),
                'min':  float(desc[col]['min']),
                '25%':  float(desc[col]['25%']),
                '50%':  float(desc[col]['50%']),
                '75%':  float(desc[col]['75%']),
                'max':  float(desc[col]['max']),
            }
            for col in desc.columns
        }

        # Skewness — ключевой сигнал для log-трансформации
        skewness = {}
        for col in num_df.columns:
            try:
                s = float(scipy_stats.skew(num_df[col].dropna()))
                if not np.isnan(s):
                    skewness[col] = round(s, 2)
            except Exception:
                pass
        stats['skewness'] = skewness
        stats['highly_skewed'] = [k for k, v in skewness.items() if abs(v) > 2]

        # Корреляции с outcome
        if outcome_col in df.columns:
            corr = num_df.corrwith(df[outcome_col]).round(4)
            stats['corr_with_outcome'] = {
                k: float(v) for k, v in corr.dropna().items()
                if k not in service_cols
            }

        # Корреляции с treatment
        if treatment_col in df.columns:
            corr_t = num_df.corrwith(df[treatment_col]).round(4)
            stats['corr_with_treatment'] = {
                k: float(v) for k, v in corr_t.dropna().items()
                if k not in service_cols
            }

        # Доля нулей по каждому признаку
        zero_rates = (num_df == 0).mean().round(4)
        stats['zero_rates'] = {
            k: float(v) for k, v in zero_rates.items()
            if v > 0.1 and k not in service_cols
        }

    # Категориальные
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    stats['categorical_cols'] = [c for c in cat_cols if c not in service_cols]
    stats['cat_top5'] = {}
    for col in stats['categorical_cols']:
        stats['cat_top5'][col] = df[col].value_counts().head(5).to_dict()

    # Outcome
    stats['outcome_zero_rate'] = float((df[outcome_col] == 0).mean())
    stats['outcome_describe']  = {
        k: float(v) for k, v in df[outcome_col].describe().items()
    }
    stats['outcome_is_binary'] = bool(df[outcome_col].nunique() <= 2)
    try:
        stats['outcome_skewness'] = float(scipy_stats.skew(df[outcome_col].dropna()))
    except Exception:
        stats['outcome_skewness'] = 0.0

    # Treatment баланс
    vc = df[treatment_col].value_counts(normalize=True)
    stats['treatment_balance'] = {str(k): round(float(v), 4) for k, v in vc.items()}

    # Топ признаков по корреляции с outcome
    if 'corr_with_outcome' in stats:
        sorted_corr = sorted(
            stats['corr_with_outcome'].items(),
            key=lambda x: abs(x[1]), reverse=True
        )
        stats['top_features_by_corr'] = dict(sorted_corr[:20])
        stats['high_corr_features']   = [k for k, v in sorted_corr if abs(v) > 0.1]

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# ПРЕПРОЦЕССИНГ
# ══════════════════════════════════════════════════════════════════════════════

_PREPROC_SYSTEM = """You are an expert ML engineer specializing in causal inference and uplift modeling.

Your task: generate Python preprocessing code for uplift modeling datasets.

CRITICAL RULES:
1. NO data leakage: fit_preprocess() learns ALL parameters from train only.
   apply_preprocess() uses those saved parameters for train AND test.
2. NEVER use outcome_col or treatment_col as input features.
3. Handle ALL NaN values explicitly.
4. Return ONLY valid Python code, no markdown, no comments outside code.
5. stats dict must be JSON-serializable (convert numpy types to Python native).

Generate exactly:
- fit_preprocess(df, treatment_col, outcome_col) -> dict
- apply_preprocess(df, stats) -> pd.DataFrame

Steps to implement:
1. Identify service columns (user_id, treatment_col, outcome_col) — exclude from features
2. Drop columns with >50% missing
3. Add binary missing indicator for columns with 5-50% missing (before imputing)
4. Fill numeric NaN with median (computed on train, saved in stats)
5. Fill binary/flag NaN with 0
6. Label encode categorical columns (save mapping in stats, use -1 for unseen)
7. Ensure all remaining features are numeric
"""

_PREPROC_USER = """Dataset statistics:

{stats_json}

Service columns to exclude: {service_cols}
High missing columns (>50%, drop): {high_missing}
Categorical columns: {cat_cols}

Generate fit_preprocess() and apply_preprocess(). Return only code."""


def _verify_code(
    code: str,
    df_sample: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    fn_names: List[str] = None,
) -> Tuple[bool, str]:
    """Верифицирует сгенерированный код на срезе данных."""
    if fn_names is None:
        fn_names = ['fit_preprocess', 'apply_preprocess']

    namespace = {'pd': pd, 'np': np}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f'Syntax/runtime error: {e}\n{traceback.format_exc()}'

    for fn in fn_names:
        if fn not in namespace:
            return False, f'Function {fn} not found in generated code'

    try:
        if 'fit_preprocess' in fn_names:
            result_stats = namespace['fit_preprocess'](
                df_sample.copy(), treatment_col, outcome_col
            )
        if 'apply_preprocess' in fn_names:
            result_df = namespace['apply_preprocess'](df_sample.copy(), result_stats)
    except Exception as e:
        return False, f'Execution failed: {e}\n{traceback.format_exc()}'

    # NaN check
    service = ['user_id', treatment_col, outcome_col]
    feat_cols = [c for c in result_df.columns if c not in service]
    nan_count = result_df[feat_cols].isnull().sum().sum()
    if nan_count > 0:
        bad = result_df[feat_cols].columns[result_df[feat_cols].isnull().any()].tolist()
        return False, f'NaN remains in features: {bad}'

    # JSON check
    try:
        json.dumps(result_stats)
    except Exception as e:
        return False, f'stats not JSON-serializable: {e}'

    # Non-numeric check
    non_num = [c for c in feat_cols if result_df[c].dtype == 'object']
    if non_num:
        return False, f'Non-numeric features remain: {non_num}'

    return True, 'OK'


def _run_agent(
    system_prompt: str,
    user_prompt: str,
    df_sample: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    max_iterations: int = 3,
    max_tokens: int = 3000,
    fn_names: List[str] = None,
    verbose: bool = True,
    step_name: str = 'Agent',
) -> Tuple[str, dict]:
    """Универсальный запуск агента с итеративной верификацией."""
    messages = [{'role': 'user', 'content': user_prompt}]
    code = None

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f'{step_name}: генерация кода (итерация {iteration}/{max_iterations})...')

        response = _CLIENT.messages.create(
            model=_MODEL,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=messages,
        )
        code = response.content[0].text.strip()

        # Убираем markdown
        for tag in ['```python', '```']:
            if code.startswith(tag):
                code = code[len(tag):]
        if code.endswith('```'):
            code = code[:-3]
        code = code.strip()

        if verbose:
            print(f'{step_name}: верификация...')

        ok, error_msg = _verify_code(
            code, df_sample, treatment_col, outcome_col, fn_names
        )

        if ok:
            if verbose:
                print(f'{step_name}: код верифицирован ✓')
            return code
        else:
            if verbose:
                print(f'{step_name}: ошибка — {error_msg[:150]}')
                print(f'{step_name}: исправляю...')
            messages.append({'role': 'assistant', 'content': code})
            messages.append({
                'role': 'user',
                'content': f'Code failed:\n\n{error_msg}\n\nFix and return corrected code only.'
            })

    return code  # возвращаем последний вариант даже если не прошёл


def generate_preprocess(
    train_df: pd.DataFrame,
    treatment_col: str  = 'treatment_flg',
    outcome_col: str    = 'rec_spend',
    max_iterations: int = 3,
    verbose: bool       = True,
) -> Tuple[str, dict]:
    """
    Шаг 1: Базовый препроцессинг через LLM-агента.
    Очищает данные, энкодирует категориальные, обрабатывает пропуски.
    """
    if _CLIENT is None:
        raise RuntimeError('ANTHROPIC_API_KEY не найден в .env')

    if verbose:
        print('EDA Agent: сбор статистики...')

    raw_stats = _collect_stats(train_df, treatment_col, outcome_col)
    df_sample = train_df.sample(
        n=min(500, len(train_df)), random_state=42
    ).reset_index(drop=True)

    stats_for_prompt = {
        'shape':            raw_stats['shape'],
        'missing_rate':     {k: v for k, v in raw_stats['missing_rate'].items() if v > 0},
        'dtypes':           raw_stats['dtypes'],
        'categorical_cols': raw_stats['categorical_cols'],
        'cat_top5':         raw_stats['cat_top5'],
        'outcome_zero_rate':raw_stats['outcome_zero_rate'],
        'outcome_is_binary':raw_stats['outcome_is_binary'],
        'service_cols':     raw_stats['service_cols'],
        'treatment_balance':raw_stats.get('treatment_balance', {}),
    }

    user_prompt = _PREPROC_USER.format(
        stats_json   = json.dumps(stats_for_prompt, indent=2, ensure_ascii=False),
        service_cols = raw_stats['service_cols'],
        high_missing = [k for k, v in raw_stats['missing_rate'].items() if v > 0.5][:10],
        cat_cols     = raw_stats['categorical_cols'],
    )

    code = _run_agent(
        system_prompt  = _PREPROC_SYSTEM,
        user_prompt    = user_prompt,
        df_sample      = df_sample,
        treatment_col  = treatment_col,
        outcome_col    = outcome_col,
        max_iterations = max_iterations,
        max_tokens     = 3000,
        fn_names       = ['fit_preprocess', 'apply_preprocess'],
        verbose        = verbose,
        step_name      = 'EDA Agent',
    )

    namespace = {'pd': pd, 'np': np}
    exec(code, namespace)
    preproc_stats = namespace['fit_preprocess'](
        train_df.copy(), treatment_col, outcome_col
    )

    if verbose:
        sample_result = namespace['apply_preprocess'](df_sample.copy(), preproc_stats)
        service = ['user_id', treatment_col, outcome_col]
        n_feat = len([c for c in sample_result.columns if c not in service])
        print(f'EDA Agent: готово. Признаков после препроцессинга: {n_feat}')

    # Сохраняем raw_stats для FE шага
    preproc_stats['_raw_stats'] = {
        'skewness':             raw_stats.get('skewness', {}),
        'highly_skewed':        raw_stats.get('highly_skewed', []),
        'corr_with_outcome':    raw_stats.get('corr_with_outcome', {}),
        'top_features_by_corr': raw_stats.get('top_features_by_corr', {}),
        'high_corr_features':   raw_stats.get('high_corr_features', []),
        'zero_rates':           raw_stats.get('zero_rates', {}),
        'outcome_skewness':     raw_stats.get('outcome_skewness', 0.0),
        'treatment_balance':    raw_stats.get('treatment_balance', {}),
    }

    return code, preproc_stats


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

_FE_SYSTEM = """You are an expert ML engineer specializing in feature engineering for causal inference and uplift modeling.

Your task: generate feature engineering code for already-preprocessed uplift modeling data.

CRITICAL RULES:
1. NO data leakage: fit_features() computes ALL parameters from train only (medians, bin edges, skewness thresholds).
   apply_features() uses saved parameters for train AND test.
2. NEVER use outcome_col or treatment_col as new features.
3. All new feature names must be unique and not collide with existing columns.
4. stats dict must be JSON-serializable.
5. Return ONLY valid Python code, no markdown.

FEATURE ENGINEERING APPROACHES (apply selectively based on data signals):

1. LOG TRANSFORMS (for highly skewed features, skewness > 2):
   - np.log1p(x) where x >= 0
   - Save: which columns were log-transformed (list)
   - New column: f"{col}_log"

2. SQRT TRANSFORMS (for moderately skewed features, 1 < skewness <= 2):
   - np.sqrt(x) where x >= 0
   - New column: f"{col}_sqrt"

3. INTERACTION TERMS (top-3 pairs by correlation with outcome):
   - Multiply top correlated features pairwise
   - New column: f"{col1}_x_{col2}"
   - Only create if both cols have |corr_with_outcome| > 0.05

4. RATIO FEATURES (for semantically meaningful pairs):
   - col_a / (col_b + 1e-8) where col_b represents a denominator
   - Examples: rto/n_trn (avg per transaction), atv/mtv (avg vs median ratio)
   - New column: f"{col_a}_per_{col_b}"
   - Only if ratio has clear business meaning

5. POLYNOMIAL FEATURES (squared terms for top features):
   - x**2 for top-2 features by |corr_with_outcome|
   - New column: f"{col}_sq"
   - Only if relationship is likely nonlinear

6. QUANTILE BINS (for continuous features with clear segments):
   - pd.qcut into 4 bins → 0,1,2,3
   - Save: bin edges in stats
   - New column: f"{col}_qbin"
   - Use for age-like or recency-like features

7. MISSING INDICATORS (already done in preprocessing, skip)

8. PERCENTILE RANK (for heavy-tailed distributions):
   - Compute rank/len(train) for top features
   - Save: sorted values for rank lookup
   - New column: f"{col}_prank"

9. ZERO FLAG (for zero-inflated features):
   - Binary: (col > 0).astype(int)
   - New column: f"{col}_nonzero"
   - Only for features with >30% zeros

Generate exactly:
- fit_features(df, treatment_col, outcome_col) -> dict (fe_stats)
- apply_features(df, fe_stats) -> pd.DataFrame

IMPORTANT:
- Be selective — quality over quantity. Max 15-20 new features total.
- Each transformation must be justified by the data signals provided.
- Clip extreme values after transformation to avoid inf/nan.
"""

_FE_USER = """Preprocessed dataset statistics (after basic preprocessing):

Shape: {shape}
Features available (excluding service cols): {n_features} features
Service columns (exclude from FE): {service_cols}

Key signals for feature engineering:

SKEWNESS (candidates for log/sqrt transform):
{skewness_info}

TOP FEATURES by |correlation with outcome| (candidates for interactions/polynomials):
{top_corr_info}

ZERO RATES (candidates for zero flags):
{zero_rates_info}

OUTCOME info:
- Zero rate: {outcome_zero_rate:.1%}
- Is binary: {outcome_is_binary}
- Skewness: {outcome_skewness:.2f}

TREATMENT balance: {treatment_balance}

Sample column names (first 30): {sample_cols}

Based on these signals, generate fit_features() and apply_features() with the most impactful transformations.
Return only Python code."""


def _verify_fe_code(
    code: str,
    df_sample: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> Tuple[bool, str]:
    """Верификация FE кода."""
    namespace = {'pd': pd, 'np': np}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, f'Syntax error: {e}\n{traceback.format_exc()}'

    for fn in ['fit_features', 'apply_features']:
        if fn not in namespace:
            return False, f'Function {fn} not found'

    try:
        fe_stats = namespace['fit_features'](
            df_sample.copy(), treatment_col, outcome_col
        )
        result = namespace['apply_features'](df_sample.copy(), fe_stats)
    except Exception as e:
        return False, f'Execution failed: {e}\n{traceback.format_exc()}'

    # NaN check
    service = ['user_id', treatment_col, outcome_col]
    feat_cols = [c for c in result.columns if c not in service]
    nan_count = result[feat_cols].isnull().sum().sum()
    if nan_count > 0:
        bad = result[feat_cols].columns[result[feat_cols].isnull().any()].tolist()
        return False, f'NaN in features after FE: {bad}'

    # Inf check
    num_result = result[feat_cols].select_dtypes(include=[np.number])
    inf_count = np.isinf(num_result.values).sum()
    if inf_count > 0:
        return False, 'Inf values found after FE — clip values properly'

    # JSON check
    try:
        json.dumps(fe_stats)
    except Exception as e:
        return False, f'fe_stats not JSON-serializable: {e}'

    # Non-numeric check
    non_num = [c for c in feat_cols if result[c].dtype == 'object']
    if non_num:
        return False, f'Non-numeric features: {non_num}'

    return True, 'OK'


def generate_features(
    train_proc: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    preproc_stats: dict = None,
    max_iterations: int = 3,
    verbose: bool = True,
) -> Tuple[str, dict]:
    """
    Шаг 2: Feature engineering через LLM-агента.

    Запускается ПОСЛЕ generate_preprocess() на уже очищенных данных.
    Агент видит реальные статистики чистых данных и генерирует
    только те трансформации которые оправданы данными.

    Parameters
    ----------
    train_proc    : уже препроцессированный train
    treatment_col : колонка treatment
    outcome_col   : колонка outcome
    preproc_stats : stats от generate_preprocess() (содержит _raw_stats)

    Returns
    -------
    (fe_code, fe_stats)
    """
    if _CLIENT is None:
        raise RuntimeError('ANTHROPIC_API_KEY не найден в .env')

    if verbose:
        print('FE Agent: анализ очищенных данных...')

    # Извлекаем сигналы из preproc_stats если доступны
    raw = {}
    if preproc_stats and '_raw_stats' in preproc_stats:
        raw = preproc_stats['_raw_stats']

    # Пересчитываем на очищенных данных
    service = ['user_id', treatment_col, outcome_col]
    feat_cols = [c for c in train_proc.columns if c not in service]
    num_df = train_proc[feat_cols].select_dtypes(include=[np.number])

    # Skewness на чистых данных
    skewness = {}
    for col in num_df.columns:
        try:
            s = float(scipy_stats.skew(num_df[col].dropna()))
            if not np.isnan(s):
                skewness[col] = round(s, 2)
        except Exception:
            pass

    # Корреляции на чистых данных
    corr_with_outcome = {}
    if outcome_col in train_proc.columns:
        corr = num_df.corrwith(train_proc[outcome_col]).round(4)
        corr_with_outcome = {
            k: float(v) for k, v in corr.dropna().items()
            if not np.isnan(v)
        }

    # Zero rates
    zero_rates = {
        col: float((num_df[col] == 0).mean())
        for col in num_df.columns
        if (num_df[col] == 0).mean() > 0.3
    }

    # Топ по корреляции
    sorted_corr = sorted(corr_with_outcome.items(), key=lambda x: abs(x[1]), reverse=True)
    top_corr = dict(sorted_corr[:15])

    # Формируем промпт
    skewness_info = '\n'.join([
        f"  {col}: skew={val:.2f} ({'→ log1p' if abs(val) > 2 else '→ sqrt'})"
        for col, val in sorted(skewness.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    ])

    top_corr_info = '\n'.join([
        f"  {col}: corr={val:.4f}"
        for col, val in list(top_corr.items())[:10]
    ])

    zero_rates_info = '\n'.join([
        f"  {col}: {val:.1%} zeros {'→ zero flag' if val > 0.3 else ''}"
        for col, val in sorted(zero_rates.items(), key=lambda x: x[1], reverse=True)[:10]
    ])

    df_sample = train_proc.sample(
        n=min(500, len(train_proc)), random_state=42
    ).reset_index(drop=True)

    user_prompt = _FE_USER.format(
        shape             = list(train_proc.shape),
        n_features        = len(feat_cols),
        service_cols      = service,
        skewness_info     = skewness_info or '  (no skewed features found)',
        top_corr_info     = top_corr_info or '  (no strong correlations)',
        zero_rates_info   = zero_rates_info or '  (no zero-heavy features)',
        outcome_zero_rate = float((train_proc[outcome_col] == 0).mean()) if outcome_col in train_proc.columns else 0.0,
        outcome_is_binary = bool(train_proc[outcome_col].nunique() <= 2) if outcome_col in train_proc.columns else False,
        outcome_skewness  = raw.get('outcome_skewness', 0.0),
        treatment_balance = raw.get('treatment_balance', {}),
        sample_cols       = feat_cols[:30],
    )

    messages = [{'role': 'user', 'content': user_prompt}]
    code = None

    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f'FE Agent: генерация кода (итерация {iteration}/{max_iterations})...')

        response = _CLIENT.messages.create(
            model     = _MODEL,
            max_tokens= 4000,
            system    = _FE_SYSTEM,
            messages  = messages,
        )
        code = response.content[0].text.strip()

        for tag in ['```python', '```']:
            if code.startswith(tag):
                code = code[len(tag):]
        if code.endswith('```'):
            code = code[:-3]
        code = code.strip()

        if verbose:
            print('FE Agent: верификация...')

        ok, error_msg = _verify_fe_code(
            code, df_sample, treatment_col, outcome_col
        )

        if ok:
            if verbose:
                print('FE Agent: код верифицирован ✓')
            break
        else:
            if verbose:
                print(f'FE Agent: ошибка — {error_msg[:150]}')
                print('FE Agent: исправляю...')
            messages.append({'role': 'assistant', 'content': code})
            messages.append({
                'role': 'user',
                'content': f'Code failed:\n\n{error_msg}\n\nFix and return corrected code only.'
            })

    if code is None:
        raise RuntimeError('FE агент не сгенерировал код')

    # Финальный запуск
    namespace = {'pd': pd, 'np': np}
    exec(code, namespace)
    fe_stats = namespace['fit_features'](
        train_proc.copy(), treatment_col, outcome_col
    )

    if verbose:
        result_sample = namespace['apply_features'](df_sample.copy(), fe_stats)
        n_new = len(result_sample.columns) - len(train_proc.columns)
        n_total = len([c for c in result_sample.columns if c not in service])
        print('FE Agent: готово.')
        print(f'  Новых признаков создано: {n_new}')
        print(f'  Всего признаков: {n_total}')

    return code, fe_stats


# ══════════════════════════════════════════════════════════════════════════════
# EDA ОТЧЁТ ДЛЯ ГРАФИКОВ
# ══════════════════════════════════════════════════════════════════════════════

def generate_eda_report(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> Dict[str, Any]:
    """
    Генерирует EDA отчёт для визуализации.
    Возвращает структурированные данные для построения графиков.
    """
    report = {}
    service = ['user_id', treatment_col, outcome_col]

    # 1. Outcome распределение
    report['outcome_zero_rate']   = float((df[outcome_col] == 0).mean())
    report['outcome_nonzero_desc']= {
        k: float(v) for k, v in
        df[outcome_col][df[outcome_col] > 0].describe().items()
    }
    try:
        report['outcome_skewness'] = float(scipy_stats.skew(df[outcome_col].dropna()))
    except Exception:
        report['outcome_skewness'] = 0.0

    # 2. Баланс treatment/control
    vc = df[treatment_col].value_counts()
    report['treatment_counts']  = {str(k): int(v) for k, v in vc.items()}
    report['treatment_balance'] = float(vc.get(1, 0) / len(df))

    # 3. ATE и разбивка
    t_mean = float(df[df[treatment_col] == 1][outcome_col].mean())
    c_mean = float(df[df[treatment_col] == 0][outcome_col].mean())
    report['ATE']            = round(t_mean - c_mean, 6)
    report['outcome_mean_t'] = round(t_mean, 6)
    report['outcome_mean_c'] = round(c_mean, 6)

    # 4. Пропуски топ-20
    miss = df.isnull().mean()
    report['missing_rates'] = {
        k: round(float(v), 4)
        for k, v in miss[miss > 0].sort_values(ascending=False).head(20).items()
    }

    # 5. Топ признаков по корреляции с outcome
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in service if c in df.columns], errors='ignore'
    )
    if not num_df.empty and outcome_col in df.columns:
        corr = num_df.corrwith(df[outcome_col]).abs().sort_values(ascending=False)
        report['top_features_by_corr'] = {
            k: round(float(v), 4) for k, v in corr.head(15).items()
        }

    # 6. Баланс по топ признакам (SMD)
    report['feature_smd'] = {}
    if 'top_features_by_corr' in report:
        top5 = list(report['top_features_by_corr'].keys())[:5]
        for col in top5:
            if col in df.columns:
                t_vals = df[df[treatment_col] == 1][col].dropna()
                c_vals = df[df[treatment_col] == 0][col].dropna()
                if len(t_vals) > 0 and len(c_vals) > 0:
                    smd = abs(t_vals.mean() - c_vals.mean()) / (
                        np.sqrt((t_vals.std()**2 + c_vals.std()**2) / 2) + 1e-8
                    )
                    report['feature_smd'][col] = {
                        'treatment_mean': round(float(t_vals.mean()), 4),
                        'control_mean':   round(float(c_vals.mean()), 4),
                        'smd':            round(float(smd), 4),
                    }

    # 7. Skewness топ-15
    skewness = {}
    for col in num_df.columns:
        try:
            s = float(scipy_stats.skew(num_df[col].dropna()))
            if not np.isnan(s):
                skewness[col] = round(s, 2)
        except Exception:
            pass
    report['skewness_top15'] = dict(
        sorted(skewness.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    )

    return report
