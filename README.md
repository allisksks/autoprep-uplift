# AutoPrep-Uplift

> LLM-агент для uplift моделирования маркетинговых A/B тестов.

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/статус-research-orange.svg)]()
[![Site](https://img.shields.io/badge/сайт-GitHub%20Pages-purple.svg)](https://allisksks.github.io/autoprep-uplift/)

## Что это

Пайплайн который автоматизирует uplift моделирование для любого маркетингового A/B теста. Три LLM-агента берут на себя самую трудоёмкую часть работы — анализ данных, препроцессинг и feature engineering — а затем перебирают 108 комбинаций ансамблей и выбирают оптимальную.

**Живой сайт:** https://allisksks.github.io/autoprep-uplift/

---

## Зачем это нужно

Классические ML-модели отвечают на вопрос «кто купит?». Это неправильный вопрос для маркетинга — такая модель находит людей которые купили бы и без промо, впустую тратя бюджет.

Правильный вопрос: **кто купит именно потому что получил промо?**

Клиентов можно разделить на четыре группы:

| Сегмент | С промо | Без промо | Отправлять? |
|---------|---------|-----------|-------------|
| **Persuadables** | Купит | Не купит | ✓ Ради них всё |
| Sure things | Купит | Купит | ✗ Бюджет впустую |
| Lost causes | Не купит | Не купит | ✗ Бесполезно |
| **Sleeping dogs** | Не купит | Купит | ✗ Промо вредит |

Sleeping dogs — особенно опасная группа. Они покупали сами, а после промо перестают. На данных Magnit таких ~28% базы. Классическая модель их не видит.

Uplift моделирование находит Persuadables и отделяет их от всех остальных.

---

## Результаты на реальных данных

Метрика: uplift@10 — нижняя граница 80% bootstrap CI разности средних между treatment и control в топ-10% клиентов по uplift score. Все результаты получены с LLM-агентом feature engineering.

| Датасет | Размер | Баланс | Победитель CV | Финальный ансамбль | Holdout CI | vs random |
|---------|--------|--------|--------------|-------------------|------------|-----------|
| **Magnit** (приватный) | 355K | 50/50 | DR-learner (16.37) | T-Ridge + R + Hurdle | **22.64** | +600% |
| **Hillstrom** | 64K | 67/33 | T-Ridge (0.071) | T-Ridge + R + Hurdle | **0.097** | +126% |
| **Lenta** | 550K | 75/25 | DR-learner (0.012) | T-LGB + T-Ridge + Hurdle | **0.022** | +553% |
| **Megafon** | 600K | 50/50 | Hurdle (0.382) | DR + T-LGB + Hurdle | **0.446** | +827% |
| **Synthetic** | 50K | 50/50 | R-learner (0.904) | T-Ridge (single) | **0.898** | +157% |

**Ключевые паттерны:**
- DR-learner лучший на сбалансированных данных (Magnit, Lenta, Megafon) — doubly robust свойство
- T-Ridge лучший при дисбалансе групп (Hillstrom 67/33) — линейная регуляризация компенсирует
- Hurdle входит в winning ensemble на 4 из 5 датасетов — zero-inflated outcome везде
- CV ranking ≠ ensemble contribution: X-learner последний по CV на Lenta, но в winning ensemble
- FE агент дал +5% на Magnit, на других датасетах нейтральный или слабо отрицательный эффект
- Ансамбль стабильно обыгрывает лучшую одиночную модель на 10-38%

---

## Как работает пайплайн

```
Сырой A/B датасет
      ↓
Валидация схемы         # проверка treatment_col, outcome_col, user_id
      ↓
Валидация данных        # рандомизация, leakage, баланс групп
      ↓
LLM Агент 1: EDA        # анализ данных, генерация кода препроцессинга
      ↓
LLM Агент 2: FE         # feature engineering на очищенных данных
      ↓
Обучение 6 мета-лёрнеров  # DR / T-LGB / T-Ridge / X / R / Hurdle
      ↓
CV + антиовёрфиттинг    # k-fold, bootstrap CI, permutation test
      ↓
Перебор 108 комбинаций  # single / pairs / triples / all-6
      ↓
LLM Агент 3: ансамбль   # видит все 108 результатов, выбирает оптимальную
      ↓
predictions.csv
```

---

## Детали по каждому компоненту

### Агент 1: EDA и препроцессинг

Анализирует датасет и генерирует два Python-метода:

- `fit_preprocess(df, treatment_col, outcome_col)` — учится только на train. Вычисляет медианы для импутации, строит маппинги категориальных признаков, определяет какие колонки дропнуть.
- `apply_preprocess(df, stats)` — применяет сохранённые параметры к любому датасету без утечки данных.

Что делает:
- Дропает колонки с >50% пропусков
- Добавляет бинарные индикаторы пропусков для колонок с 5-50% missing
- Заполняет числовые пропуски медианой (из train)
- Label encoding категориальных (маппинг сохраняется, unseen → -1)
- Проверяет что нет NaN и всё числовое

Агент сам тестирует сгенерированный код на срезе данных и итеративно исправляет ошибки — до 3 итераций.

### Агент 2: Feature Engineering

Запускается после базового препроцессинга на уже очищенных данных. Видит реальные статистики и генерирует только те трансформации которые оправданы данными.

Типы трансформаций:
- **Log1p** — для признаков со skewness > 2 (нормализует правоскошенные)
- **Sqrt** — для умеренно скошенных (1 < skewness ≤ 2)
- **Interaction terms** — произведения топ-3 пар по корреляции с outcome
- **Ratio features** — смысловые отношения (средний чек = сумма / транзакции)
- **Polynomial** — квадратичные члены для топ признаков по корреляции
- **Quantile bins** — квартильные бины (0-3) для непрерывных признаков
- **Percentile rank** — ранг клиента внутри датасета для тяжёлых хвостов
- **Zero flags** — бинарный флаг активности для признаков с >30% нулей

Все параметры (границы бинов, пороги skewness, отсортированные значения для ранга) сохраняются в `fe_stats` — применение к test корректное, без утечки.

На Magnit: 89 → ~115 признаков, +5% к holdout CI.
На Hillstrom: 8 → 26 признаков.

### 6 мета-лёрнеров

| Модель | Метод | Когда лучший |
|--------|-------|-------------|
| **DR-learner** | Doubly robust pseudo-outcomes | Сбалансированные данные (50/50) |
| T-learner LGB | Две отдельные LightGBM | Быстрый baseline |
| T-learner Ridge | Две отдельные Ridge | Дисбаланс групп, линейный сигнал |
| X-learner | Двухступенчатая импутация | Дисбаланс групп, ensemble diversity |
| R-learner | Robinson decomposition | Теоретически оптимальный |
| **Hurdle** | P(Y>0) × E[Y\|Y>0] | Zero-inflated outcome (>70% нулей) |

### Антиовёрфиттинг (5 проверок)

Перед обучением:
- **Randomization check** — t-test и chi-square по каждому признаку между treatment и control
- **Leakage detection** — корреляция признаков с treatment_flg (post-treatment признаки)
- **Balance check** — соотношение treatment/control групп

После обучения:
- **Permutation test** — H0: модель не лучше случайного ранжирования (200 перестановок)
- **Repeated CV** — 3×3-fold с разными seeds, оценка дисперсии оценки

### Агент 3: выбор ансамбля

Перебирает все 108 комбинаций на holdout (20% train):
- 6 одиночных моделей
- 15 пар
- 20 троек
- 15 четвёрок
- 6 пятёрок
- 1 полный ансамбль
- 4 стратегии взвешивания для каждой комбинации

LLM агент видит полную таблицу результатов и CV-историю каждой модели. Выбирает оптимальную комбинацию учитывая holdout performance, diversity, риск переобучения. Возвращает финальные веса и объяснение на естественном языке.

Иногда лучший вариант — одна модель (Synthetic: T-Ridge бьёт все ансамбли).

---

## Быстрый старт

### Windows

```powershell
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
copy .env.example .env
notepad .env  # добавь ANTHROPIC_API_KEY=sk-ant-...
```

### macOS / Linux

```bash
git clone https://github.com/allisksks/autoprep-uplift.git
cd autoprep-uplift
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
nano .env  # добавь ANTHROPIC_API_KEY=sk-ant-...
```

### Запуск

```powershell
# любой публичный датасет
py run_pipeline.py --dataset hillstrom
py run_pipeline.py --dataset lenta
py run_pipeline.py --dataset megafon
py run_pipeline.py --dataset synthetic

# с параметрами
py run_pipeline.py --dataset hillstrom --metric auuc --cv_folds 5 --no_fast

# полный прогон с агентами (Magnit)
py run_full_pipeline.py
```

### Что нужно от твоих данных

```python
# минимальная структура
user_id | treatment_flg | outcome | feature_1 | feature_2 | ...

# требования:
# - рандомизированный A/B тест (не observational data)
# - только pre-treatment признаки
# - от 10K строк, оптимально от 50K
```

---

## Метрики

```python
from uplift.metrics import evaluate, evaluate_all

# одна метрика — нижняя граница 80% bootstrap CI
evaluate(y, w, scores, metric='uplift@10')
evaluate(y, w, scores, metric='uplift@5')
evaluate(y, w, scores, metric='uplift@20')
evaluate(y, w, scores, metric='auuc')
evaluate(y, w, scores, metric='qini')

# все сразу
evaluate_all(y, w, scores, n_boot=200)
```

---

## Структура проекта

```
uplift/
├── metrics.py         # uplift@K, AUUC, Qini, evaluate(), evaluate_all()
├── pipeline.py        # UpliftPipeline — CV по всем моделям
├── ensemble.py        # UpliftEnsemble — 6 стратегий взвешивания
├── validation.py      # рандомизация, leakage, permutation test
├── models/
│   ├── base.py
│   ├── dr_learner.py  # Kennedy (2023)
│   ├── t_learner.py   # Künzel et al. (2019)
│   ├── x_learner.py   # Künzel et al. (2019)
│   ├── r_learner.py   # Nie & Wager (2021)
│   └── hurdle.py      # Devriendt et al. (2022)
└── agent/
    ├── eda_agent.py       # Агент 1: EDA + препроцессинг
    │                      # Агент 2: feature engineering
    └── model_selector.py  # Агент 3: выбор ансамбля

experiments/results/
├── magnit/            # figures/, tables/, predictions.csv
├── hillstrom/
├── lenta/
├── megafon/
└── synthetic/

run_pipeline.py        # универсальный запуск — любой датасет
run_full_pipeline.py   # полный прогон с агентами на Magnit
docs/                  # GitHub Pages сайт
```

---

## Датасеты

| Датасет | Строк | Outcome | Доступ |
|---------|-------|---------|--------|
| Hillstrom Email | 64K | continuous spend | `sklift.datasets.fetch_hillstrom()` |
| Lenta Retail | ~687K | binary response | `sklift.datasets.fetch_lenta()` |
| Megafon Telecom | 600K | binary response | `sklift.datasets.fetch_megafon()` |
| Synthetic | 50K | continuous (known CATE) | numpy генератор |
| Magnit Retail | 355K | continuous, 90% zeros | приватный RCT |

---

## Git workflow

```
main        ← только стабильные релизы
dev         ← интеграционная ветка
feature/*   ← одна ветка на задачу, удаляется после merge
```

---

## Статус

Проект в активной разработке. Статья готовится к публикации.

## Контакт

Telegram: [@alli1ice](https://t.me/alli1ice)

## Цитирование

```bibtex
@software{autoprep_uplift_2026,
  author = {Десятникова, Алиса},
  title  = {AutoPrep-Uplift: LLM-Augmented Pipeline for Uplift Modeling},
  year   = {2026},
  url    = {https://github.com/allisksks/autoprep-uplift}
}
```

## Docker

```bash
docker build -t autoprep-uplift .
docker run --rm autoprep-uplift
```

## CI/CD

При каждом push/PR в `main` и `dev` GitHub Actions на Python 3.13 прогоняет
линтер (ruff) и тесты (pytest). Конфиг: `.github/workflows/ci.yml`.