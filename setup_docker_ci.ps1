<#
.SYNOPSIS
    Поэтапно добавляет Docker + CI/CD (ruff + pytest) в проект autoprep-uplift.

.DESCRIPTION
    Создаёт чистый requirements.txt (UTF-8), бэкап старого в requirements-dev.txt,
    Dockerfile, .dockerignore, ruff.toml, .github/workflows/ci.yml, tests/test_smoke.py,
    дописывает разделы в README, прогоняет ruff и pytest. Git-коммит/пуш — по флагам.

    ВАЖНО: все файлы пишутся в UTF-8 БЕЗ BOM — иначе Docker/CI читают requirements криво
    (это и есть та проблема, из-за которой старый requirements.txt не работал на Linux).

.PARAMETER Commit
    Сделать git-коммит после генерации файлов.

.PARAMETER Push
    Запушить ветку feature/docker-ci (подразумевает -Commit).

.EXAMPLE
    .\setup_docker_ci.ps1            # создать файлы и прогнать проверки
    .\setup_docker_ci.ps1 -Commit   # + git commit
    .\setup_docker_ci.ps1 -Push     # + git commit + git push
#>

[CmdletBinding()]
param(
    [switch]$Commit,
    [switch]$Push
)

# ВАЖНО: 'Continue', а не 'Stop'. При 'Stop' Windows PowerShell 5.1 превращает
# служебный вывод git в stderr (напр. "Switched to branch") в обрывающую ошибку
# и скрипт падает на первом же git. Ошибки записи файлов всё равно прерывают
# выполнение (это исключения .NET-методов), так что это безопасно.
$ErrorActionPreference = 'Continue'
if ($Push) { $Commit = $true }

# ────────────────────────────── Хелперы ──────────────────────────────
function Write-Utf8NoBom([string]$Path, [string]$Content) {
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
    }
    $enc = New-Object System.Text.UTF8Encoding($false)   # $false = без BOM
    [System.IO.File]::WriteAllText($Path, $Content, $enc)
    Write-Host "  + $Path" -ForegroundColor Green
}

function Step([string]$Title) {
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Cyan
}

function Get-PkgVersion([string]$Name) {
    try {
        $out = & $script:Py -m pip show $Name 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $line = $out | Where-Object { $_ -match '^Version:' } | Select-Object -First 1
        if ($line) { return ($line -replace '^Version:\s*', '').Trim() }
    } catch { }
    return $null
}

# ───────────────────────── Шаг 0 — окружение ─────────────────────────
Step "Шаг 0 — Проверка окружения"

if (-not (Test-Path "uplift" -PathType Container)) {
    Write-Host "  Не найдена папка uplift\. Запусти скрипт из КОРНЯ репозитория autoprep-uplift." -ForegroundColor Red
    exit 1
}

# определяем команду python (python / py)
$script:Py = 'python'
$pyOk = $false
try { & $script:Py --version *>$null; if ($LASTEXITCODE -eq 0) { $pyOk = $true } } catch { }
if (-not $pyOk) {
    $script:Py = 'py'
    try { & $script:Py --version *>$null; if ($LASTEXITCODE -eq 0) { $pyOk = $true } } catch { }
}

$inGit = Test-Path ".git" -PathType Container

Write-Host ("  python: " + $(if ($pyOk) { (& $script:Py --version 2>&1) } else { 'не найден (проверки пропустятся)' }))
Write-Host ("  git:    " + $(if ($inGit) { 'репозиторий обнаружен' } else { 'НЕ git-репозиторий (git-шаги пропустятся)' }))

# ─────────────────────── Шаг 1 — ветка ───────────────────────
Step "Шаг 1 — Ветка feature/docker-ci"
if ($inGit) {
    $existing = & git branch --list feature/docker-ci 2>$null
    if ([string]::IsNullOrWhiteSpace(($existing | Out-String))) {
        & git checkout -b feature/docker-ci 2>&1 | Out-Null
    } else {
        & git checkout feature/docker-ci 2>&1 | Out-Null
    }
    Write-Host "  активная ветка: feature/docker-ci"
} else {
    Write-Host "  пропуск" -ForegroundColor Yellow
}

# ──────────────── Шаг 2 — requirements.txt (чистый UTF-8) ────────────────
Step "Шаг 2 — requirements.txt + бэкап старого freeze"

if (Test-Path "requirements.txt") {
    # ReadAllText сам распознаёт BOM (в т.ч. UTF-16) и декодирует корректно
    $old = [System.IO.File]::ReadAllText((Resolve-Path "requirements.txt"))
    Write-Utf8NoBom "requirements-dev.txt" $old
    Write-Host "  старый pip freeze сохранён -> requirements-dev.txt"
}

# пытаемся подтянуть точные версии из активного окружения
$dyn = @{}
if ($pyOk) {
    foreach ($p in 'numpy', 'pandas', 'lightgbm', 'matplotlib', 'scikit-uplift') {
        $v = Get-PkgVersion $p
        $dyn[$p] = if ($v) { "$p==$v" } else { $p }
    }
} else {
    foreach ($p in 'numpy', 'pandas', 'lightgbm', 'matplotlib', 'scikit-uplift') { $dyn[$p] = $p }
}

$req = @"
# Runtime-зависимости (UTF-8 без BOM, только то, что импортирует пакет uplift)
$($dyn['numpy'])
$($dyn['pandas'])
scikit-learn==1.8.0
scipy==1.17.1
$($dyn['lightgbm'])
catboost==1.2.10
$($dyn['scikit-uplift'])
$($dyn['matplotlib'])
seaborn==0.13.2
plotly==6.7.0
pyarrow==24.0.0
anthropic==0.105.0
python-dotenv==1.2.2
"@
Write-Utf8NoBom "requirements.txt" $req
Write-Host "  непиннутые версии (если есть) подставь руками из requirements-dev.txt" -ForegroundColor DarkGray

# ──────────────────── Шаг 3 — Dockerfile + .dockerignore ────────────────────
Step "Шаг 3 — Dockerfile + .dockerignore"

$dockerfile = @'
FROM python:3.13-slim

# libgomp нужен lightgbm / catboost
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# зависимости отдельным слоем — кэшируется между сборками
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# код пакета
COPY uplift/ ./uplift/
COPY README.md .env.example ./

# self-check: образ собрался и пакет импортируется
CMD ["python", "-c", "import uplift; print('autoprep-uplift image OK')"]
'@
Write-Utf8NoBom "Dockerfile" $dockerfile

$dockerignore = @'
.git
.github
.venv
venv
__pycache__
*.pyc
.ipynb_checkpoints
*.ipynb
data/
experiments/
docs/
*.parquet
*.csv
.env
.pytest_cache
.ruff_cache
requirements-dev.txt
'@
Write-Utf8NoBom ".dockerignore" $dockerignore

# ──────────────────── Шаг 4 — ruff.toml + GitHub Actions ────────────────────
Step "Шаг 4 — ruff.toml + .github/workflows/ci.yml"

$ruff = @'
line-length = 120
target-version = "py313"

[lint]
select = ["E", "F", "W"]
ignore = ["E501"]
'@
Write-Utf8NoBom "ruff.toml" $ruff

$ci = @'
name: CI

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.13
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest ruff

      - name: Lint (ruff)
        run: ruff check uplift/ tests/

      - name: Run tests
        env:
          ANTHROPIC_API_KEY: "ci-dummy-key-not-used"
        run: pytest -q
'@
Write-Utf8NoBom ".github/workflows/ci.yml" $ci

# ──────────────────── Шаг 5 — smoke-тест ────────────────────
Step "Шаг 5 — tests/test_smoke.py"

if (-not (Test-Path "tests/__init__.py")) { Write-Utf8NoBom "tests/__init__.py" "" }

$smoke = @'
"""Smoke-тест: пакет импортируется, окружение рабочее."""
import importlib

import numpy as np


def test_core_modules_import():
    for mod in [
        "uplift",
        "uplift.metrics",
        "uplift.pipeline",
        "uplift.ensemble",
        "uplift.validation",
    ]:
        assert importlib.import_module(mod) is not None


def test_numpy_sanity():
    assert np.array([1, 2, 3]).sum() == 6
'@
Write-Utf8NoBom "tests/test_smoke.py" $smoke

# ──────────────────── Шаг 6 — README ────────────────────
Step "Шаг 6 — Разделы Docker и CI/CD в README"

if (Test-Path "README.md") {
    $readme = [System.IO.File]::ReadAllText((Resolve-Path "README.md"))
    if ($readme -notmatch '##\s*CI/CD') {
        $append = @'

## Docker

```bash
docker build -t autoprep-uplift .
docker run --rm autoprep-uplift
```

## CI/CD

При каждом push/PR в `main` и `dev` GitHub Actions на Python 3.13 прогоняет
линтер (ruff) и тесты (pytest). Конфиг: `.github/workflows/ci.yml`.
'@
        $enc = New-Object System.Text.UTF8Encoding($false)
        [System.IO.File]::AppendAllText((Resolve-Path "README.md"), $append, $enc)
        Write-Host "  разделы Docker и CI/CD дописаны в конец README.md" -ForegroundColor Green
    } else {
        Write-Host "  раздел CI/CD уже есть — пропуск" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  Бейджи вставь руками под заголовком README:" -ForegroundColor Cyan
    Write-Host '    [![CI](https://github.com/allisksks/autoprep-uplift/actions/workflows/ci.yml/badge.svg)](https://github.com/allisksks/autoprep-uplift/actions/workflows/ci.yml)'
    Write-Host '    [![Docker](https://img.shields.io/badge/docker-ready-2496ED?logo=docker&logoColor=white)](./Dockerfile)'
}

# ──────────────────── Шаг 7 — линтер и тесты локально ────────────────────
Step "Шаг 7 — Локальная проверка (ruff + pytest)"
if ($pyOk) {
    try {
        & $script:Py -m pip install -q ruff pytest
        Write-Host "  -> ruff check --fix"
        & $script:Py -m ruff check uplift tests --fix
        Write-Host "  -> ruff check"
        & $script:Py -m ruff check uplift tests
        Write-Host "  -> pytest"
        $env:ANTHROPIC_API_KEY = "ci-dummy-key-not-used"
        & $script:Py -m pytest -q
        Write-Host "  проверки прогнаны (смотри вывод выше)" -ForegroundColor Green
    } catch {
        Write-Host "  не удалось прогнать автоматически: $($_.Exception.Message)" -ForegroundColor Yellow
        Write-Host "  активируй venv и запусти вручную: ruff check uplift tests; pytest -q" -ForegroundColor Yellow
    }
} else {
    Write-Host "  python не найден — пропуск" -ForegroundColor Yellow
}

# ──────────────────── Шаг 8 — Git ────────────────────
Step "Шаг 8 — Git"
$files = @(
    'requirements.txt',
    'requirements-dev.txt',
    'Dockerfile',
    '.dockerignore',
    'ruff.toml',
    '.github/workflows/ci.yml',
    'tests/test_smoke.py',
    'tests/__init__.py',
    'README.md'
) | Where-Object { Test-Path $_ }

if ($inGit) {
    & git add $files
    Write-Host "  файлы добавлены в индекс (git add)" -ForegroundColor Green

    if ($Commit) {
        & git commit -m "Add Dockerfile + GitHub Actions CI (ruff + pytest)"
    }
    if ($Push) {
        & git push -u origin feature/docker-ci
    }
    if (-not $Commit) {
        Write-Host ""
        Write-Host "  Осталось вручную:" -ForegroundColor Cyan
        Write-Host '    git commit -m "Add Dockerfile + GitHub Actions CI (ruff + pytest)"'
        Write-Host '    git push -u origin feature/docker-ci'
    }
} else {
    Write-Host "  не git-репозиторий — пропуск" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Готово. Проверь сборку образа: docker build -t autoprep-uplift ." -ForegroundColor Green
