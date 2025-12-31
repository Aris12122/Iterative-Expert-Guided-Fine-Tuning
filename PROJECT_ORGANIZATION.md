# Организация проекта / Project Organization

[English](#english) | [Русский](#русский)

---

## English

This document describes the project organization and file structure after cleanup and reorganization.

## Changes Made

### 1. Requirements Files Consolidation

**Before:**
- `requirements.txt` - base requirements
- `requirements-gpu.txt` - GPU requirements (duplicate)
- `requirements-datasphere.txt` - DataSphere requirements (duplicate)

**After:**
- `requirements.txt` - unified requirements file with comments for different deployment scenarios

All three files had identical dependencies, only comments differed. Now there's a single file with clear instructions for CPU, GPU, and DataSphere deployments.

### 2. Documentation Organization

**Before:** All documentation files in project root
- `USAGE.md`
- `RESULTS.md`
- `PROJECT_STATUS.md`
- `CODE_REVIEW_REPORT.md`
- `DIAGNOSTIC_REPORT.md`
- `GPU_DEPLOYMENT_SUMMARY.md`
- `DATASPHERE_DEPLOYMENT.md`
- `DATASPHERE_AI_INTEGRATION.md`

**After:** Organized in `docs/` directory
```
docs/
├── README.md                    # Documentation index
├── usage.md                     # Usage guide
├── results.md                   # Experiment results
├── status.md                    # Project status
├── deployment/                  # Deployment guides
│   ├── datasphere.md           # DataSphere deployment
│   ├── gpu.md                  # GPU deployment
│   └── datasphere_ai_integration.md  # AI integration
└── reports/                     # Reports
    ├── code_review.md          # Code review
    └── diagnostic.md           # Diagnostic report
```

### 3. Scripts Updated

- `setup_datasphere.sh` - Updated to use unified `requirements.txt`
- `setup_gpu.sh` - Updated to use unified `requirements.txt`
- `notebooks/datasphere_training.ipynb` - Updated to use unified `requirements.txt`

### 4. README Updated

- Updated project structure section
- Updated links to documentation (now in `docs/`)
- Added documentation section with quick links

## Current File Structure

```
Iterative Expert-Guided Fine-Tuning/
├── src/                         # Source code
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── utils.py
│   ├── models/
│   └── training/
├── scripts/                     # CLI scripts
│   ├── train_supervised.py
│   ├── train_kd.py
│   ├── train_active_loop.py
│   ├── eval_model.py
│   └── diagnose_data.py
├── notebooks/                   # Jupyter notebooks
│   └── datasphere_training.ipynb
├── docs/                        # Documentation
│   ├── README.md
│   ├── usage.md
│   ├── results.md
│   ├── status.md
│   ├── deployment/
│   └── reports/
├── outputs/                     # Experiment outputs
│   ├── checkpoints/
│   ├── results/
│   ├── logs/
│   └── predictions/
├── requirements.txt             # Unified dependencies
├── setup_datasphere.sh          # DataSphere setup
├── setup_gpu.sh                 # GPU setup
└── README.md                    # Main README
```

## Benefits

1. **No Duplication**: Single requirements file instead of three
2. **Better Organization**: Documentation grouped by purpose
3. **Easier Navigation**: Clear structure with `docs/README.md` as index
4. **Maintainability**: Easier to update and maintain
5. **Consistency**: All scripts and notebooks use same requirements file

---

## Русский

Этот документ описывает организацию проекта и структуру файлов после очистки и реорганизации.

## Внесенные изменения

### 1. Объединение файлов зависимостей

**До:**
- `requirements.txt` - базовые зависимости
- `requirements-gpu.txt` - зависимости для GPU (дубликат)
- `requirements-datasphere.txt` - зависимости для DataSphere (дубликат)

**После:**
- `requirements.txt` - единый файл зависимостей с комментариями для разных сценариев развертывания

Все три файла имели идентичные зависимости, отличались только комментарии. Теперь есть один файл с четкими инструкциями для развертывания на CPU, GPU и DataSphere.

### 2. Организация документации

**До:** Все файлы документации в корне проекта
- `USAGE.md`
- `RESULTS.md`
- `PROJECT_STATUS.md`
- `CODE_REVIEW_REPORT.md`
- `DIAGNOSTIC_REPORT.md`
- `GPU_DEPLOYMENT_SUMMARY.md`
- `DATASPHERE_DEPLOYMENT.md`
- `DATASPHERE_AI_INTEGRATION.md`

**После:** Организованы в директории `docs/`
```
docs/
├── README.md                    # Индекс документации
├── usage.md                     # Руководство по использованию
├── results.md                   # Результаты экспериментов
├── status.md                    # Статус проекта
├── deployment/                  # Руководства по развертыванию
│   ├── datasphere.md           # Развертывание в DataSphere
│   ├── gpu.md                  # Развертывание GPU
│   └── datasphere_ai_integration.md  # Интеграция с AI
└── reports/                     # Отчеты
    ├── code_review.md          # Обзор кода
    └── diagnostic.md           # Диагностический отчет
```

### 3. Обновленные скрипты

- `setup_datasphere.sh` - Обновлен для использования единого `requirements.txt`
- `setup_gpu.sh` - Обновлен для использования единого `requirements.txt`
- `notebooks/datasphere_training.ipynb` - Обновлен для использования единого `requirements.txt`

### 4. Обновленный README

- Обновлен раздел структуры проекта
- Обновлены ссылки на документацию (теперь в `docs/`)
- Добавлен раздел документации с быстрыми ссылками

## Текущая структура файлов

```
Iterative Expert-Guided Fine-Tuning/
├── src/                         # Исходный код
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── utils.py
│   ├── models/
│   └── training/
├── scripts/                     # CLI скрипты
│   ├── train_supervised.py
│   ├── train_kd.py
│   ├── train_active_loop.py
│   ├── eval_model.py
│   └── diagnose_data.py
├── notebooks/                   # Jupyter notebooks
│   └── datasphere_training.ipynb
├── docs/                        # Документация
│   ├── README.md
│   ├── usage.md
│   ├── results.md
│   ├── status.md
│   ├── deployment/
│   └── reports/
├── outputs/                     # Результаты экспериментов
│   ├── checkpoints/
│   ├── results/
│   ├── logs/
│   └── predictions/
├── requirements.txt             # Единые зависимости
├── setup_datasphere.sh          # Настройка DataSphere
├── setup_gpu.sh                 # Настройка GPU
└── README.md                    # Главный README
```

## Преимущества

1. **Нет дублирования**: Один файл зависимостей вместо трех
2. **Лучшая организация**: Документация сгруппирована по назначению
3. **Удобная навигация**: Четкая структура с `docs/README.md` как индексом
4. **Поддерживаемость**: Легче обновлять и поддерживать
5. **Согласованность**: Все скрипты и notebooks используют один файл зависимостей

