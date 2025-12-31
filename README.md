# Iterative Expert-Guided Fine-Tuning

[English](#english) | [Русский](#русский)

---

## English

Research project comparing different approaches to model fine-tuning:
- **Baseline 1**: Supervised fine-tuning of a student model
- **Baseline 2**: Knowledge distillation from a teacher ensemble
- **Expert-Loop v1**: One iteration of active learning + distillation

### Project Structure

```
Iterative Expert-Guided Fine-Tuning/
├── src/                        # Source code
│   ├── config.py              # Configuration dataclasses
│   ├── data.py                # Dataset loading and processing
│   ├── metrics.py             # Evaluation metrics
│   ├── utils.py               # Common utilities
│   ├── models/
│   │   ├── student.py         # Student model
│   │   └── teachers.py        # Teacher ensemble
│   └── training/
│       ├── supervised.py      # Supervised training
│       ├── distillation.py    # Knowledge distillation
│       └── active_loop.py     # Active learning loop
├── scripts/                    # CLI scripts
│   ├── train_supervised.py    # CLI for Baseline 1
│   ├── train_kd.py            # CLI for Baseline 2
│   ├── train_active_loop.py   # CLI for Expert-Loop v1
│   ├── eval_model.py          # CLI for model evaluation
│   └── diagnose_data.py       # Diagnostic script
├── notebooks/                  # Jupyter notebooks
│   └── datasphere_training.ipynb  # DataSphere training notebook
├── docs/                       # Documentation
│   ├── usage.md               # Usage guide
│   ├── results.md             # Experiment results
│   ├── status.md              # Project status
│   ├── deployment/            # Deployment guides
│   │   ├── datasphere.md      # DataSphere deployment
│   │   ├── gpu.md             # GPU deployment
│   │   └── datasphere_ai_integration.md  # AI integration
│   └── reports/               # Reports
│       ├── code_review.md     # Code review
│       └── diagnostic.md      # Diagnostic report
├── outputs/                   # Experiment outputs
│   ├── checkpoints/           # Saved models
│   ├── results/               # Results JSON files
│   ├── logs/                  # Training logs
│   └── predictions/          # Model predictions
├── requirements.txt           # Python dependencies
├── setup_datasphere.sh        # DataSphere setup script
├── setup_gpu.sh               # GPU setup script
└── README.md                  # This file
```

### Installation

```bash
pip install -r requirements.txt
```

### Usage

#### Baseline 1: Supervised Fine-Tuning

```bash
python scripts/train_supervised.py \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --experiment_name "supervised_baseline" \
    --num_labels 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

#### Baseline 2: Knowledge Distillation

```bash
python scripts/train_kd.py \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --experiment_name "kd_baseline" \
    --num_labels 4 \
    --temperature 4.0 \
    --alpha 0.7
```

#### Expert-Loop v1: Active Learning + Distillation

```bash
python scripts/train_active_loop.py \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --experiment_name "active_loop_v1" \
    --num_labels 4 \
    --initial_pool_size 100 \
    --query_size 50 \
    --query_strategy "uncertainty" \
    --max_iterations 1
```

#### Model Evaluation

```bash
python scripts/eval_model.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --config_path "outputs/checkpoints/experiment_name/config.json"
```

#### Testing Model on Examples

Test model on specific examples with readable output:

```bash
# Test on predefined examples
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --use_test_examples

# Test on dataset examples
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --dataset_name "medmcqa" \
    --split "validation" \
    --num_examples 10

# Save results to file
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --use_test_examples \
    --output_file "test_results.json"
```

### Configurations

All hyperparameters are configured through dataclasses in `src/config.py`:
- `DatasetConfig`: Dataset parameters
- `ModelConfig`: Model parameters
- `TrainingConfig`: Training parameters
- `KDConfig`: Distillation parameters
- `ActiveLoopConfig`: Active learning parameters
- `ExperimentConfig`: Unified configuration

### Features

- **CPU/GPU support**: Automatic device detection, optimized for both CPU and GPU
- **Modularity**: Clear separation of responsibilities between modules
- **Configurability**: All parameters through config objects
- **Reproducibility**: Configuration saving with each experiment
- **Cloud deployment**: Ready for Yandex DataSphere deployment

### Documentation

All documentation is organized in the `docs/` directory:

- **[Usage Guide](docs/usage.md)** - Complete usage instructions
- **[Results](docs/results.md)** - Experiment results and analysis
- **[Deployment Guides](docs/deployment/)**:
  - [Yandex DataSphere](docs/deployment/datasphere.md) - DataSphere deployment
  - [GPU Deployment](docs/deployment/gpu.md) - GPU setup for Yandex Cloud
  - [AI Integration](docs/deployment/datasphere_ai_integration.md) - Cursor/ChatGPT integration
- **[Reports](docs/reports/)**:
  - [Code Review](docs/reports/code_review.md) - Code correctness analysis
  - [Diagnostic Report](docs/reports/diagnostic.md) - Data and model diagnostics

### Cloud Deployment

#### Yandex DataSphere

Quick deployment:
```bash
# Clone repository in DataSphere
git clone https://github.com/Aris12122/Iterative-Expert-Guided-Fine-Tuning.git
cd Iterative-Expert-Guided-Fine-Tuning

# Setup environment
bash setup_datasphere.sh

# Run training
python scripts/train_supervised.py --experiment_name datasphere_test
```

See [docs/deployment/datasphere.md](docs/deployment/datasphere.md) for detailed instructions.

### Status

✅ **Project is fully implemented and tested!**

- All three approaches (Baseline 1, Baseline 2, Expert-Loop v1) are implemented
- Initial test run completed successfully
- See [docs/results.md](docs/results.md) for detailed results and next steps

---

## Русский

Исследовательский проект для сравнения различных подходов к fine-tuning моделей:
- **Baseline 1**: Supervised fine-tuning студент-модели
- **Baseline 2**: Knowledge distillation от teacher ensemble
- **Expert-Loop v1**: Одна итерация active learning + distillation

### Структура проекта

```
Iterative Expert-Guided Fine-Tuning/
├── src/                        # Исходный код
│   ├── config.py              # Dataclasses конфигураций
│   ├── data.py                # Загрузка и обработка датасетов
│   ├── metrics.py             # Метрики оценки
│   ├── utils.py               # Общие утилиты
│   ├── models/
│   │   ├── student.py         # Student модель
│   │   └── teachers.py        # Teacher ensemble
│   └── training/
│       ├── supervised.py      # Supervised обучение
│       ├── distillation.py    # Knowledge distillation
│       └── active_loop.py     # Active learning цикл
├── scripts/                    # CLI скрипты
│   ├── train_supervised.py    # CLI для Baseline 1
│   ├── train_kd.py            # CLI для Baseline 2
│   ├── train_active_loop.py   # CLI для Expert-Loop v1
│   ├── eval_model.py          # CLI для оценки моделей
│   └── diagnose_data.py       # Диагностический скрипт
├── notebooks/                  # Jupyter notebooks
│   └── datasphere_training.ipynb  # Notebook для DataSphere
├── docs/                       # Документация
│   ├── usage.md               # Руководство по использованию
│   ├── results.md             # Результаты экспериментов
│   ├── status.md              # Статус проекта
│   ├── deployment/            # Руководства по развертыванию
│   │   ├── datasphere.md      # Развертывание в DataSphere
│   │   ├── gpu.md             # Развертывание GPU
│   │   └── datasphere_ai_integration.md  # Интеграция с AI
│   └── reports/               # Отчеты
│       ├── code_review.md     # Обзор кода
│       └── diagnostic.md      # Диагностический отчет
├── outputs/                   # Результаты экспериментов
│   ├── checkpoints/           # Сохраненные модели
│   ├── results/               # JSON файлы с результатами
│   ├── logs/                  # Логи обучения
│   └── predictions/           # Предсказания моделей
├── requirements.txt           # Python зависимости
├── setup_datasphere.sh        # Скрипт настройки DataSphere
├── setup_gpu.sh               # Скрипт настройки GPU
└── README.md                  # Этот файл
```

### Установка

```bash
pip install -r requirements.txt
```

### Использование

#### Baseline 1: Supervised Fine-Tuning

```bash
python scripts/train_supervised.py \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --experiment_name "supervised_baseline" \
    --num_labels 4 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

#### Baseline 2: Knowledge Distillation

```bash
python scripts/train_kd.py \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --experiment_name "kd_baseline" \
    --num_labels 4 \
    --temperature 4.0 \
    --alpha 0.7
```

#### Expert-Loop v1: Active Learning + Distillation

```bash
python scripts/train_active_loop.py \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --experiment_name "active_loop_v1" \
    --num_labels 4 \
    --initial_pool_size 100 \
    --query_size 50 \
    --query_strategy "uncertainty" \
    --max_iterations 1
```

#### Оценка модели

```bash
python scripts/eval_model.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --config_path "outputs/checkpoints/experiment_name/config.json"
```

#### Тестирование модели на примерах

Тестирование модели на конкретных примерах с читаемым выводом:

```bash
# Тест на предопределенных примерах
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --use_test_examples

# Тест на примерах из датасета
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --dataset_name "medmcqa" \
    --split "validation" \
    --num_examples 10

# Сохранение результатов в файл
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --use_test_examples \
    --output_file "test_results.json"
```

### Конфигурации

Все гиперпараметры настраиваются через dataclasses в `src/config.py`:
- `DatasetConfig`: параметры датасета
- `ModelConfig`: параметры моделей
- `TrainingConfig`: параметры обучения
- `KDConfig`: параметры distillation
- `ActiveLoopConfig`: параметры active learning
- `ExperimentConfig`: объединенная конфигурация

### Особенности

- **CPU/GPU поддержка**: Автоматическое определение устройства, оптимизировано для CPU и GPU
- **Модульность**: Четкое разделение ответственности между модулями
- **Конфигурируемость**: Все параметры через config объекты
- **Воспроизводимость**: Сохранение конфигураций с каждым экспериментом
- **Облачное развертывание**: Готово к развертыванию в Yandex DataSphere

### Документация

Вся документация организована в папке `docs/`:

- **[Руководство по использованию](docs/usage.md)** - Полные инструкции по использованию
- **[Результаты](docs/results.md)** - Результаты экспериментов и анализ
- **[Руководства по развертыванию](docs/deployment/)**:
  - [Yandex DataSphere](docs/deployment/datasphere.md) - Развертывание в DataSphere
  - [GPU развертывание](docs/deployment/gpu.md) - Настройка GPU для Yandex Cloud
  - [Интеграция с AI](docs/deployment/datasphere_ai_integration.md) - Интеграция с Cursor/ChatGPT
- **[Отчеты](docs/reports/)**:
  - [Обзор кода](docs/reports/code_review.md) - Анализ корректности кода
  - [Диагностический отчет](docs/reports/diagnostic.md) - Диагностика данных и моделей

### Облачное развертывание

#### Yandex DataSphere

Быстрое развертывание:
```bash
# Клонировать репозиторий в DataSphere
git clone https://github.com/Aris12122/Iterative-Expert-Guided-Fine-Tuning.git
cd Iterative-Expert-Guided-Fine-Tuning

# Настроить окружение
bash setup_datasphere.sh

# Запустить обучение
python scripts/train_supervised.py --experiment_name datasphere_test
```

См. [docs/deployment/datasphere.md](docs/deployment/datasphere.md) для подробных инструкций.

### Статус

✅ **Проект полностью реализован и протестирован!**

- Все три подхода (Baseline 1, Baseline 2, Expert-Loop v1) реализованы
- Первичное тестирование завершено успешно
- См. [docs/results.md](docs/results.md) для подробных результатов и следующих шагов
