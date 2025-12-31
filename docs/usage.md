# Руководство по использованию проекта / Usage Guide

[English](#english) | [Русский](#русский)

---

## English

### Prerequisites

1. **Python Environment**
   - Python 3.10 or higher
   - Virtual environment (recommended)

2. **Dependencies**
   - Install all required packages from `requirements.txt`

3. **Data**
   - Access to HuggingFace datasets (MedMCQA or MedQA)
   - Internet connection for downloading models and datasets

### Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd "Iterative Expert-Guided Fine-Tuning"
   ```

2. **Create and activate virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; import transformers; import datasets; print('All dependencies installed successfully')"
   ```

### Data Preparation

The project automatically downloads datasets from HuggingFace Hub. No manual data preparation is required.

**Supported datasets:**
- `medmcqa` - Medical Multiple Choice Question Answering
- `medqa` - Medical Question Answering

**Note:** On first run, datasets will be downloaded automatically. This may take some time depending on your internet connection.

### Project Structure

```
Iterative Expert-Guided Fine-Tuning/
├── src/                    # Source code
│   ├── config.py          # Configuration classes
│   ├── data.py            # Dataset loading and processing
│   ├── metrics.py         # Evaluation metrics
│   ├── utils.py           # Utility functions
│   ├── models/            # Model implementations
│   └── training/          # Training loops
├── scripts/                # CLI scripts
│   ├── train_supervised.py
│   ├── train_kd.py
│   ├── train_active_loop.py
│   └── eval_model.py
├── outputs/                # Output directory (created automatically)
│   ├── checkpoints/       # Saved models
│   ├── logs/              # Training logs
│   └── predictions/       # Model predictions
├── requirements.txt       # Python dependencies
└── .cursorrules           # Project rules
```

### Quick Testing

**For quick testing and verification**, the default configuration uses:
- Limited dataset size (500 samples)
- Small batch size (8)
- Single epoch (1)
- Reduced warmup steps (10)

This allows you to verify the code works in a few minutes per epoch.

**To use full dataset**, override `max_samples` in the config or set it to `None`.

### Running Experiments

#### Baseline 1: Supervised Fine-Tuning

Train a student model using only labeled data:

```bash
python scripts/train_supervised.py \
    --experiment_name "supervised_baseline_001"
```

**Note:** By default, this uses a limited dataset (500 samples) for quick testing. Training should complete in a few minutes.

**With custom parameters:**
```bash
python scripts/train_supervised.py \
    --experiment_name "supervised_custom" \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --seed 42
```

**Output:**
- Model checkpoint: `outputs/checkpoints/{experiment_name}/model.pt`
- Configuration: `outputs/checkpoints/{experiment_name}/config.json`
- Logs: `outputs/logs/{experiment_name}.log`

#### Baseline 2: Knowledge Distillation

Train a student model using knowledge distillation from a teacher ensemble:

```bash
python scripts/train_kd.py \
    --experiment_name "kd_baseline_001"
```

**With custom parameters:**
```bash
python scripts/train_kd.py \
    --experiment_name "kd_custom" \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --temperature 4.0 \
    --alpha 0.7 \
    --seed 42
```

**Output:**
- Model checkpoint: `outputs/checkpoints/{experiment_name}/model.pt`
- Configuration: `outputs/checkpoints/{experiment_name}/config.json`
- Logs: `outputs/logs/{experiment_name}.log`

#### Expert-Loop v1: Active Learning + Distillation

Run the active learning loop with teacher distillation:

```bash
python scripts/train_active_loop.py \
    --experiment_name "active_loop_v1_001"
```

**With custom parameters:**
```bash
python scripts/train_active_loop.py \
    --experiment_name "active_loop_custom" \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --temperature 4.0 \
    --alpha 0.7 \
    --unlabeled_pool_size 1000 \
    --top_k_uncertain 50 \
    --uncertainty_metric "entropy" \
    --seed 42
```

**Output:**
- Student_v0 checkpoint: `outputs/checkpoints/{experiment_name}_v0/model.pt`
- Student_v1 checkpoint: `outputs/checkpoints/{experiment_name}/student_v1/model.pt`
- Configuration: `outputs/checkpoints/{experiment_name}/student_v1/config.json`
- Logs: `outputs/logs/{experiment_name}.log`

### Evaluating Models

Evaluate a saved model on the test set:

```bash
python scripts/eval_model.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --config_path "outputs/checkpoints/experiment_name/config.json"
```

**Note:** The evaluation script is currently a placeholder. You can evaluate models by loading them and running inference manually.

### Configuration

All experiments use configuration objects defined in `src/config.py`. Default configurations are provided via `default_medqa_experiment()`.

**Key configuration classes:**
- `DatasetConfig`: Dataset parameters (name, splits, max length)
- `ModelConfig`: Model parameters (student, teachers, num labels)
- `TrainingConfig`: Training hyperparameters
- `KDConfig`: Knowledge distillation parameters (alpha, temperature)
- `ActiveLoopConfig`: Active learning parameters (pool size, top-k, uncertainty metric)
- `ExperimentConfig`: Unified experiment configuration

### Understanding Outputs

**Checkpoints:**
- `model.pt`: PyTorch state dict of the trained model
- `config.json`: Complete experiment configuration (for reproducibility)

**Logs:**
- Training metrics (loss, accuracy, expected correctness)
- Evaluation metrics per epoch
- Final test set metrics

**Metrics:**
- `accuracy`: Classification accuracy
- `expected_correctness`: Average probability assigned to correct answer
- `loss`: Training loss
- `ce_loss`: Cross-entropy loss (for KD experiments)
- `kd_loss`: Knowledge distillation loss (for KD experiments)

### Troubleshooting

**Common Issues:**

1. **Out of Memory:**
   - Reduce `batch_size` in TrainingConfig
   - Use smaller models (e.g., `distilbert-base-uncased` instead of `bert-base-uncased`)

2. **Dataset Download Fails:**
   - Check internet connection
   - Verify dataset name is correct
   - Try downloading manually: `python -c "from datasets import load_dataset; load_dataset('medmcqa', 'A')"`

3. **Model Download Fails:**
   - Check internet connection
   - Verify model names are correct
   - Models are downloaded from HuggingFace Hub on first use

4. **CUDA Errors:**
   - This project is CPU-only. Ensure `device="cpu"` in ModelConfig
   - Remove any `.cuda()` or `.to('cuda')` calls if present

5. **Import Errors:**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Verify Python version is 3.10+

### Best Practices

1. **Experiment Naming:**
   - Use descriptive names: `supervised_baseline_001`, `kd_temp4_alpha07`, etc.
   - Include key hyperparameters in names for easy identification

2. **Reproducibility:**
   - Always set `seed` parameter
   - Save configurations with each experiment
   - Document any manual changes to code

3. **Resource Management:**
   - Start with small experiments (fewer epochs, smaller batches)
   - Monitor disk space (models and datasets can be large)
   - Use CPU-friendly batch sizes (8-16 for most models)

4. **Logging:**
   - Check logs regularly: `outputs/logs/{experiment_name}.log`
   - Monitor training progress via tqdm progress bars
   - Review metrics after each epoch

### Next Steps

After running experiments:

1. **Compare Results:**
   - Compare accuracy and expected correctness across baselines
   - Analyze training curves from logs
   - Identify best hyperparameters

2. **Extend Experiments:**
   - Try different uncertainty metrics
   - Experiment with different teacher ensembles
   - Adjust active learning pool sizes

3. **Iterate:**
   - Run multiple iterations of active learning loop
   - Fine-tune hyperparameters based on results
   - Experiment with different datasets

---

## Русский

### Требования

1. **Окружение Python**
   - Python 3.10 или выше
   - Виртуальное окружение (рекомендуется)

2. **Зависимости**
   - Установить все необходимые пакеты из `requirements.txt`

3. **Данные**
   - Доступ к датасетам HuggingFace (MedMCQA или MedQA)
   - Интернет-соединение для загрузки моделей и датасетов

### Установка

1. **Перейдите в директорию проекта:**
   ```bash
   cd "Iterative Expert-Guided Fine-Tuning"
   ```

2. **Создайте и активируйте виртуальное окружение (рекомендуется):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # На Windows: venv\Scripts\activate
   ```

3. **Установите зависимости:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Проверьте установку:**
   ```bash
   python -c "import torch; import transformers; import datasets; print('Все зависимости установлены успешно')"
   ```

### Подготовка данных

Проект автоматически загружает датасеты из HuggingFace Hub. Ручная подготовка данных не требуется.

**Поддерживаемые датасеты:**
- `medmcqa` - Medical Multiple Choice Question Answering
- `medqa` - Medical Question Answering

**Примечание:** При первом запуске датасеты будут загружены автоматически. Это может занять некоторое время в зависимости от скорости интернета.

### Структура проекта

```
Iterative Expert-Guided Fine-Tuning/
├── src/                    # Исходный код
│   ├── config.py          # Классы конфигураций
│   ├── data.py            # Загрузка и обработка датасетов
│   ├── metrics.py         # Метрики оценки
│   ├── utils.py           # Утилиты
│   ├── models/            # Реализации моделей
│   └── training/          # Циклы обучения
├── scripts/              # CLI скрипты
│   ├── train_supervised.py
│   ├── train_kd.py
│   ├── train_active_loop.py
│   └── eval_model.py
├── outputs/               # Директория результатов (создается автоматически)
│   ├── checkpoints/       # Сохраненные модели
│   ├── logs/              # Логи обучения
│   └── predictions/       # Предсказания моделей
├── requirements.txt       # Python зависимости
└── .cursorrules           # Правила проекта
```

### Быстрое тестирование

**Для быстрого тестирования и проверки** конфигурация по умолчанию использует:
- Ограниченный размер датасета (500 примеров)
- Небольшой размер батча (8)
- Одна эпоха (1)
- Уменьшенное количество warmup steps (10)

Это позволяет проверить работоспособность кода за несколько минут на эпоху.

**Для использования полного датасета** переопределите `max_samples` в конфиге или установите его в `None`.

### Запуск экспериментов

#### Baseline 1: Supervised Fine-Tuning

Обучите student модель только на размеченных данных:

```bash
python scripts/train_supervised.py \
    --experiment_name "supervised_baseline_001"
```

**Примечание:** По умолчанию используется ограниченный датасет (500 примеров) для быстрого тестирования. Обучение должно завершиться за несколько минут.

**С пользовательскими параметрами:**
```bash
python scripts/train_supervised.py \
    --experiment_name "supervised_custom" \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --seed 42
```

**Результаты:**
- Чекпоинт модели: `outputs/checkpoints/{experiment_name}/model.pt`
- Конфигурация: `outputs/checkpoints/{experiment_name}/config.json`
- Логи: `outputs/logs/{experiment_name}.log`

#### Baseline 2: Knowledge Distillation

Обучите student модель через knowledge distillation от teacher ensemble:

```bash
python scripts/train_kd.py \
    --experiment_name "kd_baseline_001"
```

**С пользовательскими параметрами:**
```bash
python scripts/train_kd.py \
    --experiment_name "kd_custom" \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --temperature 4.0 \
    --alpha 0.7 \
    --seed 42
```

**Результаты:**
- Чекпоинт модели: `outputs/checkpoints/{experiment_name}/model.pt`
- Конфигурация: `outputs/checkpoints/{experiment_name}/config.json`
- Логи: `outputs/logs/{experiment_name}.log`

#### Expert-Loop v1: Active Learning + Distillation

Запустите цикл active learning с teacher distillation:

```bash
python scripts/train_active_loop.py \
    --experiment_name "active_loop_v1_001"
```

**С пользовательскими параметрами:**
```bash
python scripts/train_active_loop.py \
    --experiment_name "active_loop_custom" \
    --dataset_name "medmcqa" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --temperature 4.0 \
    --alpha 0.7 \
    --unlabeled_pool_size 1000 \
    --top_k_uncertain 50 \
    --uncertainty_metric "entropy" \
    --seed 42
```

**Результаты:**
- Чекпоинт Student_v0: `outputs/checkpoints/{experiment_name}_v0/model.pt`
- Чекпоинт Student_v1: `outputs/checkpoints/{experiment_name}/student_v1/model.pt`
- Конфигурация: `outputs/checkpoints/{experiment_name}/student_v1/config.json`
- Логи: `outputs/logs/{experiment_name}.log`

### Оценка моделей

Оцените сохраненную модель на тестовом наборе:

```bash
python scripts/eval_model.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --config_path "outputs/checkpoints/experiment_name/config.json"
```

**Примечание:** Скрипт оценки в настоящее время является заглушкой. Вы можете оценивать модели, загружая их и запуская инференс вручную.

### Конфигурация

Все эксперименты используют объекты конфигурации, определенные в `src/config.py`. Конфигурации по умолчанию предоставляются через `default_medqa_experiment()`.

**Основные классы конфигураций:**
- `DatasetConfig`: Параметры датасета (название, разбиения, максимальная длина)
- `ModelConfig`: Параметры моделей (student, teachers, количество меток)
- `TrainingConfig`: Гиперпараметры обучения
- `KDConfig`: Параметры knowledge distillation (alpha, temperature)
- `ActiveLoopConfig`: Параметры active learning (размер пула, top-k, метрика uncertainty)
- `ExperimentConfig`: Объединенная конфигурация эксперимента

### Понимание результатов

**Чекпоинты:**
- `model.pt`: PyTorch state dict обученной модели
- `config.json`: Полная конфигурация эксперимента (для воспроизводимости)

**Логи:**
- Метрики обучения (loss, accuracy, expected correctness)
- Метрики оценки на каждой эпохе
- Финальные метрики на тестовом наборе

**Метрики:**
- `accuracy`: Точность классификации
- `expected_correctness`: Средняя вероятность, назначенная правильному ответу
- `loss`: Loss обучения
- `ce_loss`: Cross-entropy loss (для KD экспериментов)
- `kd_loss`: Knowledge distillation loss (для KD экспериментов)

### Решение проблем

**Частые проблемы:**

1. **Нехватка памяти:**
   - Уменьшите `batch_size` в TrainingConfig
   - Используйте меньшие модели (например, `distilbert-base-uncased` вместо `bert-base-uncased`)

2. **Ошибка загрузки датасета:**
   - Проверьте интернет-соединение
   - Убедитесь, что название датасета правильное
   - Попробуйте загрузить вручную: `python -c "from datasets import load_dataset; load_dataset('medmcqa', 'A')"`

3. **Ошибка загрузки модели:**
   - Проверьте интернет-соединение
   - Убедитесь, что названия моделей правильные
   - Модели загружаются из HuggingFace Hub при первом использовании

4. **Ошибки CUDA:**
   - Этот проект работает только на CPU. Убедитесь, что `device="cpu"` в ModelConfig
   - Удалите любые вызовы `.cuda()` или `.to('cuda')`, если они присутствуют

5. **Ошибки импорта:**
   - Убедитесь, что все зависимости установлены: `pip install -r requirements.txt`
   - Проверьте версию Python (должна быть 3.10+)

### Рекомендации

1. **Именование экспериментов:**
   - Используйте описательные имена: `supervised_baseline_001`, `kd_temp4_alpha07` и т.д.
   - Включайте ключевые гиперпараметры в имена для легкой идентификации

2. **Воспроизводимость:**
   - Всегда устанавливайте параметр `seed`
   - Сохраняйте конфигурации с каждым экспериментом
   - Документируйте любые ручные изменения в коде

3. **Управление ресурсами:**
   - Начните с небольших экспериментов (меньше эпох, меньшие батчи)
   - Следите за дисковым пространством (модели и датасеты могут быть большими)
   - Используйте CPU-friendly размеры батчей (8-16 для большинства моделей)

4. **Логирование:**
   - Регулярно проверяйте логи: `outputs/logs/{experiment_name}.log`
   - Следите за прогрессом обучения через progress bars tqdm
   - Просматривайте метрики после каждой эпохи

### Следующие шаги

После запуска экспериментов:

1. **Сравните результаты:**
   - Сравните accuracy и expected correctness между baseline'ами
   - Проанализируйте кривые обучения из логов
   - Определите лучшие гиперпараметры

2. **Расширьте эксперименты:**
   - Попробуйте разные метрики uncertainty
   - Экспериментируйте с разными teacher ensemble
   - Настройте размеры пулов active learning

3. **Итерируйте:**
   - Запустите несколько итераций active learning цикла
   - Настройте гиперпараметры на основе результатов
   - Экспериментируйте с разными датасетами

