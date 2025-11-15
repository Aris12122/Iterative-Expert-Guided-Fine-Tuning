# Iterative Expert-Guided Fine-Tuning

Исследовательский проект для сравнения различных подходов к fine-tuning моделей:
- **Baseline 1**: Supervised fine-tuning студент-модели
- **Baseline 2**: Knowledge distillation от teacher ensemble
- **Expert-Loop v1**: Одна итерация active learning + distillation

## Структура проекта

```
Iterative Expert-Guided Fine-Tuning/
├── src/
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
├── scripts/
│   ├── train_supervised.py    # CLI для Baseline 1
│   ├── train_kd.py            # CLI для Baseline 2
│   ├── train_active_loop.py   # CLI для Expert-Loop v1
│   └── eval_model.py          # CLI для оценки моделей
└── outputs/                   # Результаты экспериментов
```

## Установка

```bash
pip install -r requirements.txt
```

## Использование

### Baseline 1: Supervised Fine-Tuning

```bash
python scripts/train_supervised.py \
    --dataset_name "imdb" \
    --student_model_name "distilbert-base-uncased" \
    --experiment_name "supervised_baseline" \
    --num_labels 2 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3
```

### Baseline 2: Knowledge Distillation

```bash
python scripts/train_kd.py \
    --dataset_name "imdb" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --experiment_name "kd_baseline" \
    --num_labels 2 \
    --temperature 4.0 \
    --alpha 0.7
```

### Expert-Loop v1: Active Learning + Distillation

```bash
python scripts/train_active_loop.py \
    --dataset_name "imdb" \
    --student_model_name "distilbert-base-uncased" \
    --teacher_model_names "bert-base-uncased" "roberta-base" \
    --experiment_name "active_loop_v1" \
    --num_labels 2 \
    --initial_pool_size 100 \
    --query_size 50 \
    --query_strategy "uncertainty" \
    --max_iterations 1
```

### Оценка модели

```bash
python scripts/eval_model.py \
    --model_path "outputs/checkpoints/experiment_name/model.pt" \
    --config_path "outputs/checkpoints/experiment_name/config.json"
```

## Конфигурации

Все гиперпараметры настраиваются через dataclasses в `src/config.py`:
- `DatasetConfig`: параметры датасета
- `ModelConfig`: параметры моделей
- `TrainingConfig`: параметры обучения
- `KDConfig`: параметры distillation
- `ActiveLoopConfig`: параметры active learning
- `ExperimentConfig`: объединенная конфигурация

## Особенности

- **CPU-only**: Проект оптимизирован для выполнения на CPU
- **Модульность**: Четкое разделение ответственности между модулями
- **Конфигурируемость**: Все параметры через config объекты
- **Воспроизводимость**: Сохранение конфигураций с каждым экспериментом

## Статус

Проект находится в стадии разработки. Основная структура создана, реализация функций помечена как TODO.

