# Тестирование модели на примерах / Testing Model on Examples

[English](#english) | [Русский](#русский)

---

## English

The `test_model_examples.py` script allows you to test a trained model on specific examples and see readable, human-friendly output showing:
- The question
- All answer options (A, B, C, D)
- Model's prediction with confidence scores
- Correct answer
- Whether the model was correct

### Usage

#### Basic Usage

Test on predefined medical examples:

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --use_test_examples
```

#### Test on Dataset Examples

Test on examples from the validation set:

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --dataset_name "medmcqa" \
    --split "validation" \
    --num_examples 10
```

#### Save Results

Save results to a JSON file:

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --use_test_examples \
    --output_file "test_results.json"
```

### Arguments

- `--model_path` (required): Path to the trained model checkpoint (.pt file)
- `--config_path` (optional): Path to config.json (default: looks in same directory as model)
- `--dataset_name` (default: "medmcqa"): Dataset to load examples from
- `--split` (default: "validation"): Dataset split to use
- `--num_examples` (default: 10): Number of examples to test
- `--use_test_examples`: Use predefined test examples instead of dataset
- `--seed` (default: 42): Random seed
- `--output_file` (optional): Path to save results JSON

### Output Format

The script outputs each example in a readable format:

```
================================================================================
Пример 1
================================================================================

❓ ВОПРОС:
   What is the most common cause of acute appendicitis?

📋 ВАРИАНТЫ ОТВЕТОВ:
   ✅ A) Bacterial infection   [Вероятность: 15.23%]
      B) Viral infection        [Вероятность: 8.45%]
      C) Obstruction of the appendiceal lumen ✓ [Вероятность: 72.10%]
      D) Dietary factors        [Вероятность: 4.22%]

🎯 ПРЕДСКАЗАНИЕ МОДЕЛИ: C
   Текст: Obstruction of the appendiceal lumen
   Уверенность: 72.10%

✅ ПРАВИЛЬНЫЙ ОТВЕТ: C
   Текст: Obstruction of the appendiceal lumen

🎉 РЕЗУЛЬТАТ: ПРАВИЛЬНО!
```

### Example Output

At the end, you'll see a summary:

```
================================================================================
📊 ИТОГОВАЯ СТАТИСТИКА
================================================================================
Всего примеров: 10
Правильных ответов: 7
Точность: 70.00%
================================================================================
```

### Predefined Test Examples

The script includes 3 predefined medical examples:
1. Cause of acute appendicitis
2. First-line hypertension treatment
3. Normal blood pressure range

These are useful for quick testing without loading a dataset.

---

## Русский

Скрипт `test_model_examples.py` позволяет протестировать обученную модель на конкретных примерах и увидеть читаемый, понятный вывод, показывающий:
- Вопрос
- Все варианты ответов (A, B, C, D)
- Предсказание модели с оценками уверенности
- Правильный ответ
- Правильно ли ответила модель

### Использование

#### Базовое использование

Тест на предопределенных медицинских примерах:

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --use_test_examples
```

#### Тест на примерах из датасета

Тест на примерах из валидационного набора:

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --dataset_name "medmcqa" \
    --split "validation" \
    --num_examples 10
```

#### Сохранение результатов

Сохранение результатов в JSON файл:

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --use_test_examples \
    --output_file "test_results.json"
```

### Аргументы

- `--model_path` (обязательно): Путь к обученной модели (.pt файл)
- `--config_path` (опционально): Путь к config.json (по умолчанию: ищет в той же директории, что и модель)
- `--dataset_name` (по умолчанию: "medmcqa"): Датасет для загрузки примеров
- `--split` (по умолчанию: "validation"): Разбиение датасета для использования
- `--num_examples` (по умолчанию: 10): Количество примеров для тестирования
- `--use_test_examples`: Использовать предопределенные тестовые примеры вместо датасета
- `--seed` (по умолчанию: 42): Случайное зерно
- `--output_file` (опционально): Путь для сохранения результатов JSON

### Формат вывода

Скрипт выводит каждый пример в читаемом формате:

```
================================================================================
Пример 1
================================================================================

❓ ВОПРОС:
   What is the most common cause of acute appendicitis?

📋 ВАРИАНТЫ ОТВЕТОВ:
   ✅ A) Bacterial infection   [Вероятность: 15.23%]
      B) Viral infection        [Вероятность: 8.45%]
      C) Obstruction of the appendiceal lumen ✓ [Вероятность: 72.10%]
      D) Dietary factors        [Вероятность: 4.22%]

🎯 ПРЕДСКАЗАНИЕ МОДЕЛИ: C
   Текст: Obstruction of the appendiceal lumen
   Уверенность: 72.10%

✅ ПРАВИЛЬНЫЙ ОТВЕТ: C
   Текст: Obstruction of the appendiceal lumen

🎉 РЕЗУЛЬТАТ: ПРАВИЛЬНО!
```

### Пример вывода

В конце вы увидите сводку:

```
================================================================================
📊 ИТОГОВАЯ СТАТИСТИКА
================================================================================
Всего примеров: 10
Правильных ответов: 7
Точность: 70.00%
================================================================================
```

### Предопределенные тестовые примеры

Скрипт включает 3 предопределенных медицинских примера:
1. Причина острого аппендицита
2. Препарат первой линии для лечения гипертонии
3. Нормальный диапазон артериального давления

Они полезны для быстрого тестирования без загрузки датасета.

### Примеры использования

#### Быстрый тест обученной модели

```bash
# Активируйте виртуальное окружение
source venv/bin/activate

# Тест на предопределенных примерах
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --use_test_examples
```

#### Тест на реальных данных

```bash
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --dataset_name "medmcqa" \
    --split "validation" \
    --num_examples 20
```

#### Сравнение разных моделей

```bash
# Baseline 1
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/supervised_500_8_2/model.pt" \
    --use_test_examples \
    --output_file "results_supervised.json"

# Baseline 2 (KD)
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/kd_500_8_2/model.pt" \
    --use_test_examples \
    --output_file "results_kd.json"

# Baseline 3 (Active Learning)
python scripts/test_model_examples.py \
    --model_path "outputs/checkpoints/active_loop_500_8_2/student_v1/model.pt" \
    --use_test_examples \
    --output_file "results_active.json"
```

### Интерпретация результатов

- **Вероятности**: Показывают уверенность модели в каждом варианте ответа
- **Предсказание**: Вариант с наивысшей вероятностью
- **Правильность**: Сравнение предсказания с правильным ответом
- **Точность**: Процент правильных ответов на всех примерах

Если модель показывает низкую точность или неправильные ответы на простых примерах, это может указывать на проблемы с обучением.

