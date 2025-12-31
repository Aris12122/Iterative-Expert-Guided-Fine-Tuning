# Развертывание проекта в Yandex DataSphere

Это руководство поможет вам запустить проект "Iterative Expert-Guided Fine-Tuning" в Yandex DataSphere.

## Что такое Yandex DataSphere?

Yandex DataSphere — это облачная платформа для работы с данными и машинным обучением, похожая на Google Colab или Kaggle. Она предоставляет:
- Jupyter notebooks с GPU поддержкой
- Предустановленные библиотеки для ML
- Терминал для работы с файлами
- Интеграцию с Yandex Object Storage

## Подготовка проекта

### Шаг 1: Создание проекта в DataSphere

1. Перейдите на [Yandex Cloud Console](https://console.cloud.yandex.ru/)
2. Выберите сервис **DataSphere**
3. Создайте новый проект:
   - Нажмите "Создать проект"
   - Укажите имя проекта (например, "Iterative-Expert-Guided-Fine-Tuning")
   - Выберите сообщество или создайте новое
   - Нажмите "Создать"

### Шаг 2: Загрузка кода проекта

**Вариант A: Через Git (рекомендуется)**

1. В DataSphere откройте терминал (Terminal в JupyterLab)
2. Клонируйте репозиторий:
```bash
git clone https://github.com/Aris12122/Iterative-Expert-Guided-Fine-Tuning.git
cd Iterative-Expert-Guided-Fine-Tuning
```

**Вариант B: Через загрузку файлов**

1. В JupyterLab используйте File Browser
2. Загрузите все файлы проекта через Upload
3. Сохраните структуру папок (`src/`, `scripts/`, и т.д.)

### Шаг 3: Установка зависимостей

DataSphere обычно имеет предустановленный PyTorch с GPU. Проверьте версию:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
```

Установите зависимости проекта:

**В Jupyter notebook:**
```python
# Установка зависимостей
%pip install -r requirements.txt
```

**В терминале:**
```bash
pip install -r requirements.txt
```

Если PyTorch не установлен или нужна другая версия:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Шаг 4: Проверка окружения

Создайте тестовую ячейку в notebook:

```python
import sys
from pathlib import Path

# Добавить корень проекта в путь
project_root = Path.cwd()
if "Iterative-Expert-Guided-Fine-Tuning" in str(project_root):
    project_root = project_root / "Iterative-Expert-Guided-Fine-Tuning"
sys.path.insert(0, str(project_root))

# Проверка импортов
from src.config import default_medqa_experiment
from src.training.supervised import SupervisedExperiment
import torch

print("✓ Все импорты успешны")
print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
```

## Запуск экспериментов

### Вариант 1: Использование готовых скриптов

В терминале DataSphere:

```bash
# Baseline 1: Supervised Fine-Tuning
python scripts/train_supervised.py \
    --experiment_name supervised_datasphere \
    --batch_size 16 \
    --num_epochs 3 \
    --max_samples 1000

# Baseline 2: Knowledge Distillation
python scripts/train_kd.py \
    --experiment_name kd_datasphere \
    --batch_size 16 \
    --num_epochs 3 \
    --max_samples 1000

# Baseline 3: Active Learning Loop
python scripts/train_active_loop.py \
    --experiment_name active_loop_datasphere \
    --batch_size 16 \
    --num_epochs 3 \
    --max_samples 1000
```

### Вариант 2: Использование Jupyter Notebook

Используйте готовый notebook `notebooks/datasphere_training.ipynb` (см. ниже).

## Работа с данными

### Загрузка датасетов

HuggingFace Datasets автоматически загружает датасеты при первом использовании. Они кэшируются локально в DataSphere.

Если нужно использовать свои данные:
1. Загрузите файлы через File Browser
2. Или используйте Yandex Object Storage для больших файлов
3. Модифицируйте `src/data.py` для загрузки ваших данных

## Сохранение результатов

Результаты сохраняются в `outputs/`:
- `outputs/checkpoints/` — сохраненные модели
- `outputs/results/` — метрики в JSON формате
- `outputs/logs/` — логи обучения

Для сохранения результатов вне DataSphere:
1. Скачайте файлы через File Browser
2. Или используйте Git для коммита результатов
3. Или загрузите в Yandex Object Storage

## Оптимизация для GPU

DataSphere автоматически определяет GPU. Убедитесь, что:

1. В `src/config.py` используется `device="auto"` (по умолчанию)
2. Batch size увеличен для GPU (например, `--batch_size 32` или `64`)
3. Используется `gradient_accumulation_steps` для больших моделей

Пример конфигурации для GPU:
```python
config.training.batch_size = 32
config.training.gradient_accumulation_steps = 2
config.model.device = "auto"  # Автоматически выберет GPU
```

## Мониторинг обучения

### Использование TensorBoard (опционально)

```bash
pip install tensorboard
```

Запустите TensorBoard в отдельной ячейке:
```python
%load_ext tensorboard
%tensorboard --logdir outputs/logs
```

### Логирование в notebook

Все эксперименты логируют метрики в `outputs/logs/`. Вы можете читать логи прямо в notebook:

```python
import json
from pathlib import Path

results_file = Path("outputs/results/supervised_datasphere.json")
if results_file.exists():
    with open(results_file) as f:
        results = json.load(f)
    print(json.dumps(results, indent=2))
```

## Интеграция с Cursor и ChatGPT

### Cursor

**Cursor** — это локальный редактор кода с AI-ассистентом. Его нельзя напрямую интегрировать в DataSphere, но вы можете:

1. **Работать локально в Cursor**:
   - Разрабатывайте код локально в Cursor
   - Тестируйте на небольших данных
   - Коммитьте изменения в Git
   - Синхронизируйте с DataSphere через `git pull`

2. **Использовать Cursor для ревью кода**:
   - Скачайте результаты из DataSphere
   - Откройте в Cursor для анализа
   - Используйте AI-ассистента для улучшения кода

### ChatGPT / Claude / другие AI-ассистенты

Вы можете использовать AI-ассистенты несколькими способами:

1. **Через API в коде**:
```python
# Пример использования OpenAI API (требует API ключ)
import openai

openai.api_key = "your-api-key"
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Analyze this training loss..."}]
)
```

2. **Для генерации кода**:
   - Скопируйте код из DataSphere
   - Вставьте в ChatGPT/Claude
   - Получите улучшенную версию
   - Вставьте обратно в DataSphere

3. **Для анализа результатов**:
   - Экспортируйте метрики в JSON
   - Загрузите в ChatGPT для анализа
   - Получите рекомендации по улучшению

### Альтернативы для AI-ассистента в облаке

1. **GitHub Copilot** (если DataSphere поддерживает VS Code):
   - Некоторые облачные платформы поддерживают VS Code
   - GitHub Copilot работает в VS Code

2. **Yandex GPT API** (нативный для Yandex Cloud):
```python
# Пример использования Yandex GPT API
import requests

def ask_yandex_gpt(prompt: str, api_key: str):
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {api_key}"}
    data = {"modelUri": "...", "completionOptions": {...}, "messages": [...]}
    response = requests.post(url, headers=headers, json=data)
    return response.json()
```

## Рекомендации по работе

1. **Используйте Git для версионирования**:
   - Коммитьте изменения регулярно
   - Используйте ветки для экспериментов
   - Синхронизируйте локально и в облаке

2. **Сохраняйте результаты**:
   - Регулярно скачивайте чекпоинты
   - Сохраняйте метрики в JSON
   - Документируйте эксперименты

3. **Оптимизируйте использование GPU**:
   - Используйте большие batch sizes
   - Включайте gradient accumulation
   - Мониторьте использование памяти

4. **Используйте предустановленные образы**:
   - DataSphere имеет образы с предустановленными библиотеками
   - Выберите подходящий образ при создании проекта

## Устранение проблем

### Проблема: ModuleNotFoundError

**Решение**: Убедитесь, что проект добавлен в `sys.path`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
```

### Проблема: CUDA out of memory

**Решение**: Уменьшите batch size или используйте gradient accumulation:
```python
config.training.batch_size = 8
config.training.gradient_accumulation_steps = 4
```

### Проблема: Медленная загрузка данных

**Решение**: Используйте кэширование HuggingFace Datasets или загрузите данные заранее.

## Дополнительные ресурсы

- [Документация Yandex DataSphere](https://yandex.cloud/ru/docs/datasphere/)
- [Руководство по быстрому старту](https://yandex.cloud/ru/docs/datasphere/quickstart)
- [Работа с GPU в DataSphere](https://yandex.cloud/ru/docs/datasphere/concepts/gpu)

## Контакты и поддержка

Если возникли проблемы:
1. Проверьте логи в `outputs/logs/`
2. Обратитесь в поддержку Yandex Cloud
3. Создайте issue в GitHub репозитории

