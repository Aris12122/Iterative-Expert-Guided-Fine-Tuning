# Развертывание в Yandex Cloud с GPU

## Выбор сервиса Yandex Cloud

### Рекомендуемый вариант: **Yandex Compute Cloud**

**Преимущества:**
- Полный контроль над окружением
- Гибкая настройка ресурсов
- Возможность использовать любые модели и библиотеки
- Оптимизация стоимости (оплата только за время использования)
- Простое масштабирование

**Альтернативы:**
- **Yandex DataSphere** — удобно для экспериментов в Jupyter, но менее гибко
- **Yandex ML Platform** — для продакшн-развертывания, но избыточно для экспериментов

## Пошаговая инструкция

### 1. Создание виртуальной машины с GPU

1. **Войдите в консоль Yandex Cloud:**
   - Перейдите на https://console.cloud.yandex.ru/
   - Выберите нужный каталог

2. **Создайте виртуальную машину:**
   - Перейдите в раздел **Compute Cloud** → **Виртуальные машины**
   - Нажмите **Создать виртуальную машину**

3. **Настройте параметры:**
   - **Имя:** `ml-training-gpu` (или любое другое)
   - **Зона доступности:** Выберите зону с GPU (обычно `ru-central1-a` или `ru-central1-b`)
   - **Платформа:** Intel Broadwell или новее
   - **vCPU:** 8-16 ядер (рекомендуется 16)
   - **RAM:** 32-64 GB (рекомендуется 64 GB для больших моделей)
   - **GPU:** 
     - **NVIDIA Tesla V100** (16 GB) — для больших моделей (BERT-large, RoBERTa-large)
     - **NVIDIA Tesla T4** (16 GB) — оптимальный баланс цена/производительность
     - **NVIDIA A100** (40/80 GB) — для очень больших моделей (если доступно)
   - **Диск:** 
     - SSD, минимум 100 GB (рекомендуется 200-500 GB для моделей и датасетов)
   - **Образ:** Ubuntu 22.04 LTS или Ubuntu 20.04 LTS

4. **Настройте сеть:**
   - Выберите существующую сеть или создайте новую
   - Настройте публичный IP (для доступа по SSH)

5. **Настройте доступ:**
   - Добавьте SSH-ключ для доступа
   - Или используйте логин/пароль

6. **Создайте виртуальную машину**

### 2. Подключение к виртуальной машине

```bash
ssh -i ~/.ssh/your_key.pem ubuntu@<PUBLIC_IP>
# или
ssh ubuntu@<PUBLIC_IP>
```

### 3. Установка NVIDIA драйверов и CUDA

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка NVIDIA драйверов
sudo apt install -y nvidia-driver-535 nvidia-utils-535

# Перезагрузка (необходима для активации драйверов)
sudo reboot
```

После перезагрузки проверьте установку:
```bash
nvidia-smi
```

Должна отобразиться информация о GPU.

### 4. Установка Python и зависимостей

```bash
# Установка Python 3.10+
sudo apt install -y python3.10 python3.10-venv python3-pip

# Создание директории для проекта
mkdir -p ~/projects
cd ~/projects

# Клонирование проекта (или загрузка файлов через scp/sftp)
# Если проект в Git:
# git clone <your-repo-url> "Iterative Expert-Guided Fine-Tuning"

# Или загрузите файлы проекта через scp:
# scp -r /local/path/to/project ubuntu@<PUBLIC_IP>:~/projects/
```

### 5. Настройка виртуального окружения с GPU-поддержкой

```bash
cd ~/projects/"Iterative Expert-Guided Fine-Tuning"

# Создание виртуального окружения
python3.10 -m venv venv
source venv/bin/activate

# Обновление pip
pip install --upgrade pip

# Установка PyTorch с CUDA поддержкой
# Для CUDA 11.8 (совместимо с большинством GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Или для CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Установка остальных зависимостей
pip install -r requirements.txt

# Проверка установки GPU
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 6. Настройка проекта для GPU

Код уже поддерживает GPU через `device="auto"` в конфигурации. Убедитесь, что:

1. **В `src/config.py`** параметр `device` установлен в `"auto"` (по умолчанию)
2. **В `default_medqa_experiment()`** можно увеличить `batch_size` для GPU:

```python
training = TrainingConfig(
    batch_size=32,  # Увеличьте для GPU (было 8 для CPU)
    num_epochs=5,   # Увеличьте для полного обучения
    # ... остальные параметры
)
```

### 7. Запуск обучения

```bash
# Активация окружения (если еще не активировано)
source venv/bin/activate

# Запуск Baseline 1 (Supervised)
python3 scripts/train_supervised.py \
    --experiment_name "supervised_gpu_full" \
    --batch_size 32 \
    --num_epochs 5 \
    --max_samples None  # Использовать весь датасет

# Запуск Baseline 2 (Knowledge Distillation)
python3 scripts/train_kd.py \
    --experiment_name "kd_gpu_full" \
    --batch_size 32 \
    --num_epochs 5 \
    --max_samples None

# Запуск Baseline 3 (Active Loop)
python3 scripts/train_active_loop.py \
    --experiment_name "active_loop_gpu_full" \
    --batch_size 32 \
    --num_epochs 5 \
    --max_samples None
```

### 8. Мониторинг обучения

В другом терминале (или через `screen`/`tmux`):

```bash
# Мониторинг GPU
watch -n 1 nvidia-smi

# Просмотр логов
tail -f outputs/logs/<experiment_name>.log

# Использование screen для долгих экспериментов
screen -S training
# Запустите обучение
# Нажмите Ctrl+A, затем D для отсоединения
# Вернуться: screen -r training
```

## Рекомендации по моделям

### Текущие модели (подходят для начала)

**Student модели:**
- `distilbert-base-uncased` — быстрая, легкая (66M параметров)
- `bert-base-uncased` — стандартная (110M параметров)
- `roberta-base` — улучшенная версия BERT (125M параметров)

**Teacher модели:**
- `bert-base-uncased` — хорошая базовая модель
- `roberta-base` — отличная альтернатива BERT

### Рекомендации для улучшения результатов

**Для Student (можно использовать более мощные модели на GPU):**
1. **`bert-base-uncased`** или **`roberta-base`** вместо DistilBERT
   - Больше параметров → лучше качество
   - На GPU обучение все равно будет быстрым

2. **Биомедицинские модели для Student:**
   - `emilyalsentzer/Bio_ClinicalBERT` — предобучена на медицинских текстах
   - `dmis-lab/biobert-base-cased-v1.2` — BioBERT
   - `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` — PubMedBERT

**Для Teacher Ensemble (рекомендуется использовать более мощные модели):**
1. **`bert-large-uncased`** (340M параметров)
   - Значительно лучше, чем base-версия
   - Требует больше памяти (16GB GPU достаточно)

2. **`roberta-large`** (355M параметров)
   - Одна из лучших моделей для MCQA

3. **Биомедицинские large-модели:**
   - `dmis-lab/biobert-large-cased-v1.1` — BioBERT Large
   - `microsoft/BiomedNLP-PubMedBERT-large-uncased-abstract-fulltext` — PubMedBERT Large

4. **Специализированные MCQA модели:**
   - `google/electra-large-discriminator` — хороша для вопросов-ответов
   - `allenai/scibert_scivocab_uncased` — для научных/медицинских текстов

### Пример конфигурации с улучшенными моделями

Создайте функцию в `src/config.py`:

```python
def improved_medqa_experiment(
    experiment_name: str = "medqa_improved",
    experiment_type: Literal["supervised", "kd", "active_loop"] = "supervised",
) -> ExperimentConfig:
    """Конфигурация с улучшенными моделями для GPU."""
    dataset = DatasetConfig(
        dataset_name="medmcqa",
        max_samples=None,  # Весь датасет
        max_seq_length=512,
    )
    
    model = ModelConfig(
        # Более мощная student модель
        student_model_name="bert-base-uncased",  # или "roberta-base"
        # Мощные teacher модели
        teacher_model_names=[
            "bert-large-uncased",
            "roberta-large",
        ] if experiment_type != "supervised" else [],
        num_labels=4,
        device="auto",  # Автоматически использует GPU
    )
    
    training = TrainingConfig(
        batch_size=32,  # Больше для GPU
        num_epochs=5,
        learning_rate=2e-5,
        warmup_steps=500,
        eval_steps=500,
        logging_steps=100,
        num_threads=4,  # Можно увеличить на GPU-машине
    )
    
    # ... остальная конфигурация
```

## Оптимизация производительности

### 1. Увеличение batch size
На GPU можно использовать большие батчи:
- **Tesla T4:** batch_size=32-64
- **Tesla V100:** batch_size=64-128
- **A100:** batch_size=128-256

### 2. Gradient Accumulation
Если модель не помещается в память с большим batch_size:
```python
training = TrainingConfig(
    batch_size=16,
    gradient_accumulation_steps=4,  # Эффективный batch_size = 16 * 4 = 64
)
```

### 3. Mixed Precision Training
Добавьте поддержку FP16 для ускорения (опционально, требует изменений в коде):
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(**inputs)
```

### 4. DataLoader оптимизация
В `src/data.py` для GPU можно использовать:
```python
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Параллельная загрузка данных
    pin_memory=True,  # Быстрее передача на GPU
)
```

## Управление ресурсами и стоимостью

### Оценка стоимости

**Примерные цены (на момент написания):**
- **Tesla T4:** ~150-200 руб/час
- **Tesla V100:** ~300-400 руб/час
- **A100:** ~500-700 руб/час

**Время обучения (примерно):**
- Baseline 1 (5 эпох, полный датасет): 2-4 часа
- Baseline 2 (5 эпох): 3-5 часов (с teacher ensemble)
- Baseline 3 (5 эпох): 4-6 часов (с активным обучением)

**Итого:** Один полный эксперимент ~10-15 часов = 1500-3000 руб (на T4)

### Экономия средств

1. **Используйте preemptible инстансы** (если доступно) — дешевле на 50-70%
2. **Останавливайте VM** когда не используете
3. **Начните с небольших экспериментов** для проверки кода
4. **Используйте checkpointing** — сохраняйте модели регулярно

## Резервное копирование результатов

```bash
# Создание архива результатов
tar -czf results_backup_$(date +%Y%m%d).tar.gz outputs/

# Загрузка на локальную машину
scp ubuntu@<PUBLIC_IP>:~/projects/*/outputs/*.tar.gz ./

# Или используйте Yandex Object Storage для автоматического бэкапа
```

## Troubleshooting

### Проблема: CUDA out of memory
**Решение:**
- Уменьшите `batch_size`
- Увеличьте `gradient_accumulation_steps`
- Используйте более легкие модели

### Проблема: Медленная загрузка данных
**Решение:**
- Увеличьте `num_workers` в DataLoader
- Используйте `pin_memory=True`
- Предзагрузите датасет на диск

### Проблема: Модели не загружаются
**Решение:**
- Проверьте интернет-соединение
- Используйте HuggingFace cache: `export HF_HOME=~/hf_cache`
- Загрузите модели заранее: `python -c "from transformers import AutoModel; AutoModel.from_pretrained('bert-base-uncased')"`

## Дополнительные ресурсы

- [Документация Yandex Compute Cloud](https://cloud.yandex.ru/docs/compute/)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [PyTorch CUDA Installation](https://pytorch.org/get-started/locally/)
- [HuggingFace Models](https://huggingface.co/models)
