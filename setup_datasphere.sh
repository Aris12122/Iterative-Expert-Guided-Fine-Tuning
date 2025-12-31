#!/bin/bash
# Скрипт для быстрой настройки проекта в Yandex DataSphere

set -e

echo "🚀 Настройка проекта для Yandex DataSphere..."

# Определение корня проекта
if [ -d "Iterative-Expert-Guided-Fine-Tuning" ]; then
    cd Iterative-Expert-Guided-Fine-Tuning
fi

PROJECT_ROOT=$(pwd)
echo "📁 Project root: $PROJECT_ROOT"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 не найден. Установите Python 3.10+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "✓ Python: $PYTHON_VERSION"

# Проверка PyTorch и CUDA
echo ""
echo "🔍 Проверка PyTorch и CUDA..."
python3 << EOF
import sys
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
    else:
        print("⚠ CUDA недоступна. Установите PyTorch с CUDA поддержкой:")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
except ImportError:
    print("❌ PyTorch не установлен")
    print("Установите PyTorch:")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)
EOF

# Установка зависимостей
echo ""
echo "📦 Установка зависимостей..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Зависимости установлены из requirements.txt"
else
    echo "❌ Файл requirements.txt не найден"
    exit 1
fi

# Проверка структуры проекта
echo ""
echo "🔍 Проверка структуры проекта..."
REQUIRED_DIRS=("src" "scripts")
MISSING_DIRS=()

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        MISSING_DIRS+=("$dir")
    else
        echo "✓ Директория $dir существует"
    fi
done

if [ ${#MISSING_DIRS[@]} -ne 0 ]; then
    echo "❌ Отсутствуют директории: ${MISSING_DIRS[*]}"
    exit 1
fi

# Проверка импортов
echo ""
echo "🧪 Проверка импортов..."
python3 << EOF
import sys
from pathlib import Path

project_root = Path("$PROJECT_ROOT")
sys.path.insert(0, str(project_root))

try:
    from src.config import default_medqa_experiment
    from src.training.supervised import SupervisedExperiment
    from src.training.distillation import KDExperiment
    from src.training.active_loop import ActiveLoopExperiment
    print("✓ Все основные импорты успешны")
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    sys.exit(1)
EOF

# Создание директорий для результатов
echo ""
echo "📁 Создание директорий для результатов..."
mkdir -p outputs/checkpoints
mkdir -p outputs/results
mkdir -p outputs/logs
mkdir -p outputs/predictions
echo "✓ Директории созданы"

# Финальная проверка
echo ""
echo "✅ Настройка завершена!"
echo ""
echo "📝 Следующие шаги:"
echo "1. Откройте notebook: notebooks/datasphere_training.ipynb"
echo "2. Или запустите скрипт: python scripts/train_supervised.py"
echo ""
echo "💡 Полезные команды:"
echo "  - Baseline 1: python scripts/train_supervised.py --experiment_name test"
echo "  - Baseline 2: python scripts/train_kd.py --experiment_name test"
echo "  - Baseline 3: python scripts/train_active_loop.py --experiment_name test"

