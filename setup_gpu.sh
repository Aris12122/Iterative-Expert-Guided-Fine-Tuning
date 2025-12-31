#!/bin/bash
# Скрипт для автоматической настройки окружения на GPU-машине в Yandex Cloud

set -e  # Остановка при ошибке

echo "=== Настройка окружения для GPU обучения ==="

# Проверка наличия GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ОШИБКА: nvidia-smi не найден. Убедитесь, что NVIDIA драйверы установлены."
    exit 1
fi

echo "Проверка GPU..."
nvidia-smi

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo "Установка Python 3.10..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv python3-pip
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python версия: $PYTHON_VERSION"

# Создание виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

echo "Активация виртуального окружения..."
source venv/bin/activate

# Обновление pip
echo "Обновление pip..."
pip install --upgrade pip

# Установка PyTorch с CUDA
echo "Установка PyTorch с CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Проверка CUDA
echo "Проверка CUDA в PyTorch..."
python3 -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}'); print(f'CUDA версия: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ПРЕДУПРЕЖДЕНИЕ: CUDA не доступна в PyTorch. Проверьте установку."
fi

# Установка остальных зависимостей
echo "Установка остальных зависимостей..."
pip install -r requirements.txt

# Установка основных библиотек
echo "Установка основных библиотек..."
pip install transformers>=4.30.0 datasets>=2.14.0 numpy>=1.24.0 tqdm>=4.65.0

# Проверка установки
echo "Проверка установки..."
python3 -c "
import torch
import transformers
import datasets
print('✓ Все зависимости установлены успешно')
print(f'✓ PyTorch версия: {torch.__version__}')
print(f'✓ Transformers версия: {transformers.__version__}')
print(f'✓ CUDA доступна: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
    print(f'✓ CUDA версия: {torch.version.cuda}')
"

echo ""
echo "=== Настройка завершена ==="
echo ""
echo "Для активации окружения выполните:"
echo "  source venv/bin/activate"
echo ""
echo "Для запуска обучения используйте:"
echo "  python3 scripts/train_supervised.py --experiment_name my_experiment --batch_size 32 --num_epochs 5"

