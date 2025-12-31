#!/bin/bash
# Быстрая установка для VM в Yandex Cloud (< 10 минут теста)

set -e

echo "🚀 Быстрая установка для VM..."

# Обновление системы
echo "📦 Обновление системы..."
sudo apt update && sudo apt upgrade -y

# Установка Python и зависимостей
echo "🐍 Установка Python..."
sudo apt install -y python3.10 python3-pip python3-venv git curl

# Клонирование проекта (если еще не клонировано)
if [ ! -d "Iterative-Expert-Guided-Fine-Tuning" ]; then
    echo "📥 Клонирование проекта..."
    git clone https://github.com/Aris12122/Iterative-Expert-Guided-Fine-Tuning.git
fi

cd Iterative-Expert-Guided-Fine-Tuning

# Виртуальное окружение
if [ ! -d "venv" ]; then
    echo "🔧 Создание виртуального окружения..."
    python3 -m venv venv
fi

echo "🔌 Активация виртуального окружения..."
source venv/bin/activate

# Зависимости
echo "📚 Установка зависимостей..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Установка завершена!"
echo ""
echo "📝 Следующие шаги:"
echo "   1. source venv/bin/activate"
echo "   2. python scripts/quick_test_vm.py"
echo ""
echo "⏱️  Ожидаемое время теста: 5-7 минут"

