# Интеграция с Cursor и ChatGPT в Yandex DataSphere

## Можно ли интегрировать Cursor или ChatGPT в DataSphere?

**Короткий ответ**: Cursor нельзя напрямую интегрировать в DataSphere, но есть несколько эффективных способов работы с AI-ассистентами.

## Варианты работы с AI-ассистентами

### 1. Локальная разработка в Cursor + синхронизация через Git

**Рекомендуемый подход** для разработки:

1. **Разрабатывайте код локально в Cursor**:
   - Используйте все возможности Cursor AI
   - Тестируйте на небольших данных локально
   - Коммитьте изменения в Git

2. **Синхронизируйте с DataSphere**:
   ```bash
   # В DataSphere терминале
   git pull origin main
   ```

3. **Запускайте обучение в DataSphere**:
   - Используйте GPU в облаке
   - Запускайте длительные эксперименты
   - Сохраняйте результаты

4. **Анализируйте результаты локально**:
   - Скачайте результаты из DataSphere
   - Откройте в Cursor для анализа
   - Используйте AI для улучшения кода

**Преимущества**:
- ✅ Полный доступ к возможностям Cursor
- ✅ Быстрая разработка локально
- ✅ GPU обучение в облаке
- ✅ Версионирование через Git

### 2. Использование ChatGPT/Claude через API

Вы можете интегрировать AI-ассистентов прямо в код:

#### Пример с OpenAI API:

```python
# Установка: pip install openai
import openai
import json

def analyze_training_results(results_file: str, api_key: str):
    """Анализ результатов обучения через ChatGPT."""
    with open(results_file) as f:
        results = json.load(f)
    
    prompt = f"""
    Проанализируй результаты обучения модели:
    {json.dumps(results, indent=2)}
    
    Предложи рекомендации по улучшению.
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        api_key=api_key
    )
    
    return response.choices[0].message.content

# Использование
analysis = analyze_training_results(
    "outputs/results/supervised_datasphere.json",
    api_key="your-openai-api-key"
)
print(analysis)
```

#### Пример с Yandex GPT API (нативный для Yandex Cloud):

```python
import requests
import json

def ask_yandex_gpt(prompt: str, api_key: str, folder_id: str):
    """Использование Yandex GPT API."""
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "modelUri": f"gcr://yandexcloud/llm/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.6,
            "maxTokens": 2000
        },
        "messages": [
            {
                "role": "user",
                "text": prompt
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()

# Использование
result = ask_yandex_gpt(
    prompt="Объясни, что такое knowledge distillation",
    api_key="your-yandex-api-key",
    folder_id="your-folder-id"
)
print(result)
```

### 3. Работа через веб-интерфейс ChatGPT/Claude

**Простой подход** для анализа и генерации кода:

1. **Скопируйте код из DataSphere**:
   - Выделите нужный фрагмент
   - Скопируйте в буфер обмена

2. **Вставьте в ChatGPT/Claude**:
   - Вставьте код
   - Задайте вопрос или попросите улучшить
   - Получите ответ

3. **Вставьте обратно в DataSphere**:
   - Скопируйте улучшенный код
   - Вставьте в notebook или скрипт

**Примеры запросов**:
- "Объясни этот код обучения модели"
- "Как оптимизировать этот цикл обучения?"
- "Почему accuracy низкая? Что можно улучшить?"
- "Сгенерируй код для визуализации метрик"

### 4. Использование GitHub Copilot (если доступно)

Если DataSphere поддерживает VS Code или имеет расширения:

1. Установите расширение GitHub Copilot
2. Используйте автодополнение кода
3. Получайте предложения прямо в редакторе

**Проверка доступности**: Откройте настройки DataSphere и проверьте доступные расширения.

## Рекомендуемый workflow

### Для разработки нового кода:

```
1. Локально (Cursor)
   ↓
   Разработка и тестирование
   ↓
2. Git commit & push
   ↓
3. DataSphere (git pull)
   ↓
   Запуск обучения на GPU
   ↓
4. Локально (Cursor)
   ↓
   Анализ результатов и улучшение
```

### Для быстрого эксперимента:

```
1. DataSphere Notebook
   ↓
   Быстрый прототип
   ↓
2. ChatGPT/Claude (веб)
   ↓
   Анализ и предложения
   ↓
3. DataSphere Notebook
   ↓
   Внедрение улучшений
```

## Практические примеры интеграции

### Пример 1: Автоматический анализ метрик

```python
# В DataSphere notebook
import json
import requests

def get_ai_analysis(metrics: dict, api_key: str) -> str:
    """Получить анализ метрик от AI."""
    prompt = f"""
    Я обучил модель для задачи множественного выбора (MCQ).
    Результаты:
    - Accuracy: {metrics.get('accuracy', 0):.4f}
    - Expected Correctness: {metrics.get('expected_correctness', 0):.4f}
    
    Что можно улучшить? Дай конкретные рекомендации.
    """
    
    # Используйте ваш предпочитаемый API
    # (OpenAI, Yandex GPT, Claude и т.д.)
    return "AI analysis here"

# После обучения
final_metrics = experiment.evaluate()
analysis = get_ai_analysis(final_metrics, api_key="your-key")
print(analysis)
```

### Пример 2: Генерация кода для визуализации

```python
# Запрос к ChatGPT через API или вручную
prompt = """
Сгенерируй код для визуализации метрик обучения:
- График loss по эпохам
- График accuracy по эпохам
- Сравнение разных экспериментов

Используй matplotlib.
"""

# Вставьте сгенерированный код сюда
```

### Пример 3: Оптимизация гиперпараметров

```python
# Используйте AI для предложения гиперпараметров
def suggest_hyperparameters(current_config: dict, results: dict) -> dict:
    """Получить предложения по гиперпараметрам от AI."""
    prompt = f"""
    Текущая конфигурация: {json.dumps(current_config, indent=2)}
    Текущие результаты: {json.dumps(results, indent=2)}
    
    Предложи улучшенные гиперпараметры для лучших результатов.
    """
    # Вызов AI API
    return suggested_config
```

## Безопасность и приватность

⚠️ **Важно**: При использовании внешних AI API:

1. **Не загружайте чувствительные данные** в публичные AI сервисы
2. **Используйте API ключи безопасно**:
   - Храните в переменных окружения
   - Не коммитьте ключи в Git
   - Используйте `.env` файлы (не в Git)

```python
# Правильно: использование переменных окружения
import os
api_key = os.getenv("OPENAI_API_KEY")

# Неправильно: хардкод ключей
api_key = "sk-..."  # ❌ НЕ ДЕЛАЙТЕ ТАК
```

## Альтернативные решения

### 1. Локальный AI (если доступно)

Если у вас есть мощная локальная машина:
- Установите локальный LLM (Llama, Mistral и т.д.)
- Используйте через API локально
- Полная приватность данных

### 2. Yandex GPT (рекомендуется для Yandex Cloud)

- Нативный сервис Yandex Cloud
- Хорошая интеграция с экосистемой
- Возможны специальные тарифы для пользователей Cloud

### 3. Codeium / Tabnine

- Бесплатные альтернативы Copilot
- Работают в веб-редакторах
- Могут быть доступны в DataSphere

## Итоговые рекомендации

1. **Для разработки**: Используйте Cursor локально + Git синхронизацию
2. **Для анализа**: Используйте ChatGPT/Claude через веб или API
3. **Для обучения**: Используйте DataSphere с GPU
4. **Для версионирования**: Всегда используйте Git

## Полезные ссылки

- [Yandex GPT API документация](https://yandex.cloud/ru/docs/foundation-models/)
- [OpenAI API документация](https://platform.openai.com/docs)
- [GitHub Copilot](https://github.com/features/copilot)
- [Cursor Editor](https://cursor.sh/)

## Вопросы?

Если у вас есть вопросы по интеграции, создайте issue в репозитории или обратитесь в поддержку Yandex Cloud.

