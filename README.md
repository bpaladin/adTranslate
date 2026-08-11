# PDF Translator v0.9

Параллельный перевод PDF-документов на другой язык с сохранением форматирования и изображений. Поддерживает режим перевода и генерации рефератов.

## Установка

```bash
pip install -r requirements.txt
```

Зависимости: `pymupdf`, `openai`, `requests`, `jinja2`, `tqdm`, `markdown`, `httpx`.

## Быстрый старт

```bash
# Перевод через Google (без ключа)
python main.py paper.pdf paper_ru.html

# Перевод через Gemini
python main.py paper.pdf paper_ru.html -t gemini --gemini-api-key YOUR_KEY

# Реферат статьи
python main.py paper.pdf summary.html -t llama --summary

# Локальный переводчик (llama.cpp)
python main.py paper.pdf paper_ru.html -t llama
```

## CLI

```
python main.py <input.pdf> <output.html> [опции]
```

### Обязательные аргументы
input.pdf		Входной PDF-файл
output.html		Выходной HTML-файл

### Опции

```
Опция                           По умолчанию    Значение
-l, --lang			ru		Целевой язык перевода
-t, --translator		google		Переводчик: google, gemini, openrouter, llama, groq
--summary			-		Режим реферата (LLM на англ. -> перевод)
--summary-translator		= -t		Переводчик для генерации реферата
--summary-lang-translator	google		Переводчик для перевода реферата
--api-key			-		API-ключ для OpenRouter
--gemini-api-key		-		API-ключ для Gemini
--gemini-model			gemini-flash-latest	Модель Gemini
--groq-api-key			-		API-ключ для Groq
--groq-model			auto		Модель Groq
--llama-model			gemma4		Модель llama.cpp
--llama-url			http://localhost:8080/v1	URL llama.cpp сервера
--proxy				http://127.0.0.1:7897	HTTP прокси для сетевых LLM
--no-proxy			-		Отключить прокси
--workers			cpu*2		Число параллельных воркеров
--task-timeout			600		Таймаут на весь перевод (сек)
-q, --quiet			-		Тихий режим
-v, --verbose			-		Подробный вывод
```

### Переменные окружения

```
GEMINI_API_KEY			API-ключ для Google Gemini
OPENROUTER_API_KEY		API-ключ для OpenRouter
OPENAI_API_KEY			API-ключ OpenAI (fallback для OpenRouter)
GROQ_API_KEY			API-ключ для Groq
```

## Переводчики

```
Имя             Proxy           Адрес                   Значение
google		Не нужен	Нет			Google Translate API (бесплатный)
gemini		Нужен		http://127.0.0.1:7897	Google Gemini через OpenAI-совместимый API
openrouter	Нужен		http://127.0.0.1:7897	OpenRouter (доступ к множеству моделей)
llama		Не нужен	Нет			Локальный llama.cpp сервер
groq		Нужен		Нет			Groq API (бесплатные модели)
```

- **Прокси:** для gemini и openrouter автоматически включается прокси `http://127.0.0.1:7897` (отключается через `--no-proxy`)
- **Кэш:** результаты переводов сохраняются в SQLite (`translation_cache.db`) на 30 дней

## Режимы работы

### Translate (по умолчанию)

Полный перевод PDF-документа с сохранением:
- Форматирования (абзацы, заголовки, списки)
- Изображений и подписей к ним
- Таблиц (не переводятся, выводятся в исходном виде)
- Списка литературы (не переводится)

Параллельный перевод через `ThreadPoolExecutor`. Короткие фразы (≤40 символов) — показывается только перевод. Длинный текст — перевод + свёрнутый блок с оригиналом.

### Summary (реферат)

Генерация структурированного реферата научной статьи:

```bash
# Реферат через llama (по умолчанию)
python main.py paper.pdf summary.html -t llama --summary

# Реферат через Groq, перевод через Google
python main.py paper.pdf summary.html -t groq --summary --summary-lang-translator google

# Реферат через Gemini, перевод через llama
python main.py paper.pdf summary.html -t gemini --summary --summary-lang-translator llama
```

Формат реферата:
```
КРАТКОЕ СОДЕРЖАНИЕ: 4-5 предложений
КЛЮЧЕВЫЕ НАХОДКИ: список
СИЛЬНЫЕ СТОРОНЫ: список
СЛАБЫЕ СТОРОНЫ: список
```

Режим `--summary` работает в два этапа:
1. LLM генерирует реферат **на английском** (чанки 6000 символов)
2. Полученный реферат переводится на целевой язык

## Батч-обработка

```bash
# Пакетный перевод всех PDF в каталоге
./batch_translator.sh

# Пакетная генерация рефератов
./batch_summary.sh [КАТАЛОГ] [РЕЖИМ] [МОДЕЛЬ]

# Примеры
./batch_summary.sh                                    # summary, gemma4
./batch_summary.sh ./pdfs summary_translate qwen3.5   # комбинированный режим
./batch_summary.sh ./pdfs summary smollm              # стандартный режим
```

## Архитектура

```
main.py                        — CLI обёртка
modules/
  ├── pipeline.py              — Пайплайн обработки PDF + фабрика переводчиков
  ├── translation_engine.py    — Все переводчики (Google, Gemini, OpenRouter, LlamaCpp, Groq)
  ├── pdf_extractor.py         — Извлечение текста из PDF (PyMuPDF)
  ├── block_classifier.py      — Классификация блоков (heading/paragraph/reference)
  ├── image_processor.py       — Извлечение изображений
  ├── table_processor.py       — Обработка таблиц
  ├── cache_manager.py         — SQLite кэш переводов
  └── html_renderer.py         — Генерация HTML (Jinja2)
```

### Пайплайн

```
[1/5] Извлечение данных из PDF (текст, таблицы, изображения)
  ↓
[2/5] Извлечение изображений и таблиц
  ↓
[3/5] Классификация блоков (heading/paragraph/reference/table)
  ↓
[4/5] Пост-обработка (объединение ссылок, удаление дубликатов)
  ↓
[5/5] Параллельный перевод или Генерация реферата
  ↓
     Генерация HTML-отчёта
```

## Автоопределение модели llama

Если модель не задана или используется `gemma4` по умолчанию, llama.cpp автоматически:
1. Запрашивает список доступных моделей через `GET /v1/models`
2. Ищет модель с `gemma` и `4` в названии
3. Если не найдено — берёт первую доступную

## Логирование

- **Консоль:** `INFO` и выше (`-v` — `DEBUG`, `-q` — `WARNING`)
- **Файл:** `pdf_translator.log` (все сообщения, `DEBUG`+)
