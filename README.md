# PDF Translator

Параллельный перевод PDF-документов на другой язык с сохранением форматирования и изображений.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

```bash
python main.py input.pdf output.html -t gemini
```

### Позиционные аргументы

| Аргумент | Описание |
|----------|----------|
| `input` | Входной PDF файл |
| `output` | Выходной HTML файл |

### Опции

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `-l, --lang` | `ru` | Целевой язык перевода |
| `-t, --translator` | `openrouter` | Переводчик: `google`, `openrouter`, `llama`, `gemini` |
| `--api-key` | — | API ключ (или переменная окружения) |
| `--gemini-model` | `gemini-2.0-flash` | Модель Gemini |
| `--llama-model` | `gemma4` | Модель llama.cpp |
| `--llama-url` | `http://localhost:8080/v1` | URL llama.cpp сервера |
| `--workers` | `auto` | Число параллельных воркеров |
| `--task-timeout` | `600` | Таймаут на перевод (сек) |
| `-q, --quiet` | — | Тихий режим |
| `-v, --verbose` | — | Подробный вывод |

### Переменные окружения

| Переменная | Для кого |
|------------|----------|
| `GEMINI_API_KEY` | Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |
| `OPENAI_API_KEY` | OpenRouter (fallback) |

### Примеры

```bash
# Перевод через Gemini
GEMINI_API_KEY=your_key python main.py paper.pdf paper_ru.html -t gemini

# Перевод через Google (без ключа)
python main.py paper.pdf paper_ru.html -t google

# Перевод на английский
python main.py paper.pdf paper_en.html -l en -t openrouter --api-key your_key
```

## Архитектура

```
main.py                    — CLI обёртка
modules/
  pdf_extractor.py         — Извлечение текста и структуры из PDF
  block_classifier.py      — Классификация блоков (heading, paragraph, reference, table...)
  table_processor.py       — Извлечение таблиц
  image_processor.py       — Извлечение изображений
  translation_engine.py    — Переводчики (Google, OpenRouter, LlamaCpp, Gemini)
  pipeline.py              — Оркестрация: извлечение → классификация → перевод → HTML
  cache_manager.py         — SQLite кэш переводов
  html_renderer.py         — Генерация HTML-отчёта
```

## Поведение

- **Таблицы** не переводятся, выводятся в исходном виде.
- **Список литературы** (после заголовка References/Bibliography/Литература) не переводится.
- **Короткие фразы** (≤40 символов) — показывается только перевод.
- **Длинный текст** — перевод + свёрнутый блок с оригиналом.
- Используется SQLite-кэш для повторных запросов.
- Автоматический fallback: основной переводчик → Google.
