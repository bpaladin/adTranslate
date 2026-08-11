#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Универсальный переводчик научных статей с использованием llama.cpp-server (OpenAI API).
Автоматически определяет доступные модели и размер контекста через API /v1/models.
Если указанная модель не найдена или не работает, автоматически выбирается первая работающая.
"""

import os
import sys
import re
import json
import logging
import argparse
import time
import requests
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

# ========== НАСТРОЙКА ЛОГИРОВАНИЯ ==========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pdf_processor.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========== КОНФИГУРАЦИЯ ПО УМОЛЧАНИЮ ==========
START_LLAMA_SCRIPT = os.path.expanduser("~/bin/start_llama.sh")
DEFAULT_TRANSLATE_MODEL = "translate"

DEFAULT_CONFIG = {
    "api_base": "http://localhost:8080/v1",
    "model": "translate",
    "temperature": 0.2,
    "max_tokens": 4096,
    "timeout": 180,
    "chunk_size": 1500,
    "overlap": 150,
    "retry_attempts": 3,
    "ctx_size": 2048,
}

# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========
def validate_folders(input_folder: str, output_folder: str) -> Tuple[bool, Optional[str]]:
    if not os.path.exists(input_folder):
        return False, f"Папка не найдена: {input_folder}"
    if not os.path.isdir(input_folder):
        return False, f"Указанный путь не является папкой: {input_folder}"
    try:
        os.makedirs(output_folder, exist_ok=True)
        test_file = os.path.join(output_folder, ".write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        return False, f"Нет прав на запись в {output_folder}: {e}"
    return True, None

def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.M)
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.I)
    text = re.sub(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', '', text, flags=re.I)
    text = re.sub(r'doi:\s*10\.\d{4,9}/[-._;()/:A-Z0-9]+', '', text, flags=re.I)
    text = re.sub(r'(Figure|Fig\.|Table)\s+\d+\.?.*?\n', '', text, flags=re.I)
    text = re.sub(r'Header|Footer|Running head|Received:|Accepted:|Published:', '', text, flags=re.I)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\[\d+(?:–\d+)?\]', '', text)
    text = re.sub(r'\(\w+,\s*\d{4}\)', '', text)
    text = re.sub(r'^\s*References?\s*$.*$', '', text, flags=re.I | re.M)
    text = re.sub(r'^\[\d+\].*$', '', text, flags=re.M)
    text = re.sub(r'ISSN\s+\d{4}-\d{4}', '', text, flags=re.I)
    text = re.sub(r'ISBN\s+\d+-\d+-\d+-\d+-\d+', '', text, flags=re.I)
    text = re.sub(r'©\s*\d{4}.*?\n', '', text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text.strip()

def extract_title_and_references(text: str) -> Tuple[Optional[str], Optional[str]]:
    title = None
    references = None
    ref_match = re.search(r'(?i)(?:References|Bibliography)\s*\n+', text)
    if ref_match:
        ref_start = ref_match.end()
        ref_text = text[ref_start:].strip()
        ref_text = re.sub(r'^\s*\d+\s*$', '', ref_text, flags=re.M)
        if len(ref_text) > 50:
            references = ref_text
    lines = text.split('\n')
    title_lines = []
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        if re.search(r'10\.\d{4,9}/', line) or re.search(r'@', line) or re.search(r'^https?://', line):
            continue
        if re.match(r'^\s*\d+\s*$', line):
            continue
        title_lines.append(line)
        if len(' '.join(title_lines)) > 100:
            break
    if title_lines:
        title = ' '.join(title_lines)[:200]
    return title, references

def extract_abstract(text: str) -> Optional[str]:
    patterns = [
        r'abstract\s*[.:]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        r'Abstract\s*[.:]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        r'ABSTRACT\s*[.:]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        r'Summary\s*[.:]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
        r'Background\s*[.:]\s*(.*?)(?=\n\n|\n[A-Z]|$)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            abstract = match.group(1).strip()
            return abstract[:3000]
    return text[:500].strip()

# ========== ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF ==========
def extract_text_pdftotext(pdf_path: str) -> str:
    try:
        import subprocess
        result = subprocess.run(
            ['pdftotext', '-l', '5', pdf_path, '-'],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout
    except Exception:
        pass
    return ""

def extract_text_pdfplumber(pdf_path: str) -> Optional[str]:
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages[:5]):
                try:
                    page_text = page.extract_text(layout=True)
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"pdfplumber ошибка на странице {i+1}: {e}")
                    continue
        return text if text.strip() else None
    except Exception as e:
        logger.warning(f"pdfplumber не удалось: {e}")
        return None

def extract_text_from_pdf(pdf_path: str, text_limit: int) -> Optional[str]:
    text = ""
    try:
        import fitz
        fitz.TOOLS.mupdf_display_errors(False)
        doc = fitz.open(pdf_path)
        total_len = 0
        for page_num in range(min(8, len(doc))):
            page = doc[page_num]
            rect = page.rect
            crop = fitz.Rect(rect.x0, rect.y0 + rect.height*0.12,
                             rect.x1, rect.y1 - rect.height*0.08)
            page_text = page.get_text("text", clip=crop)
            if page_text:
                text += page_text + "\n"
                total_len += len(page_text)
                if total_len >= text_limit:
                    break
        doc.close()
        if text.strip():
            return clean_text(text)
    except ImportError:
        logger.warning("PyMuPDF не установлен")
    except Exception as e:
        logger.warning(f"PyMuPDF ошибка: {e}")

    text = extract_text_pdftotext(pdf_path)
    if text.strip():
        return clean_text(text)

    text = extract_text_pdfplumber(pdf_path)
    if text:
        return clean_text(text)

    return None

# ========== ЗАПРОСЫ К LLM ==========
def query_llm(prompt: str, system_prompt: Optional[str] = None,
              config: Dict = DEFAULT_CONFIG, retry: int = 0) -> Tuple[Optional[str], Optional[str]]:
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": config["model"],
            "messages": messages,
            "temperature": config["temperature"],
            "max_tokens": config["max_tokens"],
            "stream": False
        }

        response = requests.post(
            f"{config['api_base']}/chat/completions",
            json=payload,
            timeout=config["timeout"]
        )

        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0].get("message", {}).get("content")
                if content:
                    return content.strip(), None
                return None, "Ответ не содержит текста"
            return None, "Неверный формат ответа"
        else:
            error_msg = f"HTTP {response.status_code}: {response.text}"
            if response.status_code == 400 and "model" in response.text.lower():
                logger.warning("Модель не найдена, возможно, требуется обновить имя модели")
            return None, error_msg

    except requests.exceptions.ConnectionError:
        return None, "Не удалось подключиться к серверу"
    except requests.exceptions.Timeout:
        return None, "Таймаут запроса"
    except Exception as e:
        if retry < config.get("retry_attempts", 2):
            wait = 2 * (retry + 1)
            logger.warning(f"Повторная попытка {retry+1} через {wait} сек.")
            time.sleep(wait)
            return query_llm(prompt, system_prompt, config, retry+1)
        return None, str(e)

# ========== ПОЛУЧЕНИЕ ИНФОРМАЦИИ О МОДЕЛЯХ ==========
def get_available_models(api_base: str) -> Tuple[List[str], Dict[str, Any]]:
    model_names = []
    models_info = {}
    try:
        resp = requests.get(f"{api_base}/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data:
                for m in data["data"]:
                    model_id = m.get("id")
                    if model_id:
                        model_names.append(model_id)
                        ctx = m.get("context_length") or m.get("context_len") or m.get("max_context_length")
                        if ctx:
                            models_info[model_id] = {"context_length": int(ctx)}
                        else:
                            models_info[model_id] = {}
        else:
            logger.warning(f"Не удалось получить список моделей: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Ошибка при запросе /v1/models: {e}")
    return model_names, models_info

def test_model(api_base: str, model_name: str) -> bool:
    """Проверяет, доступна ли модель для инференса (короткий запрос)."""
    try:
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False
        }
        resp = requests.post(f"{api_base}/chat/completions", json=payload, timeout=10)
        return resp.status_code == 200
    except Exception:
        return False

# ========== ЗАПУСК И ПРОВЕРКА СЕРВЕРА ==========
def check_server(api_base: str) -> bool:
    """Проверяет, отвечает ли llama server."""
    try:
        resp = requests.get(f"{api_base}/models", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False

def ensure_server_running(api_base: str) -> bool:
    """Проверяет сервер; если не запущен — запускает start_llama.sh и ждёт."""
    if check_server(api_base):
        print("✓ Сервер llama уже запущен")
        return True

    script_path = START_LLAMA_SCRIPT
    if not os.path.isfile(script_path):
        print(f"❌ Сервер не запущен, скрипт запуска не найден: {script_path}")
        print("   Запустите llama.cpp-server вручную.")
        return False

    print(f"🔄 Запуск llama server: {script_path}")
    try:
        import subprocess
        subprocess.Popen(
            [script_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
    except Exception as e:
        print(f"❌ Не удалось запустить скрипт: {e}")
        return False

    print("⏳ Ожидание запуска сервера (до 15 сек)...")
    for _ in range(15):
        time.sleep(1)
        if check_server(api_base):
            print("✓ Сервер llama запущен успешно")
            return True

    print("❌ Сервер не ответил за 15 секунд")
    return False

def select_default_model(api_base: str, models_info: Dict) -> str:
    """Выбирает модель translate по умолчанию или первую доступную."""
    if DEFAULT_TRANSLATE_MODEL in models_info:
        return DEFAULT_TRANSLATE_MODEL
    model_names = list(models_info.keys())
    if model_names:
        return model_names[0]
    return DEFAULT_CONFIG["model"]

# ========== ПЕРЕВОД ==========
def simple_translate(text: str, config: Dict) -> Tuple[Optional[str], Optional[str]]:
    max_chars = int(config.get("ctx_size", 2048) * 4 * 0.7)
    if len(text) > max_chars:
        logger.warning(f"Текст {len(text)} симв. превышает лимит {max_chars}, обрезаем")
        text = text[:max_chars]

    prompt = f"""Переведите следующий научный текст на русский язык.
Стиль: академический, точный, с сохранением терминологии.

Текст:
{text}

Перевод:"""
    system = "Вы профессиональный переводчик научных текстов. Отвечайте только переводом, без пояснений."
    return query_llm(prompt, system, config)

def _split_paragraph_to_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения по точке с пробелом или переносу строки."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def chunk_translate(text: str, config: Dict) -> Tuple[Optional[str], Optional[str]]:
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    chunk_size = config.get("chunk_size", 1500)
    overlap = config.get("overlap", 150)
    ctx_tokens = config.get("ctx_size", 2048)
    safe_chunk = min(chunk_size, int(ctx_tokens * 4 * 0.4), 2500)

    raw_chunks = []
    current = []
    current_len = 0
    for para in paragraphs:
        para_len = len(para)
        if para_len > safe_chunk:
            if current:
                raw_chunks.append('\n\n'.join(current))
                current = []
                current_len = 0
            sentences = _split_paragraph_to_sentences(para)
            sub_chunk = []
            sub_len = 0
            for sent in sentences:
                if sub_len + len(sent) > safe_chunk and sub_chunk:
                    raw_chunks.append('\n\n'.join(sub_chunk))
                    sub_chunk = []
                    sub_len = 0
                sub_chunk.append(sent)
                sub_len += len(sent) + 1
            if sub_chunk:
                raw_chunks.append('\n\n'.join(sub_chunk))
        elif current_len + para_len > safe_chunk:
            raw_chunks.append('\n\n'.join(current))
            current = [para]
            current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2
    if current:
        raw_chunks.append('\n\n'.join(current))

    chunks = []
    for i, chunk in enumerate(raw_chunks):
        if i > 0 and overlap > 0:
            prev = raw_chunks[i - 1]
            tail = prev[-overlap:]
            chunks.append(tail + "\n\n" + chunk)
        else:
            chunks.append(chunk)

    summaries = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Обработка чанка {i+1}/{len(chunks)} ({len(chunk)} симв.)")
        if len(chunk) > safe_chunk:
            chunk = chunk[:safe_chunk]
        prompt = f"""Сделайте краткий научный обзор этого фрагмента на русском.
Выделите: цель, методы, ключевые результаты, выводы (5-10 предложений).

Фрагмент:
{chunk}"""
        system = "Вы научный рецензент. Пишите кратко и по существу."
        summary, err = query_llm(prompt, system, config)
        if err:
            logger.warning(f"Ошибка в чанке {i+1}: {err}")
            continue
        summaries.append(summary)
        time.sleep(1)

    if not summaries:
        return None, "Не удалось обработать ни одного чанка"

    combined = '\n\n'.join(summaries)
    max_chars = int(ctx_tokens * 4 * 0.7)
    if len(combined) > max_chars:
        combined = combined[:max_chars]

    prompt = f"""На основе следующих кратких обзоров создайте единый связный перевод-обзор статьи на русском языке.
Сохраните логическую структуру: цель → методы → результаты → выводы.
Максимум 500 слов.

Обзоры:
{combined}

Итоговый перевод-обзор:"""
    system = "Вы научный редактор. Создайте связный краткий перевод."
    return query_llm(prompt, system, config)

def translate_text(text: str, config: Dict, brief: bool = False) -> Tuple[Optional[str], Optional[str], bool]:
    if not text:
        return None, "Пустой текст", False

    max_chars = int(config.get("ctx_size", 2048) * 4 * 0.7)
    if brief or len(text) > max_chars:
        logger.info(f"Текст {len(text)} симв. > {max_chars} – переключение на краткий обзор")
        translated, err = chunk_translate(text, config)
        return translated, err, False
    else:
        logger.info("Полный перевод текста")
        translated, err = simple_translate(text, config)
        return translated, err, True

# ========== АНАЛИЗ ==========
def analyze_text(text: str, config: Dict) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    if not text:
        return None, "Пустой текст"
    text_for_analysis = text[:2000]
    prompt = f"""Проанализируй научный текст и верни ТОЛЬКО JSON (без пояснений):
{{"main_topic": "основная тема", "methods": "использованные методы", "key_findings": "основные результаты", "conclusion": "выводы"}}

Текст:
{text_for_analysis}"""
    system = "Ты научный аналитик. Отвечай только JSON."
    response, err = query_llm(prompt, system, config)
    if err:
        return None, err
    try:
        response = re.sub(r'^```json\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return None, "JSON не найден"
        data = json.loads(match.group())
        return {k: clean_text(v) for k, v in data.items()}, None
    except Exception as e:
        return None, f"Ошибка парсинга JSON: {e}"

# ========== СОХРАНЕНИЕ В HTML ==========
def save_to_html(original_text: str, translated: str, analysis: Optional[Dict],
                 title: Optional[str], references: Optional[str],
                 output_path: str, config: Dict, is_full: bool) -> bool:
    try:
        html_content = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <title>Перевод статьи</title>
    <style>
        body {{ font-family: 'Times New Roman', serif; margin: 40px; line-height: 1.6; }}
        h1 {{ text-align: center; }}
        .meta {{ color: #555; font-size: 0.9em; margin-bottom: 20px; }}
        .block {{ margin: 20px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #2c3e50; }}
        .translation {{ white-space: pre-wrap; }}
        .analysis {{ background: #eef; }}
        .references {{ background: #fef; }}
        hr {{ border: 1px solid #ccc; }}
        .footer {{ font-size: 0.8em; color: #777; margin-top: 30px; }}
    </style>
</head>
<body>
    <h1>Научный перевод статьи</h1>
    <div class="meta">
        <p><strong>Модель:</strong> {config['model']} (API: {config['api_base']})</p>
        <p><strong>Дата:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Режим:</strong> {'Полный перевод' if is_full else 'Краткий обзор (суммаризация)'}</p>
        <p><strong>Контекст модели:</strong> {config.get('ctx_size', 'неизвестно')} токенов</p>
    </div>
"""
        if is_full and title:
            html_content += f"""
    <h2>Оригинальное название</h2>
    <div class="block"><em>{title}</em></div>
"""
        if is_full and references:
            html_content += f"""
    <h2>Список литературы (оригинал)</h2>
    <div class="block references"><pre>{references}</pre></div>
"""
        html_content += f"""
    <h2>Перевод</h2>
    <div class="block translation">{translated}</div>
"""
        if analysis:
            html_content += f"""
    <h2>Анализ (LLM)</h2>
    <div class="block analysis">
        <ul>
"""
            for key, value in analysis.items():
                if value:
                    html_content += f"            <li><strong>{key.replace('_', ' ').capitalize()}:</strong> {value}</li>\n"
            html_content += """        </ul>
    </div>
"""
        html_content += f"""
    <hr>
    <div class="footer">
        <p>Создано автоматически. Оригинальный текст (очищенный) не включён в вывод.</p>
    </div>
</body>
</html>
"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return True
    except Exception as e:
        logger.exception(f"Ошибка сохранения HTML: {e}")
        return False

# ========== ОБРАБОТКА ТОЛЬКО АБСТРАКТА ==========
def process_abstract_only(pdf_path: str, config: Dict) -> bool:
    filename = os.path.basename(pdf_path)
    print(f"\n📄 {filename} – извлечение абстракта")

    text = extract_text_from_pdf(pdf_path, 20000)
    if not text:
        print("❌ Не удалось извлечь текст")
        return False

    abstract = extract_abstract(text)
    if not abstract:
        print("⚠️ Абстракт не найден, показываю первые 500 символов")
        abstract = text[:500]

    print("\n" + "="*60)
    print("📌 ОРИГИНАЛЬНЫЙ АБСТРАКТ:")
    print("="*60)
    print(abstract)
    print("\n" + "="*60)

    print("🔄 Перевод абстракта...")
    translated, err = simple_translate(abstract, config)
    if err:
        print(f"❌ Ошибка перевода: {err}")
        return False

    print("📌 ПЕРЕВОД АБСТРАКТА:")
    print("="*60)
    print(translated)
    print("="*60 + "\n")
    return True

# ========== ОБРАБОТКА ОДНОГО PDF ==========
def process_pdf(pdf_path: str, output_folder: str, config: Dict, brief: bool, no_analysis: bool) -> bool:
    filename = os.path.basename(pdf_path)
    logger.info(f"Обработка: {filename}")
    print(f"\n📄 {filename}")

    text_limit = 60000 if not brief else 40000
    text = extract_text_from_pdf(pdf_path, text_limit)
    if not text:
        print("❌ Не удалось извлечь текст")
        return False

    print(f"✓ Извлечено символов: {len(text)}")

    title, references = extract_title_and_references(text)

    print("🔄 Перевод...")
    translated, err, is_full = translate_text(text, config, brief)
    if err:
        print(f"⚠️ Ошибка перевода: {err}")
        translated = None
        is_full = False
    else:
        mode_str = "полный" if is_full else "краткий (обзор)"
        print(f"✓ Перевод выполнен ({len(translated)} симв., режим: {mode_str})")

    analysis = None
    if not no_analysis and translated:
        print("🔍 Анализ...")
        analysis, err = analyze_text(translated, config)
        if err:
            print(f"⚠️ Ошибка анализа: {err}")
        else:
            print("✓ Анализ выполнен")

    base_name = os.path.splitext(filename)[0]
    suffix = "_brief" if (brief or not is_full) else "_full"
    output_file = f"{base_name}{suffix}.html"
    output_path = os.path.join(output_folder, output_file)

    if save_to_html(text, translated, analysis, title if is_full else None,
                    references if is_full else None, output_path, config, is_full):
        print(f"✅ Сохранено: {output_file}")
        return True
    else:
        print("❌ Ошибка сохранения")
        return False

# ========== ТОЧКА ВХОДА ==========
def main():
    parser = argparse.ArgumentParser(
        description="Перевод и анализ научных статей через OpenAI-совместимый LLM сервер"
    )
    parser.add_argument("input_folder", help="Папка с PDF-файлами")
    parser.add_argument("output_folder", help="Папка для сохранения HTML")
    parser.add_argument("--api_base", default=DEFAULT_CONFIG["api_base"],
                        help=f"Базовый URL API (по умолчанию {DEFAULT_CONFIG['api_base']})")
    parser.add_argument("--model", default=None,
                        help="Имя модели в сервере. Если не указано, будет выбрана первая работающая.")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"],
                        help="Температура (0.0-1.0)")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_CONFIG["max_tokens"],
                        help="Максимальное число токенов ответа")
    parser.add_argument("--ctx_size", type=int, default=None,
                        help="Размер контекста модели (токенов). Если не указан, будет получен автоматически")
    parser.add_argument("--brief", action="store_true",
                        help="Принудительно использовать краткий обзор (суммаризация)")
    parser.add_argument("--abstract", action="store_true",
                        help="Только извлечь и перевести абстракт, вывести в консоль (без сохранения)")
    parser.add_argument("--no_analysis", action="store_true",
                        help="Отключить анализ текста")
    args = parser.parse_args()

    # Обновляем конфиг
    config = DEFAULT_CONFIG.copy()
    config["api_base"] = args.api_base
    config["temperature"] = args.temperature
    config["max_tokens"] = args.max_tokens

    # Проверка папок
    valid, err = validate_folders(args.input_folder, args.output_folder)
    if not valid:
        print(f"❌ {err}")
        sys.exit(1)

    # Проверка/автозапуск сервера
    if not ensure_server_running(config["api_base"]):
        print(f"❌ Не удалось запустить сервер llama")
        sys.exit(1)

    # Получаем список доступных моделей
    print("🔍 Получение списка доступных моделей...")
    model_names, models_info = get_available_models(config["api_base"])
    if not model_names:
        print("⚠️ Не удалось получить список моделей, будем использовать указанную или модель по умолчанию")
        if args.model:
            config["model"] = args.model
        else:
            config["model"] = DEFAULT_TRANSLATE_MODEL
    else:
        print(f"✓ Доступные модели: {', '.join(model_names)}")
        if args.model and args.model in model_names:
            config["model"] = args.model
            print(f"✓ Используем указанную модель: {config['model']}")
        elif args.model and args.model not in model_names:
            print(f"⚠️ Модель '{args.model}' не найдена в списке, ищем работающую...")
            found = False
            for m in model_names:
                print(f"   Проверяем модель: {m}...")
                if test_model(config["api_base"], m):
                    config["model"] = m
                    found = True
                    print(f"✓ Выбрана работающая модель: {config['model']}")
                    break
            if not found:
                print("❌ Ни одна из доступных моделей не отвечает. Используем модель по умолчанию.")
                config["model"] = select_default_model(config["api_base"], models_info)
        else:
            config["model"] = select_default_model(config["api_base"], models_info)
            print(f"✓ Выбрана модель по умолчанию: {config['model']}")

    # Определяем ctx_size
    if args.ctx_size is not None:
        config["ctx_size"] = args.ctx_size
        print(f"✓ ctx_size задан явно: {config['ctx_size']}")
    else:
        if config["model"] in models_info and "context_length" in models_info[config["model"]]:
            config["ctx_size"] = models_info[config["model"]]["context_length"]
            print(f"✓ ctx_size определён автоматически: {config['ctx_size']} токенов")
        else:
            config["ctx_size"] = DEFAULT_CONFIG["ctx_size"]
            print(f"⚠️ Не удалось определить ctx_size для модели {config['model']}, используется значение по умолчанию: {config['ctx_size']}")

    # Сбор PDF
    pdf_files = [f for f in os.listdir(args.input_folder) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("❌ В папке нет PDF-файлов")
        sys.exit(1)

    print(f"\n📁 Найдено PDF: {len(pdf_files)}")
    print(f"🤖 Модель: {config['model']} (API: {config['api_base']})")
    print(f"📊 Контекст: {config['ctx_size']} токенов")
    print(f"📌 Режим: {'только абстракт' if args.abstract else ('краткий обзор' if args.brief else 'авто (полный/краткий)')}")
    print(f"🔍 Анализ: {'выключен' if args.no_analysis else 'включён'}\n")

    if args.abstract:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(args.input_folder, pdf_file)
            process_abstract_only(pdf_path, config)
        return

    success = 0
    for i, pdf_file in enumerate(pdf_files, 1):
        pdf_path = os.path.join(args.input_folder, pdf_file)
        if process_pdf(pdf_path, args.output_folder, config, args.brief, args.no_analysis):
            success += 1

    print(f"\n{'='*60}")
    print(f"🎉 Готово! Успешно: {success}/{len(pdf_files)}")
    print(f"📁 Результаты в: {os.path.abspath(args.output_folder)}")

if __name__ == "__main__":
    main()
