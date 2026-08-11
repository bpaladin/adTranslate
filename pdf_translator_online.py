#!/usr/bin/env python3
"""
PDF Translator
- Параллельный перевод чанков (все чанки отправляются в пул потоков)
- Retry снижен с 12 до 5, backoff ограничен 10 секундами
- Оптимизирована классификация (block_type исключает повторные regex)
- Потоковая запись HTML по мере завершения страниц
"""

import sys
import re
import os
import time
import html
import random
import base64
import logging
import warnings
import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Match
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import fitz
import requests
from requests.adapters import HTTPAdapter

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# =========================================================
# НАСТРОЙКА ЛОГГИРОВАНИЯ
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =========================================================
# ЗАГРУЗКА .env / keys.env
# =========================================================
def _parse_env_file(path: Path):
    try:
        if not path.exists():
            return
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = value
    except Exception as e:
        logger.warning(f"Ошибка загрузки {path.name}: {e}")


def load_env_file():
    script_dir = Path(__file__).parent.absolute()
    for name in ('keys.env', '.env'):
        _parse_env_file(script_dir / name)


load_env_file()

# =========================================================
# ПОДАВЛЕНИЕ ПРЕДУПРЕЖДЕНИЙ
# =========================================================
warnings.filterwarnings("ignore")

# =========================================================
# КОНФИГУРАЦИЯ
# =========================================================
SCRIPT_VERSION = "1.0"

MAX_CHUNK_SIZE = 1500
DEFAULT_MAX_RETRIES = 3
RETRY_BASE_DELAY = 3.0
MAX_BACKOFF_DELAY = 30.0
REQUEST_TIMEOUT = 120
DEFAULT_TASK_TIMEOUT = 600

MAX_WORKERS = min(8, (os.cpu_count() or 1) * 2)
CACHE_MAX_SIZE = 50000

# API ключи
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemma-4-31b-it:free")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

VERBOSE = False

# =========================================================
# КОМПИЛИРОВАННЫЕ РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ
# =========================================================
RE_METADATA = [
    re.compile(r'^(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright)'),
    re.compile(r'^\d+\s*\(?\d{4}\)?'),
    re.compile(r'^pp?\.\s*\d+'),
    re.compile(r'^Vol\.\s*\d+'),
    re.compile(r'^No\.\s*\d+'),
    re.compile(r'^doi:\s*10\.\d+'),
]
RE_JOURNAL = [
    re.compile(r'Journal\s+of\s+[A-Z]'),
    re.compile(r'[A-Z][a-z]+\s+Journal'),
    re.compile(r'Proceedings\s+of\s+the'),
    re.compile(r'IEEE\s+Transactions'),
    re.compile(r'Nature|Science|Cell|The\s+Lancet'),
    re.compile(r'International\s+Journal\s+of'),
    re.compile(r'European\s+Journal\s+of'),
    re.compile(r'American\s+Journal\s+of'),
]
RE_PROPER_NAMES = [
    re.compile(r'[А-Я][а-я]+\s+[А-Я]\.[А-Я]\.'),
    re.compile(r'[А-Я][а-я]+\s+[А-Я][а-я]+\s+[А-Я][а-я]+'),
    re.compile(r'[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z]\.'),
    re.compile(r'[A-Z][a-z]+\s+[A-Z][a-z]+\s+[A-Z][a-z]+'),
    re.compile(r'(?:Prof|Dr|Mr|Mrs|Ms|Miss|PhD|MD)\.\s+[A-Z][a-z]+'),
]
RE_REFERENCE = [
    re.compile(r'^\s*\[\d+\]'),
    re.compile(r'^\s*\d+\.\s+[A-ZА-Я]'),
    re.compile(r'^\s*\(?\d{4}\)?\.?\s+[A-ZА-Я][a-zа-я]+'),
    re.compile(r'(?:Journal|Conference|Proceedings|IEEE|Springer|Elsevier|Wiley|Oxford|Cambridge)'),
    re.compile(r'(?:Vol\.|Volume|Issue|No\.|Number|Pages|pp\.|Pg\.)\s*\d+'),
    re.compile(r'^(References|Bibliography|Литература|Список литературы)$'),
]
RE_HEADING = [
    re.compile(r'^[A-Z][A-Z\s]{3,}$'),
    re.compile(r'^\d+(?:\.\d+)*\s+[A-ZА-Я]'),
    re.compile(r'^(?:CHAPTER|SECTION|APPENDIX|ABSTRACT|INTRODUCTION|METHODS|RESULTS|DISCUSSION|CONCLUSION)'),
]
RE_TABLE = [
    re.compile(r'^\s*\+[-+]+\+\s*$'),
    re.compile(r'^\s*\|.+\|\s*$'),
    re.compile(r'\d+\s+\d+\s+\d+\s+\d+'),
]
RE_URL = re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+', re.IGNORECASE)
RE_DOI = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.IGNORECASE)
RE_WHITESPACE = re.compile(r'[ \t]+')
RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')

# =========================================================
# ПОТОКОБЕЗОПАСНЫЕ ГЛОБАЛЬНЫЕ СТРУКТУРЫ
# =========================================================
translation_cache = OrderedDict()
cache_lock = threading.Lock()
translation_stats = {"success": 0, "failed": 0, "cached": 0, "skipped": 0, "retries": 0}
stats_lock = threading.Lock()

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def get_cache_key(text: str, target_lang: str) -> tuple:
    return (text, target_lang)

def cache_put(key: tuple, value: str):
    with cache_lock:
        if key in translation_cache:
            translation_cache.move_to_end(key)
        else:
            translation_cache[key] = value
        if len(translation_cache) > CACHE_MAX_SIZE:
            translation_cache.popitem(last=False)

def cache_get(key: tuple) -> Optional[str]:
    with cache_lock:
        if key in translation_cache:
            translation_cache.move_to_end(key)
            return translation_cache[key]
    return None

# =========================================================
# КЛАССЫ ПЕРЕВОДЧИКОВ
# =========================================================
def _make_session() -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=MAX_WORKERS, pool_maxsize=MAX_WORKERS * 2)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s


class HttpTranslator:
    def __init__(self, name: str, target_lang: str, url: str, method: str = "get",
                 build_params=None, build_headers=None, parse_response=None):
        self.name = name
        self.target_lang = target_lang
        self.url = url
        self.method = method
        self._build_params = build_params or (lambda text, lang: {})
        self._build_headers = build_headers or (lambda: {})
        self._parse = parse_response or (lambda data: None)
        self.session = _make_session()
        if method == "get":
            self.session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"})

    def translate(self, text: str) -> Optional[str]:
        params = self._build_params(text, self.target_lang)
        headers = self._build_headers()
        if self.method == "post":
            resp = self.session.post(self.url, headers=headers, json=params, timeout=REQUEST_TIMEOUT)
        else:
            resp = self.session.get(self.url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return self._parse(resp.json())


def _google_params(text: str, lang: str) -> dict:
    return {"client": "gtx", "sl": "auto", "tl": lang, "dt": "t", "ie": "UTF-8", "oe": "UTF-8", "q": text}


def _google_parse(data: list) -> Optional[str]:
    parts = [p[0] for p in data[0] if p[0]]
    return " ".join(parts).strip() if parts else None


_or_last_request = 0.0
_or_lock = threading.Lock()

OPENROUTER_FREE_MODELS_FALLBACK = [
    "openrouter/free",
    "google/gemma-4-31b-it:free",
    "liquid/lfm-2.5-1.2b-thinking:free",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "cohere/north-mini-code:free"
]


class OpenRouterRotator:
    def __init__(self, target_lang: str):
        self.target_lang = target_lang
        self.name = "OpenRouter"
        self._client = None
        self._models: list[str] = []
        self._current_idx = 0
        self._exhausted_models: dict[str, float] = {}
        self._model_cooldown_sec = 60
        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL,
                    default_headers={"X-Title": "pdf-translator"},
                    timeout=REQUEST_TIMEOUT,
                    max_retries=0,
                )
                self._discover_free_models()
            except Exception:
                pass

    def _discover_free_models(self):
        if not self._client:
            return
        try:
            resp = self._client.models.list()
            self._models = sorted(
                [m.id for m in resp.data if "free" in m.id.lower()],
                key=lambda x: ("openrouter/" in x, x),
                reverse=True,
            )
            if not self._models:
                self._models = list(OPENROUTER_FREE_MODELS_FALLBACK[:3])
            logger.info(f"OpenRouter: обнаружено {len(self._models)} free-моделей: {', '.join(self._models[:8])}")
        except Exception as e:
            logger.warning(f"OpenRouter: не удалось получить список моделей: {e}, используем статический пул")
            self._models = list(OPENROUTER_FREE_MODELS_FALLBACK[:3])

    def _is_available(self, model: str) -> bool:
        if model not in self._exhausted_models:
            return True
        if time.time() - self._exhausted_models[model] > self._model_cooldown_sec:
            del self._exhausted_models[model]
            logger.info(f"OpenRouter: cooldown истёк для {model}, возвращаем в пул")
            return True
        return False

    def _next_model(self) -> Optional[str]:
        for _ in range(len(self._models)):
            model = self._models[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self._models)
            if self._is_available(model):
                return model
        return None

    def _rate_limit_wait(self):
        global _or_last_request
        with _or_lock:
            elapsed = time.time() - _or_last_request
            if elapsed < 3.0:
                time.sleep(3.0 - elapsed)
            _or_last_request = time.time()

    def translate(self, text: str) -> Optional[str]:
        if not self._client:
            return None
        attempts = 0
        while attempts < len(self._models):
            model = self._next_model()
            if not model:
                break
            self._rate_limit_wait()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": f"Translate to {self.target_lang}. Return only translation, no explanations."},
                        {"role": "user", "content": text}
                    ],
                    temperature=0.3,
                )
                content = resp.choices[0].message.content
                if content:
                    return content.strip()
                return None
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or 'rate' in err_str or 'limit' in err_str:
                    logger.warning(f"OpenRouter: rate limit на {model}, cooldown {self._model_cooldown_sec}с")
                    self._exhausted_models[model] = time.time()
                    time.sleep(5)
                    attempts += 1
                    continue
                logger.warning(f"OpenRouter: ошибка на {model}: {e}")
                attempts += 1
                continue
        return None


class LlamaCppTranslator:
    def __init__(self, target_lang: str, model: str = "gemma4",
                 api_base: str = "http://localhost:8080/v1"):
        self.target_lang = target_lang
        self.name = "LlamaCpp"
        self.model = model
        self.api_base = api_base.rstrip("/")
        self._server_running = False
        self._ensure_server()

    def _check_server(self) -> bool:
        try:
            resp = requests.get(f"{self.api_base}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                self._server_running = True
                if self.model not in models and models:
                    self.model = models[0]
                    logger.info(f"LlamaCpp: модель не найдена, выбрана: {self.model}")
                return True
        except Exception:
            pass
        return False

    def _ensure_server(self):
        if self._check_server():
            logger.info(f"LlamaCpp: сервер доступен, модель={self.model}")
            return
        logger.info("LlamaCpp: сервер не запущен, запускаем start_llama.sh ...")
        try:
            import subprocess
            subprocess.Popen(
                ["/home/ad/bin/start_llama.sh"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as e:
            logger.error(f"LlamaCpp: не удалось запустить сервер: {e}")
            return
        for i in range(15):
            time.sleep(2)
            if self._check_server():
                logger.info(f"LlamaCpp: сервер запущен ({i*2+2}с), модель={self.model}")
                return
        logger.error("LlamaCpp: сервер не запустился за 30 секунд")

    def translate(self, text: str) -> Optional[str]:
        if not self._server_running:
            self._ensure_server()
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": f"Translate to {self.target_lang}. Return only translation, no explanations."},
                        {"role": "user", "content": text}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 4096,
                    "stream": False,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content:
                    return content.strip()
            else:
                logger.warning(f"LlamaCpp: HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.ConnectionError:
            logger.warning("LlamaCpp: соединение потеряно, пытаемся перезапустить")
            self._server_running = False
        except Exception as e:
            logger.warning(f"LlamaCpp: ошибка: {e}")
        return None


def _make_google(target_lang: str):
    return HttpTranslator("Google", target_lang,
        "https://translate.googleapis.com/translate_a/single",
        build_params=_google_params, parse_response=_google_parse)


def _make_or_fallback(target_lang: str):
    if OPENAI_API_KEY:
        t = OpenRouterRotator(target_lang)
        if t._client:
            return t
    return None


def create_translator(name: str, target_lang: str, **kwargs):
    if name == "llama":
        model = kwargs.get("llama_model", "gemma4")
        url = kwargs.get("llama_url", "http://localhost:8080/v1")
        return LlamaCppTranslator(target_lang, model=model, api_base=url), None
    elif name == "openrouter":
        primary = OpenRouterRotator(target_lang)
        fallback = _make_google(target_lang)
        return primary, fallback
    else:
        primary = _make_google(target_lang)
        fallback = _make_or_fallback(target_lang)
        return primary, fallback

# =========================================================
# ТЕКСТОВЫЕ ФУНКЦИИ (оптимизированные)
# =========================================================
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = RE_HYPHEN_BREAK.sub(r'\1\2', text)
    text = RE_WHITESPACE.sub(' ', text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def is_metadata(text: str) -> bool:
    text = text.strip()
    for pat in RE_METADATA:
        if pat.search(text):
            return True
    for pat in RE_JOURNAL:
        if pat.search(text):
            return True
    if re.search(r'[A-Za-z\s]+,\s+\d+\(\d+\):\s+\d+-\d+', text):
        return True
    return False

def is_proper_name(text: str) -> bool:
    for pat in RE_PROPER_NAMES:
        if pat.search(text):
            return True
    return False

def is_reference(text: str) -> bool:
    for pat in RE_REFERENCE:
        if pat.search(text):
            return True
    if re.search(r'[A-Z][a-z]+,\s+[A-Z]\.?\s+\(?(?:19|20)\d{2}\)?\.', text):
        return True
    return False

def is_true_table(text: str) -> bool:
    if len(text) < 50:
        return False
    for pat in RE_TABLE:
        if pat.search(text):
            return True
    lines = text.split('\n')
    if len(lines) >= 3:
        numbers_per_line = []
        for line in lines[:5]:
            numbers = re.findall(r'\b\d+(?:[.,]\d+)?\b', line)
            if len(numbers) >= 3:
                numbers_per_line.append(len(numbers))
        if len(numbers_per_line) >= 2 and all(n >= 3 for n in numbers_per_line):
            return True
    return False

# Change 4: блоки heading/paragraph уже классифицированы при извлечении,
# пропускаем повторные проверки is_metadata / is_reference
def should_skip_text(text: str, block_type: Optional[str] = None) -> bool:
    text = text.strip()
    if not text or len(text) < 2:
        return True
    if re.match(r'^\s*\d+\s*$', text) or re.match(r'^\s*Page\s+\d+\s*$', text, re.IGNORECASE):
        return True
    if '@' in text or 'http' in text or 'www.' in text:
        return True
    if '10.' in text and '/' in text:
        return True
    if 'ISBN' in text or 'ISSN' in text:
        return True
    # Эти проверки уже выполнены на этапе извлечения PDF
    if block_type not in ('heading', 'paragraph'):
        if is_metadata(text):
            return True
        if is_reference(text):
            return True
    if is_proper_name(text):
        return True
    return False

def protect_special_elements(text: str) -> Tuple[str, Dict]:
    placeholders = {}
    counter = 0

    def url_repl(match: Match) -> str:
        nonlocal counter
        placeholder = f"__URL_{counter}__"
        placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    def doi_repl(match: Match) -> str:
        nonlocal counter
        placeholder = f"__DOI_{counter}__"
        placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    text = RE_URL.sub(url_repl, text)
    text = RE_DOI.sub(doi_repl, text)
    return text, placeholders

def restore_protected_elements(text: str, placeholders: Dict) -> str:
    if not placeholders:
        return text
    sorted_placeholders = sorted(placeholders.items(), key=lambda x: len(x[0]), reverse=True)
    pattern = re.compile('|'.join(re.escape(k) for k, _ in sorted_placeholders))

    def repl(match: Match) -> str:
        return placeholders[match.group(0)]

    return pattern.sub(repl, text)

def _split_by_sentences(text: str, max_len: int) -> List[str]:
    chunks = []
    cur = ""
    for token in re.split(r'(?<=[.!?;:])\s+(?=[A-ZА-Яa-zа-я\(])', text):
        if len(token) <= max_len and len(cur) + len(token) + 1 <= max_len:
            cur += (" " + token) if cur else token
        else:
            if cur:
                chunks.append(cur)
            if len(token) > max_len:
                wcur = ""
                for w in token.split():
                    if len(w) > max_len:
                        if wcur:
                            chunks.append(wcur); wcur = ""
                        chunks.append(w)
                    elif wcur and len(wcur) + len(w) + 1 > max_len:
                        chunks.append(wcur); wcur = w
                    else:
                        wcur += (" " + w) if wcur else w
                cur = wcur
            else:
                cur = token
    if cur:
        chunks.append(cur)
    return chunks


def split_text_into_chunks(text: str, max_len: int = MAX_CHUNK_SIZE) -> List[str]:
    if len(text) <= max_len:
        return [text]
    text = text.replace('\r\n', '\n')
    chunks = []
    paragraphs = re.split(r'\n{2,}', text)
    has_paragraphs = len(paragraphs) > 1
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_len:
            if chunks and len(chunks[-1]) + len(para) + 2 <= max_len:
                chunks[-1] += "\n\n" + para
            else:
                chunks.append(para)
        else:
            chunks.extend(_split_by_sentences(para, max_len))
    if not has_paragraphs and not chunks:
        chunks = _split_by_sentences(text, max_len)
    return chunks if chunks else [text[:max_len]]

def is_heading_text(text: str) -> bool:
    for pat in RE_HEADING:
        if pat.search(text):
            return True
    return False

# =========================================================
# КЛАСС PDFBlock
# =========================================================
@dataclass(slots=True)
class PDFBlock:
    text: str
    block_type: str
    page_num: int
    font_size: float = 12.0
    translated_text: Optional[str] = None

# =========================================================
# ИЗВЛЕЧЕНИЕ ИЗ PDF
# =========================================================
def extract_pdf_blocks(pdf_path: str) -> Tuple[List[PDFBlock], Dict[int, List[str]]]:
    with fitz.open(pdf_path) as doc:
        return _extract_pdf_blocks_impl(doc)


def _extract_pdf_blocks_impl(doc: fitz.Document) -> Tuple[List[PDFBlock], Dict[int, List[str]]]:
    raw_blocks: List[dict] = []
    images_per_page: Dict[int, List[str]] = {}
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        page_images: List[str] = []

        if TQDM_AVAILABLE:
            print(f"\r📄 Извлечение страницы {page_num + 1}/{total_pages}", end="", flush=True)

        # -- изображения сразу в base64 --
        try:
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list[:5]):
                try:
                    xref = img[0]
                    pix = fitz.Pixmap(page.parent, xref)
                    if pix.n - pix.alpha < 4:
                        if pix.alpha:
                            pix = fitz.Pixmap(pix, 0)
                    img_b64 = base64.b64encode(pix.tobytes("png")).decode()
                    pix = None
                    img_html = f'''<figure class="figure">
    <img src="data:image/png;base64,{img_b64}" alt="Figure {img_idx + 1}" class="figure-image" />
    <figcaption class="figure-caption">Рис. {img_idx + 1} (стр. {page_num + 1})</figcaption>
</figure>'''
                    page_images.append(img_html)
                except Exception as e:
                    logger.warning(f"Ошибка извлечения изображения: {e}")
        except Exception as e:
            logger.warning(f"Ошибка обработки изображений на странице {page_num+1}: {e}")

        if page_images:
            images_per_page[page_num + 1] = page_images

        # -- текст с определением отступа и bbox --
        text_blocks = page.get_text("dict").get("blocks", [])
        for block in text_blocks:
            if "lines" not in block:
                continue

            line_texts: List[str] = []
            line_indents: List[bool] = []
            line_y_positions: List[float] = []
            line_font_sizes: List[float] = []
            all_font_sizes: List[float] = []

            for line in block["lines"]:
                line_spans = []
                line_has_indent = False
                line_y = line.get("bbox", (0, 0, 0, 0))[1]
                line_fs = 0
                for span in line["spans"]:
                    raw_text = span.get("text", "")
                    if raw_text:
                        if not line_spans and raw_text[0] in (' ', '\t'):
                            line_has_indent = True
                        line_spans.append(raw_text.strip())
                        fs = span.get("size", 0)
                        all_font_sizes.append(fs)
                        line_fs = max(line_fs, fs)
                if line_spans:
                    line_texts.append(" ".join(line_spans))
                    line_indents.append(line_has_indent)
                    line_y_positions.append(line_y)
                    line_font_sizes.append(line_fs)

            if not line_texts:
                continue

            paragraphs: List[str] = []
            current_para = line_texts[0]

            for i in range(1, len(line_texts)):
                is_indented = line_indents[i]
                prev_y = line_y_positions[i - 1]
                cur_y = line_y_positions[i]
                gap = cur_y - prev_y
                avg_fs = (line_font_sizes[i - 1] + line_font_sizes[i]) / 2 if line_font_sizes[i - 1] and line_font_sizes[i] else 12
                real_break = is_indented and gap > avg_fs * 0.8
                if real_break:
                    paragraphs.append(current_para)
                    current_para = line_texts[i]
                else:
                    current_para += " " + line_texts[i]
            paragraphs.append(current_para)

            paragraph = "\n\n".join(paragraphs)
            paragraph = normalize_text(paragraph)

            if not paragraph or len(paragraph) < 2:
                continue

            avg_font_size = sum(all_font_sizes) / len(all_font_sizes) if all_font_sizes else 12
            bbox = block.get("bbox", (0, 0, 0, 0))

            raw_blocks.append({
                'text': paragraph,
                'font_size': avg_font_size,
                'page_num': page_num + 1,
                'indented': line_indents[0] if line_indents else False,
                'y0': bbox[1],
                'y1': bbox[3],
            })

    if TQDM_AVAILABLE:
        print()

    # -- слияние абзацев: отступ + вертикальная близость --
    merged_blocks: List[dict] = []
    for rb in raw_blocks:
        prev = merged_blocks[-1] if merged_blocks else None
        same_page = prev and prev['page_num'] == rb['page_num']
        if same_page and rb['indented']:
            prev['text'] += " " + rb['text']
            prev['y1'] = rb['y1']
        elif (same_page
              and not rb['indented']
              and abs(prev['font_size'] - rb['font_size']) < 1
              and rb['y0'] - prev['y1'] < rb['font_size'] * 0.5):
            prev['text'] += " " + rb['text']
            prev['y1'] = rb['y1']
        else:
            merged_blocks.append(dict(rb))

    # -- классификация --
    blocks: List[PDFBlock] = []
    for rb in merged_blocks:
        text = rb['text']
        font_size = rb['font_size']
        page_num = rb['page_num']

        if re.match(r'^(References|Bibliography|Литература|Список литературы)$', text, re.IGNORECASE):
            block_type = 'reference_header'
        elif is_metadata(text):
            block_type = 'metadata'
        elif is_reference(text):
            block_type = 'reference'
        elif is_true_table(text):
            block_type = 'table'
        elif is_heading_text(text) or (font_size > 14 and len(text) < 80):
            block_type = 'heading'
        else:
            block_type = 'paragraph'

        blocks.append(PDFBlock(text, block_type, page_num, font_size))

        if VERBOSE and block_type in ('paragraph', 'heading'):
            logger.debug(f"  [абзац {page_num}] {text[:120]}...")

    return blocks, images_per_page

# =========================================================
# ФУНКЦИИ ПЕРЕВОДА (Change 1: без вложенного ThreadPoolExecutor)
# =========================================================
def _backoff(attempt: int) -> None:
    if attempt == 0:
        return
    delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1), MAX_BACKOFF_DELAY)
    time.sleep(delay)


def translate_chunk_with_retry(chunk: str, translator, max_retries: int = None) -> Optional[str]:
    max_retries = max_retries or DEFAULT_MAX_RETRIES
    if len(chunk) < 3:
        return chunk
    for attempt in range(max_retries):
        _backoff(attempt)
        try:
            translated = translator.translate(chunk)
            if translated and len(translated) >= 2 and translated.strip() != chunk.strip():
                if attempt > 0:
                    with stats_lock:
                        translation_stats["retries"] += 1
                return translated.strip()
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code if e.response is not None else "?"
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] HTTP {code} после {max_retries} попыток: {e}")
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] Таймаут после {max_retries} попыток")
        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] Ошибка соединения после {max_retries} попыток: {e}")
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] Ошибка после {max_retries} попыток: {e}")
    return None


def _translate_sub(chunks: List[str], translator, fallback, max_retries: int, max_len: int, depth: int = 0) -> List[str]:
    MAX_DEPTH = 3
    results = []
    for sub in chunks:
        t = translate_chunk_with_retry(sub, translator, max_retries)
        if not t and fallback:
            t = translate_chunk_with_retry(sub, fallback, max_retries)
        if t:
            results.append(t)
        elif depth < MAX_DEPTH and len(sub) > max_len:
            results.extend(_translate_sub(split_text_into_chunks(sub, max_len), translator, fallback, max_retries, max_len // 2, depth + 1))
        elif depth < MAX_DEPTH and len(sub) > 300:
            results.extend(_translate_sub(split_text_into_chunks(sub, 300), translator, fallback, max_retries, 300, depth + 1))
        else:
            results.append(sub)
    return results


def translate_chunk_with_fallback(chunk: str, translator, fallback=None) -> str:
    t = translate_chunk_with_retry(chunk, translator)
    if t:
        with stats_lock:
            translation_stats["success"] += 1
        return t
    if fallback:
        t = translate_chunk_with_retry(chunk, fallback)
        if t:
            with stats_lock:
                translation_stats["success"] += 1
            return t
    with stats_lock:
        translation_stats["failed"] += 1
    if len(chunk) > MAX_CHUNK_SIZE:
        return " ".join(_translate_sub(split_text_into_chunks(chunk, MAX_CHUNK_SIZE), translator, fallback, DEFAULT_MAX_RETRIES, MAX_CHUNK_SIZE // 2))
    if len(chunk) > 300:
        return " ".join(_translate_sub(split_text_into_chunks(chunk, 300), translator, fallback, DEFAULT_MAX_RETRIES, 300))
    return chunk


# =========================================================
# ПАРАЛЛЕЛЬНЫЙ ПЕРЕВОД АБЗАЦЕВ
# =========================================================
@dataclass
class _BlockState:
    placeholders: dict
    cache_key: tuple
    sub_results: List[str]
    done_count: int = 0
    total_count: int = 1


def translate_blocks_parallel(
    blocks: List[PDFBlock],
    translator,
    fallback=None,
    quiet: bool = False,
    max_workers: int = MAX_WORKERS,
    timeout: int = DEFAULT_TASK_TIMEOUT,
) -> List[PDFBlock]:
    block_states: Dict[int, _BlockState] = {}
    block_tasks: List[Tuple[int, str]] = []

    for block_idx, block in enumerate(blocks):
        bt = block.block_type
        if bt in ('metadata', 'reference', 'reference_header', 'table'):
            block.translated_text = block.text
            with stats_lock:
                translation_stats["skipped"] += 1
            continue

        text = block.text
        if not text or len(text) < 3:
            block.translated_text = text
            with stats_lock:
                translation_stats["skipped"] += 1
            continue

        if should_skip_text(text, bt):
            block.translated_text = text
            with stats_lock:
                translation_stats["skipped"] += 1
            continue

        key = get_cache_key(text, translator.target_lang)
        cached = cache_get(key)
        if cached is not None:
            block.translated_text = cached
            with stats_lock:
                translation_stats["cached"] += 1
            continue

        protected_text, placeholders = protect_special_elements(text)

        if len(protected_text) <= MAX_CHUNK_SIZE:
            state = _BlockState(
                placeholders=placeholders,
                cache_key=key,
                sub_results=[],
                total_count=1,
            )
            block_states[block_idx] = state
            block_tasks.append((block_idx, protected_text))
        else:
            sub_chunks = split_text_into_chunks(protected_text, MAX_CHUNK_SIZE)
            state = _BlockState(
                placeholders=placeholders,
                cache_key=key,
                sub_results=[],
                total_count=len(sub_chunks),
            )
            block_states[block_idx] = state
            for sub in sub_chunks:
                block_tasks.append((block_idx, sub))

    total = len(block_tasks)
    if not quiet:
        logger.info(f"Перевод {total} под-чанков в {max_workers} потоках (таймаут {timeout} сек)")

    if not block_tasks:
        return blocks

    results_lock = threading.Lock()

    def _work(block_idx: int, chunk_text: str):
        translated = translate_chunk_with_fallback(chunk_text, translator, fallback)
        with results_lock:
            state = block_states[block_idx]
            state.sub_results.append(translated if translated else chunk_text)
            state.done_count += 1
            if state.done_count == state.total_count:
                joined = " ".join(state.sub_results)
                result = restore_protected_elements(joined, state.placeholders)
                blocks[block_idx].translated_text = result
                cache_put(state.cache_key, result)
        return block_idx

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_work, *task): task
            for task in block_tasks
        }

        try:
            if TQDM_AVAILABLE and not quiet:
                with tqdm(total=len(futures), desc="Перевод") as pbar:
                    for future in as_completed(futures, timeout=timeout):
                        future.result()
                        pbar.update(1)
            else:
                for future in as_completed(futures, timeout=timeout):
                    future.result()
        except TimeoutError:
            logger.warning(f"Превышен общий таймаут ({timeout} сек). Некоторые абзацы не переведены.")
            for f in futures:
                f.cancel()

    for block_idx, state in block_states.items():
        block = blocks[block_idx]
        if block.translated_text is None:
            joined = " ".join(state.sub_results) if state.sub_results else block.text
            result = restore_protected_elements(joined, state.placeholders)
            block.translated_text = result

    return blocks


# =========================================================
# ГЕНЕРАЦИЯ HTML через HTMLReportBuilder
# =========================================================
BLOCK_CSS = """
<style>
.block-content h2 { color: #60a5fa; border-bottom: 1px solid #334155; padding-bottom: 6px; margin: 20px 0 12px; }
.block-content h3 { color: #93c5fd; margin: 16px 0 10px; }
.block-content h4 { color: #bfdbfe; margin: 12px 0 8px; }
.block-content p { line-height: 1.7; margin: 0 0 12px; text-align: justify; }
.orig { color: #94a3b8; font-size: 0.9em; margin-bottom: 2px; }
.orig::before { content: "📖 "; }
.trans { color: #e2e8f0; }
.trans::before { content: "🌐 "; }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
.ref { color: #64748b; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; margin-bottom: 6px; }
.meta { color: #64748b; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #334155; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid #334155; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: #0f172a; border: 1px solid #334155; padding: 8px 10px; color: #60a5fa; }
.table-wrap td { border: 1px solid #334155; padding: 6px 10px; }
.figure { margin: 20px 0; text-align: center; }
.figure-image { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: #94a3b8; font-size: 0.85em; }
</style>
"""


def _block_to_html(block: PDFBlock, in_ref: bool = False) -> str:
    e = html.escape
    bt = block.block_type
    trans = block.translated_text or block.text
    if bt == 'reference_header':
        return f'<div class="ref-section"><h2>{e(block.text)}</h2></div>\n'
    if in_ref or bt == 'reference':
        return f"<p class='ref'>{e(block.text)}</p>\n"
    if bt == 'metadata':
        return f"<div class='meta'>{e(block.text)}</div>\n"
    if bt == 'table':
        rows = block.text.split('\n')
        if len(rows) >= 2:
            out = ['<div class="table-wrap"><table>\n']
            for idx, row in enumerate(rows[:20]):
                cells = [c.strip() for c in re.split(r'\s{2,}|\t', row) if c.strip()]
                if not cells:
                    continue
                tag = 'th' if idx == 0 else 'td'
                if idx == 0:
                    out.append('<thead><tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in cells)
                    out.append('</tr></thead><tbody>\n')
                else:
                    out.append('<tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in cells)
                    out.append('</tr>\n')
            out.append('</tbody></table>\n</div>\n')
            return ''.join(out)
        return f"<p>{e(block.text)}</p>\n"
    if bt == 'heading':
        orig = block.text
        size = block.font_size
        h = 'h1' if size > 24 else ('h2' if size > 20 else ('h3' if size > 16 else 'h4'))
        if orig != trans:
            return f'<div class="trans-head"><p class="orig">{e(orig)}</p><{h} class="trans">{e(trans)}</{h}></div>\n'
        return f"<{h}>{e(trans)}</{h}>\n"
    return f"<p>{e(trans)}</p>\n"


def _build_html_report(
    blocks: List[PDFBlock],
    images_per_page: Dict[int, List[str]],
    output_path: str,
    input_name: str,
) -> str:
    from collections import OrderedDict
    from html_report_builder import HTMLReportBuilder

    builder = HTMLReportBuilder(title=f"Перевод PDF: {input_name}", version=SCRIPT_VERSION)

    page_blocks: OrderedDict[int, List[PDFBlock]] = OrderedDict()
    for block in blocks:
        page_blocks.setdefault(block.page_num, []).append(block)

    in_reference_section = False

    builder.add_html(BLOCK_CSS, as_section=False)

    for page_num, pblocks in page_blocks.items():
        page_html_parts: List[str] = []

        if page_num in images_per_page:
            for img_html in images_per_page[page_num]:
                page_html_parts.append(img_html)

        for block in pblocks:
            if block.block_type == 'reference_header':
                in_reference_section = True
            page_html_parts.append(_block_to_html(block, in_reference_section))

        page_html = "\n".join(page_html_parts)
        builder.add_html(f"<h3>📄 Страница {page_num}</h3>\n{page_html}", as_section=True)

    builder.save(str(output_path))
    logger.info(f"   ✅ HTML сохранён: {output_path}")
    return str(output_path)


# =========================================================
# ОСНОВНАЯ ФУНКЦИЯ (с потоковой записью HTML)
# =========================================================
def main():
    global MAX_WORKERS

    parser = argparse.ArgumentParser(description=f"PDF Translator v{SCRIPT_VERSION}")
    parser.add_argument("input", help="Входной PDF файл")
    parser.add_argument("output", help="Выходной HTML файл")
    parser.add_argument("-l", "--lang", default="ru", help="Целевой язык (по умолчанию: ru)")
    parser.add_argument("-t", "--translator", default="openrouter",
                        choices=["google", "openrouter", "llama"],
                        help="Переводчик: google, openrouter, llama (по умолчанию: openrouter)")
    parser.add_argument("--llama-model", default="gemma4",
                        help="Модель llama.cpp (по умолчанию: gemma4)")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1",
                        help="URL llama.cpp сервера (по умолчанию: http://localhost:8080/v1)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод ошибок и пропусков")
    parser.add_argument("-q", "--quiet", action="store_true", help="Тихий режим (без прогресс-бара)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="Максимальное число повторных попыток")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS,
                        help="Число параллельных воркеров")
    parser.add_argument("--task-timeout", type=int, default=DEFAULT_TASK_TIMEOUT,
                        help=f"Таймаут на выполнение всех задач (сек), по умолчанию {DEFAULT_TASK_TIMEOUT}")
    args = parser.parse_args()

    MAX_WORKERS = args.workers
    task_timeout = args.task_timeout

    global VERBOSE
    if args.verbose:
        VERBOSE = True
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=" * 60)
    logger.info(f"PDF TRANSLATOR v{SCRIPT_VERSION} - {args.translator.upper()} ({MAX_WORKERS} воркеров, таймаут {task_timeout} сек)")
    logger.info("=" * 60)

    try:
        translator, fallback = create_translator(
            args.translator, args.lang,
            llama_model=args.llama_model,
            llama_url=args.llama_url,
        )
        logger.info(f"   Переводчик: {translator.name}")
        if fallback:
            logger.info(f"   Fallback: {fallback.name}")
    except Exception as e:
        logger.error(f"Ошибка инициализации переводчика: {e}")
        sys.exit(1)

    try:
        output_path = Path(args.output)

        logger.info("📖 [1/3] Извлечение данных из PDF...")
        blocks, images_per_page = extract_pdf_blocks(args.input)
        total_images = sum(len(imgs) for imgs in images_per_page.values())
        logger.info(f"   ✅ Извлечено {len(blocks)} блоков и {total_images} изображений")

        logger.info(f"🌐 [2/3] Перевод на {args.lang}...")

        blocks = translate_blocks_parallel(
            blocks, translator, fallback,
            quiet=args.quiet, max_workers=MAX_WORKERS,
            timeout=task_timeout,
        )
        logger.info("   ✅ Перевод завершён")

        logger.info("🎨 [3/3] Генерация HTML...")
        _build_html_report(blocks, images_per_page, args.output, Path(args.input).name)

        with stats_lock:
            stats = translation_stats.copy()
        logger.info("=" * 60)
        logger.info("📊 СТАТИСТИКА:")
        logger.info(f"   ✓ Переведено: {stats['success']}")
        logger.info(f"   ⚡ Из кэша: {stats['cached']}")
        logger.info(f"   ⊘ Пропущено: {stats['skipped']}")
        logger.info(f"   🔄 Повторов: {stats['retries']}")
        logger.info(f"   ✗ Ошибок: {stats['failed']}")
        logger.info("=" * 60)
        logger.info(f"✅ Готово: {args.output}")

    except KeyboardInterrupt:
        logger.warning("⚠️ Прервано пользователем")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
