#!/usr/bin/env python3
"""
PDF Translator — единый скрипт для перевода и реферирования PDF-документов.
Поддерживает Google Translate, OpenRouter и локальный llama.cpp.
ML-классификация, профилирование, dark-тема HTML.
"""
import os
import sys
import re
import time
import json
import random
import atexit
import signal
import hashlib
import logging
import threading
import subprocess
import argparse
import warnings
import html as html_mod
import base64
import contextlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import fitz
import requests
import markdown as md
from cachetools import LRUCache
from jinja2 import Template
from requests.adapters import HTTPAdapter

try:
    from pymupdf_layout import extract_layout
    HAVE_LAYOUT = True
except ImportError:
    HAVE_LAYOUT = False

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False

# ---- Логирование ----
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# =========================================================
# .env / keys.env загрузка
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

# ---- API ключи ----
OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "google/gemma-4-31b-it:free")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

# ---- Профилирование ----
@contextlib.contextmanager
def stage_timer(name: str, timings: Optional[Dict[str, float]] = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(f"⏱️  {name}: {_format_time(elapsed)}")
        if timings is not None:
            timings[name] = elapsed

# =========================================================
# РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ
# =========================================================
RE_WHITESPACE = re.compile(r'[ \t]+')
RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')

REF_HEADING_RE = re.compile(
    r'^(References?|Bibliography|Библиография|Литература|Список\s+литературы|'
    r'Список\s+использованных\s+источников)\s*$', re.IGNORECASE
)

CAPTION_RE = re.compile(r'Figure|Fig\.|Рис\.|Схема|Table|Таблица', re.I)
LIST_LINE_RE = re.compile(r'^[\s]*([•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s|[a-z]\.\s)', re.MULTILINE)

# ---- Регулярное выражение для ссылок в квадратных скобках ----
REF_PATTERN = re.compile(r'\[([^\]]+)\]')

# =========================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================================================
def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}с"
    elif seconds < 3600:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}м {s:02d}с"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}ч {m:02d}м"

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = RE_HYPHEN_BREAK.sub(r'\1\2', text)
    text = RE_WHITESPACE.sub(' ', text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _count_list_lines(text: str) -> int:
    return len(LIST_LINE_RE.findall(text))

def _chunk_text_by_sentences(text: str, max_chunk_size: int = 6000) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len + 1 > max_chunk_size and current:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len + 1

    if current:
        chunks.append(" ".join(current))

    return chunks if chunks else [text]

def _backoff(attempt: int, base: float = 3.0, max_delay: float = 30.0) -> None:
    if attempt == 0:
        return
    delay = min(base * (2 ** (attempt - 1)) + random.uniform(0, 1), max_delay)
    time.sleep(delay)

# =========================================================
# МОДЕЛИ ДАННЫХ
# =========================================================
@dataclass
class Span:
    text: str
    font: str
    size: float
    flags: int
    color: int
    origin: tuple
    bbox: tuple

@dataclass
class Line:
    spans: List[Span] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    y0: float = 0.0

@dataclass
class Block:
    type: str
    lines: List[Line] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    page_num: int = 0
    image_data: Optional[str] = None
    caption: Optional[str] = None
    table_data: Optional[List[List[str]]] = None
    translation: Optional[str] = None

    @property
    def text(self) -> str:
        return " ".join(span.text for line in self.lines for span in line.spans).strip()

@dataclass
class Page:
    num: int
    blocks: List[Block] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0

# ---- Фабрики ----
def make_span(text: str, font: str = "", size: float = 12, flags: int = 0,
              color: int = 0, origin: tuple = (0, 0), bbox: tuple = (0, 0, 0, 0)) -> Span:
    return Span(text=text, font=font, size=size, flags=flags, color=color, origin=origin, bbox=bbox)

def make_line(spans: Optional[List[Span]] = None, bbox: tuple = (0, 0, 0, 0),
              y0: float = 0.0) -> Line:
    return Line(spans=spans or [], bbox=bbox, y0=y0)

def make_block(type: str, lines: Optional[List[Line]] = None, bbox: tuple = (0, 0, 0, 0),
               page_num: int = 0, **kwargs) -> Block:
    b = Block(type=type, lines=lines or [], bbox=bbox, page_num=page_num)
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b

# =========================================================
# МЕТРИКИ И КЛАССИФИКАЦИЯ (без ML)
# =========================================================
@dataclass
class BlockMetrics:
    text: str
    max_font: float
    all_upper: bool
    has_bold: bool
    is_centered: bool
    total_spans: int
    bbox: tuple

def block_metrics(block: Block, page_width: float = 0.0) -> BlockMetrics:
    text = block.text
    max_font = 0.0
    all_upper = True
    has_bold = False
    total_spans = 0

    for line in block.lines:
        for span in line.spans:
            total_spans += 1
            if span.size > max_font:
                max_font = span.size
            if span.text and not span.text.isupper():
                all_upper = False
            if span.flags & 2**0:
                has_bold = True

    bbox = block.bbox
    center_x = (bbox[0] + bbox[2]) / 2
    is_centered = abs(center_x - page_width / 2) < page_width * 0.1 if page_width else False

    return BlockMetrics(
        text=text, max_font=max_font, all_upper=all_upper,
        has_bold=has_bold, is_centered=is_centered,
        total_spans=total_spans, bbox=bbox
    )

class BlockClassifier:
    LABELS = ["paragraph", "heading", "list", "metadata", "reference", "reference_heading", "empty"]

    def classify(self, block: Block, page_width: float, avg_font_size: float) -> str:
        if not block.lines:
            return "empty"

        m = block_metrics(block, page_width)

        if not m.text or m.total_spans == 0:
            return "empty"

        if REF_HEADING_RE.match(m.text):
            return "reference_heading"

        if re.match(r'^\[\d+\]', m.text):
            return "reference"

        if re.search(r'[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.', m.text):
            return "metadata"

        if re.search(r'(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright)', m.text, re.I):
            if len(m.text) > 100 and not re.search(r'(DOI|ISSN|ISBN)', m.text, re.I):
                return "paragraph"
            return "metadata"

        list_lines = _count_list_lines(m.text)
        if list_lines >= 2 or (list_lines >= 1 and len(m.text.split('\n')) >= 2):
            return "list"

        score = 0
        if m.max_font > avg_font_size * 1.2:
            score += 3
        if m.all_upper:
            score += 2
        if m.has_bold:
            score += 2
        if m.is_centered:
            score += 1
        if len(m.text) < 80:
            score += 1
        if re.match(r'^\d+(\.\d+)*\s+', m.text):
            score += 2

        return "heading" if score >= 5 else "paragraph"

_classifier = BlockClassifier()

# =========================================================
# ИЗВЛЕЧЕНИЕ PDF
# =========================================================
class PDFExtractor:
    def __init__(self, path: str, crop_top: float = 40.0, crop_bottom: float = 45.0):
        self.doc = fitz.open(path)
        self.use_layout = HAVE_LAYOUT
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def _crop_rect(self, page: fitz.Page) -> fitz.Rect:
        r = page.rect
        top = min(self.crop_top, r.height * 0.4)
        bottom = min(self.crop_bottom, r.height * 0.4)
        return fitz.Rect(r.x0, r.y0 + top, r.x1, r.y1 - bottom)

    def extract(self) -> List[Page]:
        pages = []

        if self.use_layout:
            try:
                layout = extract_layout(self.doc)
                for page_num, page_data in enumerate(layout.pages):
                    top_y = self.crop_top
                    bottom_y = page_data.height - self.crop_bottom
                    page_blocks = []

                    for item in page_data.items:
                        if item['type'] != 'text':
                            continue
                        bbox = item['bbox']
                        if bbox[3] < top_y or bbox[1] > bottom_y:
                            continue

                        block = make_block(type="text", page_num=page_num + 1, bbox=bbox)
                        for line_text in item['text'].split('\n'):
                            if not line_text.strip():
                                continue
                            span = make_span(text=line_text)
                            line = make_line(spans=[span], bbox=bbox, y0=bbox[1])
                            block.lines.append(line)

                        if block.lines:
                            page_blocks.append(block)

                    pages.append(Page(num=page_num + 1, blocks=page_blocks,
                                      width=page_data.width, height=page_data.height))
                return pages
            except Exception as e:
                logger.warning(f"pymupdf_layout error: {e}, falling back to standard")
                self.use_layout = False

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            clip = self._crop_rect(page)
            raw = page.get_text("dict", clip=clip)

            page_blocks = []
            for b in raw.get("blocks", []):
                if b.get("type") == 0:
                    block = make_block(type="text", page_num=page_num + 1, bbox=b.get("bbox", (0, 0, 0, 0)))

                    for line_dict in b.get("lines", []):
                        bbox = line_dict.get("bbox", (0, 0, 0, 0))
                        line = make_line(bbox=bbox, y0=bbox[1])

                        for span_dict in line_dict.get("spans", []):
                            text = span_dict.get("text", "")
                            if not text.strip():
                                continue

                            span = make_span(
                                text=text, font=span_dict.get("font", ""),
                                size=span_dict.get("size", 12),
                                flags=span_dict.get("flags", 0),
                                color=span_dict.get("color", 0),
                                origin=span_dict.get("origin", (0, 0)),
                                bbox=span_dict.get("bbox", (0, 0, 0, 0))
                            )
                            line.spans.append(span)

                        if line.spans:
                            block.lines.append(line)

                    if block.lines:
                        page_blocks.append(block)

            pages.append(Page(num=page_num + 1, blocks=page_blocks,
                              width=raw.get("width", 612), height=raw.get("height", 792)))

        return pages

def extract_tables(page: fitz.Page, page_num: int) -> List[Block]:
    tables = page.find_tables()
    table_blocks = []

    for tab in tables:
        try:
            rows = tab.extract()
            if not rows:
                continue

            cleaned = [[cell if cell else "" for cell in row] for row in rows]
            block = make_block(type="table", page_num=page_num, bbox=tuple(tab.bbox),
                               table_data=cleaned)
            table_blocks.append(block)
        except Exception:
            continue

    return table_blocks

def intersection_ratio(block_bbox, table_bbox):
    x0 = max(block_bbox[0], table_bbox[0])
    y0 = max(block_bbox[1], table_bbox[1])
    x1 = min(block_bbox[2], table_bbox[2])
    y1 = min(block_bbox[3], table_bbox[3])

    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])

    if block_area == 0:
        return 0.0

    return intersection / block_area

def mark_table_blocks(text_blocks: List[Block], table_blocks: List[Block]):
    for block in text_blocks:
        if block.type == "table":
            continue
        for table in table_blocks:
            if intersection_ratio(block.bbox, table.bbox) > 0.30:
                block.type = "table"
                break

def extract_images(page: fitz.Page, page_num: int, text_blocks: List[Block]) -> Tuple[List[Block], List[Block]]:
    image_list = page.get_images(full=True)
    figure_blocks = []
    caption_ids = set()

    for img in image_list:
        xref = img[0]
        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n - pix.alpha < 4:
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            pix = None
        except Exception:
            continue

        rects = page.get_image_rects(xref)
        if not rects:
            continue

        img_bbox = rects[0]
        caption = None

        for idx, block in enumerate(text_blocks):
            if idx in caption_ids:
                continue
            bbox = block.bbox
            if abs(bbox[3] - img_bbox[1]) < 80 or abs(img_bbox[3] - bbox[1]) < 80:
                text = block.text
                if CAPTION_RE.search(text):
                    caption = text
                    caption_ids.add(idx)
                    break

        fig_block = make_block(type="figure", page_num=page_num, bbox=img_bbox,
                               image_data=img_b64, caption=caption)
        figure_blocks.append(fig_block)

    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining

# =========================================================
# RATE LIMITER
# =========================================================
class RateLimiter:
    def __init__(self, max_requests_per_second: float = 5.0):
        self.rate = max_requests_per_second
        self.min_interval = 1.0 / max_requests_per_second
        self.last_time = time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            sleep_time = self.min_interval - (now - self.last_time)
            self.last_time = now + max(sleep_time, 0)
            if sleep_time > 0:
                time.sleep(sleep_time)

def _make_session(max_workers: int = 8) -> requests.Session:
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=max_workers, pool_maxsize=max_workers * 2)
    s.mount('https://', adapter)
    s.mount('http://', adapter)
    return s

# =========================================================
# ПРОМПТЫ И ГЛОССАРИЙ
# =========================================================
SYSTEM_TRANSLATE_PROMPT = (
    "Ты — эксперт-переводчик научных статей. Переведи текст на {lang}. "
    "Сохраняй академический стиль, точность терминов и структуру Markdown. "
    "Не переводи имена собственные, формулы, числовые данные и аббревиатуры. "
    "Верни ТОЛЬКО перевод, без пояснений."
)

GLOSSARY_PROMPT = (
    "Извлеки до 20 ключевых научных терминов из текста и дай их перевод на {lang}. "
    "Верни строго в формате JSON-объекта: {{'термин': 'перевод', ...}}. Только JSON, без пояснений."
)

REFINE_PROMPT = (
    "Проверь перевод научного текста на {lang} на пропуски, галлюцинации и смысловые ошибки. "
    "Если есть ошибки — исправь их, вернув исправленный перевод. Если ошибок нет — верни исходный перевод. "
    "Верни ТОЛЬКО исправленный текст, без пояснений."
)

def extract_glossary(text: str, translator, lang: str,
                     cache: "Optional[TranslationCache]" = None) -> Dict[str, str]:
    if not text or not getattr(translator, 'can_generate', False):
        return {}

    cache_key = f"glossary:{lang}"

    if cache is not None:
        cached = cache.get(text, cache_key)
        if cached:
            try:
                return json.loads(cached)
            except Exception:
                pass

    prompt = GLOSSARY_PROMPT.format(lang=lang) + "\n" + text[:20000]

    try:
        raw = translator.generate(prompt)
    except Exception:
        raw = None

    glossary = {}
    if raw:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                if isinstance(parsed, dict):
                    glossary = {str(k).strip(): str(v).strip() for k, v in parsed.items()}
            except Exception:
                logger.warning("   Глоссарий: не удалось распарсить JSON")

    if glossary and cache is not None:
        cache.put(text, cache_key, json.dumps(glossary, ensure_ascii=False))

    return glossary

def _system_prompt(lang: str, glossary: Optional[Dict[str, str]] = None) -> str:
    prompt = SYSTEM_TRANSLATE_PROMPT.format(lang=lang)
    if glossary:
        terms = "\n".join(f"  {k} -> {v}" for k, v in glossary.items())
        prompt += f"\nГлоссарий терминов (используй эти переводы):\n{terms}"
    return prompt

# =========================================================
# ПЕРЕВОДЧИКИ
# =========================================================
class GoogleTranslator:
    def __init__(self, target_lang: str):
        self.name = "Google"
        self.target_lang = target_lang
        self.can_generate = False
        self.session = _make_session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.rate_limiter = RateLimiter(max_requests_per_second=3)

    def translate(self, text: str) -> Optional[str]:
        self.rate_limiter.acquire()
        try:
            resp = self.session.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "gtx", "sl": "auto", "tl": self.target_lang,
                        "dt": "t", "ie": "UTF-8", "oe": "UTF-8", "q": text},
                timeout=120
            )
            resp.raise_for_status()
            data = resp.json()
            parts = [p[0] for p in data[0] if p[0]]
            result = " ".join(parts).strip()
            return result if result else None
        except Exception as e:
            logger.warning(f"   Google Translate: {type(e).__name__}: {e}")
            return None

    def generate(self, prompt: str) -> Optional[str]:
        return None

class OpenRouterRotator:
    def __init__(self, target_lang: str):
        self.target_lang = target_lang
        self.name = "OpenRouter"
        self.can_generate = True
        self._client = None
        self._models: list = []
        self._current_idx = 0
        self._exhausted_models: dict = {}
        self._model_cooldown_sec = 60
        self.rate_limiter = RateLimiter(max_requests_per_second=0.3)

        if OPENAI_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=OPENAI_API_KEY,
                    base_url=OPENAI_BASE_URL,
                    default_headers={"X-Title": "pdf-translator"},
                    timeout=120,
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
                key=lambda x: ("openrouter/" in x, x), reverse=True,
            )
            if not self._models:
                self._models = ["openrouter/free", "google/gemma-4-31b-it:free"]
            logger.debug(f"OpenRouter: {len(self._models)} free-моделей")
        except Exception as e:
            logger.debug(f"OpenRouter: не удалось получить модели: {e}")
            self._models = ["openrouter/free", "google/gemma-4-31b-it:free"]

    def _is_available(self, model: str) -> bool:
        if model not in self._exhausted_models:
            return True
        if time.time() - self._exhausted_models[model] > self._model_cooldown_sec:
            del self._exhausted_models[model]
            return True
        return False

    def _next_model(self) -> Optional[str]:
        for _ in range(len(self._models)):
            model = self._models[self._current_idx]
            self._current_idx = (self._current_idx + 1) % len(self._models)
            if self._is_available(model):
                return model
        return None

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        if not self._client:
            return None

        attempts = 0
        while attempts < len(self._models):
            model = self._next_model()
            if not model:
                break

            self.rate_limiter.acquire()

            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _system_prompt(self.target_lang, glossary)},
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
                    self._exhausted_models[model] = time.time()
                    time.sleep(5)
                    attempts += 1
                    continue
                attempts += 1

        return None

    def generate(self, prompt: str) -> Optional[str]:
        if not self._client:
            return None

        model = self._next_model()
        if not model:
            model = self._models[0] if self._models else None
        if not model:
            return None

        self.rate_limiter.acquire()

        try:
            resp = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else None
        except Exception:
            return None

class LlamaCppTranslator:
    def __init__(self, target_lang: str, api_base: str = "http://localhost:8080/v1",
                 expected_model: Optional[str] = None, auto_find: bool = True):
        self.target_lang = target_lang
        self.name = "LlamaCpp"
        self.can_generate = True
        self.expected_model = expected_model
        self.rate_limiter = RateLimiter(max_requests_per_second=1)
        self._loaded_model = None
        self._server_ok = False
        self.api_base = None

        if api_base:
            ok, models = self._check_server(api_base)
            if ok:
                server_model = models[0] if models else "unknown"
                if expected_model and server_model != expected_model:
                    logger.warning(f"   На {api_base} загружена '{server_model}', а нужна '{expected_model}'")
                else:
                    self.api_base = api_base.rstrip("/")
                    self._loaded_model = server_model
                    self._server_ok = True
                    logger.info(f"   llama-server: {self.api_base}, модель: {self._loaded_model}")

        if not self._server_ok and auto_find:
            servers = self._find_all_servers()

            if not servers:
                raise RuntimeError("llama-server не найден. Запустите: ./start_llama.sh translate 8080")

            if expected_model is None:
                self.api_base = servers[0]["url"]
                self._loaded_model = servers[0]["model"]
                self._server_ok = True
            else:
                matched = [s for s in servers if s["model"] == expected_model]
                if matched:
                    self.api_base = matched[0]["url"]
                    self._loaded_model = matched[0]["model"]
                    self._server_ok = True
                else:
                    available = "\n".join(f"   {s['url']} -> {s['model']}" for s in servers)
                    raise RuntimeError(f"Модель '{expected_model}' не найдена.\n{available}")

        if not self._server_ok:
            raise RuntimeError("Не удалось подключиться ни к одному серверу.")

    @staticmethod
    def _find_all_servers() -> List[Dict[str, str]]:
        if os.name == 'nt':
            return []

        result = []
        try:
            output = subprocess.check_output(["pgrep", "-a", "llama-server"], text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                m = re.search(r'--port\s+(\d+)', line)
                port = int(m.group(1)) if m else 8080
                url = f"http://localhost:{port}/v1"

                try:
                    resp = requests.get(f"{url}/models", timeout=2)
                    if resp.status_code == 200:
                        models = [m.get("id", "") for m in resp.json().get("data", [])]
                        if models:
                            result.append({"url": url, "model": models[0]})
                except Exception:
                    continue
        except Exception:
            pass

        return result

    def _check_server(self, url: str) -> Tuple[bool, List[str]]:
        try:
            resp = requests.get(f"{url}/models", timeout=5)
            if resp.status_code == 200:
                models = [m.get("id", "") for m in resp.json().get("data", [])]
                if models:
                    return True, models
        except Exception:
            pass
        return False, []

    def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.3,
                  max_tokens: int = 4096) -> Optional[str]:
        if not self._server_ok:
            return None

        self.rate_limiter.acquire()

        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={"model": self._loaded_model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens, "stream": False},
                timeout=120,
            )

            if resp.status_code == 200:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content")
                return content.strip() if content else None
        except Exception:
            pass

        return None

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        return self._call_api([
            {"role": "system", "content": _system_prompt(self.target_lang, glossary)},
            {"role": "user", "content": text}
        ])

    def generate(self, prompt: str) -> Optional[str]:
        return self._call_api([{"role": "user", "content": prompt}])

# ---- Фабрика переводчиков ----
def create_translator(translator_type: str, target_lang: str,
                      llama_url: str = "http://localhost:8080/v1",
                      llama_model: Optional[str] = None,
                      auto_find: bool = True):
    if translator_type == "llama":
        primary = LlamaCppTranslator(target_lang, api_base=llama_url,
                                     expected_model=llama_model, auto_find=auto_find)
        fallback = GoogleTranslator(target_lang)
        return primary, fallback

    elif translator_type == "openrouter":
        primary = OpenRouterRotator(target_lang)
        fallback = GoogleTranslator(target_lang)
        return primary, fallback

    elif translator_type == "google":
        primary = GoogleTranslator(target_lang)
        fallback = None

        if OPENAI_API_KEY:
            try:
                or_tr = OpenRouterRotator(target_lang)
                if or_tr._client:
                    fallback = or_tr
                    logger.debug(f"   Fallback на OpenRouter")
            except Exception:
                pass

        if not fallback:
            try:
                fallback = LlamaCppTranslator(target_lang, api_base=llama_url,
                                              expected_model=llama_model, auto_find=auto_find)
                logger.debug(f"   Fallback на llama: {fallback.api_base}")
            except Exception:
                pass

        return primary, fallback

    else:
        raise ValueError(f"Неизвестный переводчик: {translator_type}")

# =========================================================
# КЭШ
# =========================================================
class TranslationCache:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, cache_path: str = "translation_cache.json"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, cache_path: str = "translation_cache.json"):
        if self._initialized:
            return

        self._initialized = True
        self.cache_path = cache_path
        self._cache: LRUCache = LRUCache(maxsize=10000)
        self._dirty = False
        self._load()

        atexit.register(self.save)

        for sig in (signal.SIGTERM, signal.SIGINT):
            prev = signal.getsignal(sig)

            def _handler(signum, frame, _prev=prev):
                self.save()
                if callable(_prev):
                    _prev(signum, frame)
                elif _prev == signal.SIG_DFL:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)

            signal.signal(sig, _handler)

    def _load(self):
        if not os.path.exists(self.cache_path):
            return

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for key, value in data.items():
                self._cache[key] = value

            logger.info(f"   Кэш загружен: {len(self._cache)} записей")
        except Exception as e:
            logger.warning(f"   Ошибка загрузки кэша: {e}")

    def save(self):
        if not self._dirty:
            return

        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f, ensure_ascii=False)
            self._dirty = False
        except Exception as e:
            logger.warning(f"   Ошибка сохранения кэша: {e}")

    def _key(self, text: str, lang: str) -> str:
        h = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{lang}:{h}"

    def get(self, text: str, lang: str) -> Optional[str]:
        return self._cache.get(self._key(text, lang))

    def put(self, text: str, lang: str, translation: str):
        self._cache[self._key(text, lang)] = translation
        self._dirty = True

        if len(self._cache) % 500 == 0:
            self.save()

# =========================================================
# ПЕРЕВОД С RETRY + FALLBACK + ЗАЩИТА ССЫЛОК
# =========================================================
def translate_chunk_with_retry(chunk: str, translator, max_retries: int = 3,
                               glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
    if len(chunk) < 3:
        return chunk

    for attempt in range(max_retries):
        _backoff(attempt)

        try:
            translated = translator.translate(chunk, glossary) if glossary else translator.translate(chunk)
            if translated and len(translated) >= 2 and translated.strip() != chunk.strip():
                return translated.strip()

        except requests.exceptions.HTTPError as e:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] HTTP {e.response.status_code}: {e}")

        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] Таймаут после {max_retries} попыток")

        except requests.exceptions.ConnectionError as e:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] Ошибка соединения: {e}")

        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"[{translator.name}] Ошибка: {e}")

    return None

# =========================================================
# PIPELINE ПЕРЕВОДА
# =========================================================
class TranslationPipeline:
    def __init__(self, translator, fallback=None, cache: Optional[TranslationCache] = None,
                 max_workers: int = 8, timeout: int = 600, is_local: bool = False,
                 translator_type: str = "google", refine: bool = False):
        self.translator = translator
        self.fallback = fallback
        self.cache = cache or TranslationCache()
        self.max_workers = max_workers
        self.timeout = timeout
        self.is_local = is_local
        self.translator_type = translator_type
        self.refine = refine

        self.stats = {"success": 0, "failed": 0, "cached": 0, "skipped": 0}
        self._stats_lock = threading.Lock()
        self._start_time = 0.0
        self._block_times = []
        
        self.SKIP_TYPES = ("figure", "table", "empty", "reference", "reference_heading")
        self.MAX_SECTION_CHARS = 6000

    # ---- Защита ссылок ----
    @staticmethod
    def _protect_refs(text: str) -> Tuple[str, Dict[str, str]]:
        refs = {}
        def repl(m):
            key = f"{{{{REF_{len(refs)}}}}}"
            refs[key] = m.group(0)
            return key
        protected = REF_PATTERN.sub(repl, text)
        return protected, refs

    @staticmethod
    def _restore_refs(text: str, refs: Dict[str, str]) -> str:
        for ph, orig in refs.items():
            text = text.replace(ph, orig)
        return text

    def _translate_one(self, text: str, lang: str,
                       glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        text = normalize_text(text)
        if not text:
            return None

        cached = self.cache.get(text, lang)
        if cached:
            with self._stats_lock:
                self.stats["cached"] += 1
            return cached

        protected_text, refs = self._protect_refs(text)
        t0 = time.time()

        result = translate_chunk_with_retry(protected_text, self.translator, glossary=glossary)
        if not result and self.fallback:
            result = translate_chunk_with_retry(protected_text, self.fallback, glossary=glossary)

        if result:
            result = self._restore_refs(result, refs)

            if self.refine and self.is_local and getattr(self.translator, 'can_generate', False):
                refined = self._refine(result, lang)
                if refined:
                    result = refined

        elapsed = time.time() - t0
        with self._stats_lock:
            self._block_times.append(elapsed)

        if result:
            self.cache.put(text, lang, result)
            with self._stats_lock:
                self.stats["success"] += 1
        else:
            with self._stats_lock:
                self.stats["failed"] += 1

        return result

    def _refine(self, translated: str, lang: str) -> Optional[str]:
        prompt = REFINE_PROMPT.format(lang=lang) + "\n" + translated
        try:
            res = self.translator.generate(prompt)
            if res and len(res) >= 2:
                return res.strip()
        except Exception:
            pass
        return None

    def _translate_group(self, group: list, lang: str,
                         glossary: Optional[Dict[str, str]] = None,
                         quiet: bool = False) -> None:
        parts = [b.text for b in group]
        combined = "\n\n".join(parts)

        result = self._translate_one(combined, lang, glossary)

        if result:
            translated_parts = [p.strip() for p in re.split(r'\n\s*\n', result.strip()) if p.strip()]
            if len(translated_parts) == len(parts):
                for b, tp in zip(group, translated_parts):
                    b.translation = tp or b.text
                return

        if not quiet:
            logger.info(f"   Секция: {len(translated_parts)}/{len(parts)} абзацев, перевод по одному")

        for b in group:
            res = self._translate_one(b.text, lang, glossary)
            b.translation = res if res else b.text

    def translate_blocks(self, blocks: list, lang: str, quiet: bool = False) -> list:
        ref_buf = []
        ref_buf_page = 0
        merged_blocks = []

        def flush_refs():
            if not ref_buf:
                return
            merged_text = re.sub(r'\s+', ' ', " ".join(ref_buf)).strip()
            ref_block = make_block(type="reference", page_num=ref_buf_page, bbox=(0, 0, 0, 0),
                                   translation=merged_text)
            merged_blocks.append(ref_block)
            ref_buf.clear()

        for block in blocks:
            text = block.text

            if not text or len(text) < 2:
                block.translation = text if text else ""
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue

            if block.type in self.SKIP_TYPES:
                block.translation = text
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue

            if block.type == "reference":
                ref_buf.append(text)
                ref_buf_page = block.page_num
                with self._stats_lock:
                    self.stats["skipped"] += 1
                continue

            flush_refs()
            merged_blocks.append(block)

        flush_refs()
        blocks = merged_blocks

        translatable = [b for b in blocks if b.type not in self.SKIP_TYPES and b.translation is None]

        if not translatable:
            return blocks

        glossary = None
        if getattr(self.translator, 'can_generate', False):
            sample = "\n".join(b.text for b in translatable)[:20000]
            if sample:
                glossary = extract_glossary(sample, self.translator, lang, self.cache)

            if glossary and not quiet:
                logger.info(f"   Глоссарий: {len(glossary)} терминов")

        sections = []
        current = []

        for b in blocks:
            if b.type in self.SKIP_TYPES or b.translation is not None:
                if current:
                    sections.append(current)
                    current = []
                continue
            current.append(b)

        if current:
            sections.append(current)

        tasks = []
        for sec in sections:
            if sum(len(b.text) for b in sec) <= self.MAX_SECTION_CHARS:
                tasks.append(sec)
                continue

            group = []
            group_len = 0
            for b in sec:
                if group and group_len + len(b.text) > self.MAX_SECTION_CHARS:
                    tasks.append(group)
                    group = [b]
                    group_len = len(b.text)
                else:
                    group.append(b)
                    group_len += len(b.text)

            if group:
                tasks.append(group)

        total = len(tasks)
        done = 0
        lock = threading.Lock()
        self._start_time = time.time()

        if not quiet:
            logger.info(f"   Блоков для перевода: {len(translatable)} (секций: {total})")

        def _work(group):
            nonlocal done

            if len(group) == 1:
                b = group[0]
                result = self._translate_one(b.text, lang, glossary)
                b.translation = result if result else b.text
            else:
                self._translate_group(group, lang, glossary, quiet)

            with lock:
                done += 1

                if TQDM and not quiet:
                    elapsed = time.time() - self._start_time
                    avg = elapsed / done
                    remaining = avg * (total - done)
                    pbar.set_postfix_str(f"ост. {_format_time(remaining)}", refresh=True)
                    pbar.update(1)

                elif not quiet and done % 5 == 0:
                    elapsed = time.time() - self._start_time
                    avg = elapsed / done
                    remaining = avg * (total - done)
                    logger.info(f"   {done}/{total} ({done*100//total}%) {_format_time(elapsed)} ETA: {_format_time(remaining)}")

        if TQDM and not quiet:
            pbar = tqdm(total=total, desc="🌐 Перевод", unit="блок",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_work, g): g for g in tasks}

            try:
                for future in as_completed(futures, timeout=self.timeout):
                    future.result()

            except TimeoutError:
                logger.warning(f"Превышен таймаут ({self.timeout} сек)")
                for f in futures:
                    f.cancel()

            finally:
                if TQDM and not quiet:
                    pbar.close()

        if not quiet:
            elapsed = time.time() - self._start_time
            logger.info(f"   Перевод завершён за {_format_time(elapsed)}")

            if self._block_times:
                avg = sum(self._block_times) / len(self._block_times)
                logger.info(f"   Среднее время на блок: {avg:.1f} сек")

        return blocks

# =========================================================
# ГЕНЕРАЦИЯ РЕФЕРАТА
# =========================================================
def generate_summary(
    pages: list,
    summary_translator_type: str,
    translate_translator_type: str,
    lang: str,
    llama_url: str = "http://localhost:8080/v1",
    llama_model: Optional[str] = None,
    auto_find: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:
    all_text = []
    for page in pages:
        for block in page.blocks:
            if block.type in ("paragraph", "heading", "metadata", "text", "list"):
                text = block.text
                if text and len(text) > 10:
                    all_text.append(text)

    full_text = "\n".join(all_text)

    if not full_text:
        return {"summary_html": "<p>Не удалось извлечь текст для реферата.</p>", "stats": {"chunks": 0, "tokens": 0, "speed": 0}}

    if summary_translator_type == "google":
        logger.warning("   Google Translate не поддерживает генерацию. Переключаю на llama.")
        summary_translator_type = "llama"

    summary_translator = None
    try:
        summary_translator, _ = create_translator(
            summary_translator_type, "en",
            llama_url=llama_url, llama_model=llama_model, auto_find=auto_find,
        )
    except RuntimeError as e:
        logger.error(f"   Не удалось создать генератор ({summary_translator_type}): {e}")
        if summary_translator_type == "llama":
            summary_translator = GoogleTranslator("en")
        else:
            return {
                "summary_html": "<p>Ошибка: нет LLM для генерации реферата.</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": str(e)}
            }

    logger.info(f"   Генератор реферата: {summary_translator.name}")

    final_translator = None
    if translate_translator_type == "llama":
        try:
            final_translator, _ = create_translator("llama", lang,
                                                    llama_url=llama_url, llama_model=llama_model, auto_find=auto_find)
        except RuntimeError:
            final_translator = GoogleTranslator(lang)
    else:
        final_translator = GoogleTranslator(lang)

    logger.info(f"   Переводчик реферата: {final_translator.name}")

    if not getattr(summary_translator, 'can_generate', False):
        logger.warning(f"   {summary_translator.name} не поддерживает generate(). Попытка найти LLM...")
        try:
            summary_translator, _ = create_translator("llama", "en",
                                                      llama_url=llama_url, llama_model=llama_model, auto_find=auto_find)
            logger.info(f"   Генератор реферата: {summary_translator.name}")
        except RuntimeError:
            logger.error("   Ни один генератор не поддерживает generate().")
            return {
                "summary_html": "<p>Ошибка: нет LLM-сервера. Запустите llama-server.</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": "no LLM"}
            }

    chunks = _chunk_text_by_sentences(full_text, max_chunk_size=6000)

    if not quiet:
        logger.info(f"   Чанков для реферата: {len(chunks)}")

    cache = TranslationCache()
    cache_key = f"summary:{summary_translator_type}"

    chunk_summaries = []
    start_time = time.time()
    total_tokens = 0

    if TQDM and not quiet:
        pbar = tqdm(total=len(chunks), desc="📝 Реферат", unit="чанк",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for i, chunk in enumerate(chunks):
        cached = cache.get(chunk, cache_key)
        if cached:
            chunk_summaries.append(cached)
            total_tokens += len(cached) // 4

            if TQDM and not quiet:
                pbar.update(1)
            continue

        prompt = "Extract key facts from this text for a summary. Write the summary in English:\n" + chunk
        summary = None

        for attempt in range(2):
            try:
                summary = summary_translator.generate(prompt)
                if summary:
                    break
            except Exception as e:
                logger.warning(f"   Чанк {i+1}: ошибка (попытка {attempt+1}): {e}")

        if summary:
            cache.put(chunk, cache_key, summary)
            chunk_summaries.append(summary)
            total_tokens += len(summary) // 4
        else:
            chunk_summaries.append(chunk)
            total_tokens += len(chunk) // 4

        if TQDM and not quiet:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (len(chunks) - i - 1)
            pbar.set_postfix_str(f"ост. {_format_time(remaining)}", refresh=True)
            pbar.update(1)

    if TQDM and not quiet:
        pbar.close()

    if not chunk_summaries:
        return {"summary_html": "<p>Не удалось сгенерировать реферат.</p>", "stats": {"chunks": 0, "tokens": 0, "speed": 0}}

    summaries_to_combine = chunk_summaries

    if len(chunk_summaries) > 5:
        if not quiet:
            logger.info(f"   Иерархическая суммаризация ({len(chunk_summaries)} промежуточных)...")

        grouped = []
        for j in range(0, len(chunk_summaries), 5):
            group = chunk_summaries[j:j + 5]
            group_prompt = "Combine these key points into a concise summary:\n" + "\n---\n".join(group)
            group_summary = summary_translator.generate(group_prompt)

            if group_summary:
                grouped.append(group_summary)
                total_tokens += len(group_summary) // 4
            else:
                grouped.extend(group)

        summaries_to_combine = grouped

    final_prompt = (
        "Based on the following key points, create a final structured summary in English "
        "strictly in the format:\n"
        "BRIEF CONTENT: 4-5 sentences\n"
        "KEY FINDINGS: list\n"
        "STRENGTHS: list\n"
        "WEAKNESSES: list\n\n"
        "Key points:\n" + "\n---\n".join(summaries_to_combine)
    )

    summary_en = None
    for attempt in range(2):
        try:
            summary_en = summary_translator.generate(final_prompt)
            if summary_en:
                break
        except Exception as e:
            logger.warning(f"   Итоговый реферат: ошибка (попытка {attempt+1}): {e}")

    if not summary_en:
        summary_en = "\n".join(summaries_to_combine)

    try:
        final_translated = final_translator.translate(summary_en)
        summary_html = md.markdown(final_translated if final_translated else summary_en)
    except Exception:
        summary_html = md.markdown(summary_en)

    elapsed = time.time() - start_time
    speed = total_tokens / elapsed if elapsed > 0 else 0
    actual_model = getattr(summary_translator, '_loaded_model', summary_translator.name)

    stats = {"chunks": len(chunks), "tokens": total_tokens, "speed": speed, "model": actual_model, "time": elapsed}

    return {"summary_html": summary_html, "stats": stats}

# =========================================================
# HTML — единый Jinja2-шаблон (тёмная/светлая тема)
# =========================================================
BLOCK_CSS = """
<style>
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 0; background: #0f172a; color: #e2e8f0; }
.container { max-width: 960px; margin: 0 auto; padding: 24px; }
h1 { color: #60a5fa; border-bottom: 2px solid #1e3a5f; padding-bottom: 12px; }
h3 { color: #93c5fd; margin-top: 28px; }
.page { background: #1e293b; padding: 20px; margin-bottom: 20px; border-radius: 8px; border: 1px solid #334155; }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
.trans-head h2, .trans-head h3, .trans-head h4 { margin: 4px 0; }
p { line-height: 1.7; margin: 0 0 10px; text-align: justify; }
.orig { color: #64748b; font-size: 0.88em; }
.trans { color: #e2e8f0; }
.ref-section h2 { color: #60a5fa; border-bottom: 1px solid #334155; padding-bottom: 6px; }
.ref { color: #64748b; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; }
.meta { color: #64748b; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #334155; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid #334155; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: #0f172a; border: 1px solid #334155; padding: 8px 10px; color: #60a5fa; }
.table-wrap td { border: 1px solid #334155; padding: 6px 10px; }
.table-wrap tr:nth-child(even) { background: #1e293b; }
.figure { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: #94a3b8; font-size: 0.85em; }
.list-block { margin-left: 20px; line-height: 1.6; }
details.original-block { margin: 4px 0; }
details.original-block summary { color: #64748b; font-size: 0.85em; cursor: pointer; }
details.original-block summary:hover { color: #94a3b8; }
.ref-link { color: #60a5fa; font-weight: bold; }
</style>
"""

BLOCK_CSS_LIGHT = """
<style>
body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
.container { max-width: 960px; margin: 0 auto; }
h1 { color: #1f2937; border-bottom: 2px solid #d1d5db; padding-bottom: 12px; }
h3 { color: #374151; margin-top: 28px; }
.page { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
p { line-height: 1.7; margin: 0 0 10px; text-align: justify; color: #111827; }
.orig { color: #9ca3af; font-size: 0.88em; }
.trans { color: #111827; }
.ref-section h2 { color: #1f2937; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; }
.ref { color: #6b7280; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; }
.meta { color: #6b7280; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #d1d5db; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid #d1d5db; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: #f3f4f6; border: 1px solid #d1d5db; padding: 8px 10px; color: #1f2937; }
.table-wrap td { border: 1px solid #d1d5db; padding: 6px 10px; }
.table-wrap tr:nth-child(even) { background: #f9fafb; }
.figure { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: #6b7280; font-size: 0.85em; }
.list-block { margin-left: 20px; line-height: 1.6; }
details.original-block { margin: 4px 0; }
details.original-block summary { color: #9ca3af; font-size: 0.85em; cursor: pointer; }
details.original-block summary:hover { color: #6b7280; }
.ref-link { color: #2563eb; font-weight: bold; }
</style>
"""

def _block_html(block: Block, in_ref: bool = False, images_dir: Optional[str] = None) -> str:
    e = html_mod.escape
    bt = block.type
    trans = block.translation or block.text
    orig = block.text

    if bt == 'reference_heading':
        return f'<div class="ref-section"><h2>{e(orig)}</h2></div>\n'

    if in_ref or bt == 'reference':
        return f"<p class='ref'>{e(orig)}</p>\n"

    if bt == 'metadata':
        return f"<div class='meta'>{e(orig)}</div>\n"

    if bt == 'table' and block.table_data:
        rows = block.table_data
        if len(rows) >= 1:
            out = ['<div class="table-wrap"><table>\n']
            for idx, row in enumerate(rows[:20]):
                tag = 'th' if idx == 0 else 'td'
                if idx == 0:
                    out.append('<thead><tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr></thead>\n<tbody>\n')
                else:
                    out.append('<tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr>\n')
            out.append('</tbody></table>\n</div>\n')
            return ''.join(out)

    if bt == 'figure':
        if block.image_data:
            img_src = f"data:image/png;base64,{block.image_data}"
            if images_dir:
                try:
                    os.makedirs(images_dir, exist_ok=True)
                    img_hash = hashlib.md5(block.image_data.encode()).hexdigest()[:8]
                    img_filename = f"fig_{block.page_num}_{img_hash}.png"
                    img_path = os.path.join(images_dir, img_filename)

                    if not os.path.exists(img_path):
                        with open(img_path, "wb") as f:
                            f.write(base64.b64decode(block.image_data))

                    img_src = os.path.join(os.path.basename(images_dir), img_filename)
                except Exception:
                    pass

            parts = []
            parts.append(f'<figure class="figure"><img src="{img_src}" alt="Рисунок" />')
            if block.caption:
                parts.append(f'<figcaption class="figure-caption">{e(block.caption)}</figcaption>')
            parts.append('</figure>\n')
            return ''.join(parts)
        else:
            return '<p class="orig">(Изображение не найдено)</p>\n'

    if bt == 'heading':
        spans_text = render_spans(block)
        if orig != trans:
            return f'<div class="trans-head"><p class="orig">{e(spans_text)}</p><p class="trans">{e(trans)}</p></div>\n'
        return f"<p class='trans'><b>{e(trans)}</b></p>\n"

    if bt == 'list':
        items = orig.split('\n')
        lis = ''.join(f'<li>{e(item.lstrip("•-*►▸‣⁃◦○●▪ 0123456789.)"))}</li>' for item in items if item.strip())
        return f'<ul class="list-block">{lis}</ul>\n'

    spans_text = render_spans(block)
    
    trans_with_refs = re.sub(r'\[([^\]]+)\]', r'<span class="ref-link">[\1]</span>', trans)

    if orig != trans:
        return f'<p class="trans">{trans_with_refs}</p><details class="original-block"><summary>Оригинал</summary><p class="orig">{e(spans_text)}</p></details>\n'

    return f"<p>{trans_with_refs}</p>\n"

def render_spans(block):
    parts = []
    for line in block.lines:
        for span in line.spans:
            text = span.text
            if span.flags & 2**0:
                text = f"<b>{text}</b>"
            if span.flags & 2**1:
                text = f"<i>{text}</i>"
            parts.append(text)
        parts.append("<br/>")
    return " ".join(parts)

# =========================================================
# ГЕНЕРАЦИЯ HTML
# =========================================================
# ИСПРАВЛЕНО: Убран лишний аргумент images_dir из вызова _block_html
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{{ title | e }}</title>
{{ css | safe }}
</head>
<body>
<div class="container">
<h1>{{ title | e }}</h1>
{% set ns = namespace(in_ref=false) %}
{% for page in pages %}
<div class="page">
<h3>📄 Страница {{ page.num }}</h3>
{% for block in page.blocks %}
{% if block.type == 'reference_heading' %}
{% set ns.in_ref = true %}
{% endif %}
{{ _block_html(block, ns.in_ref) | safe }}
{% endfor %}
</div>
{% endfor %}
</div>
</body>
</html>
"""

def generate_html(pages: List[Page], title: str, output_path: str, dark: bool = True):
    css = BLOCK_CSS if dark else BLOCK_CSS_LIGHT

    # ИСПРАВЛЕНО: Добавлен os.path.abspath для корректного определения директории
    output_dir = os.path.dirname(os.path.abspath(output_path))
    images_dir = os.path.join(output_dir, "images")

    def block_renderer(block, in_ref):
        return _block_html(block, in_ref, images_dir)

    template = Template(HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, css=css,
                               _block_html=block_renderer, images_dir=images_dir)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)

SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
.summary-container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 900px; margin: 0 auto; }
.summary-block { background: #e7f3ff; padding: 20px; border-radius: 8px; border: 1px solid #b3d7ff; margin-bottom: 20px; }
.summary-block h2 { margin-top: 0; color: #004085; }
.metadata { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 2em; border: 1px solid #dee2e6; }
.metadata h1 { margin-top: 0; color: #111; font-size: 2em; }
</style>
</head>
<body>
<div class="summary-container">
<div class="metadata"><h1>{{ title }}</h1></div>
<div class="summary-block">{{ summary_html | safe }}</div>
<details><summary>Показать исходные тексты</summary>
{% for page in pages %}
<div class="page"><h3>Страница {{ page.num }}</h3>
{% for block in page.blocks %}
{% if block.type in ('paragraph', 'heading', 'metadata', 'text', 'list') %}
{% set orig = render_spans(block) %}
{% if orig | length > 40 %}
<details><summary>Оригинал (стр. {{ page.num }})</summary><span class="original">{{ orig }}</span></details>
{% endif %}
{% endif %}
{% endfor %}</div>
{% endfor %}</details></div></body></html>
"""

def generate_summary_html(pages: List[Page], summary_html: str, title: str, output_path: str):
    template = Template(SUMMARY_TEMPLATE)
    rendered = template.render(pages=pages, title=title, summary_html=summary_html,
                               render_spans=render_spans)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)

# =========================================================
# ОСНОВНОЙ ПРОЦЕСС
# =========================================================
def process_pdf(
    pdf_path: str,
    output_html: str,
    lang: str = "ru",
    translator_type: str = "google",
    llama_url: str = "http://localhost:8080/v1",
    llama_model: Optional[str] = None,
    auto_find: bool = True,
    max_workers: int = 8,
    timeout: int = 600,
    quiet: bool = False,
    summary_mode: bool = False,
    summary_translator_type: str = "llama",
    translate_translator_type: str = "google",
    dark_html: bool = True,
    refine: bool = False,
) -> Dict[str, Any]:
    timings: Dict[str, float] = {}
    total_images = 0
    total_tables = 0

    with stage_timer("1. Извлечение PDF", timings):
        extractor = PDFExtractor(pdf_path)
        pages = extractor.extract()
        doc = fitz.open(pdf_path)

        for page in pages:
            text_blocks = [b for b in page.blocks if b.type == "text"]
            fitz_page = doc[page.num - 1]

            try:
                figure_blocks, text_blocks = extract_images(fitz_page, page.num, text_blocks)
                page.blocks = [b for b in page.blocks if b.type != "text"] + text_blocks + figure_blocks
                total_images += len(figure_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения изображений: {e}")

            try:
                table_blocks = extract_tables(fitz_page, page.num)
                page.blocks.extend(table_blocks)
                total_tables += len(table_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения таблиц: {e}")

        doc.close()
        logger.info(f"   Извлечено {len(pages)} стр., {total_images} изобр., {total_tables} табл.")

    with stage_timer("2. Классификация", timings):
        for page in pages:
            all_sizes = []
            for block in page.blocks:
                if block.type in ("paragraph", "heading", "metadata"):
                    for line in block.lines:
                        for span in line.spans:
                            all_sizes.append(span.size)

            avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12.0

            for block in page.blocks:
                if block.type == "text":
                    block.type = _classifier.classify(block, page.width, avg_size)

    with stage_timer("3. Пост-обработка", timings):
        in_refs = False
        for page in pages:
            for block in page.blocks:
                if block.type == "reference_heading":
                    in_refs = True
                    continue
                if in_refs and block.type not in ("figure", "table"):
                    block.type = "reference"

        for page in pages:
            table_blocks_page = [b for b in page.blocks if b.type == "table"]
            if table_blocks_page:
                mark_table_blocks(page.blocks, table_blocks_page)

            page.blocks = [b for b in page.blocks if not (b.type == "table" and b.table_data is None)]

    if summary_mode:
        with stage_timer("4. Генерация реферата", timings):
            logger.info("📝 Режим реферата...")

            if summary_translator_type == "google":
                logger.warning("   Google Translate не поддерживает генерацию. Переключаю на llama.")
                summary_translator_type = "llama"

            result = generate_summary(
                pages, summary_translator_type=summary_translator_type,
                translate_translator_type=translate_translator_type, lang=lang,
                llama_url=llama_url, llama_model=llama_model, auto_find=auto_find, quiet=quiet,
            )

            summary_html = result["summary_html"]
            gen_stats = result["stats"]

        with stage_timer("5. HTML реферата", timings):
            generate_summary_html(pages, summary_html, f"Реферат: {os.path.basename(pdf_path)}", output_html)
            logger.info(f"   HTML сохранён: {output_html}")

        stats = {
            "chunks": gen_stats.get("chunks", 0),
            "tokens": gen_stats.get("tokens", 0),
            "speed": gen_stats.get("speed", 0),
            "model": gen_stats.get("model", ""),
            "time": gen_stats.get("time", 0),
            "summary_mode": True,
            "timings": timings,
        }

        return {"stats": stats, "pages": len(pages), "images": total_images, "tables": total_tables, "summary": True}

    with stage_timer("4. Перевод", timings):
        logger.info(f"🌐 Перевод на {lang} ({translator_type})...")

        try:
            translator, fallback = create_translator(
                translator_type, lang, llama_url=llama_url,
                llama_model=llama_model, auto_find=auto_find,
            )
        except RuntimeError as e:
            logger.error(f"   Ошибка: {e}. Fallback на Google Translate...")
            translator = GoogleTranslator(lang)
            fallback = None

        logger.info(f"   Переводчик: {translator.name}")
        if fallback:
            logger.info(f"   Fallback: {fallback.name}")

        effective_workers = max_workers
        if translator_type == "llama":
            effective_workers = min(max_workers, 3)
            if not quiet:
                logger.info(f"   Локальный сервер: ограничено до {effective_workers} воркеров")

        pipeline = TranslationPipeline(
            translator=translator, fallback=fallback,
            max_workers=effective_workers, timeout=timeout,
            is_local=(translator_type == "llama"), translator_type=translator_type,
            refine=refine,
        )

        all_blocks = []
        for page in pages:
            all_blocks.extend(page.blocks)

        pipeline.translate_blocks(all_blocks, lang, quiet=quiet)

    with stage_timer("5. HTML", timings):
        generate_html(pages, f"Перевод: {os.path.basename(pdf_path)}", output_html, dark=dark_html)
        logger.info(f"   HTML сохранён: {output_html}")

    return {
        "stats": {**pipeline.stats, "timings": timings},
        "pages": len(pages), "images": total_images, "tables": total_tables,
    }

# =========================================================
# CLI
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="PDF Translator — единый скрипт")

    parser.add_argument("input", help="Входной PDF файл")
    parser.add_argument("-l", "--lang", default="ru", help="Целевой язык (по умолчанию: ru)")

    parser.add_argument("-t", "--translator", default="google",
                        choices=["google", "llama", "openrouter"],
                        help="Переводчик (по умолчанию: google)")

    parser.add_argument("--llama-url", default="http://localhost:8080/v1", help="URL llama.cpp сервера")
    parser.add_argument("--llama-model", help="Ожидаемое имя модели")
    parser.add_argument("--no-auto-find", action="store_true", help="Отключить автоматический поиск сервера")

    parser.add_argument("--summary", action="store_true", help="Режим реферата")

    parser.add_argument("--summary-translator", choices=["google", "llama", "openrouter"], default="llama",
                        help="Генератор реферата (по умолчанию: llama)")

    parser.add_argument("--summary-lang-translator", choices=["google", "llama", "openrouter"], default="google",
                        help="Переводчик реферата (по умолчанию: google)")

    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1) * 2),
                        help="Число воркеров")

    parser.add_argument("--task-timeout", type=int, default=600, help="Таймаут перевода (сек)")

    parser.add_argument("--refine", action="store_true",
                        help="Итеративный рефайн перевода (для локальных LLM)")

    parser.add_argument("--dark-html", action="store_true", default=True,
                        help="Тёмная тема HTML (по умолчанию: вкл)")

    parser.add_argument("--light-html", action="store_true", help="Светлая тема HTML")

    parser.add_argument("-q", "--quiet", action="store_true", help="Тихий режим")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    dark_html = not args.light_html

    input_base = os.path.splitext(args.input)[0]
    output_html = f"{input_base}_summary.html" if args.summary else f"{input_base}_translate.html"

    logger.info("=" * 60)
    logger.info(f"PDF TRANSLATOR — {args.translator.upper()}")
    logger.info("=" * 60)

    try:
        result = process_pdf(
            pdf_path=args.input,
            output_html=output_html,
            lang=args.lang,
            translator_type=args.translator,
            llama_url=args.llama_url,
            llama_model=args.llama_model,
            auto_find=not args.no_auto_find,
            max_workers=args.workers,
            timeout=args.task_timeout,
            quiet=args.quiet,
            summary_mode=args.summary,
            summary_translator_type=args.summary_translator,
            translate_translator_type=args.summary_lang_translator,
            dark_html=dark_html,
            refine=args.refine,
        )

        stats = result["stats"]

        logger.info("=" * 60)

        if stats.get("summary_mode", False):
            logger.info("📊 СТАТИСТИКА ГЕНЕРАЦИИ РЕФЕРАТА:")
            logger.info(f"   ✓ Обработано чанков: {stats.get('chunks', 0)}")

            if stats.get("tokens", 0) > 0:
                logger.info(f"   📝 Сгенерировано токенов (приблиз.): {stats['tokens']}")

            if stats.get("speed", 0) > 0:
                logger.info(f"   ⚡ Скорость: {stats['speed']:.1f} токенов/сек")

            if stats.get("model"):
                logger.info(f"   🤖 Модель: {stats['model']}")

            if stats.get("time", 0) > 0:
                logger.info(f"   ⏱️ Время генерации: {_format_time(stats['time'])}")

        else:
            logger.info("📊 СТАТИСТИКА ПЕРЕВОДА:")
            logger.info(f"   ✓ Переведено: {stats['success']}")
            logger.info(f"   ⚡ Из кэша:   {stats['cached']}")
            logger.info(f"   ⊘ Пропущено:  {stats['skipped']}")
            logger.info(f"   ✗ Ошибок:     {stats['failed']}")

            timings = stats.get("timings", {})
            if timings:
                logger.info("⏱️  ПРОФИЛИРОВАНИЕ:")
                for stage, t in timings.items():
                    logger.info(f"   {stage}: {_format_time(t)}")

        logger.info("=" * 60)
        logger.info(f"✅ Готово: {output_html}")

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
