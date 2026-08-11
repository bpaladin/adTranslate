#!/usr/bin/env python3
"""
PDF Translator — единый скрипт для перевода и реферирования PDF-документов.
Поддерживает Google Translate и локальный llama.cpp (только как клиент к уже запущенному серверу).
Автоматически находит работающий сервер через pgrep, фильтрует по модели.
"""
import os
import sys
import re
import time
import json
import atexit
import signal
import hashlib
import logging
import threading
import subprocess
import argparse
import warnings
import html
import base64
import contextlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import requests
import markdown as md
from cachetools import LRUCache
from jinja2 import Template
from requests.adapters import HTTPAdapter

# ---- Попытка использовать pymupdf_layout ----
try:
    from pymupdf_layout import extract_layout
    HAVE_LAYOUT = True
except ImportError:
    HAVE_LAYOUT = False

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


# ---- ML классификатор (опционально) ----
try:
    from sklearn.ensemble import RandomForestClassifier
    import numpy as np
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


# ---- Профилирование ----
@contextlib.contextmanager
def stage_timer(name: str, timings: Optional[Dict[str, float]] = None):
    """Контекстный менеджер для замера времени этапов."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.info(f"⏱️  {name}: {_format_time(elapsed)}")
        if timings is not None:
            timings[name] = elapsed


# ---- Вспомогательные функции ----
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


def estimate_processing_time(text_length: int, is_generation: bool = False, translator_type: str = "google") -> float:
    chars_per_sec = {"google": 800, "llama": 50}
    if is_generation:
        rate = chars_per_sec.get(translator_type, 50)
        return max(5.0, text_length / rate * 4)
    else:
        rate = chars_per_sec.get(translator_type, 200)
        return max(2.0, text_length / rate)


def _chunk_text_by_sentences(text: str, max_chunk_size: int = 6000) -> List[str]:
    """Разбивает текст на чанки по границам предложений."""
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


# ---- Модели данных ----
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


# ---- Фабричные методы ----
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


# ---- Утилита метрик блока ----
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


# ---- Извлечение PDF ----
class PDFExtractor:
    def __init__(self, path: str):
        self.doc = fitz.open(path)
        self.use_layout = HAVE_LAYOUT

    def extract(self) -> List[Page]:
        pages = []
        if self.use_layout:
            try:
                layout = extract_layout(self.doc)
                for page_num, page_data in enumerate(layout.pages):
                    page_blocks = []
                    for item in page_data.items:
                        if item['type'] == 'text':
                            block = make_block(type="text", page_num=page_num + 1, bbox=item['bbox'])
                            lines = item['text'].split('\n')
                            for line_text in lines:
                                if not line_text.strip():
                                    continue
                                span = make_span(text=line_text)
                                line = make_line(spans=[span], bbox=item['bbox'], y0=item['bbox'][1])
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
            raw = page.get_text("dict")
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


# ---- Классификация блоков ----
REF_HEADING_RE = re.compile(
    r'^(References?|Bibliography|Библиография|Литература|Список\s+литературы|'
    r'Список\s+использованных\s+источников)\s*$', re.IGNORECASE
)

CAPTION_RE = re.compile(r'Figure|Fig\.|Рис\.|Схема|Table|Таблица', re.I)

# Улучшенные regex для списков
LIST_BULLET_RE = re.compile(r'^[\s]*[•\-\*►▸‣⁃◦○●▪]\s', re.MULTILINE)
LIST_NUM_RE = re.compile(r'^[\s]*(?:\d+[\.\)]\s|[a-z]\.\s|[ivxIVX]+\.\s)', re.MULTILINE)
LIST_MIXED_RE = re.compile(r'^[\s]*(?:[•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s)', re.MULTILINE)
LIST_LINE_RE = re.compile(r'^[\s]*([•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s|[a-z]\.\s)', re.MULTILINE)


def _count_list_lines(text: str) -> int:
    """Считает количество строк-элементов списка."""
    return len(LIST_LINE_RE.findall(text))


def _extract_features(block: Block, page_width: float, avg_font_size: float) -> List[float]:
    """Извлекает числовые признаки блока для ML-классификатора."""
    m = block_metrics(block, page_width)
    text = m.text
    bbox = m.bbox
    block_height = bbox[3] - bbox[1] if len(bbox) >= 4 else 0
    block_width = bbox[2] - bbox[0] if len(bbox) >= 4 else 0
    return [
        m.max_font,
        m.max_font / avg_font_size if avg_font_size > 0 else 1.0,
        float(m.all_upper),
        float(m.has_bold),
        float(m.is_centered),
        m.total_spans,
        len(text),
        len(text.split()),
        text.count('\n') + 1,
        1.0 if re.match(r'^\d+(\.\d+)*\s+', text) else 0.0,
        1.0 if re.match(r'^\[\d+\]', text) else 0.0,
        1.0 if REF_HEADING_RE.match(text) else 0.0,
        float(_count_list_lines(text)),
        block_height,
        block_width,
    ]


class BlockClassifier:
    """Rule-based + опциональный ML-классификатор блоков PDF."""

    LABELS = ["paragraph", "heading", "list", "metadata", "reference", "reference_heading", "empty"]

    def __init__(self):
        self._ml_model = None
        self._ml_available = False
        self._init_ml()

    def _init_ml(self):
        if not HAVE_SKLEARN:
            return
        try:
            self._ml_model = RandomForestClassifier(
                n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
            )
            self._ml_available = True
        except Exception:
            self._ml_available = False

    def train(self, features: List[List[float]], labels: List[str]):
        """Обучает ML-модель на размеченных данных."""
        if not self._ml_available or not features:
            return False
        try:
            X = np.array(features)
            y = np.array(labels)
            self._ml_model.fit(X, y)
            return True
        except Exception:
            return False

    def _ml_predict(self, features: List[float]) -> Optional[str]:
        if not self._ml_available or self._ml_model is None:
            return None
        try:
            X = np.array([features])
            return self._ml_model.predict(X)[0]
        except Exception:
            return None

    def classify(self, block: Block, page_width: float, avg_font_size: float) -> str:
        if not block.lines:
            return "empty"
        m = block_metrics(block, page_width)
        if not m.text or m.total_spans == 0:
            return "empty"

        # Hard rules — приоритетные
        if REF_HEADING_RE.match(m.text):
            return "reference_heading"
        if re.match(r'^\[\d+\]', m.text):
            return "reference"
        if re.search(r'[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.', m.text):
            return "metadata"
        if re.search(r'(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright)', m.text, re.I):
            return "metadata"

        # Списки — улучшенная детекция
        list_lines = _count_list_lines(m.text)
        if list_lines >= 2 or (list_lines >= 1 and len(m.text.split('\n')) >= 2):
            return "list"

        # ML-классификация (если доступна)
        features = _extract_features(block, page_width, avg_font_size)
        ml_result = self._ml_predict(features)
        if ml_result and ml_result in self.LABELS:
            return ml_result

        # Rule-based fallback
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


# Глобальный экземпляр классификатора
_classifier = BlockClassifier()


# ---- Таблицы и изображения ----
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
    """Извлекает изображения. Возвращает (figure_blocks, remaining_text_blocks)
    без мутации исходного списка text_blocks."""
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


# ---- Rate Limiter ----
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


# ---- Переводчики ----
class GoogleTranslator:
    def __init__(self, target_lang: str):
        self.name = "Google"
        self.target_lang = target_lang
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
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"   Google Translate: нет соединения — {e}")
            return None
        except requests.exceptions.Timeout:
            logger.warning("   Google Translate: превышено время ожидания")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning(f"   Google Translate: ошибка HTTP {e.response.status_code} — {e}")
            return None
        except Exception as e:
            logger.warning(f"   Google Translate: {type(e).__name__}: {e}")
            return None

    def generate(self, prompt: str) -> Optional[str]:
        return None


class LlamaCppTranslator:
    """Клиент для уже запущенного llama.cpp сервера. Находит подходящий сервер через pgrep."""
    def __init__(self, target_lang: str, api_base: str = "http://localhost:8080/v1",
                 expected_model: Optional[str] = None, auto_find: bool = True):
        self.target_lang = target_lang
        self.name = "LlamaCpp"
        self.expected_model = expected_model
        self.rate_limiter = RateLimiter(max_requests_per_second=1)
        self._loaded_model = None
        self._server_ok = False
        self.api_base = None

        # Пробуем указанный URL
        if api_base:
            ok, models = self._check_server(api_base)
            if ok:
                server_model = models[0] if models else "unknown"
                if expected_model and server_model != expected_model:
                    logger.warning(f"   На {api_base} загружена модель '{server_model}', а нужна '{expected_model}'. Пропускаю.")
                else:
                    self.api_base = api_base.rstrip("/")
                    self._loaded_model = server_model
                    self._server_ok = True
                    logger.info(f"   Подключено к llama-server: {self.api_base}, загружена модель: {self._loaded_model}")
        if not self._server_ok and auto_find:
            servers = self._find_all_servers()
            if not servers:
                raise RuntimeError(
                    "llama-server не найден ни по указанному URL, ни через pgrep.\n"
                    "Запустите сервер, например: ./start_llama.sh translate 8080"
                )
            if expected_model is None:
                self.api_base = servers[0]["url"]
                self._loaded_model = servers[0]["model"]
                self._server_ok = True
                logger.info(f"   Найден сервер: {self.api_base}, модель: {self._loaded_model}")
            else:
                matched = [s for s in servers if s["model"] == expected_model]
                if matched:
                    self.api_base = matched[0]["url"]
                    self._loaded_model = matched[0]["model"]
                    self._server_ok = True
                    logger.info(f"   Найден сервер с моделью '{expected_model}': {self.api_base}")
                else:
                    available = "\n".join(f"   {s['url']} -> {s['model']}" for s in servers)
                    raise RuntimeError(
                        f"Модель '{expected_model}' не найдена среди запущенных серверов.\n"
                        f"Доступные серверы:\n{available}\n"
                        "Запустите сервер с нужной моделью или укажите другую модель."
                    )

        if not self._server_ok:
            raise RuntimeError("Не удалось подключиться ни к одному серверу.")

    @staticmethod
    def _find_all_servers() -> List[Dict[str, str]]:
        """Ищет все запущенные llama-server через pgrep, возвращает список {url, model}."""
        if os.name == 'nt':
            return []
        result = []
        try:
            output = subprocess.check_output(["pgrep", "-a", "llama-server"], text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                port = None
                m = re.search(r'--port\s+(\d+)', line)
                if m:
                    port = int(m.group(1))
                else:
                    port = 8080
                url = f"http://localhost:{port}/v1"
                try:
                    resp = requests.get(f"{url}/models", timeout=2)
                    if resp.status_code == 200:
                        data = resp.json()
                        models = [m.get("id", "") for m in data.get("data", [])]
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
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                if models:
                    return True, models
        except Exception:
            pass
        return False, []

    def _call_api(self, messages: List[Dict[str, str]], temperature: float = 0.3,
                  max_tokens: int = 4096) -> Optional[str]:
        if not self._server_ok:
            logger.error("llama-server недоступен")
            return None
        self.rate_limiter.acquire()
        try:
            resp = requests.post(
                f"{self.api_base}/chat/completions",
                json={
                    "model": self._loaded_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                },
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                return content.strip() if content else None
        except Exception:
            pass
        return None

    def translate(self, text: str) -> Optional[str]:
        return self._call_api([
            {"role": "system", "content": f"Translate to {self.target_lang}. Return only translation."},
            {"role": "user", "content": text}
        ])

    def generate(self, prompt: str) -> Optional[str]:
        return self._call_api([{"role": "user", "content": prompt}])


# ---- Кэш (LRU в памяти + сохранение на диск) ----
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


# ---- Фабрика ----
def create_translator(translator_type: str, target_lang: str,
                      llama_url: str = "http://localhost:8080/v1",
                      llama_model: Optional[str] = None,
                      auto_find: bool = True):
    if translator_type == "llama":
        primary = LlamaCppTranslator(target_lang, api_base=llama_url,
                                     expected_model=llama_model, auto_find=auto_find)
        fallback = GoogleTranslator(target_lang)
        return primary, fallback
    elif translator_type == "google":
        primary = GoogleTranslator(target_lang)
        fallback = None
        try:
            fallback = LlamaCppTranslator(target_lang, api_base=llama_url,
                                          expected_model=llama_model, auto_find=auto_find)
            logger.info(f"   Fallback на llama: {fallback.api_base}")
        except Exception:
            pass
        return primary, fallback
    else:
        raise ValueError(f"Неизвестный переводчик: {translator_type}")


# ---- Pipeline перевода ----
class TranslationPipeline:
    def __init__(self, translator, fallback=None, cache: Optional[TranslationCache] = None,
                 max_workers: int = 8, timeout: int = 600, is_local: bool = False,
                 translator_type: str = "google"):
        self.translator = translator
        self.fallback = fallback
        self.cache = cache or TranslationCache()
        self.max_workers = max_workers
        self.timeout = timeout
        self.is_local = is_local
        self.translator_type = translator_type
        self.stats = {"success": 0, "failed": 0, "cached": 0, "skipped": 0}
        self._stats_lock = threading.Lock()
        self._start_time = 0.0
        self._block_times = []

    def _translate_one(self, text: str, lang: str) -> Optional[str]:
        cached = self.cache.get(text, lang)
        if cached:
            with self._stats_lock:
                self.stats["cached"] += 1
            return cached
        t0 = time.time()
        result = self.translator.translate(text)
        if not result and self.fallback:
            logger.info(f"   Переключение на {self.fallback.name}...")
            result = self.fallback.translate(text)
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

    def _get_eta(self, done: int, total: int) -> str:
        if done == 0 or not self._block_times:
            return ""
        elapsed = time.time() - self._start_time
        avg_per_block = elapsed / done
        remaining = (total - done) * avg_per_block
        return f"  ETA: {_format_time(remaining)}"

    def translate_blocks(self, blocks: list, lang: str, quiet: bool = False) -> list:
        try:
            from tqdm import tqdm
            TQDM = True
        except ImportError:
            TQDM = False

        ref_buf = []
        ref_buf_page = 0
        merged_blocks = []

        def flush_refs():
            if not ref_buf:
                return
            merged_text = " ".join(ref_buf)
            merged_text = re.sub(r'\s+', ' ', merged_text).strip()
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
            if block.type in ("figure", "table", "empty", "reference_heading"):
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

        translatable = [b for b in blocks if b.type not in ("figure", "table", "empty", "reference", "reference_heading") and b.translation is None]
        if not translatable:
            return blocks

        total = len(translatable)
        done = 0
        lock = threading.Lock()
        self._start_time = time.time()

        if not quiet:
            logger.info(f"   Блоков для перевода: {total}")

        def _work(block):
            nonlocal done
            text = block.text
            result = self._translate_one(text, lang)
            block.translation = result if result else text
            with lock:
                done += 1
                elapsed = time.time() - self._start_time
                if TQDM and not quiet:
                    avg = elapsed / done
                    remaining = avg * (total - done)
                    pbar.set_postfix_str(f"ост. {_format_time(remaining)}", refresh=True)
                    pbar.update(1)
                elif not quiet and done % 5 == 0:
                    eta = self._get_eta(done, total)
                    logger.info(f"   Переведено {done}/{total} ({done * 100 // total}%)  Прошло: {_format_time(elapsed)}{eta}")

        if TQDM and not quiet:
            pbar = tqdm(total=total, desc="🌐 Перевод", unit="блок",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_work, b): b for b in translatable}
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


# ---- Генерация реферата ----
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
    try:
        from tqdm import tqdm
        TQDM = True
    except ImportError:
        TQDM = False

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

    # Создание генератора реферата
    summary_translator = None
    try:
        summary_translator, _ = create_translator(
            summary_translator_type, "en",
            llama_url=llama_url,
            llama_model=llama_model,
            auto_find=auto_find,
        )
    except RuntimeError as e:
        logger.error(f"   Не удалось создать генератор ({summary_translator_type}): {e}")
        if summary_translator_type == "llama":
            logger.warning("   Попытка использовать Google для перевода (генерация невозможна)...")
            summary_translator = GoogleTranslator("en")
        else:
            return {
                "summary_html": "<p>Ошибка: не удалось подключиться к LLM-серверу для генерации реферата.</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": str(e)}
            }
    logger.info(f"   Генератор реферата: {summary_translator.name}")

    # Создание переводчика реферата
    final_translator = None
    if translate_translator_type == "llama":
        try:
            final_translator, _ = create_translator(
                "llama", lang,
                llama_url=llama_url,
                llama_model=llama_model,
                auto_find=auto_find,
            )
        except RuntimeError as e:
            logger.warning(f"   Llama-переводчик недоступен: {e}. Используем Google.")
            final_translator = GoogleTranslator(lang)
    else:
        final_translator = GoogleTranslator(lang)
    logger.info(f"   Переводчик реферата: {final_translator.name}")

    # Проверка: может ли генератор реально генерировать
    test_gen = summary_translator.generate("test")
    if test_gen is None:
        logger.warning(f"   {summary_translator.name} не поддерживает generate(). Попытка найти LLM...")
        try:
            summary_translator, _ = create_translator(
                "llama", "en",
                llama_url=llama_url,
                llama_model=llama_model,
                auto_find=auto_find,
            )
            logger.info(f"   Генератор реферата: {summary_translator.name}")
        except RuntimeError:
            logger.error("   Ни один генератор не поддерживает generate(). Реферат невозможен.")
            return {
                "summary_html": "<p>Ошибка: нет доступного LLM-сервера для генерации реферата. Запустите llama-server.</p>",
                "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": "no LLM available"}
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
                logger.warning(f"   Чанк {i + 1}/{len(chunks)}: пустой ответ (попытка {attempt + 1})")
            except Exception as e:
                logger.warning(f"   Чанк {i + 1}/{len(chunks)}: ошибка (попытка {attempt + 1}): {e}")
        if summary:
            cache.put(chunk, cache_key, summary)
            chunk_summaries.append(summary)
            total_tokens += len(summary) // 4
        else:
            logger.warning(f"   Чанк {i + 1}/{len(chunks)}: используется оригинальный текст")
            chunk_summaries.append(chunk)
            total_tokens += len(chunk) // 4
        if TQDM and not quiet:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (len(chunks) - i - 1)
            pbar.set_postfix_str(f"ост. {_format_time(remaining)}", refresh=True)
            pbar.update(1)
        elif not quiet:
            elapsed = time.time() - start_time
            avg = elapsed / (i + 1)
            remaining = avg * (len(chunks) - i - 1)
            logger.info(f"   Чанк {i + 1}/{len(chunks)} готов  Прошло: {_format_time(elapsed)}  Ост.: {_format_time(remaining)}")

    if TQDM and not quiet:
        pbar.close()

    if not chunk_summaries:
        return {"summary_html": "<p>Не удалось сгенерировать реферат.</p>", "stats": {"chunks": 0, "tokens": 0, "speed": 0}}

    # Иерархическая суммаризация: если промежуточных чанков больше 5 — группируем по 5
    summaries_to_combine = chunk_summaries
    if len(chunk_summaries) > 5:
        if not quiet:
            logger.info(f"   Иерархическая суммаризация ({len(chunk_summaries)} промежуточных)...")
        grouped = []
        for j in range(0, len(chunk_summaries), 5):
            group = chunk_summaries[j:j + 5]
            group_prompt = (
                "Combine these key points into a concise summary:\n" + "\n---\n".join(group)
            )
            group_summary = summary_translator.generate(group_prompt)
            if group_summary:
                cache.put("\n---\n".join(group), cache_key + ":hier", group_summary)
                grouped.append(group_summary)
                total_tokens += len(group_summary) // 4
            else:
                grouped.extend(group)
        summaries_to_combine = grouped

    final_prompt = (
        "Based on the following key points, create a final structured summary in English "
        "strictly in the format:\nBRIEF CONTENT: 4-5 sentences\nKEY FINDINGS: list\nSTRENGTHS: list\nWEAKNESSES: list\n\n"
        "Key points:\n" + "\n---\n".join(summaries_to_combine)
    )
    if not quiet:
        logger.info("📝 Генерация итогового реферата...")
    summary_en = None
    for attempt in range(2):
        try:
            summary_en = summary_translator.generate(final_prompt)
            if summary_en:
                break
            logger.warning(f"   Итоговый реферат: пустой ответ (попытка {attempt + 1})")
        except Exception as e:
            logger.warning(f"   Итоговый реферат: ошибка (попытка {attempt + 1}): {e}")
    if not summary_en:
        logger.warning("   Итоговый реферат не сгенерирован, собираем из промежуточных")
        summary_en = "\n\n".join(summaries_to_combine)

    if not quiet:
        logger.info("📝 Итоговый реферат сгенерирован")
        logger.info(f"📝 Перевод реферата на {lang}...")
    try:
        final_translated = final_translator.translate(summary_en)
        summary_html = md.markdown(final_translated if final_translated else summary_en)
    except Exception:
        summary_html = md.markdown(summary_en)

    elapsed = time.time() - start_time
    speed = total_tokens / elapsed if elapsed > 0 else 0
    actual_model = getattr(summary_translator, '_loaded_model', summary_translator.name)
    stats = {"chunks": len(chunks), "tokens": total_tokens, "speed": speed, "model": actual_model, "time": elapsed}
    if not quiet:
        if speed > 0:
            logger.info(f"   Скорость генерации: {speed:.1f} токенов/сек")
        logger.info(f"   Модель: {actual_model}")
    return {"summary_html": summary_html, "stats": stats}


# ---- Рендеринг HTML ----
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
        .page { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .block-heading { font-weight: bold; }
        .block-paragraph { line-height: 1.6; }
        .block-metadata { color: #666; font-size: 0.9em; }
        .block-reference { color: #888; font-size: 0.85em; margin-left: 16px; }
        .block-list { margin-left: 20px; line-height: 1.6; }
        .block-table { border-collapse: collapse; width: 100%; }
        .block-table td, .block-table th { border: 1px solid #ccc; padding: 6px; }
        .block-figure { text-align: center; }
        .block-figure img { max-width: 100%; }
        .block-figure figcaption { font-style: italic; }
        .translation { color: #333; }
        .original { color: #999; font-size: 0.9em; }
        details.original-block { margin: 4px 0; }
        details.original-block summary { color: #888; font-size: 0.85em; cursor: pointer; }
        details.original-block summary:hover { color: #555; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% for page in pages %}
    <div class="page">
        <h3>Страница {{ page.num }}</h3>
        {% for block in page.blocks %}
            {% if block.type == 'figure' %}
                <figure class="block-figure">
                    <img src="data:image/png;base64,{{ block.image_data }}" />
                    {% if block.caption %}<figcaption>{{ block.caption }}</figcaption>{% endif %}
                </figure>
            {% elif block.type == 'table' and block.table_data %}
                <table class="block-table">
                {% for row in block.table_data %}<tr>{% for cell in row %}<td>{{ cell }}</td>{% endfor %}</tr>{% endfor %}
                </table>
            {% elif block.type == 'heading' %}
                <div class="block-heading">{{ render_block(block) }}</div>
            {% elif block.type == 'reference_heading' %}
                <h2 class="block-heading">{{ render_block(block) }}</h2>
            {% elif block.type == 'metadata' %}
                <div class="block-metadata">{{ render_block(block) }}</div>
            {% elif block.type == 'reference' %}
                <div class="block-reference">{{ render_block(block) }}</div>
            {% elif block.type == 'list' %}
                <div class="block-list">{{ render_block(block) }}</div>
            {% else %}
                <p class="block-paragraph">{{ render_block(block) }}</p>
            {% endif %}
        {% endfor %}
    </div>
    {% endfor %}
</body>
</html>
"""
SHORT_THRESHOLD = 40


def render_block(block):
    orig = render_spans(block)
    trans = block.translation
    if not trans:
        return orig
    if len(orig.strip()) <= SHORT_THRESHOLD:
        return f'<span class="translation">{html.escape(trans)}</span>'
    return f'{html.escape(trans)}<details class="original-block"><summary>Оригинал</summary><span class="original">{html.escape(orig)}</span></details>'


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
        parts.append("\n")
    return " ".join(parts)


def generate_html(pages: List[Page], title: str, output_path: str):
    template = Template(HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, render_block=render_block)
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


# ---- Основной процесс ----
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
) -> Dict[str, Any]:
    import fitz

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
                pages,
                summary_translator_type=summary_translator_type,
                translate_translator_type=translate_translator_type,
                lang=lang,
                llama_url=llama_url,
                llama_model=llama_model,
                auto_find=auto_find,
                quiet=quiet,
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
        logger.info(f"🌐[5/5] Перевод на {lang} ({translator_type})...")
        try:
            translator, fallback = create_translator(
                translator_type, lang,
                llama_url=llama_url,
                llama_model=llama_model,
                auto_find=auto_find,
            )
        except RuntimeError as e:
            logger.error(f"   Ошибка создания переводчика: {e}")
            logger.info("   Fallback на Google Translate...")
            translator = GoogleTranslator(lang)
            fallback = None

        logger.info(f"   Переводчик: {translator.name}")
        if fallback:
            logger.info(f"   Fallback: {fallback.name}")

        # Динамическое число воркеров для локального сервера
        effective_workers = max_workers
        if translator_type == "llama":
            effective_workers = min(max_workers, 3)
            if not quiet:
                logger.info(f"   Локальный сервер: ограничено до {effective_workers} воркеров")

        is_local = translator_type == "llama"
        pipeline = TranslationPipeline(
            translator=translator,
            fallback=fallback,
            max_workers=effective_workers,
            timeout=timeout,
            is_local=is_local,
            translator_type=translator_type,
        )

        all_blocks = []
        for page in pages:
            all_blocks.extend(page.blocks)

        pipeline.translate_blocks(all_blocks, lang, quiet=quiet)

    with stage_timer("5. HTML", timings):
        generate_html(pages, f"Перевод: {os.path.basename(pdf_path)}", output_html)
        logger.info(f"   HTML сохранён: {output_html}")

    return {
        "stats": {**pipeline.stats, "timings": timings},
        "pages": len(pages),
        "images": total_images,
        "tables": total_tables,
    }


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="PDF Translator — единый скрипт")
    parser.add_argument("input", help="Входной PDF файл")
    parser.add_argument("-l", "--lang", default="ru", help="Целевой язык (по умолчанию: ru)")
    parser.add_argument("-t", "--translator", default="google",
                        choices=["google", "llama"], help="Переводчик (по умолчанию: google)")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1", help="URL llama.cpp сервера")
    parser.add_argument("--llama-model", help="Ожидаемое имя модели (точное совпадение)")
    parser.add_argument("--no-auto-find", action="store_true", help="Отключить автоматический поиск сервера через pgrep")
    parser.add_argument("--summary", action="store_true", help="Режим реферата")
    parser.add_argument("--summary-translator", choices=["google", "llama"], default="llama",
                        help="Генератор реферата (по умолчанию: llama)")
    parser.add_argument("--summary-lang-translator", choices=["google", "llama"], default="google",
                        help="Переводчик реферата (по умолчанию: google)")
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1) * 2),
                        help="Число воркеров")
    parser.add_argument("--task-timeout", type=int, default=600, help="Таймаут перевода (сек)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Тихий режим")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

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
