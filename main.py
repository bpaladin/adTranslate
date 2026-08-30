#!/usr/bin/env python3
"""
PDF Translator — единый скрипт для перевода и реферирования PDF-документов.
Поддерживает Google Translate, OpenRouter и локальный llama.cpp.
"""

import os
import sys
import re
import string
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
import concurrent.futures
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, TimeoutError

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

__version__ = "1.1.1-fixes"

# ---- Логирование с цветом ----
class ColorFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[36;20m"
    green = "\x1b[32;20m"
    reset = "\x1b[0m"

    def __init__(self, fmt):
        super().__init__()
        self.fmt = fmt
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)

if os.name == 'nt':
    os.system('')  # Включаем поддержку ANSI в консоли Windows

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

STOP_EVENT = threading.Event()
_watchdog_started = False

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

DEFAULT_SETTINGS = {
    "block_timeout": 900,
    "request_timeout": 600,
    "api_timeout": 120,
    "model_cooldown_sec": 60,
    "llama_max_retries": 3,
    "llm_chunk_chars": 1800,
    "llm_min_chunk_chars": 650,
    "llm_rescue_min_chars": 220,
}

def load_settings(path: str = "settings.json") -> dict:
    p = Path(path)
    if p.exists():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            merged = dict(DEFAULT_SETTINGS)
            merged.update({k: v for k, v in data.items() if k in DEFAULT_SETTINGS})
            return merged
        except Exception as e:
            logger.warning(f"Ошибка чтения {p.name}: {e} — используются умолчания")
    try:
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_SETTINGS, f, indent=2, ensure_ascii=False)
        logger.info(f"Создан файл настроек: {p.name}")
    except Exception as e:
        logger.debug(f"Не удалось создать {p.name}: {e}")
    return dict(DEFAULT_SETTINGS)

SETTINGS = load_settings()
BLOCK_TIMEOUT = int(SETTINGS["block_timeout"])
REQUEST_TIMEOUT = max(600, int(SETTINGS["request_timeout"]))
API_TIMEOUT = SETTINGS["api_timeout"]
MODEL_COOLDOWN_SEC = SETTINGS["model_cooldown_sec"]
LLAMA_MAX_RETRIES = SETTINGS["llama_max_retries"]

OPENAI_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://sprutdock.ru/v1")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "z-ai/glm-5.2:free")
SPRUTDOCK_API_KEY = os.getenv("SPRUTDOCK_API_KEY") or OPENAI_API_KEY

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

RE_WHITESPACE = re.compile(r'[ \t]+')
RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')

_REF_HEADING_ALT = (
    r'References?|Reference\s+list|References?\s+and\s+notes?|'
    r'Bibliography|Библиография|Библиографический\s+список|'
    r'Литература|Список\s+литературы|Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|Список\s+источников|'
    r'Использованная\s+литература|Источники'
)
REF_HEADING_RE = re.compile(rf'^\s*({_REF_HEADING_ALT})\s*[:.]?\s*$', re.IGNORECASE)
REF_HEADING_PREFIX_RE = re.compile(rf'^\s*({_REF_HEADING_ALT})\s*[:.]?\s+(.+)$', re.IGNORECASE)
BACK_MATTER_RE = re.compile(
    r'^\s*(Acknowledg(?:ement|ment)s?\b|Author\s+contributions?\b|Funding\b|'
    r'Competing\s+interests?\b|Additional\s+information\b|Supplementary\s+information\b|'
    r'Supplementary\s+notes?\b|Correspondence\b|Reprints?\s+and\s+permissions?\b|'
    r'Publisher\s*\'?s?\s+note\b|Open\s+Access\b|Conflict(?:s)?\s+of\s+interest\b|'
    r'Ethics\s+declarations?\b|Ethical\s+approval\b|Consent\s+to\s+participate\b|'
    r'Consent\s+for\s+publication\b|Data\s+availability\b|Code\s+availability\b|'
    r'Availability\s+of\s+data\s+and\s+materials\b|Declarations?\b|Received[:.]?\s+\d|'
    r'Presented\s+at\b)', re.IGNORECASE
)
_CIT_RUN_RE = re.compile(r'^\d{1,2}(?:\s*[,–—-]\s*\d{1,2})*\s*[,.]?$')
_REF_LABEL_RE = re.compile(
    r'\b(?:Fig(?:ure)?\.?|Table|Tab\.?|Vol(?:ume)?\.?|No\.?|Section|Chapter|Ch\.?|'
    r'Equation|Eq\.?|Ref\.?|Page|pp\.?|Appendix|Supplementary|Suppl\.?)\s*$'
)
TABLE_CAPTION_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+', re.I)
TABLE_CAPTION_STRICT_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+\s*[|:.–—-]', re.I)
CAPTION_RE = re.compile(r'Figure|Fig\.|Рис\.|Схема|Table|Таблица', re.I)
FIGURE_CAPTION_RE = re.compile(r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Рис\.?\s*\d+|Схема\s*\d+)', re.I)
FIGURE_CAPTION_STRICT_RE = re.compile(r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Рис\.?\s*\d+|Схема\s*\d+)\s*[|:.,–—-]', re.I)
CAPTION_DIST_THRESHOLD = 120.0
LIST_LINE_RE = re.compile(r'^[\s]*([•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s|[a-z]\.\s)', re.MULTILINE)

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
    text = re.sub(r'(?<=\w)-\s+(?=[a-zа-яё])', '-', text)
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

def _interruptible_sleep(delay: float, step: float = 0.5) -> None:
    remaining = delay
    while remaining > 0 and not STOP_EVENT.is_set():
        time.sleep(min(step, remaining))
        remaining -= step

def _start_keyboard_watchdog():
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    if STOP_EVENT.is_set():
        return
    if os.name == 'nt':
        try:
            import msvcrt
        except ImportError:
            return

        def _watch():
            while not STOP_EVENT.is_set():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('q', 'Q', '\x1b', '\x03'):
                            logger.warning("⏹ Прерывание по клавише, остановка...")
                            STOP_EVENT.set()
                    else:
                        time.sleep(0.1)
                except Exception:
                    break
        threading.Thread(target=_watch, daemon=True).start()

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
    image_ext: Optional[str] = None
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

def make_span(text: str, font: str = "", size: float = 12, flags: int = 0,
              color: int = 0, origin: tuple = (0, 0), bbox: tuple = (0, 0, 0, 0)) -> Span:
    return Span(text=text, font=font, size=size, flags=flags, color=color, origin=origin, bbox=bbox)

def make_line(spans: Optional[List[Span]] = None, bbox: tuple = (0, 0, 0, 0), y0: float = 0.0) -> Line:
    return Line(spans=spans or [], bbox=bbox, y0=y0)

def make_block(type: str, lines: Optional[List[Line]] = None, bbox: tuple = (0, 0, 0, 0),
               page_num: int = 0, **kwargs) -> Block:
    b = Block(type=type, lines=lines or [], bbox=bbox, page_num=page_num)
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b

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
    return BlockMetrics(text=text, max_font=max_font, all_upper=all_upper, has_bold=has_bold, is_centered=is_centered, total_spans=total_spans, bbox=bbox)

class BlockClassifier:
    LABELS = ["paragraph", "heading", "list", "metadata", "reference", "reference_heading", "empty"]
    METADATA_WORD_LIMIT = 8

    def classify(self, block: Block, page_width: float, avg_font_size: float) -> str:
        if not block.lines:
            return "empty"
        m = block_metrics(block, page_width)
        if not m.text or m.total_spans == 0:
            return "empty"
        if not re.search(r'[A-Za-zА-Яа-я]', m.text):
            return "empty"

        # Фильтрация технического мусора типа "g()" или строк из одних символов
        if re.fullmatch(r'\s*(?:[a-zA-Z_]+\(\)|[\W\d]+)\s*', m.text):
            return "empty"

        word_count = len(m.text.split())
        if REF_HEADING_RE.match(m.text):
            return "reference_heading"
        if _is_table_caption(m.text):
            return "table"
        if word_count <= self.METADATA_WORD_LIMIT:
            if re.match(r'^\[\d+\]', m.text):
                return "reference"
            if re.search(r'[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.', m.text):
                return "metadata"
            if re.search(r'(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright)', m.text, re.I):
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
                    pages.append(Page(num=page_num + 1, blocks=page_blocks, width=page_data.width, height=page_data.height))
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
                    block = make_block(type="text", page_num=page_num + 1, bbox=b.get("bbox", (0, 0,0,0)))
                    for line_dict in b.get("lines", []):
                        bbox = line_dict.get("bbox", (0, 0, 0, 0))
                        line = make_line(bbox=bbox, y0=bbox[1])
                        for span_dict in line_dict.get("spans", []):
                            text = span_dict.get("text", "")
                            if not text.strip():
                                continue
                            span = make_span(text=text, font=span_dict.get("font", ""), size=span_dict.get("size", 12), flags=span_dict.get("flags", 0), color=span_dict.get("color", 0), origin=span_dict.get("origin", (0, 0)), bbox=span_dict.get("bbox", (0, 0, 0, 0)))
                            line.spans.append(span)
                        if line.spans:
                            block.lines.append(line)
                    if block.lines:
                        page_blocks.append(block)
            pages.append(Page(num=page_num + 1, blocks=page_blocks, width=raw.get("width", 612), height=raw.get("height", 792)))
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
            block = make_block(type="table", page_num=page_num, bbox=tuple(tab.bbox), table_data=cleaned)
            table_blocks.append(block)
        except Exception:
            continue
    return table_blocks

def _row_columns(block: Block, tol: float = 8.0) -> Optional[List[float]]:
    if not block.lines:
        return None
    if len(block.lines) == 1:
        xs = [s.bbox[0] for s in block.lines[0].spans if s.text.strip()]
        if len(xs) < 2:
            return None
        gaps = [b - a for a, b in zip(sorted(xs), sorted(xs)[1:])]
        if gaps and min(gaps) < 5:
            return None
    else:
        xs = [ln.bbox[0] for ln in block.lines if ln.spans]
        if len(xs) < 2:
            return None
    clusters = []
    for x in sorted(xs):
        if clusters and x - clusters[-1][-1] <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    if len(clusters) < 2:
        return None
    return [sum(c) / len(c) for c in clusters]

def _columns_align(ref: List[float], cols: List[float], tol: float = 10.0, min_frac: float = 0.5) -> bool:
    if not ref or not cols:
        return False
    matched = 0
    for x in cols:
        if any(abs(x - y) <= tol for y in ref):
            matched += 1
    return matched / len(cols) >= min_frac

def _build_table_row(block: Block, columns: List[float]) -> List[str]:
    cells = [""] * len(columns)
    def _cell_text(u) -> str:
        if isinstance(u, Span):
            return u.text.strip()
        return " ".join(s.text for s in u.spans if s.text.strip()).strip()

    def _cell_x(u) -> float:
        if isinstance(u, Span):
            return u.bbox[0]
        return u.bbox[0] if u.spans else 0.0

    units = block.lines[0].spans if len(block.lines) == 1 else block.lines
    for u in units:
        text = _cell_text(u)
        if not text:
            continue
        k = min(range(len(columns)), key=lambda kk: abs(_cell_x(u) - columns[kk]))
        if cells[k]:
            cells[k] += " " + text
        else:
            cells[k] = text
    return cells

def _merge_span_table(caption: Block, row_blocks: List[Block]) -> Block:
    columns = []
    for rb in row_blocks:
        cols = _row_columns(rb) or []
        for c in cols:
            if not any(abs(c - x) <= 10 for x in columns):
                columns.append(c)
    columns.sort()
    data = [_build_table_row(rb, columns) for rb in row_blocks]
    xs = [caption.bbox[0], *[rb.bbox[0] for rb in row_blocks]]
    ys = [caption.bbox[1], *[rb.bbox[1] for rb in row_blocks]]
    xe = [caption.bbox[2], *[rb.bbox[2] for rb in row_blocks]]
    ye = [caption.bbox[3], *[rb.bbox[3] for rb in row_blocks]]
    bbox = (min(xs), min(ys), max(xe), max(ye))
    block = make_block(type="table", page_num=caption.page_num, bbox=bbox, lines=list(caption.lines), caption=caption.text, table_data=data)
    for rb in row_blocks:
        block.lines.extend(rb.lines)
    return block

def _find_span_table_blocks(blocks: List[Block]) -> Tuple[List[Block], List[Block]]:
    table_blocks = []
    merged_list = []
    i = 0
    n = len(blocks)
    while i < n:
        b = blocks[i]
        if b.type == "text" and _is_table_caption(b.text):
            rows = []
            ref = None
            j = i + 1
            while j < n:
                nb = blocks[j]
                if nb.type != "text":
                    break
                cols = _row_columns(nb)
                if cols is None:
                    break
                if ref is None:
                    ref = cols
                elif not _columns_align(ref, cols):
                    break
                rows.append(nb)
                j += 1
            if rows:
                merged = _merge_span_table(b, rows)
                table_blocks.append(merged)
                merged_list.append(merged)
                i = j
                continue
        merged_list.append(b)
        i += 1
    return table_blocks, merged_list

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

# ИСПРАВЛЕНИЕ 2: Улучшенная детекция таблиц (снижены пороги)
def _heuristic_table_data(text: str) -> Optional[List[List[str]]]:
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if len(lines) < 2:
        return None
    rows = []
    for ln in lines:
        cells = [c.strip() for c in re.split(r'\s{2,}|\t', ln) if c.strip()]
        if len(cells) < 2:
            return None
        rows.append(cells)
    col_counts = {len(r) for r in rows}
    if len(col_counts) != 1:
        return None
    return rows

def _looks_like_table_data(text: str) -> bool:
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if len(lines) < 2:
        return False
    if any(re.search(r'[.!?]\s*$', ln) for ln in lines):
        return False
    col_counts = {len(re.split(r'\s{2,}|\t', ln)) for ln in lines}
    if len(col_counts) == 1 and next(iter(col_counts)) >= 2:
        return True
    if all(len(ln) < 60 for ln in lines):
        return True
    return False

def consolidate_tables(blocks: List[Block], found_table_blocks: Optional[List[Block]] = None) -> List[Block]:
    if found_table_blocks:
        for block in blocks:
            if block.type == "table":
                continue
            for table in found_table_blocks:
                if intersection_ratio(block.bbox, table.bbox) > 0.30:
                    block.type = "table"
                    break
    for b in blocks:
        if b.type in ("table", "figure"):
            continue
        rows = _heuristic_table_data(b.text)
        if rows:
            b.type = "table"
            b.table_data = rows
    i = 0
    n = len(blocks)
    while i < n:
        b = blocks[i]
        if b is None or b.type not in ("text", "heading", "table"):
            i += 1
            continue
        if b.table_data:
            i += 1
            continue
        if not _is_table_caption(b.text):
            i += 1
            continue
        j = i + 1
        nb = blocks[j] if j < n else None
        if nb is not None and nb.type == "table" and nb.table_data:
            nb.caption = b.text
            blocks[i] = None
            i = j
            continue
        if b.type != "table":
            b.type = "table"
        rows = []
        ref = None
        j = i + 1
        while j < n:
            nb = blocks[j]
            if nb is None or nb.type not in ("text", "paragraph", "list"):
                break
            cols = _row_columns(nb)
            if cols is not None and (ref is None or _columns_align(ref, cols)):
                if ref is None:
                    ref = cols
                rows.append(nb)
                j += 1
                continue
            if _heuristic_table_data(nb.text) is not None or _looks_like_table_data(nb.text):
                rows.append(nb)
                j += 1
            else:
                break
        if rows:
            merged = _merge_span_table(b, rows)
            b.bbox = merged.bbox
            b.lines = merged.lines
            b.caption = merged.caption
            b.table_data = merged.table_data
        i = j
    return [b for b in blocks if b is not None and b.type != "empty" and not (b.type == "table" and b.table_data is None and not b.text.strip())]

def _is_table_caption(text: str) -> bool:
    if not text:
        return False
    if TABLE_CAPTION_STRICT_RE.match(text):
        return True
    if not TABLE_CAPTION_RE.match(text):
        return False
    return len(text) <= 120 and len(text.split()) <= 15

def _is_figure_caption(text: str) -> bool:
    if not text:
        return False
    if not FIGURE_CAPTION_RE.match(text):
        return False
    if FIGURE_CAPTION_STRICT_RE.match(text):
        return True
    rest = FIGURE_CAPTION_RE.sub('', text, count=1).lstrip()
    if not rest:
        return False
    if len(text) > 160:
        return False
    if re.search(r'\b(?:illustrates?|shows?|depicts?|displays?|demonstrates?|presents?|изобража\w*|показыва\w*|иллюстрир\w*)\b', text, re.I):
        return False
    return True

def _rect_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)

def _union_rects(rects) -> fitz.Rect:
    return fitz.Rect(min(r.x0 for r in rects), min(r.y0 for r in rects), max(r.x1 for r in rects), max(r.y1 for r in rects))

def _cluster_drawing_rects(rects: List[fitz.Rect], gap: float = 6.0) -> List[List[fitz.Rect]]:
    clusters = []
    for r in sorted(rects, key=lambda x: (x.y0, x.x0)):
        placed = False
        for cl in clusters:
            if _rect_gap(_union_rects(cl), (r.x0, r.y0, r.x1, r.y1)) <= gap:
                cl.append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])
    return clusters

def _overlaps_any(rect: fitz.Rect, others, ratio: float = 0.4) -> bool:
    a = rect.get_area() or 1.0
    for o in others:
        inter = fitz.Rect(rect) & fitz.Rect(o)
        if not inter.is_empty and inter.get_area() / a > ratio:
            return True
    return False

def _nearest_caption(rect, candidates, caption_ids) -> Tuple[Optional[str], int]:
    best_dist = CAPTION_DIST_THRESHOLD
    caption = None
    cap_idx = -1
    for idx, block in candidates:
        if idx in caption_ids:
            continue
        d = _rect_gap(rect, block.bbox)
        if d < best_dist:
            best_dist = d
            caption = block.text
            cap_idx = idx
    return caption, cap_idx

def extract_images(page: fitz.Page, page_num: int, text_blocks: List[Block]) -> Tuple[List[Block], List[Block]]:
    image_list = page.get_images(full=True)
    figure_blocks = []
    caption_ids = set()
    candidates = [(i, b) for i, b in enumerate(text_blocks) if _is_figure_caption(b.text)]
    for img in image_list:
        xref = img[0]
        try:
            info = page.parent.extract_image(xref)
        except Exception as e:
            logger.warning(f"   Изобр. xref={xref} (стр. {page_num}): {e}")
            continue
        data = info.get("image")
        ext = info.get("ext", "png")
        if not data:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        img_bbox = fitz.Rect(rects[0])
        if img_bbox.get_area() < 300:
            continue
        image_data = base64.b64encode(data).decode()
        caption, cap_idx = _nearest_caption(img_bbox, candidates, caption_ids)
        if cap_idx >= 0:
            caption_ids.add(cap_idx)
        fig_block = make_block(type="figure", page_num=page_num, bbox=tuple(img_bbox), image_data=image_data, image_ext=ext, caption=caption)
        figure_blocks.append(fig_block)
    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining

def extract_vector_figures(page: fitz.Page, page_num: int, text_blocks: List[Block], used_bboxes=(), table_bboxes=(), dpi: int = 150) -> Tuple[List[Block], List[Block]]:
    draw_rects = []
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        if r.width < 2 or r.height < 2:
            continue
        if r.width > page.rect.width * 0.95 and r.height > page.rect.height * 0.95:
            continue
        draw_rects.append(r)
    if not draw_rects:
        return [], text_blocks
    clusters = _cluster_drawing_rects(draw_rects, gap=6)
    table_rects = [fitz.Rect(b) for b in table_bboxes]
    used_rects = [fitz.Rect(b) for b in used_bboxes]
    page_w, page_h = page.rect.width, page.rect.height
    candidates = [(i, b) for i, b in enumerate(text_blocks) if _is_figure_caption(b.text)]
    figure_blocks = []
    caption_ids = set()
    for cl in clusters:
        rect = _union_rects(cl)
        if rect.width < 30 or rect.height < 20 or rect.get_area() < 1500:
            continue
        if rect.width > page_w * 0.6 and rect.height < page_h * 0.15:
            continue
        if _overlaps_any(rect, table_rects, 0.4) or _overlaps_any(rect, used_rects, 0.4):
            continue
        caption, cap_idx = _nearest_caption(rect, candidates, caption_ids)
        if caption is None:
            continue
        caption_ids.add(cap_idx)

        image_data = None
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False, colorspace=fitz.csRGB)
            image_data = base64.b64encode(pix.tobytes("png")).decode()
        except Exception as e1:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False)
                image_data = base64.b64encode(pix.tobytes("png")).decode()
            except Exception as e2:
                logger.warning(f"   Рендер векторной фигуры (стр. {page_num}): {e2}")
                continue

        figure_blocks.append(make_block(type="figure", page_num=page_num, bbox=tuple(rect), image_data=image_data, image_ext="png", caption=caption))
    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining

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

SYSTEM_TRANSLATE_PROMPT = (
    "You are a professional scientific translator. "
    "Translate ONLY the TARGET text from {source_lang} into {lang}. "
    "Do not summarize, explain, comment, or add information. "
    "Do not reproduce the source text. "
    "Do not create headings, labels, quotes, Markdown fences, or alternative translations. "
    "Preserve paragraph boundaries. "
    "Preserve every protected token exactly as written. "
    "Return ONLY the translation of TARGET."
)

TABLE_TRANSLATE_PROMPT = (
    "Translate ONLY the text inside table cells into {lang}. "
    "Preserve the exact number of rows and columns. "
    "Preserve numbers, units, abbreviations and protected tokens. "
    "Return ONLY the table."
)

_CYRILLIC_LANGS = {"ru", "uk", "bg", "sr", "be", "mk", "kk", "uz", "ky", "mn"}
_CYRILLIC_RE = re.compile(r'[а-яё]', re.I)
_LANG_NAMES = {
    "ru": "Russian", "en": "English", "de": "German", "fr": "French",
    "es": "Spanish", "it": "Italian", "zh": "Chinese", "ja": "Japanese",
    "ko": "Korean", "pt": "Portuguese", "pl": "Polish", "tr": "Turkish",
    "ar": "Arabic", "hi": "Hindi", "uk": "Ukrainian", "bg": "Bulgarian",
}

def _translation_system_prompt(source_lang: str, lang: str, glossary: Optional[Dict[str, str]] = None) -> str:
    prompt = SYSTEM_TRANSLATE_PROMPT.format(source_lang=source_lang or "the source language", lang=_LANG_NAMES.get(lang, lang))
    if glossary:
        terms = "\n".join(f"- {k} => {v}" for k, v in glossary.items())
        prompt += "\n\nTerminology memory. Use these translations when the corresponding source term occurs. Do not output this list:\n" + terms
    return prompt

def _strip_llm_wrappers(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^\s*```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.I)
    marker_line = re.compile(r"^\s*<<<\s*(?:END_)?[A-Z_][A-Z0-9_]*\s*>>>\s*$", re.I)
    lines = text.splitlines()
    while lines and marker_line.match(lines[0]):
        lines.pop(0)
    while lines and marker_line.match(lines[-1]):
        lines.pop()
    text = "\n".join(lines).strip()
    text = re.sub(r"<<<\s*(?:END_)?[A-Z_][A-Z0-9_]*\s*>>>", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    lines = text.splitlines()
    if lines:
        first = lines[0].strip().lower()
        preambles = ("translation:", "translated text:", "here is the translation:", "here's the translation:", "перевод:", "вот перевод:")
        if first in preambles:
            lines = lines[1:]
    return "\n".join(lines).strip()

def _norm_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()

def _extract_numbers(text: str) -> list:
    pattern = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?|\.\d+)(?:\s*[%‰])?(?!\w)")
    return pattern.findall(text or "")

# ИСПРАВЛЕНИЕ 1: Защита академических цитат и HTML-ссылок целиком
def _extract_protected_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    protected: Dict[str, str] = {}
    counter = 0

    def add_value(value: str, prefix: str = "P") -> str:
        nonlocal counter
        if not value or not value.strip():
            return value
        key = f"[[{prefix}_{counter:04d}]]"
        counter += 1
        protected[key] = value
        return f" {key} "

    result = text

    # Защита HTML ссылок <a href="...">...</a>
    html_link_pattern = re.compile(r"<a\s+[^>]*href=['\"][^'\"]+['\"][^>]*>.*?</a>", re.I | re.DOTALL)
    for m in html_link_pattern.finditer(result):
        value = m.group(0).strip()
        result = result[:m.start()] + add_value(value, "LINK") + result[m.end():]

    # Защита атрибутов href="..."
    href_pattern = re.compile(r"href=['\"][^'\"]+['\"]", re.I)
    for m in href_pattern.finditer(result):
        value = m.group(0).strip()
        result = result[:m.start()] + add_value(value, "LINK") + result[m.end():]

    # Защита всей академической цитаты (Smith et al., 2015) -> [[CITE_0001]]
    surname = r"[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+"
    citation_pattern = re.compile(
        rf"\([^\n(){{}}]{{0,220}}(?:19|20)\d{{2}}[^\n(){{}}]{{0,40}}\)"
        rf"|\b{surname}(?:\s+et\s+al\.?)?\s*\((?:19|20)\d{{2}}[^)]*\)",
        re.UNICODE | re.I,
    )
    spans = []
    for cm in citation_pattern.finditer(result):
        value = cm.group(0).strip()
        spans.append((cm.start(), cm.end(), value))
    for a, b, value in sorted(set(spans), reverse=True):
        result = result[:a] + add_value(value, "CITE") + result[b:]

    patterns = [
        re.compile(r"\[\[REF\d+\]\]"),
        REF_PATTERN,
        re.compile(r"https?://[^\s<>\]\)]+", re.I),
        re.compile(r"\b(?:doi:\s*)?10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I),
        re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9_]*(?:\s*[=<>≤≥±]\s*[^,.;\n]+)?"),
    ]

    def add(match):
        value = match.group(0)
        if not value.strip():
            return value
        if re.fullmatch(r"[A-Za-z]{1,20}", value):
            return value
        return add_value(value)

    for pat in patterns:
        result = pat.sub(add, result)
    return result, protected

def _restore_protected(text: str, protected: Dict[str, str]) -> str:
    result = text or ""
    for token, original in protected.items():
        pattern = re.escape(token).replace(r"\[\[", r"\[\[\s*").replace(r"\]\]", r"\s*\]\]")
        result = re.sub(pattern, lambda m: original, result, flags=re.I | re.DOTALL)
    result = re.sub(r"\[\[\s*(?:AUTHOR|P|REF|CITE|LINK)_\d+\s*\]\]", "", result, flags=re.I | re.DOTALL)
    return result

def _repetition_ratio(text: str, n: int = 6) -> float:
    words = re.findall(r"\w+", (text or "").casefold(), flags=re.UNICODE)
    if len(words) < n * 2:
        return 0.0
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    counts = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(1, len(grams))

def _looks_like_meta_response(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    lines = [x.strip().casefold() for x in t.splitlines() if x.strip()]
    if not lines:
        return False
    bad_prefixes = ("analysis:", "reasoning:", "chain of thought:", "thoughts:", "translation note:", "translator note:", "here is the translation:", "вот перевод:", "анализ:", "рассуждение:", "объяснение:", "я не могу перевести", "не могу перевести")
    if any(line.startswith(bad_prefixes) for line in lines[:2]):
        return True
    meta = re.search(r"\b(i cannot|i can't|i am unable|as an ai|as a language model)\b", t, re.I)
    return bool(meta)

def validate_translation(source: str, candidate: str, target_lang: str, protected: Optional[Dict[str, str]] = None) -> Tuple[bool, Dict[str, Any]]:
    source = (source or "").strip()
    candidate = _strip_llm_wrappers(candidate)
    info: Dict[str, Any] = {"echo": False, "too_short": False, "too_long": False, "repetition": False, "numbers_changed": False, "protected_missing": [], "language_mismatch": False, "meta_response": False}
    if not candidate:
        return False, info
    info["meta_response"] = _looks_like_meta_response(candidate)

    src_norm = _norm_for_compare(source)
    cand_norm = _norm_for_compare(candidate)

    if len(source) >= 20:
        if cand_norm == src_norm:
            info["echo"] = True
        elif src_norm in cand_norm and len(src_norm) > 100 and len(src_norm) > len(cand_norm) * 0.8:
            info["echo"] = True

    if len(source) >= 150 and len(candidate) < max(30, int(len(source) * 0.15)):
        info["too_short"] = True
    if len(source) >= 100 and len(candidate) > int(len(source) * 3.2):
        info["too_long"] = True

    if _repetition_ratio(candidate) > 0.30:
        info["repetition"] = True

    def _norm_num(x: str):
        x = x.strip().replace(" ", "").replace("−", "-")
        x = re.sub(r"[%‰]$", "", x)
        try:
            return ("num", float(x.replace(",", ".")))
        except ValueError:
            try:
                return ("num", float(x.replace(",", "")))
            except ValueError:
                return ("txt", x.casefold())

    src_nums = sorted(_norm_num(x) for x in _extract_numbers(source))
    cand_nums = sorted(_norm_num(x) for x in _extract_numbers(candidate))
    if src_nums != cand_nums:
        info["numbers_changed"] = True

    if protected:
        info["protected_missing"] = [token for token in protected if token not in candidate]

    if target_lang in _CYRILLIC_LANGS and len(candidate) > 30:
        letters = [c for c in candidate if c.isalpha()]
        cyr = sum(1 for c in letters if _CYRILLIC_RE.match(c))
        if letters and cyr / len(letters) < 0.12:
            info["language_mismatch"] = True

    hard = bool(info["echo"] or info["too_short"] or info["too_long"] or info["meta_response"] or not candidate)
    return (not hard), info

REF_PATTERN = re.compile(r'\[(?:[0-9][0-9,\s;\-–—]*|[A-Za-z][A-Za-z0-9-]*)\]')
_REF_PH_RE = re.compile(r'(?<![A-Za-z0-9_])\[?\[?REF_?(\d+)\]?\]?(?![A-Za-z0-9_])')

def protect_refs(text: str) -> Tuple[str, Dict[str, str]]:
    refs: Dict[str, str] = {}
    def repl(m) -> str:
        key = f"[[REF{len(refs)}]]"
        refs[key] = m.group(0)
        return f" {key} "
    return REF_PATTERN.sub(repl, text), refs

def restore_refs(text: str, refs: Dict[str, str]) -> str:
    if not refs:
        return text
    for ph, orig in refs.items():
        text = text.replace(ph, orig)
    by_index = {i: orig for i, orig in enumerate(refs.values())}
    def rep(m) -> str:
        idx = int(m.group(1))
        return by_index.get(idx, m.group(0))
    return _REF_PH_RE.sub(rep, text)

def highlight_refs(escaped_text: str) -> str:
    return REF_PATTERN.sub(lambda m: f'<span class="ref-link">{m.group(0)}</span>', escaped_text)

def _wrap_citation_spans(block: Block) -> None:
    for line in block.lines:
        spans = line.spans
        if not spans:
            continue
        new_spans = []
        i = 0
        n = len(spans)
        while i < n:
            s = spans[i]
            if (s.flags & 1) and s.text.strip():
                j = i
                run_parts = []
                while j < n and spans[j].flags & 1:
                    run_parts.append(spans[j].text)
                    j += 1
                run_text = "".join(run_parts)
                if _CIT_RUN_RE.match(run_text):
                    prev_text = "".join(x.text for x in spans[:i])
                    if not _REF_LABEL_RE.search(prev_text):
                        cit = re.sub(r'\s*,\s*', ', ', re.sub(r'\s*[–—-]\s*', '–', run_text.strip('., ')))
                        cit = cit.rstrip('., ')
                        new_spans.append(make_span(text="[" + cit + "]", font=s.font, size=s.size, flags=s.flags, color=s.color, origin=s.origin, bbox=s.bbox))
                        i = j
                        continue
                new_spans.extend(spans[i:j])
                i = j
            else:
                new_spans.append(s)
                i += 1
        line.spans = new_spans

def _split_line(line: Line, idx: int) -> Tuple[Line, Line]:
    head_spans: List[Span] = []
    rest_spans: List[Span] = []
    pos = 0
    for s in line.spans:
        t = s.text
        if pos + len(t) <= idx:
            head_spans.append(s)
        elif pos >= idx:
            rest_spans.append(s)
        else:
            cut = idx - pos
            head_spans.append(make_span(text=t[:cut], font=s.font, size=s.size, flags=s.flags, color=s.color, origin=s.origin, bbox=s.bbox))
            rest_spans.append(make_span(text=t[cut:], font=s.font, size=s.size, flags=s.flags, color=s.color, origin=s.origin, bbox=s.bbox))
        pos += len(t)
    return (make_line(spans=head_spans, bbox=line.bbox, y0=line.y0), make_line(spans=rest_spans, bbox=line.bbox, y0=line.y0))

def _split_ref_heading(block: Block) -> List[Block]:
    if not block.lines:
        return [block]
    lines = block.lines
    first = lines[0]
    first_text = " ".join(s.text for s in first.spans).strip()
    if REF_HEADING_RE.match(first_text):
        if len(lines) > 1:
            head_block = make_block(type="reference_heading", page_num=block.page_num, bbox=block.bbox, lines=lines[:1])
            rest_block = make_block(type="reference", page_num=block.page_num, bbox=block.bbox, lines=lines[1:])
            return [head_block, rest_block]
        return [block]
    m = REF_HEADING_PREFIX_RE.match(first_text)
    if m and m.group(2) and len(first_text) > len(m.group(1)):
        idx = first_text.find(m.group(1).strip()) + len(m.group(1).strip())
        head_line, rest_line = _split_line(first, idx)
        head_block = make_block(type="reference_heading", page_num=block.page_num, bbox=block.bbox, lines=[head_line])
        rest_block = make_block(type="reference", page_num=block.page_num, bbox=block.bbox, lines=[rest_line] + lines[1:])
        return [head_block, rest_block]
    return [block]

class GoogleTranslator:
    def __init__(self, target_lang: str):
        self.name = "Google"
        self.target_lang = target_lang
        self.can_generate = False
        self.session = _make_session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0"})
        self.rate_limiter = RateLimiter(max_requests_per_second=0.5)
        self._rate_limit_count = 0
        self._429_max_retries = 8

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        self.rate_limiter.acquire()
        attempt = 0
        while True:
            try:
                resp = self.session.get(
                    "https://translate.googleapis.com/translate_a/single",
                    params={"client": "gtx", "sl": "auto", "tl": self.target_lang, "dt": "t", "ie": "UTF-8", "oe": "UTF-8", "q": text},
                    timeout=120
                )
                if resp.status_code == 429:
                    raise requests.exceptions.HTTPError(f"429 Client Error", response=resp)
                resp.raise_for_status()
                data = resp.json()
                parts = [p[0] for p in data[0] if p[0]]
                result = " ".join(parts).strip()
                self._rate_limit_count = 0
                return result if result else None
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    attempt += 1
                    self._rate_limit_count += 1
                    retry_after = e.response.headers.get('Retry-After')
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after)
                    else:
                        delay = min(10.0 * (2 ** (self._rate_limit_count - 1)), 300.0) + random.uniform(0, 2)
                    logger.warning(f"   Google Translate: 429, пауза {delay:.0f} сек (попытка {attempt}/{self._429_max_retries})")
                    _interruptible_sleep(delay)
                    if STOP_EVENT.is_set() or attempt >= self._429_max_retries:
                        self._rate_limit_count = 0
                        return None
                    continue
                else:
                    status = e.response.status_code if e.response is not None else "?"
                    logger.warning(f"   Google Translate: HTTP {status}")
                    return None
            except Exception as e:
                logger.warning(f"   Google Translate: {type(e).__name__}")
                return None

    def generate(self, prompt: str) -> Optional[str]:
        return None

class SprutRotator:
    def __init__(self, target_lang: str, model: Optional[str] = None):
        self.target_lang = target_lang
        self.name = "SprutDock"
        self.can_generate = True
        self._client = None
        self._models: list = []
        self._current_idx = 0
        self._exhausted_models: dict = {}
        self._model_cooldown_sec = MODEL_COOLDOWN_SEC
        self._forced_model = model
        self.rate_limiter = RateLimiter(max_requests_per_second=0.3)
        if SPRUTDOCK_API_KEY:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=SPRUTDOCK_API_KEY,
                    base_url=OPENAI_BASE_URL,
                    default_headers={"X-Title": "pdf-translator"},
                    timeout=API_TIMEOUT,
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
            available = [m.id for m in resp.data] if resp and resp.data else []

            if self._forced_model:
                if not available or self._forced_model in available:
                    self._models = [self._forced_model]
                    logger.info(f"SprutDock: принудительно используется модель {self._forced_model}")
                else:
                    logger.warning(f"SprutDock: запрошенная модель {self._forced_model} недоступна. Доступные: {available}")
                    self._models = available or [DEFAULT_MODEL]
            elif DEFAULT_MODEL in available:
                self._models = [DEFAULT_MODEL]
            elif available:
                free = sorted([m for m in available if "free" in m.lower()], key=lambda x: (DEFAULT_MODEL.split("/")[0] in x, x), reverse=True)
                self._models = free or [DEFAULT_MODEL]
            else:
                self._models = [DEFAULT_MODEL]
            logger.info(f"SprutDock: доступно моделей: {len(self._models)}")
        except Exception as e:
            logger.warning(f"SprutDock: не удалось получить список моделей ({e}), используется {DEFAULT_MODEL}")
            self._models = [self._forced_model or DEFAULT_MODEL]

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
        max_attempts = len(self._models) * 3
        while attempts < max_attempts:
            if STOP_EVENT.is_set(): return None
            model = self._next_model()
            if not model:
                break
            self.rate_limiter.acquire()
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": _translation_system_prompt("English", self.target_lang, glossary)},
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
                    _interruptible_sleep(5)
                    attempts += 1
                    continue
                elif '502' in err_str or '503' in err_str or '504' in err_str or 'bad gateway' in err_str or 'server error' in err_str:
                    logger.warning(f"   SprutDock: {model} -> {type(e).__name__}: {str(e)[:200]}, повтор...")
                    _interruptible_sleep(3)
                    attempts += 1
                    continue
                logger.warning(f"   SprutDock translate: {type(e).__name__}: {str(e)[:200]}")
                attempts += 1
        return None

    def generate(self, prompt: str) -> Optional[str]:
        if not self._client:
            return None
        for attempt in range(3):
            if STOP_EVENT.is_set(): return None
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
            except Exception as e:
                err_str = str(e).lower()
                if '429' in err_str or '502' in err_str or '503' in err_str or '504' in err_str or 'bad gateway' in err_str:
                    logger.warning(f"   SprutDock generate: {type(e).__name__}: {str(e)[:200]}, повтор {attempt+1}/3...")
                    _interruptible_sleep(3)
                    continue
                logger.warning(f"   SprutDock generate: {type(e).__name__}: {str(e)[:200]}")
                return None
        return None

def _model_matches(expected: Optional[str], actual: Optional[str]) -> bool:
    if not expected or not actual:
        return False
    e = expected.lower()
    a = actual.lower()
    if e == a:
        return True
    e_base = os.path.basename(e).lower()
    a_base = os.path.basename(a).lower()
    return any(sub in other for sub, other in ((e, a), (a, e), (e_base, a_base), (a_base, e_base)))

def _is_short_citation(text: str) -> bool:
    t = normalize_text(text)
    if not t or len(t) > 140:
        return False
    citation = re.compile(
        r"^\s*(?:\(?[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s+(?:et\s+al\.?|and\s+[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+|&\s*[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+))?\s*,?\s*(?:19|20)\d{2}[a-z]?\)?|\(?(?:[A-ZÀ-ÖØ-Þ][A-Za-zÀ-ÖØ-öø-ÿ'’\-]+(?:\s*,\s*|\s*&\s*)?)+\s*,\s*(?:19|20)\d{2}[a-z]?\)?)[.,;]?\s*$",
        re.UNICODE,
    )
    if citation.match(t):
        return True
    if re.fullmatch(r"\s*(?:\d{1,3}(?:\s*[-–—,;]\s*\d{1,3})*|\[[0-9,\-–— ]+\])\s*[.]?\s*", t):
        return True
    if len(t) <= 50 and re.fullmatch(r"(?:[A-Z][A-Za-z'’\-]+(?:\s+et\s+al\.?)?)(?:\s*\(?(?:19|20)\d{2}\)?)?\.?", t):
        return bool(re.search(r"(?:19|20)\d{2}|et\s+al", t, re.I))
    return False

class LlamaCppTranslator:
    def __init__(self, target_lang: str, api_base: str = "http://localhost:8080/v1", expected_model: Optional[str] = None, auto_find: bool = True, max_retries: int = 3, source_lang: str = "English"):
        self.target_lang = target_lang
        self.name = "LlamaCpp"
        self.can_generate = True
        self.max_retries = max(1, min(int(max_retries), 6))
        self.source_lang = source_lang
        self.expected_model = expected_model
        self.rate_limiter = RateLimiter(max_requests_per_second=1)
        self._loaded_model = None
        self._server_ok = False
        self.api_base = None

        if api_base:
            ok, models = self._check_server(api_base)
            if ok:
                server_model = models[0] if models else "unknown"
                if expected_model and not _model_matches(expected_model, server_model):
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
                matched = [x for x in servers if _model_matches(expected_model, x["model"])]
                if matched:
                    self.api_base = matched[0]["url"]
                    self._loaded_model = matched[0]["model"]
                    self._server_ok = True
                else:
                    available = "\n".join(f"   {x['url']} -> {x['model']}" for x in servers)
                    raise RuntimeError(f"Модель '{expected_model}' не найдена.\n{available}")

        if not self._server_ok:
            raise RuntimeError("Не удалось подключиться ни к одному серверу.")

    @staticmethod
    def _find_all_servers() -> List[Dict[str, str]]:
        if os.name == "nt":
            return []
        result = []
        try:
            output = subprocess.check_output(["pgrep", "-a", "llama-server"], text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                m = re.search(r"--port\s+(\d+)", line)
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
            resp = requests.get(f"{url.rstrip('/')}/models", timeout=5)
            if resp.status_code == 200:
                models = [m.get("id", "") for m in resp.json().get("data", [])]
                if models:
                    return True, models
        except Exception:
            pass
        return False, []

    def _call_completion(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2048, timeout: int = REQUEST_TIMEOUT, stop: Optional[List[str]] = None) -> Optional[str]:
        if not self._server_ok:
            return None
        timeout = max(1, int(timeout))
        self.rate_limiter.acquire()
        try:
            payload = {
                "model": self._loaded_model,
                "messages": [
                    {"role": "system", "content": _translation_system_prompt(self.source_lang, self.target_lang)},
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens, "stream": False,
            }
            if stop:
                payload["stop"] = stop
            resp = requests.post(f"{self.api_base}/chat/completions", json=payload, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                choices = data.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content:
                        return content.strip()
                content = data.get("content")
                if content:
                    return str(content).strip()

            if resp.status_code in (404, 405, 501):
                payload2 = {"model": self._loaded_model, "prompt": prompt, "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens, "stream": False}
                if stop:
                    payload2["stop"] = stop
                resp2 = requests.post(f"{self.api_base}/completions", json=payload2, timeout=timeout)
                if resp2.status_code == 200:
                    data = resp2.json()
                    choices = data.get("choices") or [{}]
                    content = data.get("content")
                    if not content and isinstance(choices[0], dict):
                        content = choices[0].get("text")
                    return content.strip() if content else None
            return None
        except Exception:
            return None

    def _translation_prompt(self, source: str, lang: str, glossary: Optional[Dict[str, str]] = None, context_before: str = "", context_after: str = "", strict: bool = False) -> Tuple[str, Dict[str, str]]:
        protected, tokens = _extract_protected_tokens(source)
        sys_prompt = _translation_system_prompt(self.source_lang, lang, glossary)
        parts = [sys_prompt, ""]
        if context_before:
            parts += ["CONTEXT BEFORE (READ ONLY; DO NOT TRANSLATE):", "<<<CONTEXT_BEFORE>>>", context_before, "<<<END_CONTEXT_BEFORE>>>", ""]
        parts += ["TARGET (TRANSLATE ONLY THIS TEXT):", "<<<TARGET>>>", protected, "<<<END_TARGET>>>", ""]
        if context_after:
            parts += ["CONTEXT AFTER (READ ONLY; DO NOT TRANSLATE):", "<<<CONTEXT_AFTER>>>", context_after, "<<<END_CONTEXT_AFTER>>>", ""]
        parts += ["OUTPUT ONLY THE TRANSLATION OF TARGET:"]
        return "\n".join(parts), tokens

    def translate(self, text: str, glossary: Optional[Dict[str, str]] = None, context_before: str = "", context_after: str = "", strict: bool = False, deadline: Optional[float] = None) -> Optional[str]:
        source = normalize_text(text)
        if not source:
            return None

        prompt, protected = self._translation_prompt(source, self.target_lang, glossary, context_before=context_before, context_after=context_after, strict=strict)

        temperatures = (0.0, 0.1, 0.2)
        for attempt in range(self.max_retries):
            if STOP_EVENT.is_set():
                return None
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 1.0:
                return None
            req_timeout = REQUEST_TIMEOUT if remaining is None else max(1, min(REQUEST_TIMEOUT, int(remaining)))
            max_tokens = max(512, min(4096, int(len(source) / 2.0) + 512))

            raw = self._call_completion(prompt, temperature=temperatures[attempt % len(temperatures)], max_tokens=max_tokens, timeout=req_timeout)
            candidate_raw = _strip_llm_wrappers(raw)
            candidate = _restore_protected(candidate_raw, protected)
            ok, reason = validate_translation(source, candidate, self.target_lang)
            if ok:
                return candidate.strip()

            if attempt + 1 < self.max_retries:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 1.0:
                    return None
                _interruptible_sleep(min(1.5 * (2 ** attempt), 8.0))
        return None

    def generate(self, prompt: str) -> Optional[str]:
        return self._call_completion(prompt, temperature=0.2, max_tokens=1024)

def create_translator(translator_type: str, target_lang: str, llama_url: str = "http://localhost:8080/v1", llama_model: Optional[str] = None, auto_find: bool = True, sprut_model: Optional[str] = None):
    if translator_type == "llama":
        return LlamaCppTranslator(target_lang, api_base=llama_url, expected_model=llama_model, auto_find=auto_find), None
    elif translator_type in ("openrouter", "sprut"):
        return SprutRotator(target_lang, model=sprut_model), None
    elif translator_type == "google":
        primary = GoogleTranslator(target_lang)
        fallback = None
        if OPENAI_API_KEY:
            try:
                or_tr = SprutRotator(target_lang, model=sprut_model)
                if or_tr._client:
                    fallback = or_tr
            except Exception:
                pass
        if not fallback:
            try:
                fallback = LlamaCppTranslator(target_lang, api_base=llama_url, expected_model=llama_model, auto_find=auto_find)
            except Exception:
                pass
        return primary, fallback
    else:
        raise ValueError(f"Неизвестный переводчик: {translator_type}")

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
                STOP_EVENT.set()
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
        except Exception:
            pass

    def save(self):
        if not self._dirty:
            return
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(dict(self._cache), f, ensure_ascii=False)
            self._dirty = False
        except Exception:
            pass

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

def translate_chunk_with_retry(chunk: str, translator, max_retries: int = 3, glossary: Optional[Dict[str, str]] = None, context_before: str = "", context_after: str = "", deadline: Optional[float] = None) -> Optional[str]:
    if not chunk or len(chunk.strip()) < 3:
        return chunk
    attempts = max(1, min(int(max_retries), 6))
    for attempt in range(attempts):
        try:
            if hasattr(translator, "translate"):
                if isinstance(translator, LlamaCppTranslator):
                    translated = translator.translate(chunk, glossary=glossary, context_before=context_before, context_after=context_after, deadline=deadline)
                else:
                    translated = translator.translate(chunk, glossary) if glossary else translator.translate(chunk)
            else:
                translated = None

            if translated and translated.strip():
                return _strip_llm_wrappers(translated).strip()

        except Exception as e:
            logger.warning(f"[{getattr(translator, 'name', 'translator')}] {type(e).__name__}: {str(e)[:160]}")

        if attempt + 1 < attempts:
            _interruptible_sleep(min(1.5 * (2 ** attempt), 60.0))

    return None

def _split_text_for_llm(text: str, max_chars: int = 1800, min_chunk_chars: int = 650) -> List[str]:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]
    protected_abbr = re.compile(r"\b(?:e\.g|i\.e|et al|fig|figs|eq|eqs|dr|mr|mrs|ms|prof|no|nos|vs|approx|ca|p|pp)\.$", re.I)
    parts = []
    start = 0
    for m in re.finditer(r"[.!?。！？](?:[\"'”’\)\]]*)\s+(?=[A-ZА-ЯЁ0-9])", text):
        left = text[start:m.end()].strip()
        if not left:
            continue
        if protected_abbr.search(left[-12:]):
            continue
        parts.append(left)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)

    if len(parts) <= 1:
        words = text.split()
        chunks, cur = [], []
        cur_len = 0
        for word in words:
            add_len = len(word) + (1 if cur else 0)
            if cur and cur_len + add_len > max_chars:
                chunks.append(" ".join(cur))
                cur, cur_len = [word], len(word)
            else:
                cur.append(word)
                cur_len += add_len
        if cur:
            chunks.append(" ".join(cur))
        return chunks

    chunks, cur, cur_len = [], [], 0
    for part in parts:
        plen = len(part)
        if cur and cur_len + plen + 1 > max_chars:
            chunks.append(" ".join(cur))
            cur, cur_len = [], 0
        cur.append(part)
        cur_len += plen + (1 if cur_len else 0)
    if cur:
        chunks.append(" ".join(cur))

    merged = []
    for ch in chunks:
        if merged and len(ch) < min_chunk_chars and len(merged[-1]) + len(ch) + 1 <= max_chars:
            merged[-1] += " " + ch
        else:
            merged.append(ch)
    return merged

class TranslationPipeline:
    def __init__(self, translator, fallback=None, cache: Optional[TranslationCache] = None, max_workers: int = 4, timeout: int = BLOCK_TIMEOUT, is_local: bool = False, translator_type: str = "google"):
        self.translator = translator
        self.fallback = fallback
        self.cache = cache or TranslationCache()
        self.max_workers = max_workers
        self.block_timeout = max(30, int(timeout or BLOCK_TIMEOUT))
        self.is_local = is_local
        self.translator_type = translator_type
        self.stats = {"success": 0, "failed": 0, "cached": 0, "skipped": 0, "rejected": 0}
        self._stats_lock = threading.Lock()
        self._start_time = 0.0
        self._block_times = []

    BASE_SKIP_TYPES = ("figure", "table", "empty", "metadata")
    MAX_SECTION_CHARS = 2500
    MAX_LLM_CHUNK_CHARS = 1800
    MIN_LLM_CHUNK_CHARS = 650
    RESCUE_MIN_CHARS = int(SETTINGS.get("llm_rescue_min_chars", 220))

    def _skip_types(self) -> Tuple[str, ...]:
        return self.BASE_SKIP_TYPES + ("reference", "reference_heading")

    @staticmethod
    def _safe_cache_get(cache, text, lang):
        value = cache.get(text, lang)
        if value and value.strip():
            ok, _ = validate_translation(text, value, lang)
            return value if ok else None
        return None

    def _translate_one(self, text: str, lang: str, glossary: Optional[Dict[str, str]] = None, context_before: str = "", context_after: str = "", _allow_split: bool = True, _deadline: Optional[float] = None) -> Optional[str]:
        text = normalize_text(text)
        if not text:
            return None

        deadline = _deadline if _deadline is not None else time.monotonic() + self.block_timeout
        if time.monotonic() >= deadline:
            return None

        if _is_short_citation(text):
            return text

        if _allow_split and len(text) > self.MAX_LLM_CHUNK_CHARS:
            chunks = _split_text_for_llm(text, self.MAX_LLM_CHUNK_CHARS, self.MIN_LLM_CHUNK_CHARS)
            if len(chunks) > 1:
                out = []
                for i, chunk in enumerate(chunks):
                    r = self._translate_one(
                        chunk, lang, glossary,
                        context_before=context_before[-500:] if i == 0 else "",
                        context_after=context_after[:500] if i == len(chunks)-1 else "",
                        _allow_split=False, _deadline=deadline,
                    )
                    if not r:
                        r = self._rescue_block(chunk, lang, glossary, "", "", deadline)
                        if not r:
                            return None
                    out.append(r.strip())
                return " ".join(out)

        protected, tokens = _extract_protected_tokens(text)
        cached = self._safe_cache_get(self.cache, text, lang)
        if cached:
            with self._stats_lock:
                self.stats["cached"] += 1
            return cached

        result = translate_chunk_with_retry(
            text if isinstance(self.translator, LlamaCppTranslator) else protected,
            self.translator,
            max_retries=1 if isinstance(self.translator, LlamaCppTranslator) else 2,
            glossary=glossary, context_before=context_before, context_after=context_after, deadline=deadline,
        )

        if result:
            result = _strip_llm_wrappers(_restore_protected(result, tokens)).strip()
            valid, reason = validate_translation(text, result, lang)
            if not valid:
                result = None
                with self._stats_lock:
                    self.stats["rejected"] += 1

        if not result and self.fallback and time.monotonic() < deadline:
            result = translate_chunk_with_retry(
                protected if not isinstance(self.fallback, LlamaCppTranslator) else text,
                self.fallback, max_retries=2, glossary=glossary, deadline=deadline,
            )
            if result:
                result = _strip_llm_wrappers(_restore_protected(result, tokens)).strip()
                valid, _ = validate_translation(text, result, lang)
                if not valid:
                    result = None

        if result:
            self.cache.put(text, lang, result)
            with self._stats_lock:
                self.stats["success"] += 1
            return result

        if _allow_split and time.monotonic() < deadline:
            rescued = self._rescue_block(text, lang, glossary, context_before, context_after, deadline)
            if rescued:
                self.cache.put(text, lang, rescued)
                with self._stats_lock:
                    self.stats["success"] += 1
                return rescued

        with self._stats_lock:
            self.stats["failed"] += 1
        return None

    def _rescue_block(self, text: str, lang: str, glossary: Optional[Dict[str, str]], context_before: str, context_after: str, deadline: float) -> Optional[str]:
        if time.monotonic() >= deadline:
            return None
        sizes = [900, 600, 420, self.RESCUE_MIN_CHARS]
        for size in sizes:
            if len(text) <= size:
                continue
            chunks = _split_text_for_llm(text, size, max(80, min(self.MIN_LLM_CHUNK_CHARS, size // 3)))
            if len(chunks) <= 1:
                continue
            out = []
            ok = True
            for i, chunk in enumerate(chunks):
                if time.monotonic() >= deadline:
                    return None
                r = self._translate_one(
                    chunk, lang, glossary,
                    context_before=context_before[-300:] if i == 0 else "",
                    context_after=context_after[:300] if i == len(chunks)-1 else "",
                    _allow_split=False, _deadline=deadline,
                )
                if not r:
                    ok = False
                    break
                out.append(r.strip())
            if ok and out:
                return " ".join(out)
        return None

    def _translate_table(self, block: Block, lang: str, quiet: bool = False) -> None:
        rows = block.table_data or []
        if not rows:
            return
        md_lines = []
        for i, row in enumerate(rows):
            cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0 and len(rows) > 1:
                md_lines.append("| " + " | ".join(["---"] * len(row)) + " |")
        md_table = "\n".join(md_lines)

        prompt = TABLE_TRANSLATE_PROMPT.format(lang=_LANG_NAMES.get(lang, lang)) + "\n\nTABLE:\n<<<\n" + md_table + "\n>>>"
        try:
            raw = self.translator.generate(prompt) if getattr(self.translator, "can_generate", False) else None
        except Exception:
            raw = None

        if not raw:
            return

        parsed = self._parse_md_table(_strip_llm_wrappers(raw))
        if parsed and len(parsed) == len(rows):
            expected_cols = max((len(r) for r in rows), default=0)
            if all(len(r) == expected_cols for r in parsed):
                block.table_data = parsed
                if not quiet:
                    logger.info(f"   Таблица (стр. {block.page_num}): переведена")

    @staticmethod
    def _parse_md_table(raw: str) -> Optional[List[List[str]]]:
        rows = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if not ln.startswith("|"):
                ln = "|" + ln + "|"
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if cells and all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue
            rows.append(cells)
        return rows if rows else None

    def _translate_section(self, section: list, lang: str, glossary: Optional[Dict[str, str]] = None) -> None:
        for idx, block in enumerate(section):
            if STOP_EVENT.is_set():
                return
            before = section[idx - 1].text if idx > 0 else ""
            after = section[idx + 1].text if idx + 1 < len(section) else ""
            result = self._translate_one(
                block.text, lang, glossary,
                context_before=before[-600:],
                context_after=after[:600],
            )
            block.translation = result

    def translate_blocks(self, blocks: list, lang: str, quiet: bool = False) -> list:
        merged_blocks = []
        skip_types = self._skip_types()

        for block in blocks:
            text = block.text
            if not text or len(text) < 2 or block.type in skip_types:
                block.translation = text if text else ""
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue

            if block.type in ("paragraph", "list", "text"):
                _wrap_citation_spans(block)
            merged_blocks.append(block)

        blocks = merged_blocks

        if getattr(self.translator, "can_generate", False):
            for block in blocks:
                if block.type == "table" and block.table_data:
                    self._translate_table(block, lang, quiet)

        translatable = [b for b in blocks if b.type not in skip_types and b.translation is None]
        if not translatable:
            return blocks

        glossary = None
        sections = []
        current = []
        cur_len = 0
        for b in translatable:
            blen = len(b.text)
            if current and cur_len + blen > self.MAX_SECTION_CHARS:
                sections.append(current)
                current = []
                cur_len = 0
            current.append(b)
            cur_len += blen
        if current:
            sections.append(current)

        total = len(sections)
        done = 0
        lock = threading.Lock()
        self._start_time = time.time()

        if not quiet:
            logger.info(f"   Блоков для перевода: {len(translatable)} (секций контекста: {total})")

        def _work(group):
            nonlocal done
            if STOP_EVENT.is_set():
                return
            self._translate_section(group, lang, glossary)
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
            pbar = tqdm(total=total, desc="🌐 Перевод", unit="секция", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        _start_keyboard_watchdog()
        executor = ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, 4)))
        futures = {executor.submit(_work, g): g for g in sections}

        try:
            while futures and not STOP_EVENT.is_set():
                done_futures, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                for future in done_futures:
                    futures.pop(future, None)
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning(f"   Ошибка секции: {type(e).__name__}: {e}")
        except KeyboardInterrupt:
            STOP_EVENT.set()
            logger.warning("⏹ Прервано пользователем (Ctrl+C)")
        finally:
            for f in futures:
                f.cancel()
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            if TQDM and not quiet:
                pbar.close()

        if not quiet:
            elapsed = time.time() - self._start_time
            logger.info(f"   Перевод завершён за {_format_time(elapsed)}")
            logger.info(f"   LLM guard: успешно={self.stats['success']}, кэш={self.stats['cached']}, отклонено={self.stats['rejected']}, ошибки={self.stats['failed']}")

        return blocks

def generate_summary(pages: list, summary_translator_type: str, translate_translator_type: str, lang: str, llama_url: str = "http://localhost:8080/v1", llama_model: Optional[str] = None, auto_find: bool = True, sprut_model: Optional[str] = None, quiet: bool = False) -> Dict[str, Any]:
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
        summary_translator_type = "llama"

    try:
        summary_translator, _ = create_translator(summary_translator_type, "en", llama_url=llama_url, llama_model=llama_model, auto_find=auto_find, sprut_model=sprut_model)
    except Exception as e:
        return {"summary_html": f"<p>Ошибка LLM: {e}</p>", "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": str(e)}}

    try:
        final_translator, _ = create_translator(translate_translator_type, lang, llama_url=llama_url, llama_model=llama_model, auto_find=auto_find, sprut_model=sprut_model)
    except Exception as e:
        return {"summary_html": f"<p>Ошибка переводчика: {e}</p>", "stats": {"chunks": 0, "tokens": 0, "speed": 0, "error": str(e)}}

    chunks = _chunk_text_by_sentences(full_text, max_chunk_size=6000)
    cache = TranslationCache()
    cache_key = f"summary:{summary_translator_type}"
    chunk_summaries = []
    start_time = time.time()
    total_tokens = 0

    for i, chunk in enumerate(chunks):
        cached = cache.get(chunk, cache_key)
        if cached:
            chunk_summaries.append(cached)
            continue
        prompt = "Extract key facts from this text for a summary. Write the summary in English:\n" + chunk
        summary = summary_translator.generate(prompt)
        if summary:
            cache.put(chunk, cache_key, summary)
            chunk_summaries.append(summary)
            total_tokens += len(summary) // 4

    if not chunk_summaries:
        return {"summary_html": "<p>Не удалось сгенерировать реферат.</p>", "stats": {"chunks": 0, "tokens": 0, "speed": 0}}

    summaries_to_combine = chunk_summaries
    if len(chunk_summaries) > 5:
        grouped = []
        for j in range(0, len(chunk_summaries), 5):
            group = chunk_summaries[j:j + 5]
            group_prompt = "Combine these key points into a concise summary:\n" + "\n---\n".join(group)
            group_summary = summary_translator.generate(group_prompt)
            grouped.append(group_summary if group_summary else "\n".join(group))
        summaries_to_combine = grouped

    final_prompt = "Based on the following key points, create a final structured summary in English strictly in the format:\nBRIEF CONTENT: 4-5 sentences\nKEY FINDINGS: list\nSTRENGTHS: list\nWEAKNESSES: list\n\nKey points:\n" + "\n---\n".join(summaries_to_combine)
    summary_en = summary_translator.generate(final_prompt) or "\n\n".join(summaries_to_combine)

    try:
        final_translated = final_translator.translate(summary_en)
        summary_html = md.markdown(final_translated if final_translated else summary_en)
    except Exception:
        summary_html = md.markdown(summary_en)

    elapsed = time.time() - start_time
    speed = total_tokens / elapsed if elapsed > 0 else 0
    return {"summary_html": summary_html, "stats": {"chunks": len(chunks), "tokens": total_tokens, "speed": speed, "model": getattr(summary_translator, '_loaded_model', summary_translator.name), "time": elapsed}}

# ИСПРАВЛЕНИЕ 4: Узкие поля в CSS
_CSS_TEMPLATE = string.Template("""
<style>
body { font-family: $BODY_FONT; margin: $BODY_MARGIN; background: $BG; color: $TEXT; }
.container { max-width: 1100px; margin: 0 auto; padding: 20px 40px; }
h1 { color: $H1_COLOR; border-bottom: 2px solid $H1_BORDER; padding-bottom: 12px; }
h3 { color: $H3_COLOR; margin-top: 28px; }
.page { background: $PAGE_BG; padding: 15px 20px; margin-bottom: 20px; border-radius: 8px; $PAGE_BOX }
.page > summary { cursor: pointer; font-size: 1.05em; font-weight: bold; color: $SUMMARY_COLOR; padding: 4px 0 8px; user-select: none; }
.page > summary:hover { color: $SUMMARY_HOVER; }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
.trans-head h2, .trans-head h3, .trans-head h4 { margin: 4px 0; }
p { line-height: 1.7; margin: 0 0 10px; text-align: justify; }
.orig { color: $ORIG_COLOR; font-size: 0.88em; }
.trans { color: $TEXT; }
.ref-section h2 { color: $H1_COLOR; border-top: 2px solid $TABLEWRAP_BORDER; margin-top: 24px; padding-top: 16px; padding-bottom: 6px; }
.ref { color: $REF_COLOR; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; margin-bottom: 8px; }
.meta { color: $META_COLOR; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid $TABLEWRAP_BORDER; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid $TABLEWRAP_BORDER; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: $TH_BG; border: 1px solid $TABLEWRAP_BORDER; padding: 8px 10px; color: $TH_COLOR; }
.table-wrap td { border: 1px solid $TABLEWRAP_BORDER; padding: 6px 10px; }
.figure { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: $CAPTION_COLOR; font-size: 0.85em; }
.list-block { margin-left: 20px; line-height: 1.6; }
.ref-link { color: $REFLINK_COLOR; font-weight: bold; }
.table-wrap tbody tr:nth-child(even) td { background: $TBODY_EVEN; }
.table-pre { white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 8px 0; }
.table-caption { font-style: italic; color: $CAPTION_COLOR; font-size: 0.85em; margin: 8px 0; }
.toc { background: $TOC_BG; border: 1px solid $TOC_BORDER; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px; }
.toc ul { list-style: none; margin: 6px 0 0; padding-left: 18px; }
.toc > ul { padding-left: 0; }
.toc a { color: $TOC_A; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.btn-orig { background: $BTN_BG; border: 1px solid $BTN_BORDER; color: $BTN_COLOR; border-radius: 6px; padding: 6px 12px; margin: 0 4px 12px 0; cursor: pointer; }
.btn-orig:hover { color: $BTN_HOVER; }
details.original-block { margin: 4px 0; }
details.original-block summary { color: $DETAILS_COLOR; font-size: 0.85em; cursor: pointer; }
details.original-block summary:hover { color: $DETAILS_HOVER; }
#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999; align-items: center; justify-content: center; cursor: zoom-out; }
#lightbox img { max-width: 92%; max-height: 92%; border-radius: 6px; }
</style>
""")

_CSS_PALETTES = {
    True: {
        "BODY_FONT": "'Segoe UI', Arial, sans-serif", "BODY_MARGIN": "0; padding: 0", "BG": "#0f172a", "TEXT": "#e2e8f0", "H1_COLOR": "#60a5fa", "H1_BORDER": "#1e3a5f", "H3_COLOR": "#93c5fd",
        "PAGE_BG": "#1e293b", "PAGE_BOX": "border: 1px solid #334155", "SUMMARY_COLOR": "#93c5fd", "SUMMARY_HOVER": "#60a5fa",
        "ORIG_COLOR": "#64748b", "REF_COLOR": "#64748b", "META_COLOR": "#64748b", "TABLEWRAP_BORDER": "#334155", "TH_BG": "#0f172a", "TH_COLOR": "#60a5fa",
        "TBODY_EVEN": "#16213a", "CAPTION_COLOR": "#94a3b8", "REFLINK_COLOR": "#60a5fa", "TOC_BG": "#1e293b", "TOC_BORDER": "#334155", "TOC_A": "#93c5fd",
        "BTN_BG": "#1e293b", "BTN_BORDER": "#334155", "BTN_COLOR": "#94a3b8", "BTN_HOVER": "#e2e8f0", "DETAILS_COLOR": "#64748b", "DETAILS_HOVER": "#94a3b8", "CONTAINER_PAD": "padding: 24px;",
    },
    False: {
        "BODY_FONT": "Arial, sans-serif", "BODY_MARGIN": "0; padding: 20px", "BG": "#f0f2f5", "TEXT": "#111827", "H1_COLOR": "#1f2937", "H1_BORDER": "#d1d5db", "H3_COLOR": "#374151",
        "PAGE_BG": "white", "PAGE_BOX": "box-shadow: 0 2px 4px rgba(0,0,0,0.1);", "SUMMARY_COLOR": "#1f2937", "SUMMARY_HOVER": "#2563eb",
        "ORIG_COLOR": "#9ca3af", "REF_COLOR": "#6b7280", "META_COLOR": "#6b7280", "TABLEWRAP_BORDER": "#d1d5db", "TH_BG": "#f3f4f6", "TH_COLOR": "#1f2937",
        "TBODY_EVEN": "#f9fafb", "CAPTION_COLOR": "#6b7280", "REFLINK_COLOR": "#2563eb", "TOC_BG": "#f8fafc", "TOC_BORDER": "#e5e7eb", "TOC_A": "#2563eb",
        "BTN_BG": "#fff", "BTN_BORDER": "#d1d5db", "BTN_COLOR": "#6b7280", "BTN_HOVER": "#111827", "DETAILS_COLOR": "#9ca3af", "DETAILS_HOVER": "#6b7280", "CONTAINER_PAD": "",
    }
}

def build_css(dark: bool = True) -> str:
    return _CSS_TEMPLATE.safe_substitute(_CSS_PALETTES[bool(dark)])

def _reference_entries(block: Block) -> List[str]:
    entries: List[List[str]] = []
    cur: List[str] = []
    for line in block.lines:
        txt = " ".join(s.text for s in line.spans).strip()
        if not txt:
            continue
        if re.match(r'^\d+\.', txt) and cur:
            entries.append(cur)
            cur = []
        cur.append(txt)
    if cur:
        entries.append(cur)
    return [" ".join(e) for e in entries]

def _block_html(block: Block) -> str:
    e = html_mod.escape
    bt = block.type
    trans = block.translation if block.translation is not None else "[ПЕРЕВОД НЕ ПОЛУЧЕН]"
    orig = block.text

    def trans_html(text: str) -> str:
        return highlight_refs(e(text))

    def orig_html(spans_text: str) -> str:
        return highlight_refs(e(spans_text))

    if bt == 'reference_heading':
        return f'<div class="ref-section"><h2>{trans_html(trans)}</h2></div>\n'
    if bt == 'empty':
        return ''
    if bt == 'reference':
        entries = _reference_entries(block)
        if len(entries) > 1:
            return ''.join(f'<p class="ref">{trans_html(x)}</p>\n' for x in entries)
        return f'<p class="ref">{trans_html(trans)}</p>\n'
    if bt == 'metadata':
        if orig != trans:
            return (f'<div class="meta">{trans_html(trans)}</div>'
                    f'<details class="original-block"><summary>Оригинал</summary>'
                    f'<p class="orig">{orig_html(orig)}</p></details>\n')
        return f"<div class='meta'>{orig_html(orig)}</div>\n"
    if bt == 'table':
        if block.table_data:
            rows = block.table_data
            out = []
            if block.caption:
                out.append(f'<div class="table-caption">{trans_html(block.caption)}</div>\n')
            out.append('<div class="table-wrap"><table>\n')
            for idx, row in enumerate(rows[:20]):
                tag = 'th' if idx == 0 else 'td'
                if idx == 0:
                    out.append('<thead><tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr></thead><tbody>\n')
                else:
                    out.append('<tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr>\n')
            out.append('</tbody></table>\n</div>\n')
            return ''.join(out)
        if orig.strip():
            return f'<div class="table-wrap"><pre class="table-pre">{e(orig)}</pre></div>\n'
    if bt == 'figure':
        parts = []
        if block.image_data:
            img_src = f"data:image/{block.image_ext or 'png'};base64,{block.image_data}"
            alt = block.caption or f"Рисунок {block.page_num}"
            parts.append(f'<figure class="figure"><img src="{img_src}" alt="{e(alt)}" onclick="zoomImg(this)" />')
            if block.caption:
                parts.append(f'<figcaption class="figure-caption">{e(block.caption)}</figcaption>')
            parts.append('</figure>\n')
            return ''.join(parts)
        return '<p class="orig">(Изображение не извлечено)</p>\n'
    if bt == 'heading':
        spans_text = render_spans(block)
        anchor = f' id="{e(getattr(block, "_anchor", ""))}"' if getattr(block, "_anchor", None) else ''
        if orig != trans:
            return f'<div class="trans-head"{anchor}><p class="orig">{orig_html(spans_text)}</p><p class="trans">{trans_html(trans)}</p></div>\n'
        return f"<p class='trans'><b>{trans_html(trans)}</b></p>\n"
    if bt == 'list':
        items = orig.split('\n')
        numbered = bool(re.match(r'^\s*\d+[\.\)]', items[0])) if items and items[0].strip() else False
        tag = 'ol' if numbered else 'ul'
        lis = ''.join(f'<li>{e(item.lstrip("•-*►▸‣⁃◦○●▪ 0123456789.)"))}</li>' for item in items if item.strip())
        return f'<{tag} class="list-block">{lis}</{tag}>\n'

    spans_text = render_spans(block)
    if orig != trans:
        return f'<p class="trans">{trans_html(trans)}</p><details class="original-block"><summary>Оригинал</summary><p class="orig">{orig_html(spans_text)}</p></details>\n'
    return f"<p>{trans_html(trans)}</p>\n"

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
    {% if toc %}
    <div class="toc"><strong>Содержание</strong>
        <ul>
        {% for item in toc %}
            <li><a href="#{{ item.anchor }}">{{ item.text | e }}</a></li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    <div>
        <button class="btn-orig" onclick="toggleAllOriginals(true)">Показать все оригиналы</button>
        <button class="btn-orig" onclick="toggleAllOriginals(false)">Скрыть оригиналы</button>
    </div>
    {% for page in pages %}
    <details class="page" open>
        <summary>📄 Страница {{ page.num }}</summary>
        {% for block in page.blocks %}
            {{ _block_html(block) | safe }}
        {% endfor %}
    </details>
    {% endfor %}
</div>
<div id="lightbox" onclick="closeLightbox()"></div>
<script>
function toggleAllOriginals(open) {
    document.querySelectorAll('details.original-block').forEach(function(d){ d.open = open; });
}
function zoomImg(img) {
    var lb = document.getElementById('lightbox');
    lb.innerHTML = '';
    var c = document.createElement('img');
    c.src = img.src;
    lb.appendChild(c);
    lb.style.display = 'flex';
}
function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
}
</script>
</body>
</html>
"""

def _build_toc(pages: List[Page]) -> List[Dict[str, str]]:
    toc = []
    idx = 0
    for page in pages:
        for block in page.blocks:
            if block.type == "heading":
                block._anchor = f"h{idx}"
                toc.append({"anchor": f"h{idx}", "text": block.translation or block.text})
                idx += 1
    return toc

def generate_html(pages: List[Page], title: str, output_path: str, dark: bool = True):
    css = build_css(dark)
    toc = _build_toc(pages)
    template = Template(HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, css=css, toc=toc, _block_html=_block_html)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)

SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
.summary-container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 95%; margin: 0 auto; }
.summary-block { background: #e7f3ff; padding: 20px; border-radius: 8px; border: 1px solid #b3d7ff; margin-bottom: 20px; }
.summary-block h2 { margin-top: 0; color: #004085; }
.metadata { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 2em; border: 1px solid #dee2e6; }
.metadata h1 { margin-top: 0; color: #111; font-size: 2em; }
.page > summary { cursor: pointer; font-size: 1.05em; font-weight: bold; color: #1f2937; padding: 4px 0 8px; user-select: none; }
.page > summary:hover { color: #2563eb; }
</style>
</head>
<body>
<div class="summary-container">
<div class="metadata"><h1>{{ title }}</h1></div>
<div class="summary-block">{{ summary_html | safe }}</div>
<details><summary>Показать исходные тексты</summary>
{% for page in pages %}
<details class="page" open><summary>Страница {{ page.num }}</summary>
{% for block in page.blocks %}
    {% if block.type in ('paragraph', 'heading', 'metadata', 'text', 'list') %}
        {% set orig = render_spans(block) %}
        {% if orig | length > 40 %}
        <details><summary>Оригинал (стр. {{ page.num }})</summary><span class="original">{{ orig }}</span></details>
        {% endif %}
    {% endif %}
{% endfor %}</details>
{% endfor %}</details></div></body></html>
"""

def generate_summary_html(pages: List[Page], summary_html: str, title: str, output_path: str):
    template = Template(SUMMARY_TEMPLATE)
    rendered = template.render(pages=pages, title=title, summary_html=summary_html, render_spans=render_spans)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)

def process_pdf(
    pdf_path: str,
    output_html: str,
    lang: str = "ru",
    translator_type: str = "google",
    llama_url: str = "http://localhost:8080/v1",
    llama_model: Optional[str] = None,
    auto_find: bool = True,
    max_workers: int = 8,
    timeout: int = BLOCK_TIMEOUT,
    quiet: bool = False,
    summary_mode: bool = False,
    summary_translator_type: str = "llama",
    translate_translator_type: str = "google",
    dark_html: bool = True,
    sprut_model: Optional[str] = None,
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
            non_text = [b for b in page.blocks if b.type != "text"]
            fitz_page = doc[page.num - 1]
            table_blocks = []
            table_bboxes = []
            try:
                table_blocks = extract_tables(fitz_page, page.num)
                table_bboxes.extend(tb.bbox for tb in table_blocks)
                total_tables += len(table_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения таблиц: {e}")
            try:
                span_tables, text_blocks = _find_span_table_blocks(text_blocks)
                if span_tables and table_blocks:
                    covered = [i for i, tb in enumerate(table_blocks) if any(_rect_gap(tb.bbox, st.bbox) <= 5 for st in span_tables)]
                    for i in reversed(covered):
                        table_blocks.pop(i)
                    total_tables -= len(covered)
                table_bboxes.extend(tb.bbox for tb in span_tables)
                total_tables += len(span_tables)
            except Exception as e:
                logger.warning(f"Ошибка распознавания текстовых таблиц: {e}")
            figure_blocks = []
            try:
                figure_blocks, text_blocks = extract_images(fitz_page, page.num, text_blocks)
                total_images += len(figure_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения изображений: {e}")
            try:
                used_bboxes = [fb.bbox for fb in figure_blocks]
                vector_blocks, text_blocks = extract_vector_figures(fitz_page, page.num, text_blocks, used_bboxes=used_bboxes, table_bboxes=table_bboxes)
                figure_blocks.extend(vector_blocks)
                total_images += len(vector_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения векторных фигур: {e}")
            page.blocks = non_text + text_blocks + table_blocks + figure_blocks

        total_text = "".join(block.text for page in pages for block in page.blocks)
        if len(total_text.strip()) < 10:
            logger.error("❌ В PDF не найден текст (возможно, файл состоит только из изображений или отсканирован).")
            sys.exit(1)
        doc.close()
    logger.info(f"   Извлечено {len(pages)} стр., {total_images} изобр., {total_tables} табл.")

    with stage_timer("2. Классификация", timings):
        for page in pages:
            all_sizes = [span.size for block in page.blocks if block.type in ("paragraph", "heading", "metadata") for line in block.lines for span in line.spans]
            avg_size = sum(all_sizes) / len(all_sizes) if all_sizes else 12.0
            for block in page.blocks:
                if block.type == "text":
                    block.type = _classifier.classify(block, page.width, avg_size)

    with stage_timer("3. Пост-обработка", timings):
        for page in pages:
            new_blocks = []
            for block in page.blocks:
                new_blocks.extend(_split_ref_heading(block))
            page.blocks = new_blocks
        in_refs = False
        for page in pages:
            for block in page.blocks:
                if block.type == "reference_heading":
                    in_refs = True
                    continue
                if in_refs:
                    if BACK_MATTER_RE.match(block.text):
                        in_refs = False
                    elif block.type not in ("figure", "table"):
                        block.type = "reference"
        for page in pages:
            table_blocks_page = [b for b in page.blocks if b.type == "table"]
            page.blocks = consolidate_tables(page.blocks, table_blocks_page)

    if summary_mode:
        with stage_timer("4. Генерация реферата", timings):
            logger.info("📝 Режим реферата...")
            if summary_translator_type == "google":
                summary_translator_type = "llama"
            result = generate_summary(pages, summary_translator_type=summary_translator_type, translate_translator_type=translate_translator_type, lang=lang, llama_url=llama_url, llama_model=llama_model, auto_find=auto_find, sprut_model=sprut_model, quiet=quiet)
            summary_html = result["summary_html"]
            gen_stats = result["stats"]

        with stage_timer("5. HTML реферата", timings):
            generate_summary_html(pages, summary_html, f"Реферат: {os.path.basename(pdf_path)}", output_html)
            logger.info(f"   HTML сохранён: {output_html}")

        return {"stats": {"chunks": gen_stats.get("chunks", 0), "tokens": gen_stats.get("tokens", 0), "speed": gen_stats.get("speed", 0), "model": gen_stats.get("model", ""), "time": gen_stats.get("time", 0), "summary_mode": True, "timings": timings}, "pages": len(pages), "images": total_images, "tables": total_tables, "summary": True}

    with stage_timer("4. Перевод", timings):
        logger.info(f"🌐 Перевод на {lang} ({translator_type})...")
        try:
            translator, fallback = create_translator(translator_type, lang, llama_url=llama_url, llama_model=llama_model, auto_find=auto_find, sprut_model=sprut_model)
        except RuntimeError as e:
            logger.error(f"   Не удалось инициализировать переводчик '{translator_type}': {e}")
            return {"stats": {"error": str(e)}, "pages": 0, "images": 0, "tables": 0}

        effective_workers = 1 if translator_type == "llama" else max_workers
        if translator_type == "llama" and not quiet:
            logger.info(f"   Локальный сервер: ограничено до 1 воркера (параллельные запросы вызывают зацикливание/галлюцинации)")

        pipeline = TranslationPipeline(
            translator=translator, fallback=fallback,
            max_workers=effective_workers, timeout=timeout,
            is_local=(translator_type == "llama"), translator_type=translator_type,
        )

        all_blocks = [block for page in pages for block in page.blocks]
        pipeline.translate_blocks(all_blocks, lang, quiet=quiet)

    with stage_timer("5. HTML", timings):
        generate_html(pages, f"Перевод: {os.path.basename(pdf_path)}", output_html, dark=dark_html)
        logger.info(f"   HTML сохранён: {output_html}")

    return {"stats": {**pipeline.stats, "timings": timings}, "pages": len(pages), "images": total_images, "tables": total_tables}

def main():
    parser = argparse.ArgumentParser(description="PDF Translator — единый скрипт")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="Входной PDF файл")
    parser.add_argument("-l", "--lang", default="ru", help="Целевой язык (по умолчанию: ru)")
    parser.add_argument("-t", "--translator", default="google", choices=["google", "llama", "openrouter", "sprut"], help="Переводчик (по умолчанию: google)")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1", help="URL llama.cpp сервера")
    parser.add_argument("--llama-model", help="Ожидаемое имя модели")
    parser.add_argument("--sprut-model", default=os.getenv("SPRUT_MODEL"), help="Модель для SprutDock (например, z-ai/glm-5.2:free)")
    parser.add_argument("--no-auto-find", action="store_true", help="Отключить автоматический поиск сервера")
    parser.add_argument("--summary", action="store_true", help="Режим реферата")
    parser.add_argument("--summary-translator", choices=["google", "llama", "openrouter", "sprut"], default="llama", help="Генератор реферата (по умолчанию: llama)")
    parser.add_argument("--summary-lang-translator", choices=["google", "llama", "openrouter", "sprut"], default="google", help="Переводчик реферата (по умолчанию: google)")
    parser.add_argument("--workers", type=int, default=min(8, (os.cpu_count() or 1) * 2), help="Число воркеров")
    parser.add_argument("--block-timeout", type=int, default=BLOCK_TIMEOUT, help="Максимум секунд на один блок")
    parser.add_argument("--dark-html", action="store_true", default=True, help="Тёмная тема HTML (по умолчанию: вкл)")
    parser.add_argument("--light-html", action="store_true", help="Светлая тема HTML")
    parser.add_argument("-q", "--quiet", action="store_true", help="Тихий режим")
    parser.add_argument("-v", "--verbose", action="store_true", help="Подробный вывод")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    dark_html = not args.light_html
    input_base = os.path.splitext(args.input)[0]
    output_html = f"{input_base}_summary.html" if args.summary else f"{input_base}_translate.html"

    logger.info("=" * 60)
    logger.info(f"PDF TRANSLATOR — {args.translator.upper()} v{__version__}")
    logger.info("=" * 60)

    try:
        result = process_pdf(
            pdf_path=args.input, output_html=output_html, lang=args.lang, translator_type=args.translator,
            llama_url=args.llama_url, llama_model=args.llama_model, auto_find=not args.no_auto_find,
            max_workers=args.workers, timeout=args.block_timeout, quiet=args.quiet, summary_mode=args.summary,
            summary_translator_type=args.summary_translator, translate_translator_type=args.summary_lang_translator,
            dark_html=dark_html, sprut_model=args.sprut_model,
        )

        stats = result["stats"]
        logger.info("=" * 60)
        if stats.get("summary_mode", False):
            logger.info("📊 СТАТИСТИКА ГЕНЕРАЦИИ РЕФЕРАТА:")
            logger.info(f"   ✓ Обработано чанков: {stats.get('chunks', 0)}")
            if stats.get("tokens", 0) > 0: logger.info(f"   📝 Сгенерировано токенов (приблиз.): {stats['tokens']}")
            if stats.get("speed", 0) > 0: logger.info(f"   ⚡ Скорость: {stats['speed']:.1f} токенов/сек")
            if stats.get("model"): logger.info(f"   🤖 Модель: {stats['model']}")
            if stats.get("time", 0) > 0: logger.info(f"   ⏱️ Время генерации: {_format_time(stats['time'])}")
        else:
            logger.info("📊 СТАТИСТИКА ПЕРЕВОДА:")
            logger.info(f"   ✓ Переведено: {stats.get('success', 0)}")
            logger.info(f"   ⚡ Из кэша:   {stats.get('cached', 0)}")
            logger.info(f"   ⊘ Пропущено:  {stats.get('skipped', 0)}")
            logger.info(f"   ✗ Ошибок:     {stats.get('failed', 0)}")
            if stats.get("error"): logger.error(f"   Причина: {stats['error']}")

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