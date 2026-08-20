#!/usr/bin/env python3
"""
PDF Translator — единый скрипт для перевода и реферирования PDF-документов.
Поддерживает Google Translate, OpenRouter и локальный llama.cpp.
Объединяет main2.py (ML-классификация, профилирование) и
pdf_translator_online.py (protect/restore, retry, dark-тема HTML).
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

__version__ = "1.0.0"

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
    r'^\s*(References?|Reference\s+list|References?\s+and\s+notes?|'
    r'Bibliography|Библиография|Библиографический\s+список|'
    r'Литература|Список\s+литературы|Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|Список\s+источников|'
    r'Использованная\s+литература|Источники)\s*[:.]?\s*$', re.IGNORECASE
)
# Заголовок литературы, за которым сразу идёт первая запись списка
REF_HEADING_PREFIX_RE = re.compile(
    r'^\s*(References?|Reference\s+list|References?\s+and\s+notes?|'
    r'Bibliography|Библиография|Библиографический\s+список|'
    r'Литература|Список\s+литературы|Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|Список\s+источников|'
    r'Использованная\s+литература|Источники)\s*[:.]?\s+(.+)$', re.IGNORECASE
)
# Заголовки «хвоста» статьи, после которых список литературы заканчивается
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
# Номерные цитаты в тексте (например 5, 1–3, 11, 13–18)
_CIT_RUN_RE = re.compile(r'^\d{1,2}(?:\s*[,–—-]\s*\d{1,2})*\s*[,.]?$')
_REF_LABEL_RE = re.compile(
    r'\b(?:Fig(?:ure)?\.?|Table|Tab\.?|Vol(?:ume)?\.?|No\.?|Section|Chapter|Ch\.?|'
    r'Equation|Eq\.?|Ref\.?|Page|pp\.?|Appendix|Supplementary|Suppl\.?)\s*$'
)
TABLE_CAPTION_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+', re.I)
TABLE_CAPTION_STRICT_RE = re.compile(
    r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+\s*[|:.–—-]', re.I)
CAPTION_RE = re.compile(r'Figure|Fig\.|Рис\.|Схема|Table|Таблица', re.I)
FIGURE_CAPTION_RE = re.compile(
    r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Рис\.?\s*\d+|Схема\s*\d+)', re.I
)
FIGURE_CAPTION_STRICT_RE = re.compile(
    r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Рис\.?\s*\d+|Схема\s*\d+)\s*[|:.,–—-]', re.I
)
# Максимальная дистанция между фигурой/рисунком и её подписью
CAPTION_DIST_THRESHOLD = 120.0
LIST_LINE_RE = re.compile(r'^[\s]*([•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s|[a-z]\.\s)', re.MULTILINE)


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


@dataclass
class PDFBlock:
    text: str
    block_type: str
    page_num: int
    font_size: float = 12.0
    translated_text: Optional[str] = None


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
# МЕТРИКИ И КЛАССИФИКАЦИЯ
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
    # Блоки длиннее этого считаются содержательными и переводятся,
    # даже если поверхностно похожи на метаданные/референсы.
    METADATA_WORD_LIMIT = 8

    def classify(self, block: Block, page_width: float, avg_font_size: float) -> str:
        if not block.lines:
            return "empty"
        m = block_metrics(block, page_width)
        if not m.text or m.total_spans == 0:
            return "empty"
        if not re.search(r'[A-Za-zА-Яа-я]', m.text):
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


def _row_columns(block: Block, tol: float = 8.0) -> Optional[List[float]]:
    """Колонки строки таблицы по x0 строк/спанов (ячейки = отдельные строки/спаны)."""
    if not block.lines:
        return None
    if len(block.lines) == 1:
        xs = [s.bbox[0] for s in block.lines[0].spans if s.text.strip()]
        if len(xs) < 3:
            return None
        gaps = [b - a for a, b in zip(sorted(xs), sorted(xs)[1:])]
        if gaps and min(gaps) < 5:
            return None
    else:
        xs = [ln.bbox[0] for ln in block.lines if ln.spans]
        if len(xs) < 3:
            return None
    clusters = []
    for x in sorted(xs):
        if clusters and x - clusters[-1][-1] <= tol:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    if len(clusters) < 3:
        return None
    return [sum(c) / len(c) for c in clusters]


def _columns_align(ref: List[float], cols: List[float],
                   tol: float = 10.0, min_frac: float = 0.5) -> bool:
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
    block = make_block(type="table", page_num=caption.page_num, bbox=bbox,
                       lines=list(caption.lines), caption=caption.text, table_data=data)
    for rb in row_blocks:
        block.lines.extend(rb.lines)
    return block


def _find_span_table_blocks(blocks: List[Block]) -> Tuple[List[Block], List[Block]]:
    """Текстовые таблицы: подпись «Table N» + следующие выровненные по колонкам строки.
    Возвращает (merged_table_blocks, new_blocks), где подпись заменена на блок таблицы."""
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


def mark_table_blocks(text_blocks: List[Block], table_blocks: List[Block]):
    for block in text_blocks:
        if block.type == "table":
            continue
        for table in table_blocks:
            if intersection_ratio(block.bbox, table.bbox) > 0.30:
                block.type = "table"
                break


def _heuristic_table_data(text: str) -> Optional[List[List[str]]]:
    """Пытается распознать таблицу по структуре текста, когда find_tables() не сработал."""
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if len(lines) < 3:
        return None
    rows = []
    for ln in lines:
        cells = [c.strip() for c in re.split(r'\s{2,}|\t', ln) if c.strip()]
        if len(cells) < 3:
            return None
        rows.append(cells)
    col_counts = {len(r) for r in rows}
    if len(col_counts) != 1:
        return None
    return rows


def heuristic_table_blocks(blocks: List[Block]) -> List[Block]:
    converted = []
    for b in blocks:
        if b.type == "table" or b.type == "figure":
            continue
        rows = _heuristic_table_data(b.text)
        if rows:
            b.type = "table"
            b.table_data = rows
            converted.append(b)
    return converted


def _looks_like_table_data(text: str) -> bool:
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if len(lines) < 3:
        return False
    if any(re.search(r'[.!?]\s*$', ln) for ln in lines):
        return False
    col_counts = {len(re.split(r'\s{2,}|\t', ln)) for ln in lines}
    if len(col_counts) == 1 and next(iter(col_counts)) >= 2:
        return True
    if all(len(ln) < 60 for ln in lines):
        return True
    return False


def mark_table_regions(blocks: List[Block]) -> int:
    """Помечает как таблицы блоки «Table N. ...» и следующие за ними ячейки."""
    marked = 0
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
            marked += 1
            i = j
            continue
        if b.type != "table":
            b.type = "table"
            marked += 1
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
                marked += 1
                j += 1
                continue
            if _heuristic_table_data(nb.text) is not None or _looks_like_table_data(nb.text):
                rows.append(nb)
                marked += 1
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
    return marked


def _is_table_caption(text: str) -> bool:
    """Подпись таблицы: «Table N | ...» (или короткий заголовок без разделителя)."""
    if not text:
        return False
    if TABLE_CAPTION_STRICT_RE.match(text):
        return True
    if not TABLE_CAPTION_RE.match(text):
        return False
    return len(text) <= 120 and len(text.split()) <= 15


def _is_figure_caption(text: str) -> bool:
    """Подпись рисунка: «Fig. N | ...», «Figure 3. ...» (якорно, не просто упоминание)."""
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
    if re.search(r'\b(?:illustrates?|shows?|depicts?|displays?|demonstrates?|presents?|'
                 r'изобража\w*|показыва\w*|иллюстрир\w*)\b', text, re.I):
        return False
    return True


def _rect_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def _union_rects(rects) -> fitz.Rect:
    return fitz.Rect(min(r.x0 for r in rects), min(r.y0 for r in rects),
                     max(r.x1 for r in rects), max(r.y1 for r in rects))


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
        caption = None
        cap_idx = -1
        best_dist = CAPTION_DIST_THRESHOLD
        for idx, block in candidates:
            if idx in caption_ids:
                continue
            d = _rect_gap(img_bbox, block.bbox)
            if d < best_dist:
                best_dist = d
                caption = block.text
                cap_idx = idx
        if cap_idx >= 0:
            caption_ids.add(cap_idx)
        fig_block = make_block(type="figure", page_num=page_num, bbox=tuple(img_bbox),
                               image_data=image_data, image_ext=ext, caption=caption)
        figure_blocks.append(fig_block)
    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining


def extract_vector_figures(page: fitz.Page, page_num: int, text_blocks: List[Block],
                           used_bboxes=(), table_bboxes=(), dpi: int = 150) -> Tuple[List[Block], List[Block]]:
    """Извлекает векторные фигуры (drawings): кластеризует области отрисовки,
    исключает таблицы/растровые изображения и рендерит регион как PNG."""
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
        caption = None
        cap_idx = -1
        best_dist = CAPTION_DIST_THRESHOLD
        for idx, b in candidates:
            if idx in caption_ids:
                continue
            d = _rect_gap(rect, b.bbox)
            if d < best_dist:
                best_dist = d
                caption = b.text
                cap_idx = idx
        if caption is None:
            continue
        caption_ids.add(cap_idx)
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False)
            image_data = base64.b64encode(pix.tobytes("png")).decode()
        except Exception as e:
            logger.warning(f"   Рендер векторной фигуры (стр. {page_num}): {e}")
            continue
        figure_blocks.append(make_block(type="figure", page_num=page_num, bbox=tuple(rect),
                                        image_data=image_data, image_ext="png", caption=caption))
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
    "Ссылки на литературу вида [[REFn]] копируй в перевод в том же виде, не меняя и не удаляя их. "
    "Верни ТОЛЬКО перевод, без пояснений."
)

GLOSSARY_PROMPT = (
    "Извлеки до 20 ключевых научных терминов из текста и дай их перевод на {lang}. "
    "Верни строго в формате JSON-объекта: {{'термин': 'перевод', ...}}. Только JSON, без пояснений."
)

TABLE_TRANSLATE_PROMPT = (
    "Переведи содержимое ячеек таблицы на {lang}. Сохрани структуру таблицы: те же строки, "
    "колонки и разделители (| и ---). Не переводи числа, единицы измерения и аббревиатуры. "
    "Верни ТОЛЬКО таблицу, без пояснений."
)

REFINE_PROMPT = (
    "Проверь перевод научного текста на {lang} на пропуски, галлюцинации и смысловые ошибки. "
    "Если есть ошибки — исправь их, вернув исправленный перевод. Если ошибок нет — верни исходный перевод. "
    "Ссылки на литературу вида [[REFn]] копируй без изменений. "
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
    prompt = GLOSSARY_PROMPT.format(lang=lang) + "\n\n" + text[:20000]
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
        prompt += f"\n\nГлоссарий терминов (используй эти переводы):\n{terms}"
    return prompt


# =========================================================
# ЗАЩИТА ССЫЛОК НА ЛИТЕРАТУРУ
# =========================================================
REF_PATTERN = re.compile(r'\[(?:[0-9][0-9,\s;\-–—]*|[A-Za-z][A-Za-z0-9-]*)\]')
REF_HIGHLIGHT_RE = re.compile(r'\[(?:[0-9][0-9,\s;\-–—]*|[A-Za-z][A-Za-z0-9-]*)\]')
_REF_PH_RE = re.compile(r'(?<![A-Za-z0-9_])\[?\[?REF_?(\d+)\]?\]?(?![A-Za-z0-9_])')


def protect_refs(text: str) -> Tuple[str, Dict[str, str]]:
    refs: Dict[str, str] = {}

    def repl(m) -> str:
        key = f"[[REF{len(refs)}]]"
        refs[key] = m.group(0)
        return key

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
    return REF_HIGHLIGHT_RE.sub(
        lambda m: f'<span class="ref-link">{m.group(0)}</span>', escaped_text
    )


# =========================================================
# ЦИТАТЫ-НОМЕРА (жирный/надстрочный шрифт) → СКВОЗНЫЕ СКОБКИ []
# =========================================================
def _wrap_citation_spans(block: Block) -> None:
    """Оборачивает в [..] номерные цитаты (1–3, 5, 11, 13–18) в тексте блока."""
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
                        new_spans.append(make_span(
                            text="[" + cit + "]",
                            font=s.font, size=s.size, flags=s.flags, color=s.color,
                            origin=s.origin, bbox=s.bbox
                        ))
                        i = j
                        continue
                new_spans.extend(spans[i:j])
                i = j
            else:
                new_spans.append(s)
                i += 1
        line.spans = new_spans


def _split_line(line: Line, idx: int) -> Tuple[Line, Line]:
    """Разделяет строку по позиции idx внутри текста спанов."""
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
            head_spans.append(make_span(text=t[:cut], font=s.font, size=s.size,
                                        flags=s.flags, color=s.color, origin=s.origin, bbox=s.bbox))
            rest_spans.append(make_span(text=t[cut:], font=s.font, size=s.size,
                                        flags=s.flags, color=s.color, origin=s.origin, bbox=s.bbox))
        pos += len(t)
    return (make_line(spans=head_spans, bbox=line.bbox, y0=line.y0),
            make_line(spans=rest_spans, bbox=line.bbox, y0=line.y0))


def _split_ref_heading(block: Block) -> List[Block]:
    """Если блок начинается с заголовка списка литературы, разделяет его
    на блок-заголовок и блок с записями списка."""
    if not block.lines:
        return [block]
    lines = block.lines
    first = lines[0]
    first_text = " ".join(s.text for s in first.spans).strip()
    if REF_HEADING_RE.match(first_text):
        if len(lines) > 1:
            head_block = make_block(type="reference_heading", page_num=block.page_num,
                                    bbox=block.bbox, lines=lines[:1])
            rest_block = make_block(type="reference", page_num=block.page_num,
                                    bbox=block.bbox, lines=lines[1:])
            return [head_block, rest_block]
        return [block]
    m = REF_HEADING_PREFIX_RE.match(first_text)
    if m and m.group(2) and len(first_text) > len(m.group(1)):
        idx = first_text.find(m.group(1).strip()) + len(m.group(1).strip())
        head_line, rest_line = _split_line(first, idx)
        head_block = make_block(type="reference_heading", page_num=block.page_num,
                                bbox=block.bbox, lines=[head_line])
        rest_block = make_block(type="reference", page_num=block.page_num,
                                bbox=block.bbox, lines=[rest_line] + lines[1:])
        return [head_block, rest_block]
    return [block]


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
            logger.info(f"OpenRouter: {len(self._models)} free-моделей")
        except Exception as e:
            logger.warning(f"OpenRouter: не удалось получить модели: {e}")
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
                    logger.info(f"   Fallback на OpenRouter")
            except Exception:
                pass
        if not fallback:
            try:
                fallback = LlamaCppTranslator(target_lang, api_base=llama_url,
                                              expected_model=llama_model, auto_find=auto_find)
                logger.info(f"   Fallback на llama: {fallback.api_base}")
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
# ПЕРЕВОД С RETRY + FALLBACK
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

    BASE_SKIP_TYPES = ("figure", "table", "empty")
    MAX_SECTION_CHARS = 6000

    def _skip_types(self) -> Tuple[str, ...]:
        return self.BASE_SKIP_TYPES + ("reference", "reference_heading")

    def _translate_one(self, text: str, lang: str,
                       glossary: Optional[Dict[str, str]] = None) -> Optional[str]:
        text = normalize_text(text)
        if not text:
            return None
        protected, refs = protect_refs(text)
        cached = None if refs else self.cache.get(text, lang)
        if cached:
            with self._stats_lock:
                self.stats["cached"] += 1
            return cached
        t0 = time.time()
        result = translate_chunk_with_retry(protected, self.translator, glossary=glossary)
        if not result and self.fallback:
            result = translate_chunk_with_retry(protected, self.fallback, glossary=glossary)
        if result and refs:
            result = restore_refs(result, refs)
            missing = [orig for orig in refs.values() if orig not in result]
            if missing:
                logger.warning(f"   Ссылки {missing} потеряны при переводе, повторная попытка")
                result2 = translate_chunk_with_retry(text, self.translator, glossary=glossary)
                if result2:
                    result2 = restore_refs(result2, refs)
                    if all(orig in result2 for orig in refs.values()):
                        result = result2
                    else:
                        missing2 = [orig for orig in refs.values() if orig not in result2]
                        logger.warning(f"   Ссылки {missing2} всё ещё потеряны после повтора")
        if result and self.refine and self.is_local and getattr(self.translator, 'can_generate', False):
            refined = self._refine(result, lang)
            if refined:
                if refs and all(orig in refined for orig in refs.values()):
                    result = refined
                elif not refs:
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
        prompt = REFINE_PROMPT.format(lang=lang) + "\n\n" + translated
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

    def _translate_table(self, block: Block, lang: str, quiet: bool = False) -> None:
        rows = block.table_data or []
        if not rows:
            return
        md_lines = []
        for i, row in enumerate(rows[:20]):
            cells = [str(c).replace("|", "\\|").replace("\n", " ") for c in row]
            md_lines.append("| " + " | ".join(cells) + " |")
            if i == 0 and len(rows) > 1:
                md_lines.append("| " + " | ".join(["---"] * len(row)) + " |")
        md_table = "\n".join(md_lines)
        prompt = TABLE_TRANSLATE_PROMPT.format(lang=lang) + "\n\n" + md_table
        try:
            raw = self.translator.generate(prompt)
        except Exception:
            raw = None
        if not raw:
            return
        parsed = self._parse_md_table(raw)
        if parsed:
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
            if cells and all(re.fullmatch(r':?-+:?', c) for c in cells):
                continue
            rows.append(cells)
        return rows if rows else None

    def translate_blocks(self, blocks: list, lang: str, quiet: bool = False) -> list:
        merged_blocks = []
        skip_types = self._skip_types()

        for block in blocks:
            text = block.text
            if not text or len(text) < 2:
                block.translation = text if text else ""
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue
            if block.type in ("figure", "table", "empty"):
                block.translation = text
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue
            if block.type in ("reference", "reference_heading"):
                block.translation = text
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue
            if block.type in ("paragraph", "list", "text"):
                _wrap_citation_spans(block)
            merged_blocks.append(block)
        blocks = merged_blocks

        if getattr(self.translator, 'can_generate', False):
            for block in blocks:
                if block.type == "table" and block.table_data:
                    self._translate_table(block, lang, quiet)

        translatable = [b for b in blocks if b.type not in skip_types and b.translation is None]
        if not translatable:
            return blocks

        glossary = None
        if getattr(self.translator, 'can_generate', False):
            sample = "\n\n".join(b.text for b in translatable)[:20000]
            if sample:
                glossary = extract_glossary(sample, self.translator, lang, self.cache)
                if glossary and not quiet:
                    logger.info(f"   Глоссарий: {len(glossary)} терминов")

        sections = []
        current = []
        for b in blocks:
            if b.type in skip_types or b.translation is not None:
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
        "strictly in the format:\nBRIEF CONTENT: 4-5 sentences\nKEY FINDINGS: list\nSTRENGTHS: list\nWEAKNESSES: list\n\n"
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
        summary_en = "\n\n".join(summaries_to_combine)

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
.page > summary { cursor: pointer; font-size: 1.05em; font-weight: bold; color: #93c5fd; padding: 4px 0 8px; user-select: none; }
.page > summary:hover { color: #60a5fa; }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
.trans-head h2, .trans-head h3, .trans-head h4 { margin: 4px 0; }
p { line-height: 1.7; margin: 0 0 10px; text-align: justify; }
.orig { color: #64748b; font-size: 0.88em; }
.trans { color: #e2e8f0; }
.ref-section h2 { color: #60a5fa; border-top: 2px solid #334155; margin-top: 24px; padding-top: 16px; padding-bottom: 6px; }
.ref { color: #64748b; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; margin-bottom: 8px; }
.meta { color: #64748b; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #334155; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid #334155; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: #0f172a; border: 1px solid #334155; padding: 8px 10px; color: #60a5fa; }
.table-wrap td { border: 1px solid #334155; padding: 6px 10px; }
.figure { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: #94a3b8; font-size: 0.85em; }
.list-block { margin-left: 20px; line-height: 1.6; }
.ref-link { color: #60a5fa; font-weight: bold; }
.table-wrap tbody tr:nth-child(even) td { background: #16213a; }
.table-pre { white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 8px 0; }
.table-caption { font-style: italic; color: #94a3b8; font-size: 0.85em; margin: 8px 0; }
.toc { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px; }
.toc ul { list-style: none; margin: 6px 0 0; padding-left: 18px; }
.toc > ul { padding-left: 0; }
.toc a { color: #93c5fd; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.btn-orig { background: #1e293b; border: 1px solid #334155; color: #94a3b8; border-radius: 6px; padding: 6px 12px; margin: 0 4px 12px 0; cursor: pointer; }
.btn-orig:hover { color: #e2e8f0; }
details.original-block { margin: 4px 0; }
details.original-block summary { color: #64748b; font-size: 0.85em; cursor: pointer; }
details.original-block summary:hover { color: #94a3b8; }
#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999; align-items: center; justify-content: center; cursor: zoom-out; }
#lightbox img { max-width: 92%; max-height: 92%; border-radius: 6px; }
</style>
"""

BLOCK_CSS_LIGHT = """
<style>
body { font-family: Arial, sans-serif; margin: 40px; background: #f0f2f5; }
.container { max-width: 960px; margin: 0 auto; }
h1 { color: #1f2937; border-bottom: 2px solid #d1d5db; padding-bottom: 12px; }
h3 { color: #374151; margin-top: 28px; }
.page { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.page > summary { cursor: pointer; font-size: 1.05em; font-weight: bold; color: #1f2937; padding: 4px 0 8px; user-select: none; }
.page > summary:hover { color: #2563eb; }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
p { line-height: 1.7; margin: 0 0 10px; text-align: justify; color: #111827; }
.orig { color: #9ca3af; font-size: 0.88em; }
.trans { color: #111827; }
.ref-section h2 { color: #1f2937; border-top: 2px solid #d1d5db; margin-top: 24px; padding-top: 16px; padding-bottom: 6px; }
.ref { color: #6b7280; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; margin-bottom: 8px; }
.meta { color: #6b7280; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid #d1d5db; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid #d1d5db; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: #f3f4f6; border: 1px solid #d1d5db; padding: 8px 10px; color: #1f2937; }
.table-wrap td { border: 1px solid #d1d5db; padding: 6px 10px; }
.figure { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: #6b7280; font-size: 0.85em; }
.list-block { margin-left: 20px; line-height: 1.6; }
.ref-link { color: #2563eb; font-weight: bold; }
.table-wrap tbody tr:nth-child(even) td { background: #f9fafb; }
.table-pre { white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 8px 0; }
.table-caption { font-style: italic; color: #6b7280; font-size: 0.85em; margin: 8px 0; }
.toc { background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px; }
.toc ul { list-style: none; margin: 6px 0 0; padding-left: 18px; }
.toc > ul { padding-left: 0; }
.toc a { color: #2563eb; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.btn-orig { background: #fff; border: 1px solid #d1d5db; color: #6b7280; border-radius: 6px; padding: 6px 12px; margin: 0 4px 12px 0; cursor: pointer; }
.btn-orig:hover { color: #111827; }
details.original-block { margin: 4px 0; }
details.original-block summary { color: #9ca3af; font-size: 0.85em; cursor: pointer; }
details.original-block summary:hover { color: #6b7280; }
#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999; align-items: center; justify-content: center; cursor: zoom-out; }
#lightbox img { max-width: 92%; max-height: 92%; border-radius: 6px; }
</style>
"""


def _reference_entries(block: Block) -> List[str]:
    """Разбивает блок списка литературы на отдельные записи."""
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


def _block_html(block: Block, in_ref: bool = False) -> str:
    e = html_mod.escape
    bt = block.type
    trans = block.translation or block.text
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


# =========================================================
# ГЕНЕРАЦИЯ HTML
# =========================================================
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
            {{ _block_html(block, False) | safe }}
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
                toc.append({
                    "anchor": f"h{idx}",
                    "text": block.translation or block.text,
                })
                idx += 1
    return toc


def generate_html(pages: List[Page], title: str, output_path: str, dark: bool = True):
    css = BLOCK_CSS if dark else BLOCK_CSS_LIGHT
    toc = _build_toc(pages)
    template = Template(HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, css=css, toc=toc,
                               _block_html=_block_html)
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
                    covered = [i for i, tb in enumerate(table_blocks)
                               if any(_rect_gap(tb.bbox, st.bbox) <= 5 for st in span_tables)]
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
                vector_blocks, text_blocks = extract_vector_figures(
                    fitz_page, page.num, text_blocks,
                    used_bboxes=used_bboxes, table_bboxes=table_bboxes)
                figure_blocks.extend(vector_blocks)
                total_images += len(vector_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения векторных фигур: {e}")
            page.blocks = non_text + text_blocks + table_blocks + figure_blocks
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
            if table_blocks_page:
                mark_table_blocks(page.blocks, table_blocks_page)
            heuristic_table_blocks(page.blocks)
            mark_table_regions(page.blocks)
            page.blocks = [b for b in page.blocks
                           if b is not None
                           and b.type != "empty"
                           and not (b.type == "table" and b.table_data is None and not b.text.strip())]

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
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    logger.info(f"PDF TRANSLATOR — {args.translator.upper()} v{__version__}")
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
