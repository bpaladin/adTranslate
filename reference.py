#!/usr/bin/env python3
"""
reference.py — Генератор структурированных рефератов научных статей (PDF).
Поддерживает входные файлы на русском и английском языках.
Использует llama.cpp сервер для генерации рефератов.
"""

import os
import sys
import re
import time
import json
import logging
import argparse
import statistics
import contextlib
import threading
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import fitz
import requests
import markdown as md
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

# ---- Logging ----
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
    os.system('')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.propagate = False

warnings_module = __import__('warnings')
warnings_module.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

STOP_EVENT = threading.Event()


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
        logger.warning(f"Error loading {path.name}: {e}")


def load_env_file():
    script_dir = Path(__file__).parent.absolute()
    for name in ('keys.env', '.env'):
        _parse_env_file(script_dir / name)


load_env_file()

REQUEST_TIMEOUT = 600
API_TIMEOUT = 120

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
        logger.info(f"  {name}: {_format_time(elapsed)}")
        if timings is not None:
            timings[name] = elapsed


RE_WHITESPACE = re.compile(r'[ \t]+')
RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')

REF_HEADING_ALT = (
    r'References?|Reference\s+list|References?\s+and\s+notes?|'
    r'Bibliography|Библиография|Библиографический\s+список|'
    r'Литература|Список\s+литературы|Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|Список\s+источников|'
    r'Использованная\s+литература|Источники'
)
REF_HEADING_RE = re.compile(rf'^\s*({REF_HEADING_ALT})\s*[:.]?\s*$', re.IGNORECASE)
REF_HEADING_PREFIX_RE = re.compile(rf'^\s*({REF_HEADING_ALT})\s*[:.]?\s+(.+)$', re.IGNORECASE)
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
TABLE_CAPTION_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+', re.I)
TABLE_CAPTION_STRICT_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+\s*[|:.–—-]', re.I)
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


def _count_list_lines(text: str) -> int:
    return len(LIST_LINE_RE.findall(text))


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
    return BlockMetrics(text=text, max_font=max_font, all_upper=all_upper, has_bold=has_bold,
                        is_centered=is_centered, total_spans=total_spans, bbox=bbox)


def _detect_table_text(text: str) -> Optional[List[List[str]]]:
    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
    if len(lines) < 2:
        return None
    punct_endings = sum(1 for ln in lines[:-1] if re.search(r'[.!?;]\s*$', ln))
    if punct_endings > len(lines) * 0.4:
        return None
    rows = []
    col_counts = set()
    for ln in lines:
        cells = [c.strip() for c in re.split(r'\s{3,}|\t+', ln) if c.strip()]
        if len(cells) < 2:
            return None
        rows.append(cells)
        col_counts.add(len(cells))
    if len(col_counts) != 1:
        return None
    return rows


def _is_table_caption(text: str) -> bool:
    if not text:
        return False
    if TABLE_CAPTION_STRICT_RE.match(text):
        return True
    if not TABLE_CAPTION_RE.match(text):
        return False
    return len(text) <= 120 and len(text.split()) <= 15


class BlockClassifier:
    LABELS = ["paragraph", "heading", "list", "metadata", "reference", "reference_heading", "empty", "table"]
    METADATA_WORD_LIMIT = 10

    def classify(self, block: Block, page_width: float, avg_font_size: float) -> str:
        if not block.lines:
            return "empty"
        m = block_metrics(block, page_width)
        if not m.text or m.total_spans == 0:
            return "empty"
        if not re.search(r'[A-Za-zА-Яа-я]', m.text):
            return "empty"
        if re.fullmatch(r'\s*(?:[a-zA-Z_]+\(\)|[\W\d]+)\s*', m.text):
            return "empty"

        word_count = len(m.text.split())

        if _detect_table_text(m.text) is not None:
            return "table"

        if REF_HEADING_RE.match(m.text):
            return "reference_heading"
        if _is_table_caption(m.text):
            return "table"

        if word_count <= self.METADATA_WORD_LIMIT:
            if re.match(r'^\[\d+\]', m.text):
                return "reference"
            if re.search(r'[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.', m.text):
                return "metadata"
            if re.search(r'(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright|Email|Corresponding|Received|Accepted|Published)', m.text, re.I):
                return "metadata"
            if re.fullmatch(r'[\w\s\-]+,\s*\d{4}', m.text):
                return "metadata"

        list_lines = _count_list_lines(m.text)
        if list_lines >= 2 or (list_lines >= 1 and len(m.text.split('\n')) >= 2):
            return "list"

        score = 0
        if m.max_font > avg_font_size * 1.25:
            score += 3
        if m.all_upper:
            score += 2
        if m.has_bold:
            score += 2
        if m.is_centered:
            score += 1
        if len(m.text) < 100:
            score += 1
        if re.match(r'^\d+(\.\d+)*\s+', m.text):
            score += 2
        if word_count <= 4 and m.max_font >= avg_font_size * 1.1:
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


_TABLE_STRATEGIES = [
    {"strategy": "lines_strict", "vertical_strategy": "lines_strict",
     "horizontal_strategy": "lines_strict", "snap_tolerance": 5, "join_tolerance": 5},
    {"strategy": "text", "vertical_strategy": "text",
     "horizontal_strategy": "text", "snap_tolerance": 5, "join_tolerance": 5},
]


def _extract_rows_safely(tab) -> Optional[List[List[str]]]:
    try:
        rows = tab.extract()
    except Exception:
        return None
    if not rows:
        return None
    cleaned = [[cell if cell else "" for cell in (row or [])] for row in rows]
    if not cleaned or all(not str(c).strip() for row in cleaned for c in row):
        return None
    if len(cleaned) < 2 or len(cleaned[0]) < 2:
        return None
    return cleaned


def _rect_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return max(dx, dy)


def _rects_overlap(a, b, threshold: float = 0.85) -> bool:
    ra, rb = fitz.Rect(a), fitz.Rect(b)
    inter = ra & rb
    if inter.is_empty:
        return False
    area = max(ra.get_area(), 1.0)
    return inter.get_area() / area > threshold


def extract_tables(page: fitz.Page, page_num: int) -> List[Block]:
    table_blocks: List[Block] = []
    seen_bboxes: List[fitz.Rect] = []

    for params in _TABLE_STRATEGIES:
        try:
            tables = page.find_tables(**params)
        except Exception as e:
            logger.debug(f"find_tables (page {page_num}, params={params['strategy']}): {e}")
            continue

        for tab in tables:
            try:
                rows = _extract_rows_safely(tab)
                if not rows:
                    continue
                tab_rect = fitz.Rect(tab.bbox)
                if any(_rect_gap(tab_rect, r) <= 3 or _rects_overlap(tab_rect, r, 0.85)
                       for r in seen_bboxes):
                    continue
                seen_bboxes.append(tab_rect)
                table_blocks.append(make_block(
                    type="table", page_num=page_num, bbox=tuple(tab.bbox), table_data=rows))
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
    matched = sum(1 for x in cols if any(abs(x - y) <= tol for y in ref))
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


def intersection_ratio(block_bbox, table_bbox) -> float:
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


def consolidate_tables(blocks: List[Block], found_table_blocks: Optional[List[Block]] = None) -> List[Block]:
    if found_table_blocks:
        for block in blocks:
            if block.type == "table":
                continue
            for table in found_table_blocks:
                if intersection_ratio(block.bbox, table.bbox) > 0.50:
                    rows = _detect_table_text(block.text)
                    if rows:
                        block.type = "table"
                        block.table_data = rows
                    break

    for b in blocks:
        if b.type in ("table", "figure", "empty"):
            continue
        rows = _detect_table_text(b.text)
        if rows:
            b.type = "table"
            b.table_data = rows
            continue

    i = 0
    n = len(blocks)
    while i < n:
        b = blocks[i]
        if b is None or b.type in ("table", "figure", "empty") or not _is_table_caption(b.text):
            i += 1
            continue
        nb = blocks[i + 1] if i + 1 < n else None
        if nb is not None and nb.type == "table" and nb.table_data:
            nb.caption = b.text
            blocks[i] = None
        i += 1

    return [b for b in blocks if b is not None and b.type != "empty"
            and not (b.type == "table" and b.table_data is None and not b.text.strip())]


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
    return [block]


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


# ---- LLM Client ----

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


def _strip_llm_wrappers(text: Optional[str]) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^\s*```(?:text|markdown)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.I)
    marker_line = re.compile(r"^\s*(?:<<<\s*(?:END_)?[A-Z_][A-Z0-9_]*\s*>>>|\[\/?(?:END_)?(?:CTX|SRC|OUT)\])\s*$", re.I)
    lines = text.splitlines()
    while lines and marker_line.match(lines[0]):
        lines.pop(0)
    while lines and marker_line.match(lines[-1]):
        lines.pop()
    text = "\n".join(lines).strip()
    text = re.sub(r"<<<\s*(?:END_)?[A-Z_][A-Z0-9_]*\s*>>>", "", text, flags=re.I)
    text = re.sub(r"\[\/?(?:END_)?(?:CTX|SRC|OUT)\]", "", text, flags=re.I)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    lines = text.splitlines()
    if lines:
        first = lines[0].strip().lower()
        preambles = ("translation:", "translated text:", "here is the translation:", "here's the translation:",
                     "перевод:", "вот перевод:", "summary:", "реферат:", "abstract:")
        if first in preambles:
            lines = lines[1:]
    return "\n".join(lines).strip()


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


class LlamaCppClient:
    def __init__(self, api_base: str = "http://localhost:8080/v1",
                 expected_model: Optional[str] = None, auto_find: bool = True):
        self.name = "LlamaCpp"
        self.api_base = None
        self._loaded_model = None
        self._server_ok = False
        self.expected_model = expected_model
        self.rate_limiter = RateLimiter(max_requests_per_second=1)
        self._consecutive_failures = 0
        self._last_health_check = 0.0

        if api_base:
            ok, models, found_url = self._check_server(api_base)
            if ok:
                server_model = models[0] if models else "unknown"
                if expected_model and not _model_matches(expected_model, server_model):
                    logger.warning(f"   On {found_url} loaded '{server_model}', but expected '{expected_model}'")
                else:
                    self.api_base = found_url
                    self._loaded_model = server_model
                    self._server_ok = True
                    logger.info(f"   llama-server: {self.api_base}, model: {self._loaded_model}")

        if not self._server_ok and auto_find:
            servers = self._find_all_servers()
            if not servers:
                raise RuntimeError("llama-server not found. Run: ./start_llama.sh 8080")
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
                    raise RuntimeError(f"Model '{expected_model}' not found.\n{available}")

        if not self._server_ok:
            raise RuntimeError("Failed to connect to any server.")

    @staticmethod
    def _find_all_servers() -> List[Dict[str, str]]:
        if os.name == 'nt':
            return []
        result = []
        try:
            output = subprocess.check_output(["pgrep", "-a", "llama-server"], text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                m = re.search(r"--port\s+(\d+)", line)
                port = int(m.group(1)) if m else 8080
                url = f"http://localhost:{port}/v1"
                ok, models, found_url = LlamaCppClient._check_server(url)
                if ok and models:
                    result.append({"url": found_url, "model": models[0]})
        except Exception:
            pass
        return result

    @staticmethod
    def _check_server(url: str) -> Tuple[bool, List[str], str]:
        base = url.rstrip("/")
        candidates = [base]
        if base.endswith("/v1"):
            candidates.append(base[:-3])
        else:
            candidates.append(base + "/v1")
        for b in candidates:
            try:
                resp = requests.get(f"{b}/models", timeout=(2, 5))
                if resp.status_code == 200:
                    models = [m.get("id", "") for m in resp.json().get("data", [])]
                    if models:
                        return True, models, b
            except Exception:
                pass
        return False, [], url

    def _health_check(self) -> bool:
        now = time.monotonic()
        if now - self._last_health_check < 30:
            return self._server_ok
        self._last_health_check = now
        ok, _, found_url = self._check_server(self.api_base)
        if not ok:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                self._server_ok = False
                logger.warning("   llama-server unavailable (3+ consecutive errors).")
        else:
            self._consecutive_failures = 0
            self._server_ok = True
            if found_url != self.api_base:
                self.api_base = found_url
        return self._server_ok

    def generate(self, prompt: str, temperature: float = 0.2, max_tokens: int = 2048,
                 timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
        if not self._server_ok:
            return None
        timeout = max(1, int(timeout))
        io_timeout = (10, timeout)
        self.rate_limiter.acquire()

        base = self.api_base
        alt_base = base + "/v1" if not base.endswith("/v1") else base[:-3]
        urls_to_try = [base, alt_base]

        payload = {
            "model": self._loaded_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens, "stream": False,
        }

        for url in urls_to_try:
            try:
                resp = requests.post(f"{url}/chat/completions", json=payload, timeout=io_timeout)
                if resp.status_code == 200:
                    self._consecutive_failures = 0
                    if url != base:
                        self.api_base = url
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
                    payload2 = {"model": self._loaded_model, "prompt": prompt, "temperature": temperature,
                                "top_p": 0.9, "max_tokens": max_tokens, "stream": False}
                    resp2 = requests.post(f"{url}/completions", json=payload2, timeout=io_timeout)
                    if resp2.status_code == 200:
                        self._consecutive_failures = 0
                        if url != base:
                            self.api_base = url
                        data = resp2.json()
                        choices = data.get("choices") or [{}]
                        content = data.get("content")
                        if not content and isinstance(choices[0], dict):
                            content = choices[0].get("text")
                        return content.strip() if content else None
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                continue

        self._consecutive_failures += 1
        return None


# ---- Summary Generation ----

def _extract_full_text(pages: List[Page]) -> str:
    all_text = []
    for page in pages:
        for block in page.blocks:
            if block.type in ("paragraph", "heading", "metadata", "text", "list"):
                text = block.text
                if text and len(text) > 10:
                    all_text.append(text)
    return "\n".join(all_text)


def _detect_source_language(text: str) -> str:
    cyrillic_chars = len(re.findall(r'[а-яёА-ЯЁ]', text))
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    total = cyrillic_chars + latin_chars
    if total == 0:
        return "en"
    return "ru" if cyrillic_chars / total > 0.3 else "en"


SUMMARY_PROMPT_TEMPLATE = """Ты - научный референт. Составь структурированный реферат научной статьи.

Требования:
- Длина реферата: ~{length} слов
- Язык реферата: {target_lang_name}
- Выдели особенности исследования
- Укажи сильные стороны
- Укажи слабые стороны

Формат ответа (строго соблюдай структуру, пиши на языке {target_lang_name}):

## КРАТКОЕ СОДЕРЖАНИЕ
(4-5 предложений о сути работы)

## КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ
(список основных результатов)

## ОСОБЕННОСТИ ИССЛЕДОВАНИЯ
(что делает эту работу уникальной)

## СИЛЬНЫЕ СТОРОНЫ
(сильные стороны методологии, результатов, оформления)

## СЛАБЫЕ СТОРОНЫ
(ограничения, недостатки, области для улучшения)

Текст статьи:
{chunk}"""

SUMMARY_PROMPT_EN = """You are a scientific reviewer. Create a structured summary of a scientific paper.

Requirements:
- Summary length: ~{length} words
- Highlight research peculiarities
- Identify strengths
- Identify weaknesses

Strictly follow this format (write in {target_lang_name}):

## BRIEF CONTENT
(4-5 sentences about the essence of the work)

## KEY FINDINGS
(list of main results)

## RESEARCH PECULIARITIES
(what makes this work unique)

## STRENGTHS
(methodology, results, presentation strengths)

## WEAKNESSES
(limitations, shortcomings, areas for improvement)

Paper text:
{chunk}"""

def generate_reference(
    pdf_path: str,
    output_html: str,
    length: int = 500,
    target_lang: str = "ru",
    llama_url: str = "http://localhost:8080/v1",
    llama_model: Optional[str] = None,
    auto_find: bool = True,
    quiet: bool = False,
) -> Dict[str, Any]:

    timings: Dict[str, float] = {}

    # Stage 1: Extract PDF
    with stage_timer("1. PDF extraction", timings):
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
            except Exception as e:
                logger.warning(f"Table extraction error: {e}")
            try:
                span_tables, text_blocks = _find_span_table_blocks(text_blocks)
                table_bboxes.extend(tb.bbox for tb in span_tables)
            except Exception as e:
                logger.warning(f"Span table error: {e}")
            page.blocks = non_text + text_blocks + table_blocks

        total_text = "".join(block.text for page in pages for block in page.blocks)
        if len(total_text.strip()) < 10:
            logger.error("No text found in PDF (possibly image-only or scanned).")
            sys.exit(1)
        doc.close()
    logger.info(f"   Extracted {len(pages)} pages")

    # Stage 2: Classify blocks
    with stage_timer("2. Classification", timings):
        for page in pages:
            all_sizes = [span.size for block in page.blocks
                         if block.type in ("paragraph", "heading", "metadata")
                         for line in block.lines for span in line.spans]
            avg_size = statistics.median(all_sizes) if all_sizes else 12.0
            for block in page.blocks:
                if block.type == "text":
                    block.type = _classifier.classify(block, page.width, avg_size)

    # Stage 3: Post-processing
    with stage_timer("3. Post-processing", timings):
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

    # Stage 4: Extract text and detect language
    full_text = _extract_full_text(pages)
    source_lang = _detect_source_language(full_text)
    target_lang_name = {"ru": "русском", "en": "English", "de": "Deutsch", "fr": "Français"}.get(target_lang, target_lang)
    source_lang_name = {"ru": "русском", "en": "English"}.get(source_lang, source_lang)

    if not quiet:
        logger.info(f"   Detected language: {source_lang_name}")
        logger.info(f"   Target language: {target_lang_name}")

    # Stage 5: Generate summary with LLM
    with stage_timer("4. Summary generation", timings):
        try:
            client = LlamaCppClient(api_base=llama_url, expected_model=llama_model, auto_find=auto_find)
        except RuntimeError as e:
            logger.error(f"   Failed to connect to llama-server: {e}")
            return {"error": str(e)}

        chunks = _chunk_text_by_sentences(full_text, max_chunk_size=6000)
        chunk_summaries = []
        start_time = time.time()

        if not quiet:
            logger.info(f"   Processing {len(chunks)} chunks...")

        pbar = tqdm(total=len(chunks), desc=" Summarizing", unit="chunk",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]") if (TQDM and not quiet) else None

        for i, chunk in enumerate(chunks):
            if STOP_EVENT.is_set():
                break
            if source_lang == "ru":
                prompt = SUMMARY_PROMPT_TEMPLATE.format(length=length // max(1, len(chunks)),
                                                        target_lang_name=target_lang_name, chunk=chunk)
            else:
                prompt = SUMMARY_PROMPT_EN.format(length=length // max(1, len(chunks)),
                                                  target_lang_name=target_lang_name, chunk=chunk)
            summary = client.generate(prompt, temperature=0.2, max_tokens=2048)
            if summary:
                chunk_summaries.append(_strip_llm_wrappers(summary))
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

    if not chunk_summaries:
        return {"error": "Failed to generate summary", "pages": len(pages)}

    # Stage 6: Combine summaries
    with stage_timer("5. Combining summaries", timings):
        if len(chunk_summaries) > 3:
            combined = "\n\n".join(chunk_summaries)
            combine_prompt = (
                f"Объедини эти фрагменты реферата в единый структурированный реферат на {target_lang_name} языке. "
                f"Сохрани формат: КРАТКОЕ СОДЕРЖАНИЕ, КЛЮЧЕВЫЕ РЕЗУЛЬТАТЫ, ОСОБЕННОСТИ, СИЛЬНЫЕ СТОРОНЫ, СЛАБЫЕ СТОРОНЫ.\n\n"
                if target_lang == "ru" else
                f"Combine these summary fragments into a single structured summary in {target_lang_name}. ..."
            )
            final_summary = client.generate(combine_prompt + combined, temperature=0.2, max_tokens=4096)
            if final_summary:
                summary_md = _strip_llm_wrappers(final_summary)
            else:
                summary_md = combined
        else:
            summary_md = "\n\n".join(chunk_summaries)

        summary_html = md.markdown(summary_md)

    elapsed = time.time() - start_time

    # Stage 7: Generate HTML
    with stage_timer("6. HTML generation", timings):
        title = f"Реферат: {os.path.basename(pdf_path)}" if target_lang == "ru" else f"Summary: {os.path.basename(pdf_path)}"
        html_content = _build_summary_html(title, summary_html, source_lang_name, target_lang_name, length, timings)

        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"   HTML saved: {output_html}")

    return {
        "pages": len(pages),
        "chunks": len(chunks),
        "source_lang": source_lang,
        "target_lang": target_lang,
        "time": elapsed,
        "timings": timings,
        "model": getattr(client, '_loaded_model', client.name),
    }


def _build_summary_html(title: str, summary_html: str, source_lang: str, target_lang: str,
                        length: int, timings: Dict[str, float]) -> str:
    timings_html = "".join(
        f"<tr><td>{stage}</td><td>{_format_time(t)}</td></tr>"
        for stage, t in timings.items()
    )

    return f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --accent: #60a5fa;
            --accent-light: #93c5fd;
            --border: #334155;
            --success: #34d399;
            --warning: #fbbf24;
            --danger: #f87171;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.7;
            padding: 20px;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}

        .header {{
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-primary));
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
        }}

        .header h1 {{
            font-size: 2em;
            color: var(--accent);
            margin-bottom: 12px;
        }}

        .meta-info {{
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            color: var(--text-secondary);
            font-size: 0.9em;
        }}

        .meta-info span {{
            display: flex;
            align-items: center;
            gap: 6px;
        }}

        .summary-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 32px;
            margin-bottom: 24px;
        }}

        .summary-card h2 {{
            color: var(--accent);
            font-size: 1.5em;
            margin-bottom: 20px;
            padding-bottom: 12px;
            border-bottom: 2px solid var(--border);
        }}

        .summary-card h3 {{
            color: var(--accent-light);
            font-size: 1.2em;
            margin: 24px 0 12px;
        }}

        .summary-card ul {{
            margin-left: 24px;
            margin-bottom: 16px;
        }}

        .summary-card li {{
            margin-bottom: 8px;
            color: var(--text-primary);
        }}

        .summary-card p {{
            margin-bottom: 16px;
            text-align: justify;
        }}

        .stats-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
        }}

        .stats-table th,
        .stats-table td {{
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        .stats-table th {{
            color: var(--accent);
            font-weight: 600;
        }}

        .stats-table tr:hover {{
            background: rgba(96, 165, 250, 0.05);
        }}

        .footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85em;
            margin-top: 32px;
            padding-top: 16px;
            border-top: 1px solid var(--border);
        }}

        @media (max-width: 640px) {{
            .header {{ padding: 20px; }}
            .header h1 {{ font-size: 1.5em; }}
            .summary-card {{ padding: 20px; }}
            .meta-info {{ flex-direction: column; gap: 8px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <div class="meta-info">
                <span>Язык оригинала: {source_lang}</span>
                <span>Язык реферата: {target_lang}</span>
                <span>Целевая длина: ~{length} слов</span>
            </div>
        </div>

        <div class="summary-card">
            {summary_html}
        </div>

        <div class="summary-card">
            <h2>Статистика генерации</h2>
            <table class="stats-table">
                <thead>
                    <tr><th>Этап</th><th>Время</th></tr>
                </thead>
                <tbody>
                    {timings_html}
                </tbody>
            </table>
        </div>

        <div class="footer">
            Generated by reference.py v{__version__} | llama.cpp
        </div>
    </div>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(
        description="Research paper summarizer — generates structured reference summaries from PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reference.py paper.pdf
  python reference.py paper.pdf -l en --length 800
  python reference.py paper.pdf --llama-url http://localhost:8081/v1
        """
    )
    parser.add_argument("input", help="Input PDF file")
    parser.add_argument("-o", "--output", help="Output HTML file (default: <input>_reference.html)")
    parser.add_argument("-l", "--lang", default="ru", choices=["ru", "en", "de", "fr", "es"],
                        help="Target language for summary (default: ru)")
    parser.add_argument("--length", type=int, default=500,
                        help="Approximate summary length in words (default: 500)")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1",
                        help="llama.cpp server URL (default: http://localhost:8080/v1)")
    parser.add_argument("--llama-model", help="Expected model name")
    parser.add_argument("--no-auto-find", action="store_true",
                        help="Disable automatic server discovery")
    parser.add_argument("-q", "--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
        for h in logger.handlers:
            h.setLevel(logging.DEBUG)

    if not os.path.isfile(args.input):
        logger.error(f"File not found: {args.input}")
        sys.exit(1)

    output_html = args.output or f"{os.path.splitext(args.input)[0]}_reference.html"

    logger.info("=" * 60)
    logger.info(f"REFERENCE GENERATOR v{__version__}")
    logger.info("=" * 60)

    try:
        result = generate_reference(
            pdf_path=args.input,
            output_html=output_html,
            length=args.length,
            target_lang=args.lang,
            llama_url=args.llama_url,
            llama_model=args.llama_model,
            auto_find=not args.no_auto_find,
            quiet=args.quiet,
        )

        if "error" in result:
            logger.error(f"Error: {result['error']}")
            sys.exit(1)

        logger.info("=" * 60)
        logger.info("GENERATION STATISTICS:")
        logger.info(f"   Pages processed: {result['pages']}")
        logger.info(f"   Text chunks: {result['chunks']}")
        logger.info(f"   Source language: {result['source_lang']}")
        logger.info(f"   Target language: {result['target_lang']}")
        logger.info(f"   Model: {result['model']}")
        logger.info(f"   Generation time: {_format_time(result['time'])}")
        timings = result.get("timings", {})
        if timings:
            logger.info("  PROFILING:")
            for stage, t in timings.items():
                logger.info(f"   {stage}: {_format_time(t)}")
        logger.info("=" * 60)
        logger.info(f"Done: {output_html}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
