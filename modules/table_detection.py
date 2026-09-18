import re
import logging
from typing import List, Optional, Tuple

from .models import Block, make_block, Span

logger = logging.getLogger(__name__)

TABLE_CAPTION_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+', re.I)
TABLE_CAPTION_STRICT_RE = re.compile(r'^\s*(?:Table|Таблица|Табл\.?)\s+\d+\s*[|:.–—-]', re.I)


def _is_reversed_text(text: str) -> bool:
    """Check if text appears to be reversed (e.g., 'eulav-p' instead of 'p-value')."""
    if not text or len(text) < 3:
        return False
    reversed_text = text[::-1]
    common_words = {'the', 'and', 'for', 'with', 'from', 'that', 'this', 'are', 'was',
                    'not', 'but', 'have', 'has', 'had', 'one', 'two', 'all', 'can',
                    'her', 'was', 'our', 'out', 'day', 'had', 'his', 'how', 'its',
                    'let', 'may', 'new', 'now', 'old', 'see', 'way', 'who', 'did',
                    'get', 'got', 'him', 'hit', 'man', 'put', 'say', 'she', 'too',
                    'use', 'mean', 'table', 'value', 'index', 'group',
                    'total', 'score', 'range', 'level', 'count', 'percent', 'rate',
                    'sign', 'test', 'data', 'type', 'name', 'case', 'high', 'low'}
    words_in_text = set(re.findall(r'[a-zA-Z]{3,}', text.lower()))
    words_in_reversed = set(re.findall(r'[a-zA-Z]{3,}', reversed_text.lower()))
    normal_score = len(words_in_text & common_words)
    reversed_score = len(words_in_reversed & common_words)
    if reversed_score > normal_score and reversed_score >= 1:
        return True
    if re.match(r'^[\d.,]+[<>≤≥]$', text):
        return True
    if re.match(r'^[<>≤≥][\d.,]+$', reversed_text):
        return True
    if '(' in text and ')' not in text and ')' in reversed_text:
        return True
    return False


def _fix_reversed_cell(cell: str) -> str:
    """Fix a reversed cell value."""
    if not cell:
        return cell
    lines = cell.split('\n')
    fixed_lines = [ln[::-1] for ln in lines]
    fixed_lines.reverse()
    return ' '.join(ln for ln in fixed_lines if ln.strip())


def _fix_reversed_table(rows: List[List[str]]) -> List[List[str]]:
    """Fix reversed text in table cells if detected."""
    if not rows or not rows[0]:
        return rows
    sample_cells = []
    for row in rows[:4]:
        for cell in row[:6]:
            if cell and len(cell.strip()) > 1:
                sample_cells.append(cell.strip())
    if not sample_cells:
        return rows
    reversed_count = sum(1 for c in sample_cells if _is_reversed_text(c))
    if reversed_count < 2:
        return rows
    fixed_rows = []
    for row in rows:
        fixed_row = [_fix_reversed_cell(cell) for cell in row]
        fixed_rows.append(fixed_row)
    return fixed_rows


_REF_CELL_RE = re.compile(
    r'(?:DOI:|Vol\.|Pp?\.\s|Art\.\s|http|//\s*\w+\.\w+|р\.\s|Том\s|№\s|С\.\s|\d{4}\.\s|'
    r'\w+\.\s+\w+\.\s*\d{4}|Ser\.\s|pp\.\s|pp\.,|pp\.)', re.I)
_REF_STYLE_RE = re.compile(
    r'(?:[A-Z][a-z]+\s+[A-Z]\.[A-Z]?\.|[A-Z][a-z]+\s+[A-Z]\.?\s+[A-Z][a-z]+)')
_REF_NUM_RE = re.compile(r'^\d+\.\s+[A-Z]')


def _is_reference_table(rows: List[List[str]]) -> bool:
    """Check if table cells contain reference-like entries (journal citations)."""
    if not rows:
        return False
    ref_signals = 0
    total_cells = 0
    for row in rows:
        for cell in row:
            cell = cell.strip()
            if not cell:
                continue
            total_cells += 1
            if _REF_CELL_RE.search(cell):
                ref_signals += 1
            if _REF_STYLE_RE.search(cell):
                ref_signals += 1
            if _REF_NUM_RE.match(cell):
                ref_signals += 1
    if total_cells == 0:
        return False
    return ref_signals / total_cells > 0.3


def extract_tables(page, page_num: int) -> List[Block]:
    """Детекция таблиц через pdfplumber. Принимает pdfplumber Page."""
    table_blocks: List[Block] = []
    seen_bboxes: List[tuple] = []

    try:
        tables = page.find_tables()
    except Exception as e:
        logger.debug(f"pdfplumber find_tables (стр. {page_num}): {e}")
        return table_blocks

    for tab in tables:
        try:
            raw = tab.extract()
            if not raw:
                continue
            rows = []
            for row in raw:
                cells = [(cell or "").strip() for cell in row]
                rows.append(cells)
            if not rows or all(not c for row in rows for c in row):
                continue
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            if _is_reference_table(rows):
                continue
            rows = _fix_reversed_table(rows)
            tab_rect = tuple(tab.bbox)
            if any(_rect_gap(tab_rect, r) <= 3 or
                   _rects_overlap(tab_rect, r, 0.85)
                   for r in seen_bboxes):
                continue
            seen_bboxes.append(tab_rect)
            table_blocks.append(make_block(
                type="table", page_num=page_num, bbox=tab_rect, table_data=rows))
        except Exception:
            continue
    return table_blocks


def _rect_gap(a: tuple, b: tuple) -> float:
    """Минимальное расстояние между двумя прямоугольниками (x0, y0, x1, y1)."""
    dx = max(0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx ** 2 + dy ** 2) ** 0.5


def _rects_overlap(a: tuple, b: tuple, min_frac: float = 0.5) -> bool:
    """Проверяет, что пересечение a и b составляет >= min_frac от площади a."""
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return False
    intersection = (x1 - x0) * (y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    if area_a == 0:
        return False
    return intersection / area_a >= min_frac


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


def _heuristic_table_data(text: str) -> Optional[List[List[str]]]:
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if len(lines) < 2:
        return None
    # Прозаический абзац с justification-пробелами — не таблица:
    # длинные строки без пунктуации на концах.
    avg_len = sum(len(ln) for ln in lines) / len(lines)
    if avg_len > 90 and not any(re.search(r'[.!?;]\s*$', ln) for ln in lines):
        return None
    ends_with_punct = sum(1 for ln in lines if re.search(r'[.!?;]\s*$', ln))
    if ends_with_punct > len(lines) * 0.4:
        return None
    rows = []
    for ln in lines:
        cells = [c.strip() for c in re.split(r'\s{3,}|\t+', ln.strip()) if c.strip()]
        if len(cells) < 2:
            cells = [c.strip() for c in re.split(r'\s{2,}', ln.strip()) if c.strip()]
        if len(cells) < 2:
            return None
        rows.append(cells)
    if len(rows) < 2:
        return None
    col_counts = {len(r) for r in rows}
    if len(col_counts) == 1 and next(iter(col_counts)) >= 2:
        return rows
    if len(col_counts) <= 2:
        max_cols = max(col_counts)
        normalized = []
        for r in rows:
            if len(r) < max_cols:
                r = r + [""] * (max_cols - len(r))
            normalized.append(r[:max_cols])
        if all(len(r) == max_cols for r in normalized):
            return normalized
    return None


def _looks_like_table_data(text: str) -> bool:
    lines = [ln for ln in text.split('\n') if ln.strip()]
    if len(lines) < 2:
        return False
    if any(re.search(r'[.!?]\s*$', ln) for ln in lines[:-1]):
        return False
    col_counts = {len(re.split(r'\s{3,}|\t+', ln)) for ln in lines}
    if len(col_counts) == 1 and next(iter(col_counts)) >= 2:
        return True
    if all(len(ln) < 60 for ln in lines):
        return True
    return False


_NUM_PATTERN = re.compile(r'\d+[.,:]\d+|\d+%|\d+\.\d+[eE][+-]?\d+|\b\d{1,3}(?:[,\s]\d{3})*(?:\.\d+)?\b')
_LABEL_DATA_RE = re.compile(r'^[A-Za-zА-Яа-я][\w\s\-()]{0,30}$')


def _detect_table_by_content(blocks: List[Block]) -> List[Block]:
    """Detect table-like blocks by analyzing content patterns across consecutive blocks.
    Looks for sequences of blocks with numeric data or label:value patterns."""
    if len(blocks) < 2:
        return []
    candidates = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b.type not in ("text", "paragraph"):
            i += 1
            continue
        group = [b]
        j = i + 1
        while j < len(blocks):
            nb = blocks[j]
            if nb.type not in ("text", "paragraph"):
                break
            b_text = group[-1].text
            nb_text = nb.text
            b_nums = len(_NUM_PATTERN.findall(b_text))
            nb_nums = len(_NUM_PATTERN.findall(nb_text))
            if b_nums >= 2 and nb_nums >= 2:
                group.append(nb)
                j += 1
                continue
            b_cols = len(re.split(r'\s{3,}|\t+', b_text.strip()))
            nb_cols = len(re.split(r'\s{3,}|\t+', nb_text.strip()))
            if b_cols >= 2 and nb_cols >= 2 and abs(b_cols - nb_cols) <= 1:
                group.append(nb)
                j += 1
                continue
            if (_LABEL_DATA_RE.match(b_text) or _LABEL_DATA_RE.match(nb_text)) and len(group) >= 2:
                group.append(nb)
                j += 1
                continue
            break
        if len(group) >= 3:
            candidates.extend(group)
        i = j if j > i + 1 else i + 1
    return candidates


def _cluster_columns_by_x(blocks: List[Block], eps: float = 15.0, min_samples: int = 2) -> List[float]:
    all_x = []
    for b in blocks:
        if b.type in ("table", "figure", "empty", "reference", "reference_heading", "metadata"):
            continue
        for ln in b.lines:
            line_text = "".join(s.text for s in ln.spans).strip()
            if len(line_text) > 60:
                continue
            for s in ln.spans:
                if s.text.strip():
                    all_x.append(s.bbox[0])
    if not all_x:
        return []
    all_x.sort()
    clusters = []
    for x in all_x:
        if clusters and x - clusters[-1][-1] <= eps:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    result = []
    for c in clusters:
        if len(c) >= min_samples:
            result.append(sum(c) / len(c))
    if len(result) < 2:
        return result
    filtered = []
    for i, col in enumerate(result):
        if i == 0 or col - filtered[-1] > 60:
            filtered.append(col)
    return filtered


def consolidate_tables(blocks: List[Block], found_table_blocks: Optional[List[Block]] = None) -> List[Block]:
    if found_table_blocks:
        for block in blocks:
            if block.type == "table":
                continue
            for table in found_table_blocks:
                if intersection_ratio(block.bbox, table.bbox) > 0.50:
                    if _heuristic_table_data(block.text) or _looks_like_table_data(block.text):
                        block.type = "table"
                        break
    for b in blocks:
        if b.type in ("table", "figure", "reference", "reference_heading", "metadata"):
            continue
        rows = _heuristic_table_data(b.text)
        if rows:
            b.type = "table"
            b.table_data = rows
            continue
        if b.type in ("text", "paragraph") and len(b.lines) >= 2:
            # Таблица без линеек и подписи: >=2 строк с широким gutter
            # (>=15pt) между спанами — разделителем колонок. У прозы
            # межспановые зазоры — обычные межсловные пробелы (замер по
            # корпусу: макс. ~8pt), даже в двухколоночной вёрстке.
            gutter_lines = 0
            for ln in b.lines:
                xs = sorted(
                    (s.bbox[0], s.bbox[2])
                    for s in ln.spans if s.text.strip()
                )
                if any(nxt - prev_x1 >= 15.0 for (_, prev_x1), (nxt, _)
                       in zip(xs, xs[1:])):
                    gutter_lines += 1
            if gutter_lines >= 2:
                b.type = "table"
    content_table_blocks = _detect_table_by_content(blocks)
    for b in content_table_blocks:
        if b.type not in ("text", "paragraph"):
            continue
        rows = _heuristic_table_data(b.text)
        if rows:
            b.type = "table"
            b.table_data = rows
        elif b.type == "text":
            b.type = "table"
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
    return [b for b in blocks if b is not None and b.type != "empty"
            and not (b.type == "table" and b.table_data is None and not b.text.strip())]


def _is_table_caption(text: str) -> bool:
    if not text:
        return False
    if TABLE_CAPTION_STRICT_RE.match(text):
        return True
    if not TABLE_CAPTION_RE.match(text):
        return False
    return len(text) <= 120 and len(text.split()) <= 15
