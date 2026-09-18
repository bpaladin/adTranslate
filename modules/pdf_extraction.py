import logging
from typing import List, Optional
from collections import defaultdict

import fitz

from .models import make_block, make_span, make_line, Page

try:
    from pymupdf_layout import extract_layout
    HAVE_LAYOUT = True
except ImportError:
    HAVE_LAYOUT = False

logger = logging.getLogger(__name__)


def _merge_fragmented_lines(lines, page: fitz.Page, block_bbox: tuple, y_tol: float = 2.0):
    """Склейка фрагментированных строк.

    Проблема: в некоторых PDF ligature/специальные шрифты (fi, fl, ffi)
    разбивают одну строку на множество крошечных фрагментов (по 1-2 символа),
    каждый из которых PyMuPDF выдаёт как отдельную "строку" с одним спаном.

    Решение: группируем строки с одинаковым y0 (±y_tol), и если в группе
    больше одного фрагмента — склеиваем их в одну строку, извлекая полный
    текст через clip на всю ширину блока.
    """
    if len(lines) <= 1:
        return lines

    # Группировка по y0 (округлённому)
    y_groups = defaultdict(list)
    for line in lines:
        y_key = round(line.y0 / y_tol) * y_tol
        y_groups[y_key].append(line)

    merged = []
    for y_key in sorted(y_groups.keys()):
        group = y_groups[y_key]
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Проверяем, фрагментирована ли группа:
        # много строк с коротким текстом (медиана < 5 символов)
        texts = []
        for ln in group:
            t = "".join(s.text for s in ln.spans).strip()
            texts.append(t)
        avg_len = sum(len(t) for t in texts) / len(texts) if texts else 0
        max_len = max(len(t) for t in texts) if texts else 0

        # Если есть хотя бы одна длинная строка (>20 символов) — это нормальная строка,
        # не фрагмент. Просто берём её.
        if max_len > 20:
            best = max(group, key=lambda ln: len("".join(s.text for s in ln.spans)))
            merged.append(best)
            continue

        # Если группа из 1-2 фрагментов и текст короткий — это可能是 нормальная строка
        # с неполным текстом (например, конец абзаца). Не склеиваем.
        if len(group) <= 2 and avg_len > 5:
            # Берём строку с наибольшим текстом
            best = max(group, key=lambda ln: len("".join(s.text for s in ln.spans)))
            merged.append(best)
            continue

        # Фрагментация: много крошечных кусочков на одном y0
        # Сортируем по x0, склеиваем текст, извлекаем через широкий clip
        group_sorted = sorted(group, key=lambda ln: ln.bbox[0])
        x0 = min(ln.bbox[0] for ln in group_sorted)
        y0 = min(ln.y0 for ln in group_sorted)
        x1 = max(ln.bbox[2] for ln in group_sorted)
        y1 = max(ln.bbox[3] for ln in group_sorted)
        union_bbox = (x0, y0, x1, y1)

        # Извлекаем полный текст через clip на всю ширину блока
        # (чтобы захватить все фрагменты, даже если PyMuPDF разбил их)
        wide_clip = fitz.Rect(block_bbox[0], y0 - 1, block_bbox[2], y1 + 1)
        full_text = page.get_text("text", clip=wide_clip).strip()

        if full_text and len(full_text) > avg_len * 2:
            # Успешно извлекли полный текст
            line = make_line(bbox=union_bbox, y0=y0)
            span = make_span(text=full_text, size=12)
            line.spans.append(span)
            merged.append(line)
        else:
            # Не удалось — склеиваем фрагменты вручную (сортировка по x)
            combined = " ".join(t for t in texts if t)
            line = make_line(bbox=union_bbox, y0=y0)
            span = make_span(text=combined, size=12)
            line.spans.append(span)
            merged.append(line)

    return merged


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
                    block_bbox = b.get("bbox", (0, 0, 0, 0))
                    block = make_block(type="text", page_num=page_num + 1, bbox=block_bbox)
                    for line_dict in b.get("lines", []):
                        bbox = line_dict.get("bbox", (0, 0, 0, 0))
                        spans = line_dict.get("spans", [])
                        dict_line_text = "".join(s.get("text", "") for s in spans)
                        plain_line_text = page.get_text("text", clip=fitz.Rect(bbox)).strip()
                        if plain_line_text and len(plain_line_text) > len(dict_line_text):
                            line = make_line(bbox=bbox, y0=bbox[1])
                            span = make_span(text=plain_line_text, size=12)
                            line.spans.append(span)
                        else:
                            line = make_line(bbox=bbox, y0=bbox[1])
                            for span_dict in spans:
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
                    # Склейка фрагментированных строк: ligature/特殊 шрифт
                    # разбивает одну строку на множество крошечных спанов.
                    # Группируем по y0 (±2pt) и склеиваем в одну строку.
                    if block.lines and len(block.lines) > 3:
                        block.lines = _merge_fragmented_lines(block.lines, page, block_bbox)
                    if block.lines:
                        page_blocks.append(block)
            pages.append(Page(num=page_num + 1, blocks=page_blocks,
                              width=raw.get("width", 612), height=raw.get("height", 792)))
        return pages