import re
import base64
import logging
from typing import List, Optional, Tuple

import fitz

from .models import make_block

logger = logging.getLogger(__name__)

FIGURE_CAPTION_RE = re.compile(r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Рис\.?\s*\d+|Схема\s*\d+)', re.I)
FIGURE_CAPTION_STRICT_RE = re.compile(r'^\s*(?:Fig(?:ure)?\.?\s*\d+|Рис\.?\s*\d+|Схема\s*\d+)\s*[|:.,–—-]', re.I)
CAPTION_DIST_THRESHOLD = 120.0


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


def extract_images(page: fitz.Page, page_num: int, text_blocks) -> Tuple[List, List]:
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
        fig_block = make_block(type="figure", page_num=page_num, bbox=tuple(img_bbox),
                               image_data=image_data, image_ext=ext, caption=caption)
        figure_blocks.append(fig_block)
    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining


def extract_vector_figures(page: fitz.Page, page_num: int, text_blocks,
                           used_bboxes=(), table_bboxes=(), dpi: int = 150) -> Tuple[List, List]:
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
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect,
                                  alpha=False, colorspace=fitz.csRGB)
            image_data = base64.b64encode(pix.tobytes("png")).decode()
        except Exception:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False)
                image_data = base64.b64encode(pix.tobytes("png")).decode()
            except Exception as e2:
                logger.warning(f"   Рендер векторной фигуры (стр. {page_num}): {e2}")
                continue

        figure_blocks.append(make_block(type="figure", page_num=page_num, bbox=tuple(rect),
                                        image_data=image_data, image_ext="png", caption=caption))
    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining