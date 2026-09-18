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


def _cluster_figure_blocks(blocks, gap: float = 35.0):
    """Кластеризация figure-блоков по близости bbox для запекания составных рисунков.

    Однопроходная кластеризация зависит от порядка: вложенный блок может
    образовать отдельный кластер раньше, чем union соседа его накроет.
    Поэтому после прохода сливаем кластеры с пересекающимися/вложенными
    union'ами до неподвижной точки (иначе дубли изображений в HTML).
    """
    clusters: List[List] = []
    for b in sorted(blocks, key=lambda x: (x.bbox[1], x.bbox[0])):
        r = fitz.Rect(b.bbox)
        placed = False
        for cl in clusters:
            union = _union_rects([fitz.Rect(x.bbox) for x in cl])
            if _rect_gap(union, r) <= gap:
                cl.append(b)
                placed = True
                break
        if not placed:
            clusters.append([b])
    # Пост-проход: слить кластеры с пересекающимися union'ами
    changed = True
    while changed and len(clusters) > 1:
        changed = False
        unions = [_union_rects([fitz.Rect(x.bbox) for x in cl]) for cl in clusters]
        used = [False] * len(clusters)
        merged: List[List] = []
        for i, cl in enumerate(clusters):
            if used[i]:
                continue
            acc = list(cl)
            acc_union = unions[i]
            used[i] = True
            for j in range(i + 1, len(clusters)):
                if used[j]:
                    continue
                if _rect_gap(acc_union, unions[j]) <= gap:
                    acc.extend(clusters[j])
                    acc_union = _union_rects(
                        [fitz.Rect(x.bbox) for x in acc])
                    used[j] = True
                    changed = True
            merged.append(acc)
        clusters = merged
    return clusters


def bake_composite_figures(page: fitz.Page, figure_blocks, gap: float = 35.0, dpi: int = 150):
    """Запекание близких растровых/векторных фигур в один pixmap.
    Составной рисунок (несколько панелей A,B,C рядом) должен стать одним изображением,
    а не набором отдельных <figure>. Дубликаты уже отфильтрованы в extract_images."""
    if len(figure_blocks) <= 1:
        return figure_blocks
    clusters = _cluster_figure_blocks(figure_blocks, gap=gap)
    if len(clusters) == len(figure_blocks):
        return figure_blocks  # ничего не кластеризовалось
    baked: List = []
    page_w, page_h = page.rect.width, page.rect.height
    for cl in clusters:
        if len(cl) == 1:
            baked.append(cl[0])
            continue
        # не запекаем кластеры, раскиданные по всей странице (например, логотипы на разных углах)
        # если union слишком вытянут и покрывает >85% ширины и >70% высоты — пропускаем
        union = _union_rects([fitz.Rect(b.bbox) for b in cl])
        if union.width > page_w * 0.85 and union.height > page_h * 0.55:
            baked.extend(cl)
            continue
        # отбрасываем слишком мелкие / вырожденные кластеры
        if union.width < 30 or union.height < 20 or union.get_area() < 1200:
            baked.extend(cl)
            continue
        # выбираем подпись: первая непустая в кластере, иначе ближайшая к union
        caption = next((b.caption for b in cl if b.caption), None)
        # рендерим union
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=union,
                                  alpha=False, colorspace=fitz.csRGB)
            image_data = base64.b64encode(pix.tobytes("png")).decode()
        except Exception:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=union, alpha=False)
                image_data = base64.b64encode(pix.tobytes("png")).decode()
            except Exception as e:
                logger.warning(f"   Запекание составного рисунка (стр. {page.number+1}): {e}")
                baked.extend(cl)
                continue
        # определяем страницу для блока (у всех одинаковая)
        page_num = cl[0].page_num
        baked_block = make_block(type="figure", page_num=page_num, bbox=tuple(union),
                                 image_data=image_data, image_ext="png", caption=caption)
        baked.append(baked_block)
    return baked


def extract_images(page: fitz.Page, page_num: int, text_blocks) -> Tuple[List, List]:
    image_list = page.get_images(full=True)
    figure_blocks = []
    caption_ids = set()
    candidates = [(i, b) for i, b in enumerate(text_blocks) if _is_figure_caption(b.text)]
    seen_xrefs = set()
    seen_hash_to_bboxes: dict = {}
    seen_bboxes = []
    for img in image_list:
        xref = img[0]
        if xref != 0 and xref in seen_xrefs:
            continue
        if xref != 0:
            seen_xrefs.add(xref)
        try:
            info = page.parent.extract_image(xref)
        except Exception as e:
            logger.warning(f"   Изобр. xref={xref} (стр. {page_num}): {e}")
            continue
        data = info.get("image")
        ext = info.get("ext", "png")
        if not data:
            continue
        h = hash(data)
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        for rect in rects:
            img_bbox = fitz.Rect(rect)
            if img_bbox.get_area() < 300:
                continue
            duplicate = False
            # проверка дубликата по bbox (перекрытие >70% или почти совпадающие координаты)
            for b in seen_bboxes:
                inter = img_bbox & fitz.Rect(b)
                if not inter.is_empty:
                    small_area = min(img_bbox.get_area(), fitz.Rect(b).get_area())
                    if small_area and inter.get_area() / small_area > 0.7:
                        duplicate = True
                        break
                if _rect_gap(img_bbox, b) < 1.0 and abs(img_bbox.get_area() - fitz.Rect(b).get_area()) / max(1, img_bbox.get_area()) < 0.05:
                    duplicate = True
                    break
            if duplicate:
                continue
            # дедуп по содержимому: тот же хеш уже встречался и bbox перекрывается — пропускаем
            # если тот же хеш но bbox далеко — считаем отдельным вхождением (разные места)
            if h in seen_hash_to_bboxes:
                for prev_bbox in seen_hash_to_bboxes[h]:
                    inter = img_bbox & fitz.Rect(prev_bbox)
                    if not inter.is_empty and inter.get_area() / min(img_bbox.get_area(), fitz.Rect(prev_bbox).get_area()) > 0.5:
                        duplicate = True
                        break
                    if _rect_gap(img_bbox, prev_bbox) < 2.0:
                        duplicate = True
                        break
                if duplicate:
                    continue
            seen_bboxes.append(tuple(img_bbox))
            seen_hash_to_bboxes.setdefault(h, []).append(tuple(img_bbox))
            caption, cap_idx = _nearest_caption(img_bbox, candidates, caption_ids)
            if cap_idx >= 0:
                caption_ids.add(cap_idx)
            else:
                caption = None
            image_data = base64.b64encode(data).decode()
            fig_block = make_block(type="figure", page_num=page_num, bbox=tuple(img_bbox),
                                   image_data=image_data, image_ext=ext, caption=caption)
            figure_blocks.append(fig_block)
    remaining = [b for i, b in enumerate(text_blocks) if i not in caption_ids]
    return figure_blocks, remaining


_CAPTION_LIKE_RE = re.compile(
    r"^\s*(?:fig\.?|figure|рис\.?|рисунок|схема|scheme|табл\w*|table|"
    r"график|chart|diagram|диаграмма)\b", re.IGNORECASE,
)


def _looks_like_figure_label(text: str, nlines: int = 1,
                             font_size: float = 0.0,
                             body_median: float = 0.0) -> bool:
    """Короткая подпись внутри/у края графика: тики осей, n=24, единицы,
    однострочные ряды меток («Passive viewing Cued recall …»).

    Длинный текст (>80 символов) — всегда абзац, не метка.
    Подписи «Fig./Рис. N» и ссылки вида «Table S1.» обрабатывают другие
    этапы, сюда не входят. Однострочный текст без цифр принимаем, только
    если кегль не крупнее основного (защита от заголовков под рисунком).
    """
    t = (text or "").strip()
    if not t or len(t) > 80:
        return False
    if _is_figure_caption(t) or _CAPTION_LIKE_RE.match(t):
        return False
    if re.search(r"\d", t):
        return True
    if len(t) <= 12 and re.fullmatch(r"\(?[A-Za-zА-Яа-яЁё]{1,12}\)?", t):
        return True
    if (nlines <= 1 and len(t) <= 60 and body_median > 0
            and font_size <= body_median + 0.5):
        return True
    return False


def _block_overlap_ratio(block_bbox, rect: fitz.Rect) -> float:
    """Доля площади текстового блока, покрытая rect."""
    try:
        bb = fitz.Rect(block_bbox)
    except Exception:
        return 0.0
    inter = bb & rect
    if inter.is_empty:
        return 0.0
    area = bb.get_area() or 1.0
    return inter.get_area() / area


def _render_rect(page: fitz.Page, rect: fitz.Rect, dpi: int) -> Optional[str]:
    """Рендер rect в base64 PNG; None при ошибке."""
    try:
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect,
                              alpha=False, colorspace=fitz.csRGB)
        return base64.b64encode(pix.tobytes("png")).decode()
    except Exception:
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=rect, alpha=False)
            return base64.b64encode(pix.tobytes("png")).decode()
        except Exception as e2:
            logger.warning(f"   Рендер векторной фигуры: {e2}")
            return None


def _cluster_passes_size(rect: fitz.Rect, page_w: float, page_h: float) -> bool:
    """Базовый размерный фильтр кластера (общий для обоих этапов)."""
    if rect.width < 30 or rect.height < 20 or rect.get_area() < 1500:
        return False
    if rect.width > page_w * 0.6 and rect.height < page_h * 0.15:
        return False
    return True


def extract_vector_figures(page: fitz.Page, page_num: int, text_blocks,
                           used_bboxes=(), table_bboxes=(), dpi: int = 150) -> Tuple[List, List]:
    """
    Извлечение векторных фигур (diagram, charts, plots).

    Этап 1: кластеры с подписями «Fig./Рис./Схема N» — как раньше.
    Этап 2 (строгий): кластеры без подписи извлекаются, только если рядом
      есть короткие метки графика (тики осей, n=24, единицы измерения).
      Линейки/подчёркивания и кластеры, накрывающие абзац целиком,
      отбрасываются. Найденные метки включаются в clip фигуры и удаляются
      из текстового потока (они видимы на изображении).
    """
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
    absorbed_ids = set()  # индексы меток, включённых в clip фигуры

    # ── Этап 1: кластеры с подписями (без изменений) ──
    for cl in clusters:
        rect = _union_rects(cl)
        if not _cluster_passes_size(rect, page_w, page_h):
            continue
        if _overlaps_any(rect, table_rects, 0.4) or _overlaps_any(rect, used_rects, 0.4):
            continue
        caption, cap_idx = _nearest_caption(rect, candidates, caption_ids)
        if caption is None:
            continue
        caption_ids.add(cap_idx)

        image_data = _render_rect(page, rect, dpi)
        if image_data is None:
            continue

        figure_blocks.append(make_block(type="figure", page_num=page_num, bbox=tuple(rect),
                                        image_data=image_data, image_ext="png", caption=caption))

    # ── Этап 2: кластеры без подписи — только с метками графика ──
    try:
        sizes = sorted(float(getattr(b, "font_size", 0) or 0) for b in text_blocks)
        body_median = sizes[len(sizes) // 2] if sizes else 0.0
    except Exception:
        body_median = 0.0

    def _label_info(i, b):
        # Считаем ВИЗУАЛЬНЫЕ строки (кластеризация центров по Y):
        # PyMuPDF дробит ряд меток («Passive viewing Cued recall …»)
        # на несколько dict-строк с одинаковым y.
        lines = getattr(b, "lines", None) or []
        try:
            fs = float(getattr(b, "font_size", 0) or 0)
        except Exception:
            fs = 0.0
        try:
            tol = max(2.0, 0.6 * (fs or 10.0))
            centers = sorted(
                (float(ln.y0) + float(ln.y1)) * 0.5 for ln in lines
            )
            nrows = 0
            last = None
            for cy in centers:
                if last is None or abs(cy - last) > tol:
                    nrows += 1
                last = cy
            nlines = nrows or 1
        except Exception:
            nlines = len(lines) or 1
        return nlines, fs

    for cl in clusters:
        rect = _union_rects(cl)
        tiny = not _cluster_passes_size(rect, page_w, page_h)
        strip = rect.width > page_w * 0.6 and rect.height < page_h * 0.15
        # Волосяные линейки — никогда не фигуры; тонкие полосы (оси
        # таймлайнов) и мелкие кластеры — только при сильных метках.
        if rect.width < 8 or rect.height < 8:
            continue
        if _overlaps_any(rect, table_rects, 0.4) or _overlaps_any(rect, used_rects, 0.4):
            continue

        # Пропускаем кластеры, которые уже извлечены на этапе 1
        if any(_rect_gap(rect, fitz.Rect(fb.bbox)) < 1.0 for fb in figure_blocks):
            continue

        # Линейка/подчёркивание, накрывшее абзац: длинный текстовый блок
        # почти целиком внутри rect — это не фигура (защита от избыточного
        # запекания: elibrary «История изучения...» + колонтитулы).
        veto = False
        for i, b in enumerate(text_blocks):
            if i in caption_ids or i in absorbed_ids:
                continue
            btext = getattr(b, "text", "") or ""
            if len(btext.strip()) > 80 and not _is_figure_caption(btext):
                if _block_overlap_ratio(b.bbox, rect) > 0.6:
                    veto = True
                    break
        if veto:
            continue

        # Ищем метки: пересечение с rect (+6pt) либо близость (<=14pt)
        # с перекрытием по X. Абзацы (длинный текст) не трогаем.
        # Мелким/полосным кластерам доверяем только пересекающиеся метки
        # (номер страницы под линейкой колонтитула — не метка).
        near = fitz.Rect(rect.x0 - 6, rect.y0 - 6, rect.x1 + 6, rect.y1 + 6)
        absorbed = []
        absorbed_overlap = []
        for i, b in enumerate(text_blocks):
            if i in caption_ids or i in absorbed_ids:
                continue
            btext = (getattr(b, "text", "") or "").strip()
            if not btext:
                continue
            nlines, fs = _label_info(i, b)
            if not _looks_like_figure_label(btext, nlines, fs, body_median):
                continue
            try:
                bb = fitz.Rect(b.bbox)
            except Exception:
                continue
            if _block_overlap_ratio(b.bbox, near) >= 0.35:
                absorbed.append(i)
                absorbed_overlap.append(i)
                continue
            if tiny or strip:
                continue
            gap = _rect_gap((bb.x0, bb.y0, bb.x1, bb.y1),
                            (rect.x0, rect.y0, rect.x1, rect.y1))
            x_overlap = max(0.0, min(bb.x1, rect.x1) - max(bb.x0, rect.x0))
            if gap <= 14.0 and x_overlap >= 0.5 * max(1.0, bb.width):
                absorbed.append(i)

        # Без меток фигуру не создаём: голые линейки не запекаем.
        # Мелким/полосным кластерам нужны именно пересекающиеся метки.
        if not absorbed or ((tiny or strip) and not absorbed_overlap):
            continue

        # Расширяем clip, чтобы метки попали в изображение
        clip = fitz.Rect(rect)
        for i in absorbed:
            try:
                clip |= fitz.Rect(text_blocks[i].bbox)
            except Exception:
                pass
        absorbed_ids.update(absorbed)

        image_data = _render_rect(page, clip, dpi)
        if image_data is None:
            absorbed_ids.difference_update(absorbed)
            continue

        figure_blocks.append(make_block(
            type="figure", page_num=page_num, bbox=tuple(clip),
            image_data=image_data, image_ext="png", caption=None,
        ))

    remaining = [b for i, b in enumerate(text_blocks)
                 if i not in caption_ids and i not in absorbed_ids]
    return figure_blocks, remaining