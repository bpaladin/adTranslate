"""Main PDF processing pipeline orchestrator.

Text analysis algorithm is ported from the single-file reference
``pdf2html.py`` (taken as ground truth):
- text source is a single ``page.get_text("dict")`` pass via
  :mod:`modules.text_extraction` (no ``clip`` re-extraction fallback);
- Unicode-aware hyphen-join incl. cross-block merges;
- line-level, caption-protected figure overlap removal;
- table validation with paragraph-overlap veto + dedup;
- repeating-header/footer detection with canonical running heads.
"""

import os
import sys
import re
import statistics
from typing import Dict, Any, List, Optional

import fitz
import pdfplumber

from .utils import stage_timer, _format_time
from .logging_setup import logger
from .models import Page
from .config import BLOCK_TIMEOUT
from .pdf_extraction import PDFExtractor
from .table_detection import extract_tables, _find_span_table_blocks, consolidate_tables
from .figure_extraction import extract_images, extract_vector_figures, bake_composite_figures
from .classifier import _classifier
from .citation import _split_ref_heading
from .summary import generate_summary
from .html_generation import generate_html, generate_summary_html, generate_raw_html
from .translator_factory import create_translator
from .translation_pipeline import TranslationPipeline
from .text_extraction import (
    extract_text_blocks,
    merge_hyphen_split_blocks,
    remove_figure_overlapping_text,
    _text_quality,
)

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

_FOOTER_MIN_PAGES = 3
_FIGURE_BAKE_GAP = 35.0  # same as pdf2html.py (was 45.0)


def _intersection_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    area_a = max(0.0, a.get_area())
    if area_a <= 0:
        return 0.0
    return inter.get_area() / area_a


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _canonical_running_head(text: str) -> str:
    """Canonical running-head form: lowercase, collapsed spaces, page numbers cut.

    ``Current Biology 26, ... 2016 1723`` and ``... 1724`` on neighbour
    pages share one key although page numbers differ.
    """
    t = _normalize_for_dedup(text or "")
    t = re.sub(r"^\d{1,4}\s+", "", t)
    t = re.sub(r"\s+\d{1,4}$", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _is_short_dataless_table(block) -> bool:
    if getattr(block, "type", "") != "table":
        return False
    if getattr(block, "table_data", None):
        return False
    try:
        if len((getattr(block, "text", "") or "")) > 250:
            return False
    except Exception:
        return False
    return True


def _detect_repeating_blocks(pages: List['Page']) -> None:
    """Mark headers/footers/journal running heads as metadata."""
    total_pages = len(pages)
    if total_pages < _FOOTER_MIN_PAGES:
        return

    def _eligible(block) -> bool:
        if block.type in ("paragraph", "text", "heading"):
            return True
        return _is_short_dataless_table(block)

    text_to_pages: dict = {}
    canon_to_pages: dict = {}
    for page in pages:
        for block in page.blocks:
            if not _eligible(block):
                continue
            norm = _normalize_for_dedup(block.text)
            if not norm:
                continue
            if re.fullmatch(r"\d{1,4}", norm):
                canon_to_pages.setdefault("__pagenum__", set()).add(page.num)
                continue
            if len(norm) < 5 or len(norm) > 250:
                continue
            text_to_pages.setdefault(norm, set()).add(page.num)
            canon = _canonical_running_head(block.text)
            if 10 <= len(canon) <= 250 and canon != norm:
                canon_to_pages.setdefault(canon, set()).add(page.num)

    min_pages = max(3, total_pages // 2)
    repeat_keys = {n for n, s in text_to_pages.items() if len(s) >= min_pages}
    repeat_keys |= {c for c, s in canon_to_pages.items() if len(s) >= min_pages}
    if not repeat_keys and "__pagenum__" not in canon_to_pages:
        return
    # bare page numbers always count once seen on enough pages
    if "__pagenum__" in canon_to_pages and len(canon_to_pages["__pagenum__"]) >= min_pages:
        repeat_keys.add("__pagenum__")
    if not repeat_keys:
        return
    for page in pages:
        for block in page.blocks:
            if not _eligible(block):
                continue
            norm = _normalize_for_dedup(block.text)
            if not norm:
                continue
            key = "__pagenum__" if re.fullmatch(r"\d{1,4}", norm) else None
            if key is None:
                if norm in repeat_keys:
                    key = norm
                else:
                    canon = _canonical_running_head(block.text)
                    if canon in repeat_keys:
                        key = canon
            if key is not None and key in repeat_keys:
                block.type = "metadata"


_FOOTER_RE = re.compile(
    r'^\s*(?:Page|Страница|P\.?\s*p?\.?|S\.?\s*p?\.?)\s*\d+\s*$',
    re.IGNORECASE
)
_AUTHOR_RE = re.compile(
    r'^\s*[A-Z][a-z]+\s+(?:et\s+al\.|and\s+[A-Z][a-z]+)\s*$'
)
_JOURNAL_LINE_RE = re.compile(
    r'(?:Author\s+manuscript|PMC\s+\d{4}|NIH-PA|Clin\s+Neurophysiol|Available\s+in\s+PMC)',
    re.IGNORECASE
)


def _detect_footers(pages: List['Page']) -> None:
    """Remove footer blocks (Page N, journal names) and everything below them."""
    for page in pages:
        footer_idx = None
        for i, block in enumerate(page.blocks):
            text = block.text.strip()
            if _FOOTER_RE.match(text) or _AUTHOR_RE.match(text):
                footer_idx = i
                break
            if _JOURNAL_LINE_RE.match(text) and i > 0:
                prev_texts = [page.blocks[j].text.strip() for j in range(max(0, i - 2), i)]
                if any(_FOOTER_RE.match(pt) or _AUTHOR_RE.match(pt) for pt in prev_texts):
                    footer_idx = i - 1
                    break
        if footer_idx is not None:
            page.blocks = page.blocks[:footer_idx]
            continue
        if page.blocks:
            last = page.blocks[-1]
            if _JOURNAL_LINE_RE.match(last.text.strip()):
                page.blocks = page.blocks[:-1]


def _block_font_size(block) -> float:
    fs = float(getattr(block, "font_size", 0.0) or 0.0)
    if fs > 0:
        return fs
    sizes = []
    for line in getattr(block, "lines", []) or []:
        for span in getattr(line, "spans", []) or []:
            s = float(getattr(span, "size", 0) or 0)
            if s > 0:
                sizes.append(s)
    if sizes:
        return statistics.median(sizes)
    return 0.0


def _run_extraction(pdf_path: str, debug_text: bool = False):
    """Shared stage 1: pdf2html-grade text/tables/figures extraction.

    Returns (pages, doc, pdf, total_images, total_tables).
    Callers must close doc/pdf (done here on success path via returned
    handles — actually closed inside; kept open only during processing).
    """
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract()
    doc = fitz.open(pdf_path)
    pdf = pdfplumber.open(pdf_path)
    total_images = 0
    total_tables = 0

    for page in pages:
        fitz_page = doc[page.num - 1]
        # 1. Single text source (pdf2html): ignore cropped extractor blocks,
        #    re-extract full-page dict blocks.
        text_blocks = extract_text_blocks(fitz_page, page.num)
        # Cross-block hyphen merges while blocks are still line-accurate.
        text_blocks = merge_hyphen_split_blocks(text_blocks)
        non_text = [b for b in page.blocks if b.type != "text"]

        # 2. pdfplumber tables
        plumber_tables = []
        try:
            plumber_tables = extract_tables(pdf.pages[page.num - 1], page.num)
        except Exception as e:
            logger.warning(f"Ошибка извлечения таблиц (pdfplumber): {e}")

        # 3. span tables (text structure)
        span_tables = []
        try:
            span_tables, text_blocks = _find_span_table_blocks(text_blocks)
        except Exception as e:
            logger.warning(f"Ошибка распознавания текстовых таблиц: {e}")

        # 4. Validation + dedup (pdf2html): drop tables overlapping paragraphs,
        #    return false positives back to text.
        original_text_blocks = list(text_blocks)

        def is_likely_real_table(tbl, is_plumber=False):
            if getattr(tbl, 'starts_with_bold', False):
                return False
            tbl_rect = fitz.Rect(tbl.bbox)
            for tb in original_text_blocks:
                txt_rect = fitz.Rect(tb.bbox)
                inter_ratio = max(_intersection_ratio(tbl_rect, txt_rect),
                                  _intersection_ratio(txt_rect, tbl_rect))
                if inter_ratio > 0.7:
                    return False
            if is_plumber:
                for st in span_tables:
                    st_rect = fitz.Rect(st.bbox)
                    inter_ratio = max(_intersection_ratio(tbl_rect, st_rect),
                                      _intersection_ratio(st_rect, tbl_rect))
                    if inter_ratio > 0.7:
                        return False
            return True

        final_span_tables = []
        for st in span_tables:
            if is_likely_real_table(st):
                st.type = "table"
                final_span_tables.append(st)
            else:
                st_rect = fitz.Rect(st.bbox)
                is_dup = any(_intersection_ratio(st_rect, fitz.Rect(tb.bbox)) > 0.9
                             for tb in text_blocks)
                if not is_dup:
                    text_blocks.append(st)

        final_plumber_tables = []
        for pt in plumber_tables:
            if is_likely_real_table(pt, is_plumber=True):
                if getattr(pt, 'type', '') != "table":
                    pt.type = "table"
                final_plumber_tables.append(pt)

        table_blocks = []
        for pt in final_plumber_tables:
            pt_rect = fitz.Rect(pt.bbox)
            if not any(_intersection_ratio(pt_rect, fitz.Rect(vt.bbox)) > 0.7
                       for vt in final_span_tables):
                table_blocks.append(pt)
        for st in final_span_tables:
            st_rect = fitz.Rect(st.bbox)
            if not any(_intersection_ratio(st_rect, fitz.Rect(vt.bbox)) > 0.7
                       for vt in table_blocks):
                table_blocks.append(st)

        table_bboxes = [tb.bbox for tb in table_blocks]
        total_tables += len(table_blocks)

        # 5. Figures (raster + vector + composite bake)
        if debug_text:
            logger.debug("PAGE %s: extracted %d text blocks", page.num, len(text_blocks))
            for bi, tb in enumerate(text_blocks):
                try:
                    bb = tuple(round(float(v), 2) for v in tb.bbox)
                except Exception:
                    bb = tb.bbox
                logger.debug("PAGE %s BLOCK %d bbox=%s quality=%.3f text=%r",
                             page.num, bi, bb, _text_quality(tb.text), tb.text)

        figure_blocks = []
        try:
            raster_figs, text_blocks = extract_images(fitz_page, page.num, text_blocks)
            figure_blocks.extend(raster_figs)
            used_bboxes = [fitz.Rect(b.bbox) for b in raster_figs]
            vec_figs, text_blocks = extract_vector_figures(
                fitz_page, page.num, text_blocks,
                used_bboxes=used_bboxes, table_bboxes=table_bboxes)
            figure_blocks.extend(vec_figs)
            figure_blocks = bake_composite_figures(fitz_page, figure_blocks, gap=_FIGURE_BAKE_GAP)
            total_images += len(figure_blocks)
        except Exception as e:
            logger.warning(f"Ошибка извлечения изображений: {e}")

        # Line-level, caption-protected removal (pdf2html).
        figure_rects = [fitz.Rect(b.bbox) for b in figure_blocks]
        text_blocks = remove_figure_overlapping_text(text_blocks, figure_rects,
                                                     margin_x=2, margin_y=2)

        page.blocks = non_text + text_blocks + table_blocks + figure_blocks

    return pages, doc, pdf, total_images, total_tables


def _run_classify_and_post(pages: List[Page]) -> None:
    with stage_timer("2. Классификация", {}) as _t:
        pass  # placeholder replaced below (timings handled by caller)


def _classify_pages(pages: List[Page]) -> None:
    for page in pages:
        all_sizes = []
        for block in page.blocks:
            if block.type == "text":
                fs = _block_font_size(block)
                if fs > 0:
                    all_sizes.append(fs)
            elif block.type in ("paragraph", "heading", "metadata"):
                if getattr(block, 'lines', None):
                    for line in block.lines:
                        spans = getattr(line, 'spans', []) or []
                        for span in spans:
                            size = getattr(span, 'size', 0) or 0
                            if size > 0:
                                all_sizes.append(float(size))
                else:
                    fs = _block_font_size(block)
                    if fs > 0:
                        all_sizes.append(fs)
        avg_size = statistics.median(all_sizes) if all_sizes else 12.0
        for block in page.blocks:
            if block.type == "text":
                block.type = _classifier.classify(block, page.width, avg_size)


def _post_process(pages: List[Page]) -> None:
    _detect_repeating_blocks(pages)
    _detect_footers(pages)
    for page in pages:
        new_blocks = []
        for block in page.blocks:
            if not hasattr(block, 'page_num') or not getattr(block, 'page_num', 0):
                block.page_num = page.num
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
    debug_text: bool = False,
) -> Dict[str, Any]:

    timings: Dict[str, float] = {}
    total_images = 0
    total_tables = 0

    with stage_timer("1. Извлечение PDF", timings):
        pages, doc, pdf, total_images, total_tables = _run_extraction(
            pdf_path, debug_text=debug_text)
        total_text = "".join(block.text for page in pages for block in page.blocks)
        if len(total_text.strip()) < 10:
            logger.error(" В PDF не найден текст (возможно, файл состоит только из изображений или отсканирован).")
            doc.close()
            pdf.close()
            sys.exit(1)
        doc.close()
        pdf.close()
    logger.info(f"   Извлечено {len(pages)} стр., {total_images} изобр., {total_tables} табл.")

    with stage_timer("2. Классификация", timings):
        _classify_pages(pages)

    with stage_timer("3. Пост-обработка", timings):
        _post_process(pages)

    if summary_mode:
        with stage_timer("4. Генерация реферата", timings):
            logger.info(" Режим реферата...")
            if summary_translator_type == "google":
                summary_translator_type = "llama"
            result = generate_summary(
                pages, summary_translator_type=summary_translator_type,
                translate_translator_type=translate_translator_type, lang=lang,
                llama_url=llama_url, llama_model=llama_model, auto_find=auto_find,
                sprut_model=sprut_model, quiet=quiet)
            summary_html = result["summary_html"]
            gen_stats = result["stats"]

        with stage_timer("5. HTML реферата", timings):
            generate_summary_html(pages, summary_html, f"Реферат: {os.path.basename(pdf_path)}", output_html)
            logger.info(f"   HTML сохранён: {output_html}")

        return {"stats": {"chunks": gen_stats.get("chunks", 0), "tokens": gen_stats.get("tokens", 0),
                          "speed": gen_stats.get("speed", 0), "model": gen_stats.get("model", ""),
                          "time": gen_stats.get("time", 0), "summary_mode": True, "timings": timings},
                "pages": len(pages), "images": total_images, "tables": total_tables, "summary": True}

    with stage_timer("4. Перевод", timings):
        logger.info(f" Перевод на {lang} ({translator_type})...")
        try:
            translator, fallback = create_translator(
                translator_type, lang, llama_url=llama_url, llama_model=llama_model,
                auto_find=auto_find, sprut_model=sprut_model)
        except RuntimeError as e:
            logger.error(f"   Не удалось инициализировать переводчик '{translator_type}': {e}")
            return {"stats": {"error": str(e)}, "pages": 0, "images": 0, "tables": 0}

        effective_workers = 1 if translator_type == "llama" else max_workers
        if translator_type == "llama" and not quiet:
            logger.info("   Локальный сервер: ограничено до 1 воркера (параллельные запросы вызывают зацикливание/галлюцинации)")

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

    return {"stats": {**pipeline.stats, "timings": timings}, "pages": len(pages),
            "images": total_images, "tables": total_tables}


def process_pdf_raw(
    pdf_path: str,
    output_html: str,
    dark_html: bool = True,
    debug_text: bool = False,
) -> Dict[str, Any]:
    """Raw HTML export without translation (pdf2html-grade)."""

    timings: Dict[str, float] = {}
    total_images = 0
    total_tables = 0

    with stage_timer("1. Извлечение PDF", timings):
        pages, doc, pdf, total_images, total_tables = _run_extraction(
            pdf_path, debug_text=debug_text)
        total_text = "".join(block.text for page in pages for block in page.blocks)
        if len(total_text.strip()) < 10:
            logger.error(" В PDF не найден текст (возможно, файл состоит только из изображений или отсканирован).")
            doc.close()
            pdf.close()
            sys.exit(1)
        doc.close()
        pdf.close()
    logger.info(f"   Извлечено {len(pages)} стр., {total_images} изобр., {total_tables} табл.")

    with stage_timer("2. Классификация", timings):
        _classify_pages(pages)

    with stage_timer("3. Пост-обработка", timings):
        _post_process(pages)

    with stage_timer("4. HTML (без перевода)", timings):
        generate_raw_html(pages, f"Оригинал: {os.path.basename(pdf_path)}", output_html, dark=dark_html)
        logger.info(f"   HTML сохранён: {output_html}")

    return {"stats": {"timings": timings}, "pages": len(pages),
            "images": total_images, "tables": total_tables}
