"""Main PDF processing pipeline orchestrator."""

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
from .figure_extraction import extract_images, extract_vector_figures, _rect_gap
from .classifier import _classifier
from .citation import _split_ref_heading
from .summary import generate_summary
from .html_generation import generate_html, generate_summary_html, generate_raw_html
from .translator_factory import create_translator
from .translation_pipeline import TranslationPipeline

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


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _detect_repeating_blocks(pages: List['Page']) -> None:
    """Detect blocks that repeat across many pages (headers, footers, journal names)
    and mark them as metadata so they are not translated repeatedly."""
    total_pages = len(pages)
    if total_pages < _FOOTER_MIN_PAGES:
        return

    text_to_pages = {}
    for page in pages:
        for block in page.blocks:
            if block.type not in ("paragraph", "text"):
                continue
            norm = _normalize_for_dedup(block.text)
            if len(norm) < 5 or len(norm) > 200:
                continue
            key = norm
            if key not in text_to_pages:
                text_to_pages[key] = set()
            text_to_pages[key].add(page.num)

    min_pages = max(3, total_pages // 2)
    for norm, page_set in text_to_pages.items():
        if len(page_set) < min_pages:
            continue
        for page in pages:
            for block in page.blocks:
                if block.type not in ("paragraph", "text"):
                    continue
                if _normalize_for_dedup(block.text) == norm:
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
    """Remove footer blocks (Page N, journal names) and everything below them on each page."""
    for page in pages:
        footer_idx = None
        for i, block in enumerate(page.blocks):
            text = block.text.strip()
            if _FOOTER_RE.match(text):
                footer_idx = i
                break
            if _AUTHOR_RE.match(text):
                footer_idx = i
                break
            if _JOURNAL_LINE_RE.search(text) and i > 0:
                prev_texts = [page.blocks[j].text.strip() for j in range(max(0, i - 2), i)]
                if any(_FOOTER_RE.match(pt) or _AUTHOR_RE.match(pt) for pt in prev_texts):
                    footer_idx = i - 1
                    break
        if footer_idx is not None:
            page.blocks = page.blocks[:footer_idx]
            continue
        for i, block in enumerate(page.blocks):
            text = block.text.strip()
            if _JOURNAL_LINE_RE.search(text):
                if i >= len(page.blocks) - 3:
                    page.blocks = page.blocks[:i]
                    break


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
        pdf = pdfplumber.open(pdf_path)
        for page in pages:
            text_blocks = [b for b in page.blocks if b.type == "text"]
            non_text = [b for b in page.blocks if b.type != "text"]
            fitz_page = doc[page.num - 1]
            table_blocks = []
            table_bboxes = []
            try:
                table_blocks = extract_tables(pdf.pages[page.num - 1], page.num)
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
                    fitz_page, page.num, text_blocks, used_bboxes=used_bboxes, table_bboxes=table_bboxes)
                figure_blocks.extend(vector_blocks)
                total_images += len(vector_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения векторных фигур: {e}")
            page.blocks = non_text + text_blocks + table_blocks + figure_blocks

        total_text = "".join(block.text for page in pages for block in page.blocks)
        if len(total_text.strip()) < 10:
            logger.error(" В PDF не найден текст (возможно, файл состоит только из изображений или отсканирован).")
            sys.exit(1)
        doc.close()
        pdf.close()
    logger.info(f"   Извлечено {len(pages)} стр., {total_images} изобр., {total_tables} табл.")

    with stage_timer("2. Классификация", timings):
        for page in pages:
            all_sizes = [span.size for block in page.blocks
                         if block.type in ("paragraph", "heading", "metadata")
                         for line in block.lines for span in line.spans]
            avg_size = statistics.median(all_sizes) if all_sizes else 12.0
            for block in page.blocks:
                if block.type == "text":
                    block.type = _classifier.classify(block, page.width, avg_size)

    with stage_timer("3. Пост-обработка", timings):
        _detect_repeating_blocks(pages)
        _detect_footers(pages)
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
) -> Dict[str, Any]:

    timings: Dict[str, float] = {}
    total_images = 0
    total_tables = 0

    with stage_timer("1. Извлечение PDF", timings):
        extractor = PDFExtractor(pdf_path)
        pages = extractor.extract()
        doc = fitz.open(pdf_path)
        pdf = pdfplumber.open(pdf_path)
        for page in pages:
            text_blocks = [b for b in page.blocks if b.type == "text"]
            non_text = [b for b in page.blocks if b.type != "text"]
            fitz_page = doc[page.num - 1]
            table_blocks = []
            table_bboxes = []
            try:
                table_blocks = extract_tables(pdf.pages[page.num - 1], page.num)
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
                    fitz_page, page.num, text_blocks, used_bboxes=used_bboxes, table_bboxes=table_bboxes)
                figure_blocks.extend(vector_blocks)
                total_images += len(vector_blocks)
            except Exception as e:
                logger.warning(f"Ошибка извлечения векторных фигур: {e}")
            page.blocks = non_text + text_blocks + table_blocks + figure_blocks

        total_text = "".join(block.text for page in pages for block in page.blocks)
        if len(total_text.strip()) < 10:
            logger.error(" В PDF не найден текст (возможно, файл состоит только из изображений или отсканирован).")
            sys.exit(1)
        doc.close()
        pdf.close()
    logger.info(f"   Извлечено {len(pages)} стр., {total_images} изобр., {total_tables} табл.")

    with stage_timer("2. Классификация", timings):
        for page in pages:
            all_sizes = [span.size for block in page.blocks
                         if block.type in ("paragraph", "heading", "metadata")
                         for line in block.lines for span in line.spans]
            avg_size = statistics.median(all_sizes) if all_sizes else 12.0
            for block in page.blocks:
                if block.type == "text":
                    block.type = _classifier.classify(block, page.width, avg_size)

    with stage_timer("3. Пост-обработка", timings):
        _detect_repeating_blocks(pages)
        _detect_footers(pages)
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

    with stage_timer("4. HTML (без перевода)", timings):
        generate_raw_html(pages, f"Оригинал: {os.path.basename(pdf_path)}", output_html, dark=dark_html)
        logger.info(f"   HTML сохранён: {output_html}")

    return {"stats": {"timings": timings}, "pages": len(pages),
            "images": total_images, "tables": total_tables}
