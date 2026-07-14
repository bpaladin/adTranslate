import os
import re
import logging
import threading
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from modules.pdf_extractor import PDFExtractor, Page, Block
from modules.block_classifier import classify_block
from modules.table_processor import extract_tables, mark_table_blocks
from modules.image_processor import extract_images
from modules.translation_engine import (
    GeminiTranslator, GoogleTranslator, OpenRouterTranslator,
    LlamaCppTranslator, RateLimiter,
)
from modules.cache_manager import TranslationCache
from modules.html_renderer import generate_html

logger = logging.getLogger(__name__)


# =========================================================
# Фабрика переводчиков
# =========================================================
def create_translator(
    translator_type: str,
    target_lang: str,
    api_key: Optional[str] = None,
    llama_model: str = "gemma4",
    llama_url: str = "http://localhost:8080/v1",
    gemini_model: str = "gemini-2.0-flash",
):
    if translator_type == "gemini":
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("Для Gemini нужен GEMINI_API_KEY")
        primary = GeminiTranslator(target_lang, key, model=gemini_model)
        fallback = GoogleTranslator(target_lang)
        return primary, fallback

    elif translator_type == "openrouter":
        key = (
            api_key
            or os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not key:
            raise ValueError("Для OpenRouter нужен OPENROUTER_API_KEY")
        primary = OpenRouterTranslator(target_lang, key)
        fallback = GoogleTranslator(target_lang)
        return primary, fallback

    elif translator_type == "llama":
        primary = LlamaCppTranslator(
            target_lang, model=llama_model, api_base=llama_url,
        )
        fallback = GoogleTranslator(target_lang)
        return primary, fallback

    elif translator_type == "google":
        primary = GoogleTranslator(target_lang)
        return primary, None

    else:
        raise ValueError(f"Неизвестный переводчик: {translator_type}")


# =========================================================
# Перевод блоков с кэшем и ограничителем
# =========================================================
class TranslationPipeline:
    def __init__(
        self,
        translator,
        fallback=None,
        cache: Optional[TranslationCache] = None,
        max_workers: int = 8,
        timeout: int = 600,
    ):
        self.translator = translator
        self.fallback = fallback
        self.cache = cache or TranslationCache()
        self.max_workers = max_workers
        self.timeout = timeout
        self.stats = {"success": 0, "failed": 0, "cached": 0, "skipped": 0}
        self._stats_lock = threading.Lock()

    def _translate_one(self, text: str, lang: str) -> Optional[str]:
        cached = self.cache.get(text, lang)
        if cached:
            with self._stats_lock:
                self.stats["cached"] += 1
            return cached

        result = self.translator.translate(text)
        if not result and self.fallback:
            result = self.fallback.translate(text)

        if result:
            self.cache.put(text, lang, result)
            with self._stats_lock:
                self.stats["success"] += 1
        else:
            with self._stats_lock:
                self.stats["failed"] += 1
        return result

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
            merged_text = "\n".join(ref_buf)
            merged_text = re.sub(r'\s+', ' ', merged_text).strip()
            ref_block = Block(
                type="reference",
                page_num=ref_buf_page,
                bbox=(0, 0, 0, 0),
            )
            ref_block.translation = merged_text
            ref_block.lines = []
            merged_blocks.append(ref_block)
            ref_buf.clear()

        for block in blocks:
            text = " ".join(
                span.text for line in block.lines for span in line.spans
            ).strip()

            # Пустые блоки и блоки без текста (например, «настоящие» таблицы
            # с table_data, но без lines) — пропускаем перевод,
            # но ОБЯЗАТЕЛЬНО сохраняем в merged_blocks, чтобы рендерер их увидел.
            if not text or len(text) < 2:
                block.translation = text if text else ""
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)   # ← ИСПРАВЛЕНО: раньше было просто continue
                continue

            # figure / table / empty / reference_heading — не переводим
            if block.type in ("figure", "table", "empty", "reference_heading"):
                block.translation = text
                with self._stats_lock:
                    self.stats["skipped"] += 1
                merged_blocks.append(block)
                continue

            # reference — буферизуем
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

        # Отбор блоков для перевода
        translatable = [
            b for b in blocks
            if b.type not in ("figure", "table", "empty",
                              "reference", "reference_heading")
            and b.translation is None
        ]

        if not translatable:
            return blocks

        total = len(translatable)
        done = 0
        lock = threading.Lock()

        def _work(block):
            nonlocal done
            text = " ".join(
                span.text for line in block.lines for span in line.spans
            ).strip()
            result = self._translate_one(text, lang)
            block.translation = result if result else text
            with lock:
                done += 1
                if TQDM and not quiet:
                    pbar.update(1)
                elif not quiet and done % 10 == 0:
                    logger.info(f"   Переведено {done}/{total}")

        if TQDM and not quiet:
            pbar = tqdm(total=total, desc="🌐 Перевод", unit="блок")

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

        return blocks


def process_pdf(
    pdf_path: str,
    output_html: str,
    lang: str = "ru",
    translator_type: str = "openrouter",
    api_key: Optional[str] = None,
    llama_model: str = "gemma4",
    llama_url: str = "http://localhost:8080/v1",
    gemini_model: str = "gemini-2.0-flash",
    max_workers: int = 8,
    timeout: int = 600,
    quiet: bool = False,
) -> Dict[str, Any]:

    import fitz

    # 1. Извлечение текста
    logger.info("🏷️ [1/5] Извлечение данных из PDF...")
    extractor = PDFExtractor(pdf_path)
    pages = extractor.extract()

    # 2. Извлечение изображений и таблиц
    total_images = 0
    total_tables = 0
    doc = fitz.open(pdf_path)

    for page in pages:
        text_blocks = [b for b in page.blocks if b.type == "text"]
        fitz_page = doc[page.num - 1]

        try:
            figure_blocks = extract_images(fitz_page, page.num, text_blocks)
            page.blocks.extend(figure_blocks)
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
    logger.info(
        f"🏷️ [2/5] Извлечено {len(pages)} страниц, "
        f"{total_images} изображений, {total_tables} таблиц"
    )

    # 3. Классификация блоков
    logger.info("🏷️ [3/5] Классификация блоков...")
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
                old_type = block.type
                block.type = classify_block(block, page.width, avg_size)
                if block.type != old_type:
                    text_preview = " ".join(
                        span.text for line in block.lines for span in line.spans
                    )[:60]
                    logger.debug(
                        f"  page {page.num}: {old_type} -> {block.type}: "
                        f"{text_preview}"
                    )

    # 4. Пост-обработка
    logger.info("🏷️ [4/5] Пост-обработка (References + таблицы)...")

    # 4a. Всё после первого reference_heading -> reference (сквозь все страницы)
    in_refs = False
    for page in pages:
        for block in page.blocks:
            if block.type == "reference_heading":
                in_refs = True
                continue
            if in_refs and block.type not in ("figure", "table"):
                block.type = "reference"

    # 4b. Текстовые блоки, попавшие внутрь bbox таблицы -> table
    #     Затем удаляем «призраков»: блоки type=="table" без table_data
    #     (это текстовые куски из ячеек, которые дублируют настоящую таблицу).
    for page in pages:
        table_blocks_page = [b for b in page.blocks if b.type == "table"]
        if table_blocks_page:
            mark_table_blocks(page.blocks, table_blocks_page)

        # ← КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: удаляем дубликаты
        page.blocks = [
            b for b in page.blocks
            if not (b.type == "table" and b.table_data is None)
        ]

    # 5. Перевод
    logger.info(f"🌐[5/5] Перевод на {lang} ({translator_type})...")
    translator, fallback = create_translator(
        translator_type, lang,
        api_key=api_key,
        llama_model=llama_model,
        llama_url=llama_url,
        gemini_model=gemini_model,
    )
    logger.info(f"   Переводчик: {translator.name}")
    if fallback:
        logger.info(f"   Fallback: {fallback.name}")

    pipeline = TranslationPipeline(
        translator=translator,
        fallback=fallback,
        max_workers=max_workers,
        timeout=timeout,
    )

    all_blocks = []
    for page in pages:
        all_blocks.extend(page.blocks)

    pipeline.translate_blocks(all_blocks, lang, quiet=quiet)

    # 6. HTML
    logger.info("🎨 Генерация HTML...")
    generate_html(pages, f"Перевод: {os.path.basename(pdf_path)}", output_html)
    logger.info(f"   HTML сохранён: {output_html}")

    return {
        "stats": pipeline.stats,
        "pages": len(pages),
        "images": total_images,
        "tables": total_tables,
    }
