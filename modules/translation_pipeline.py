import re
import time
import logging
import threading
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from .models import Block
from .translation_utils import validate_translation, _strip_llm_wrappers, _restore_protected, _extract_protected_tokens
from .translator_llama import LlamaCppTranslator, _is_short_citation
from .translation_cache import TranslationCache
from .utils import normalize_text, _interruptible_sleep, _start_keyboard_watchdog, _format_time, STOP_EVENT
from .citation import _wrap_citation_spans
from .config import SETTINGS, BLOCK_TIMEOUT, REQUEST_TIMEOUT, MAX_SECTION_CHARS, MAX_LLM_CHUNK_CHARS, MIN_LLM_CHUNK_CHARS

logger = logging.getLogger(__name__)

try:
    from tqdm import tqdm
    TQDM = True
except ImportError:
    TQDM = False


def translate_chunk_with_retry(chunk: str, translator, max_retries: int = 3,
                               glossary: Optional[Dict[str, str]] = None,
                               context_before: str = "", context_after: str = "",
                               deadline: Optional[float] = None) -> Optional[str]:
    if not chunk or len(chunk.strip()) < 3:
        return chunk
    # Пункт 2: уменьшаем число попыток и паузу
    attempts = max(1, min(int(max_retries), 3))
    for attempt in range(attempts):
        try:
            if hasattr(translator, "translate"):
                if isinstance(translator, LlamaCppTranslator):
                    translated = translator.translate(chunk, glossary=glossary, context_before=context_before,
                                                      context_after=context_after, deadline=deadline)
                else:
                    translated = translator.translate(chunk, glossary) if glossary else translator.translate(chunk)
            else:
                translated = None

            if translated and translated.strip():
                return _strip_llm_wrappers(translated).strip()

        except Exception as e:
            logger.warning(f"[{getattr(translator, 'name', 'translator')}] {type(e).__name__}: {str(e)[:160]}")

        if attempt + 1 < attempts:
            _interruptible_sleep(min(1.0 * (2 ** attempt), 10.0))  # макс 10 сек вместо 60

    return None


def _split_text_by_paragraphs(text: str, max_chars: int = 1800) -> List[str]:
    # Пункт 3: убираем normalize_text, т.к. текст уже нормализован
    if len(text) <= max_chars:
        return [text]
    paragraphs = re.split(r'\n\s*\n', text)
    if len(paragraphs) <= 1:
        return []
    result = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            result.append(para)
        else:
            result.extend(_split_text_for_llm(para, max_chars, max(80, max_chars // 3)))
    return result if len(result) > 1 else []


def _split_text_for_llm(text: str, max_chars: int = 1800, min_chunk_chars: int = 650) -> List[str]:
    # Пункт 3: убираем normalize_text, используем предкомпилированную регулярку из класса
    if len(text) <= max_chars:
        return [text]
    # используем self._protected_abbr, но здесь нет self; сделаем глобальную или передадим
    # Для простоты оставим локальную, но можно передать из класса.
    protected_abbr = re.compile(r"\b(?:e\.g|i\.e|et al|fig|figs|eq|eqs|dr|mr|mrs|ms|prof|no|nos|vs|approx|ca|p|pp)\.$", re.I)
    parts = []
    start = 0
    for m in re.finditer(r"[.!?](?:[\"'""'\)\]]*)\s+(?=[A-ZА-ЯЁ0-9])", text):
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
    def __init__(self, translator, fallback=None, cache: Optional[TranslationCache] = None,
                 max_workers: int = 4, timeout: int = BLOCK_TIMEOUT,
                 is_local: bool = False, translator_type: str = "google"):
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
        self._pending_blocks: List[Tuple[Block, str, Optional[Dict[str, str]]]] = []
        self._switched_to_fallback = False
        self._original_translator = None

        # Пункт 3: предкомпилированная регулярка для аббревиатур
        self._protected_abbr = re.compile(r"\b(?:e\.g|i\.e|et al|fig|figs|eq|eqs|dr|mr|mrs|ms|prof|no|nos|vs|approx|ca|p|pp)\.$", re.I)

        # Пункт 7: in-memory кэш для быстрого доступа
        self._mem_cache = {}  # key: (text, lang) -> translation
        self._mem_cache_limit = 1000

        # Пункт 15: кэш для _extract_protected_tokens
        self._protected_cache = {}

    BASE_SKIP_TYPES = ("figure", "table", "empty", "metadata")
    RESCUE_MIN_CHARS = int(SETTINGS.get("llm_rescue_min_chars", 220))

    def _skip_types(self) -> Tuple[str, ...]:
        return self.BASE_SKIP_TYPES + ("reference", "reference_heading")

    @staticmethod
    def _safe_cache_get(cache, text, lang):
        value = cache.get(text, lang)
        if value and value.strip():
            ok, _ = validate_translation(text, value, lang)  # пока оставим, но можно убрать
            return value if ok else None
        return None

    def _translate_one(self, text: str, lang: str, glossary: Optional[Dict[str, str]] = None,
                       context_before: str = "", context_after: str = "",
                       _allow_split: bool = True, _deadline: Optional[float] = None) -> Optional[str]:
        # Пункт 4 и 11: text уже нормализован, убираем normalize_text
        if not text:
            return None

        deadline = _deadline if _deadline is not None else time.monotonic() + self.block_timeout
        if time.monotonic() >= deadline:
            return None

        if _is_short_citation(text):
            return text

        if _allow_split and len(text) > MAX_LLM_CHUNK_CHARS:
            para_chunks = _split_text_by_paragraphs(text, MAX_LLM_CHUNK_CHARS)
            chunks = para_chunks if para_chunks else _split_text_for_llm(text, MAX_LLM_CHUNK_CHARS, MIN_LLM_CHUNK_CHARS)
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

        # Пункт 15: кэшируем защищённые токены
        cache_key = (text, lang)  # для protected токенов не зависит от языка, но добавим для общности
        if cache_key in self._protected_cache:
            protected, tokens = self._protected_cache[cache_key]
        else:
            protected, tokens = _extract_protected_tokens(text)
            self._protected_cache[cache_key] = (protected, tokens)
            # ограничим размер кэша, чтобы не расти бесконечно
            if len(self._protected_cache) > 2000:
                # удаляем первые 500 записей (простейшая эвристика)
                for _ in range(500):
                    self._protected_cache.pop(next(iter(self._protected_cache)))

        # Пункт 7: сначала проверяем in-memory кэш
        mem_key = (text, lang)
        if mem_key in self._mem_cache:
            cached = self._mem_cache[mem_key]
            if cached:
                with self._stats_lock:
                    self.stats["cached"] += 1
                return cached

        # Потом основной кэш
        cached = self._safe_cache_get(self.cache, text, lang)
        if cached:
            self._mem_cache[mem_key] = cached
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
            # Пункт 4: упрощаем валидацию – только проверяем не пусто и длина не слишком мала
            if result and len(result) > 2:
                valid = True
            else:
                valid = False
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
                if result and len(result) > 2:
                    valid = True
                else:
                    valid = False
                if not valid:
                    result = None

        if result:
            self.cache.put(text, lang, result)
            self._mem_cache[mem_key] = result
            # ограничиваем размер mem_cache
            if len(self._mem_cache) > self._mem_cache_limit:
                # удаляем первый добавленный (просто для примера)
                for _ in range(100):
                    self._mem_cache.pop(next(iter(self._mem_cache)))
            with self._stats_lock:
                self.stats["success"] += 1
            return result

        if _allow_split and time.monotonic() < deadline:
            rescued = self._rescue_block(text, lang, glossary, context_before, context_after, deadline)
            if rescued:
                self.cache.put(text, lang, rescued)
                self._mem_cache[mem_key] = rescued
                with self._stats_lock:
                    self.stats["success"] += 1
                return rescued

        with self._stats_lock:
            self.stats["failed"] += 1
        return None

    def _rescue_block(self, text: str, lang: str, glossary: Optional[Dict[str, str]],
                      context_before: str, context_after: str, deadline: float) -> Optional[str]:
        # Пункт 5: оставляем только один размер
        if time.monotonic() >= deadline:
            return None
        size = 600  # единственный размер
        if len(text) <= size:
            return None
        chunks = _split_text_for_llm(text, size, max(80, min(MIN_LLM_CHUNK_CHARS, size // 3)))
        if len(chunks) <= 1:
            return None
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

    def _deduplicate_blocks(self, blocks: list) -> list:
        # Пункт 6: ускоренная дедупликация через словарь
        skip_types = self._skip_types()
        translatable = []
        for i, b in enumerate(blocks):
            if b.type not in skip_types and b.translation is None and len(b.text.strip()) > 20:
                translatable.append((i, b))

        # Строим словарь нормализованных текстов для быстрого поиска
        norm_texts = {}  # нормализованный текст -> список индексов
        for idx, block in translatable:
            norm = getattr(block, 'normalized_text', normalize_text(block.text))
            if norm not in norm_texts:
                norm_texts[norm] = []
            norm_texts[norm].append(idx)

        to_skip = set()
        # Ищем дубли (когда один текст содержит другой)
        keys = list(norm_texts.keys())
        for i in range(len(keys)):
            if keys[i] in to_skip:  # такого быть не может, но для безопасности
                continue
            for j in range(i+1, len(keys)):
                if keys[j] in to_skip:
                    continue
                shorter, longer = (keys[i], keys[j]) if len(keys[i]) <= len(keys[j]) else (keys[j], keys[i])
                if shorter in longer:
                    # помечаем все индексы, соответствующие короткому тексту
                    for idx in norm_texts[shorter]:
                        to_skip.add(idx)

        for idx in to_skip:
            blocks[idx].translation = blocks[idx].text
            with self._stats_lock:
                self.stats["skipped"] += 1

        if to_skip:
            logger.info(f"   Дедупликация: исключено {len(to_skip)} дублирующих блоков")

        return blocks

    def _translate_section(self, section: list, lang: str, glossary: Optional[Dict[str, str]] = None) -> None:
        for idx, block in enumerate(section):
            if STOP_EVENT.is_set():
                return
            # Пункт 11: используем нормализованные тексты
            before = section[idx - 1].normalized_text if idx > 0 else ""
            after = section[idx + 1].normalized_text if idx + 1 < len(section) else ""
            result = self._translate_one(
                block.normalized_text, lang, glossary,
                context_before=before[-600:],
                context_after=after[:600],
            )
            block.translation = result
            if result is None and block.text and len(block.text.strip()) > 10:
                with self._stats_lock:
                    self._pending_blocks.append((block, lang, glossary))

    def translate_blocks(self, blocks: list, lang: str, quiet: bool = False) -> list:
        merged_blocks = []
        skip_types = self._skip_types()

        # Пункт 11: предварительная нормализация и добавление атрибута
        for block in blocks:
            text = block.text
            if text:
                block.normalized_text = normalize_text(text)
            else:
                block.normalized_text = ""

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

        blocks = self._deduplicate_blocks(blocks)

        translatable = [b for b in blocks if b.type not in skip_types and b.translation is None]
        if not translatable:
            logger.info("   Все блоки уже переведены или пропущены")
            return blocks

        glossary = None
        sections = []
        current = []
        cur_len = 0
        for b in translatable:
            blen = len(b.normalized_text)
            if current and cur_len + blen > MAX_SECTION_CHARS:
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
                if pbar:
                    elapsed = time.time() - self._start_time
                    avg = elapsed / done
                    remaining = avg * (total - done)
                    pbar.set_postfix_str(f"ост. {_format_time(remaining)}", refresh=True)
                    pbar.update(1)
                elif not quiet and done % 10 == 0:  # Пункт 10: реже логируем
                    elapsed = time.time() - self._start_time
                    avg = elapsed / done
                    remaining = avg * (total - done)
                    logger.info(f"   {done}/{total} ({done*100//total}%) {_format_time(elapsed)} ETA: {_format_time(remaining)}")

        pbar = None
        if TQDM and not quiet:
            pbar = tqdm(total=total, desc=" Перевод", unit="секция",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                        mininterval=1.0)  # Пункт 10

        _start_keyboard_watchdog()
        executor = ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, 2)))  # Пункт 13 (дополнительно уменьшаем)
        futures = {executor.submit(_work, g): g for g in sections}
        _last_health_check = time.monotonic()

        try:
            while futures and not STOP_EVENT.is_set():
                done_futures, _ = wait(futures, timeout=0.5, return_when=FIRST_COMPLETED)
                # Пункт 8: отключаем health-check
                # now = time.monotonic()
                # if now - _last_health_check >= 30 and isinstance(self.translator, LlamaCppTranslator):
                #     _last_health_check = now
                #     if not self.translator._health_check() and self.fallback and not self._switched_to_fallback:
                #         logger.warning("   llama-server недоступен. Переключение на fallback...")
                #         self._switched_to_fallback = True
                #         self._original_translator = self.translator
                #         self.translator = self.fallback
                #         self.fallback = None
                for future in done_futures:
                    futures.pop(future, None)
                    try:
                        future.result()
                    except Exception as e:
                        logger.warning(f"   Ошибка секции: {type(e).__name__}: {e}")
        except KeyboardInterrupt:
            STOP_EVENT.set()
            logger.warning(" Прервано пользователем (Ctrl+C)")
        finally:
            for f in futures:
                f.cancel()
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except TypeError:
                executor.shutdown(wait=False)
            if pbar:
                pbar.close()

        if not quiet:
            elapsed = time.time() - self._start_time
            logger.info(f"   Перевод завершён за {_format_time(elapsed)}")
            logger.info(f"   LLM guard: успешно={self.stats['success']}, кэш={self.stats['cached']}, "
                        f"отклонено={self.stats['rejected']}, ошибки={self.stats['failed']}")

        if self._pending_blocks and not STOP_EVENT.is_set():
            if not quiet:
                logger.info(f"   Повторная обработка {len(self._pending_blocks)} отложенных блоков...")
            pending = list(self._pending_blocks)
            self._pending_blocks.clear()
            for block, lang, glossary in pending:
                if STOP_EVENT.is_set():
                    break
                if block.translation is not None:
                    continue
                result = translate_chunk_with_retry(
                    block.normalized_text,  # используем нормализованный
                    self.translator,
                    max_retries=3,
                    glossary=glossary, deadline=time.monotonic() + self.block_timeout,
                )
                if result:
                    result = _strip_llm_wrappers(result).strip()
                    if result and len(result) > 2:
                        valid = True
                    else:
                        valid = False
                    if valid:
                        block.translation = result
                        self.cache.put(block.normalized_text, lang, result)
                        self._mem_cache[(block.normalized_text, lang)] = result
                        with self._stats_lock:
                            self.stats["success"] += 1
                            self.stats["failed"] -= 1
            if not quiet:
                retried = sum(1 for b, _, _ in pending if b.translation is not None)
                if retried:
                    logger.info(f"   Повторно переведено: {retried}/{len(pending)}")
        return blocks