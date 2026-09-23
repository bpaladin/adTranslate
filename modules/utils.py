"""Utility functions for PDF Translator."""

import os
import re
import time
import threading
import contextlib
from typing import Optional, Dict, List

from .logging_setup import logger

__version__ = "1.3.0"

STOP_EVENT = threading.Event()
_watchdog_started = False

RE_WHITESPACE = re.compile(r'[ \t]+')
RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')

LIST_LINE_RE = re.compile(r'^[\s]*([•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s|[a-z]\.\s)', re.MULTILINE)

_PROTECTED_ABBR = re.compile(r"\b(?:e\.g|i\.e|et al|fig|figs|eq|eqs|dr|mr|mrs|ms|prof|no|nos|vs|approx|ca|p|pp)\.$", re.I)


@contextlib.contextmanager
def stage_timer(name: str, timings: Optional[Dict[str, float]] = None):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        logger.debug(f"  {name}: {_format_time(elapsed)}")
        if timings is not None:
            timings[name] = elapsed


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


def _count_list_lines(text: str) -> int:
    return len(LIST_LINE_RE.findall(text))


def split_by_sentences(text: str, max_chars: int = 1800, min_chunk_chars: int = 650) -> List[str]:
    """Разбить текст по предложениям на куски ≤ max_chars.

    Единая реализация для translation_pipeline, summary и transformers-бэкенда:
    защита аббревиатур (e.g., Fig., et al.), fallback по словам при отсутствии
    границ предложений, склейка коротких хвостов (< min_chunk_chars).
    При min_chunk_chars=0 склейка отключена (поведение старых наивных сплиттеров).
    """
    if len(text) <= max_chars:
        return [text]
    parts = []
    start = 0
    for m in re.finditer(r'''[.!?](?:[]"'""')]*)\s+(?=[A-ZА-ЯЁ0-9])''', text):
        left = text[start:m.end()].strip()
        if not left:
            continue
        if _PROTECTED_ABBR.search(left[-12:]):
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


def _chunk_text_by_sentences(text: str, max_chunk_size: int = 6000) -> List[str]:
    # min_chunk_chars=0: для реферата склейка хвостов не нужна, как раньше.
    return split_by_sentences(text, max_chunk_size, 0)


def _interruptible_sleep(delay: float, step: float = 0.5) -> None:
    remaining = delay
    while remaining > 0 and not STOP_EVENT.is_set():
        time.sleep(min(step, remaining))
        remaining -= step


def _start_keyboard_watchdog():
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    if STOP_EVENT.is_set():
        return
    if os.name == 'nt':
        try:
            import msvcrt
        except ImportError:
            return

        def _watch():
            while not STOP_EVENT.is_set():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        if ch in ('q', 'Q', '\x1b', '\x03'):
                            logger.warning(" Прерывание по клавише, остановка...")
                            STOP_EVENT.set()
                    else:
                        time.sleep(0.1)
                except Exception:
                    break
        threading.Thread(target=_watch, daemon=True).start()
