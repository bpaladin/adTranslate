"""Utility functions for PDF Translator."""

import os
import re
import time
import threading
import contextlib
from typing import Optional, Dict, List

from .logging_setup import logger

__version__ = "1.2.0"

STOP_EVENT = threading.Event()
_watchdog_started = False

RE_WHITESPACE = re.compile(r'[ \t]+')
RE_HYPHEN_BREAK = re.compile(r'(\w+)-\s*\n\s*(\w+)')

LIST_LINE_RE = re.compile(r'^[\s]*([•\-\*►▸‣⁃◦○●▪]|\d+[\.\)]\s|[a-z]\.\s)', re.MULTILINE)


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
