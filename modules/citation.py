"""Citation handling functions for PDF Translator."""

import re
from typing import List, Tuple

from .models import Span, Line, Block, make_span, make_line, make_block

# Reference-heading keywords.  Order matters: longer alternatives come
# first because ``re.finditer`` tries alternatives left-to-right and does
# not backtrack across them — without this, "Reference list" would be
# reduced to a bare "Reference", and "Список литературы" — to "Литература".
_REF_HEADING_ALT = (
    r'Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|'
    r'Библиографический\s+список|'
    r'References?\s+and\s+notes?|'
    r'Reference\s+list|'
    r'Использованная\s+литература|'
    r'Список\s+литературы|'
    r'Список\s+источников|'
    r'References?|'
    r'Bibliography|'
    r'Библиография|'
    r'Литература|'
    r'Источники'
)
_REF_HEADING_KEYWORD_RE = re.compile(_REF_HEADING_ALT, re.IGNORECASE)
# Публичный «строгий» вариант оставлен для совместимости.
_REF_HEADING_RE = re.compile(
    rf'^\s*(?:{_REF_HEADING_ALT})\s*[:.]?\s*$', re.IGNORECASE
)

# Символы, допустимые вокруг заголовка, не превращающие его в «контент».
_REF_HEADING_TRIM = ' \t:.,;—-–\u2014'

_CIT_RUN_RE = re.compile(r'^\d{1,2}(?:\s*[,–—-]\s*\d{1,2})*\s*[,.]?$')
_REF_LABEL_RE = re.compile(
    r'\b(?:Fig(?:ure)?\.?|Table|Tab\.?|Vol(?:ume)?\.?|No\.?|Section|Chapter|Ch\.?|'
    r'Equation|Eq\.?|Ref\.?|Page|pp\.?|Appendix|Supplementary|Suppl\.?)\s*$'
)


def _wrap_citation_spans(block: Block) -> None:
    for line in block.lines:
        spans = line.spans
        if not spans:
            continue
        new_spans = []
        i = 0
        n = len(spans)
        while i < n:
            s = spans[i]
            if (s.flags & 1) and s.text.strip():
                j = i
                run_parts = []
                while j < n and spans[j].flags & 1:
                    run_parts.append(spans[j].text)
                    j += 1
                run_text = "".join(run_parts)
                if _CIT_RUN_RE.match(run_text):
                    prev_text = "".join(x.text for x in spans[:i])
                    if not _REF_LABEL_RE.search(prev_text):
                        cit = re.sub(r'\s*,\s*', ', ',
                                     re.sub(r'\s*[–—-]\s*', '–', run_text.strip('., ')))
                        cit = cit.rstrip('., ')
                        new_spans.append(make_span(
                            text="[" + cit + "]", font=s.font, size=s.size,
                            flags=s.flags, color=s.color, origin=s.origin, bbox=s.bbox))
                        i = j
                        continue
                new_spans.extend(spans[i:j])
                i = j
            else:
                new_spans.append(s)
                i += 1
        line.spans = new_spans


def _split_line(line: Line, idx: int) -> Tuple[Line, Line]:
    """Утилита: разрезать строку на две по индексу (оставлена для совместимости)."""
    head_spans: List[Span] = []
    rest_spans: List[Span] = []
    pos = 0
    for s in line.spans:
        t = s.text
        if pos + len(t) <= idx:
            head_spans.append(s)
        elif pos >= idx:
            rest_spans.append(s)
        else:
            cut = idx - pos
            head_spans.append(make_span(text=t[:cut], font=s.font, size=s.size,
                                        flags=s.flags, color=s.color,
                                        origin=s.origin, bbox=s.bbox))
            rest_spans.append(make_span(text=t[cut:], font=s.font, size=s.size,
                                        flags=s.flags, color=s.color,
                                        origin=s.origin, bbox=s.bbox))
        pos += len(t)
    return (make_line(spans=head_spans, bbox=line.bbox, y0=line.y0),
            make_line(spans=rest_spans, bbox=line.bbox, y0=line.y0))


def _find_ref_heading(text: str):
    """Вернуть ``(start, end)`` span заголовка списка литературы или ``None``.

    Правила:

    * Если в строке встретилось несколько ключевых слов из
      :data:`_REF_HEADING_ALT`, **последнее** вхождение определяет конец
      заголовка.
    * Строка считается заголовком только если вне найденных ключевых
      фраз нет других слов — допускаются лишь пунктуация и пробелы.
      Поэтому ``"Список литературы приведён в конце"`` — не заголовок,
      а ``"Литература. Библиография"`` — заголовок.
    """
    matches = list(_REF_HEADING_KEYWORD_RE.finditer(text))
    if not matches:
        return None

    first, last = matches[0], matches[-1]

    # Между/до/после найденных ключей — только пунктуация и пробелы.
    if text[:first.start()].strip(_REF_HEADING_TRIM):
        return None
    for a, b in zip(matches, matches[1:]):
        if text[a.end():b.start()].strip(_REF_HEADING_TRIM):
            return None
    if text[last.end():].strip(_REF_HEADING_TRIM):
        return None

    return first.start(), last.end()


def _split_ref_heading(block: Block) -> List[Block]:
    if not block.lines:
        return [block]
    lines = block.lines
    first = lines[0]
    first_text = " ".join(s.text for s in first.spans).strip()

    if _find_ref_heading(first_text) is None:
        return [block]

    # Заголовок занимает всю первую строку, остальные строки — сам список.
    if len(lines) > 1:
        head_block = make_block(type="reference_heading", page_num=block.page_num,
                                bbox=block.bbox, lines=lines[:1])
        rest_block = make_block(type="reference", page_num=block.page_num,
                                bbox=block.bbox, lines=lines[1:])
        return [head_block, rest_block]
    return [block]
