"""Citation handling functions for PDF Translator."""

import re
from typing import List, Tuple

from .models import Span, Line, Block, make_span, make_line, make_block

_REF_HEADING_ALT = (
    r'References?|Reference\s+list|References?\s+and\s+notes?|'
    r'Bibliography|Библиография|Библиографический\s+список|'
    r'Литература|Список\s+литературы|Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|Список\s+источников|'
    r'Использованная\s+литература|Источники'
)
REF_HEADING_RE = re.compile(rf'^\s*({_REF_HEADING_ALT})\s*[:.]?\s*$', re.IGNORECASE)
REF_HEADING_PREFIX_RE = re.compile(rf'^\s*({_REF_HEADING_ALT})\s*[:.]?\s+(.+)$', re.IGNORECASE)
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
                        cit = re.sub(r'\s*,\s*', ', ', re.sub(r'\s*[–—-]\s*', '–', run_text.strip('., ')))
                        cit = cit.rstrip('., ')
                        new_spans.append(make_span(text="[" + cit + "]", font=s.font, size=s.size,
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
            head_spans.append(make_span(text=t[:cut], font=s.font, size=s.size, flags=s.flags,
                                        color=s.color, origin=s.origin, bbox=s.bbox))
            rest_spans.append(make_span(text=t[cut:], font=s.font, size=s.size, flags=s.flags,
                                        color=s.color, origin=s.origin, bbox=s.bbox))
        pos += len(t)
    return (make_line(spans=head_spans, bbox=line.bbox, y0=line.y0),
            make_line(spans=rest_spans, bbox=line.bbox, y0=line.y0))


def _split_ref_heading(block: Block) -> List[Block]:
    if not block.lines:
        return [block]
    lines = block.lines
    first = lines[0]
    first_text = " ".join(s.text for s in first.spans).strip()
    if REF_HEADING_RE.match(first_text):
        if len(lines) > 1:
            head_block = make_block(type="reference_heading", page_num=block.page_num, bbox=block.bbox, lines=lines[:1])
            rest_block = make_block(type="reference", page_num=block.page_num, bbox=block.bbox, lines=lines[1:])
            return [head_block, rest_block]
        return [block]
    m = REF_HEADING_PREFIX_RE.match(first_text)
    if m and m.group(2) and len(first_text) > len(m.group(1)):
        idx = first_text.find(m.group(1).strip()) + len(m.group(1).strip())
        head_line, rest_line = _split_line(first, idx)
        head_block = make_block(type="reference_heading", page_num=block.page_num, bbox=block.bbox, lines=[head_line])
        rest_block = make_block(type="reference", page_num=block.page_num, bbox=block.bbox, lines=[rest_line] + lines[1:])
        return [head_block, rest_block]
    return [block]