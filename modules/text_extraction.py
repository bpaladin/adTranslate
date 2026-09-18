"""Text extraction ported from pdf2html.py (single-file reference).

Key differences from the old ``PDFExtractor`` path:
- single ``page.get_text("dict")`` source, no ``get_text("text", clip=...)``
  fallback that scrambles two-column/table/caption order;
- whitespace is normalized, soft hyphens stripped, regular hyphens kept;
- line-break hyphen detection handles Unicode hyphens, pyphen wordlists
  and compound prefixes (word-level, not whole-line);
- lines are grouped into paragraph blocks by size/color/bold/gap;
- hyphen splits across *block* boundaries are merged
  ("представ-" + "лений");
- figure overlap removal is line-level and caption-protected.

Produces :class:`modules.models.Block` objects with extra attributes
``font_size`` / ``font_flags`` / ``color`` / ``starts_with_bold`` and
``text_override`` (hyphen-repaired text, kept separately from spans).
"""

import re
import statistics
from pathlib import Path
from typing import List, Tuple, Optional, Set

import fitz

from .logging_setup import logger
from .models import Block, Line, Span, make_block, make_span, make_line

# ── Optional hyphenation deps ─────────────────────────────────────────────
try:
    import pyphen  # type: ignore
    _PYPHEN = {
        "ru": pyphen.Pyphen(lang="ru_RU"),
        "en": pyphen.Pyphen(lang="en_US"),
    }
except Exception:
    _PYPHEN = {}

_WORDS_RU: Set[str] = set()
_WORDS_EN: Set[str] = set()


def _load_wordlist(path: str) -> Set[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return {w.strip().lower() for w in f if w.strip()}
    except OSError:
        return set()


def _init_wordlists(base_dir: Optional[str] = None) -> None:
    global _WORDS_RU, _WORDS_EN
    candidates = []
    if base_dir:
        candidates.append(Path(base_dir))
    candidates.append(Path(__file__).resolve().parent.parent / "data")
    candidates.append(Path.cwd() / "data")
    for base in candidates:
        ru = base / "words_ru.txt"
        en = base / "words_en.txt"
        if not _WORDS_RU and ru.exists():
            _WORDS_RU = _load_wordlist(str(ru))
        if not _WORDS_EN and en.exists():
            _WORDS_EN = _load_wordlist(str(en))
        if _WORDS_RU or _WORDS_EN:
            break


_init_wordlists()

# ── Constants (from pdf2html.py) ──────────────────────────────────────────
_RE_FIG_MARKER = re.compile(r'^\s*(рисунок|рис\.?|figure|fig\.?)\s*\d+', re.I)
_RE_TBL_MARKER = re.compile(r'^\s*(продолжение\s+)?(table|таблиц\w*)\s*\d+', re.I)

_CAPTION_RE = re.compile(
    r'^\s*(?:рисунок|рис\.?|figure|fig\.?|табл\w*|table|схема|scheme|'
    r'график|chart|diagram|диаграмма)\s*\d+',
    re.IGNORECASE,
)

_HYPHEN_CHARS = (
    "\u002d" "\u2010" "\u2011" "\u2012" "\u2013" "\u2014"
    "\u2015" "\u2212" "\u2043" "\ufe58" "\ufe63"
)

_COMPOUND_PREFIXES = re.compile(
    r"^(?:non|pre|post|anti|auto|co|de|dis|ex|extra|hyper|"
    r"inter|intra|macro|meta|micro|mid|mini|multi|over|"
    r"pseudo|re|semi|sub|super|trans|ultra|under|uni|"
    r"well|high|low|long|short|self|cross|deep|fast|real|"
    r"un|mis|counter|"
    r"де|ре|пост|пре|анти|авто|со|экс|гипер|интер|"
    r"макро|микро|мульти|сверх|суб|транс|ультра|"
    r"высоко|низко|долго|само|взаимо|полу|недо|пере)$",
    re.IGNORECASE,
)

_WORD_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿА-Яа-яЁё]+(?:['’][A-Za-zА-Яа-яЁё]+)?")

_FUNCTION_TAIL_WORDS = frozenset(
    "of the a an in on and or to".split()
    + "в на с к о у об от до по из за над под".split()
)


# ── Text helpers ──────────────────────────────────────────────────────────
def _normalize_whitespace(text: str) -> str:
    text = (text or "").replace("\r", " ").replace("\n", " ")
    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.strip()


def _clean_span_text(text: str) -> str:
    text = (text or "").replace("\u00ad", "")
    text = text.replace("\r", " ").replace("\n", " ")
    return text


def _split_trailing_hyphen(s: str) -> Tuple[str, Optional[str]]:
    s = (s or "").rstrip()
    if s and s[-1] in _HYPHEN_CHARS:
        return s[:-1].rstrip(), s[-1]
    return s, None


def _pick_wordlist(left: str, right: str) -> Set[str]:
    sample = left + right
    cyr = sum(1 for c in sample if "а" <= c.lower() <= "я" or c.lower() == "ё")
    lat = sum(1 for c in sample if "a" <= c.lower() <= "z")
    if cyr > lat and _WORDS_RU:
        return _WORDS_RU
    if lat > cyr and _WORDS_EN:
        return _WORDS_EN
    return _WORDS_RU | _WORDS_EN


def _pick_pyphen(left: str, right: str):
    sample = left + right
    cyr = sum(1 for c in sample if "а" <= c.lower() <= "я" or c.lower() == "ё")
    lat = sum(1 for c in sample if "a" <= c.lower() <= "z")
    if cyr >= lat and "ru" in _PYPHEN:
        return _PYPHEN["ru"]
    if "en" in _PYPHEN:
        return _PYPHEN["en"]
    return None


def _is_valid_hyphenation_pyphen(left: str, right: str) -> Optional[bool]:
    dic = _pick_pyphen(left, right)
    if dic is None:
        return None
    joined = (left + right).lower()
    if not joined.isalpha():
        return None
    try:
        inserted = dic.inserted(joined)
    except Exception:
        return None
    positions = set()
    offset = 0
    for ch in inserted:
        if ch == "-":
            positions.add(offset)
        else:
            offset += 1
    return len(left) in positions


def _last_word(s: str) -> str:
    m = list(_WORD_TOKEN_RE.finditer(s or ""))
    return m[-1].group(0) if m else ""


def _first_word(s: str) -> str:
    m = _WORD_TOKEN_RE.search(s or "")
    return m.group(0) if m else ""


def _looks_like_real_hyphen(prefix: str, next_line: str) -> bool:
    """True → keep hyphen (real compound word); False → join (line break)."""
    if not prefix or not next_line:
        return True
    left, hyphen = _split_trailing_hyphen(prefix)
    if hyphen is None:
        return False
    right = next_line.lstrip()
    if not right:
        return True
    raw = prefix.rstrip()
    if len(raw) >= 2 and raw[-1] in _HYPHEN_CHARS and raw[-2].isspace():
        return True
    if left and (left[-1].isdigit() or right[0].isdigit()):
        return True
    if right[0] in ")]}»\"'.,;:!?-–—•·":
        return True
    if not (left and left[-1].isalpha() and right[0].isalpha()):
        return True
    if re.search(r"\b(?:DE|IE|NE|SD|SE|EEG|EOG|EMG|ECG|P\d+|N\d+)$", left, re.I):
        return True
    lw = _last_word(left) or left
    rw = _first_word(right) or right
    if len(lw) <= 1 or len(rw) <= 1:
        return True
    words = _pick_wordlist(lw, rw)
    if words:
        joined_is_word = (lw + rw).lower() in words
        left_is_word = lw.lower() in words
        right_is_word = rw.lower() in words
        if joined_is_word:
            return False
        if left_is_word and right_is_word:
            return True
    if _COMPOUND_PREFIXES.match(lw):
        return True
    if rw.lower() in _FUNCTION_TAIL_WORDS:
        return True
    try:
        pyphen_verdict = _is_valid_hyphenation_pyphen(lw, rw)
    except Exception:
        pyphen_verdict = None
    if pyphen_verdict is True and rw[0].islower():
        return False
    if rw[0].isupper():
        return True
    return False


def _join_lines(lines: List[str]) -> str:
    prepared = []
    for ln in lines:
        ln = _clean_span_text(ln).strip()
        if ln:
            prepared.append(ln)
    if not prepared:
        return ""
    result = [prepared[0]]
    for ln in prepared[1:]:
        prev = result[-1].rstrip()
        cur = ln.lstrip()
        left, hyphen = _split_trailing_hyphen(prev)
        if hyphen is not None and left and cur and left[-1].isalpha() and cur[0].isalpha():
            if not _looks_like_real_hyphen(prev, cur):
                result[-1] = left + cur
                continue
            if cur[0].islower():
                result[-1] = left + hyphen + cur
                continue
            result.append(cur)
            continue
        result.append(cur)
    return _normalize_whitespace(" ".join(result))


def _span_text(s) -> str:
    if isinstance(s, dict):
        return s.get("text", "")
    return getattr(s, "text", "") or ""


def _span_bbox(s) -> tuple:
    if isinstance(s, dict):
        return tuple(s.get("bbox", (0, 0, 0, 0)))
    return tuple(getattr(s, "bbox", (0, 0, 0, 0)))


def _span_x0(s) -> float:
    try:
        return float(_span_bbox(s)[0])
    except Exception:
        return 0.0


def _build_line_text(spans: list) -> str:
    """Conservative line rebuild: sort by X, concatenate, no bbox spacing."""
    if not spans:
        return ""
    indexed = list(enumerate(spans))
    indexed.sort(key=lambda pair: (round(_span_x0(pair[1]), 3), pair[0]))
    return _normalize_whitespace("".join(_clean_span_text(_span_text(s)) for _, s in indexed))


def _text_quality(text: str) -> float:
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    letters = sum(c.isalpha() for c in chars)
    digits = sum(c.isdigit() for c in chars)
    spaces = len(re.findall(r"\s+", text))
    single_tokens = sum(1 for token in text.split() if len(token) == 1 and token.isalpha())
    n = len(chars)
    score = min(0.45, (letters + digits) / n * 0.45)
    words = text.split()
    if len(words) >= 2:
        score += 0.25
    if len(words) >= 5:
        score += 0.15
    if words:
        score -= min(0.45, (single_tokens / len(words)) * 0.65)
    if spaces > max(2, n * 0.35):
        score -= 0.15
    return max(0.0, min(1.0, score))


def _span_sort_key(span: dict, index: int = 0):
    bbox = span.get("bbox", (0, 0, 0, 0)) if isinstance(span, dict) else getattr(span, "bbox", (0, 0, 0, 0))
    return (round(float(bbox[0]), 3), round(float(bbox[1]), 3), index)


def _create_text_block_from_lines(lines_info, page_num: int = 0) -> Optional[Block]:
    if not lines_info:
        return None
    x0 = min(li["bbox"][0] for li in lines_info)
    y0 = min(li["bbox"][1] for li in lines_info)
    x1 = max(li["bbox"][2] for li in lines_info)
    y1 = max(li["bbox"][3] for li in lines_info)
    lines_text: List[str] = []
    max_font = 0.0
    max_flags = 0
    max_color = lines_info[0].get("color", 0)
    model_lines: List[Line] = []
    for li in lines_info:
        spans = li["spans"]
        line_text = _build_line_text(spans)
        if line_text:
            lines_text.append(line_text)
        model_spans: List[Span] = []
        for s in spans:
            max_font = max(max_font, float(s.get("size", 0) or 0))
            max_flags |= int(s.get("flags", 0) or 0)
            if "bold" in s.get("font", "").lower():
                max_flags |= 16
            model_spans.append(make_span(
                text=s.get("text", ""), font=s.get("font", ""),
                size=float(s.get("size", 0) or 0), flags=int(s.get("flags", 0) or 0),
                color=int(s.get("color", 0) or 0),
                origin=tuple(s.get("origin", (0, 0))),
                bbox=tuple(s.get("bbox", (0, 0, 0, 0))),
            ))
        model_lines.append(make_line(spans=model_spans, bbox=tuple(li["bbox"]), y0=float(li["bbox"][1])))
        for s in spans:
            max_font = max(max_font, float(s.get("size", 0) or 0))
    text = _join_lines(lines_text)
    if not text:
        return None
    starts_with_bold = False
    if lines_info and lines_info[0].get("spans"):
        first_span = lines_info[0]["spans"][0]
        starts_with_bold = ("bold" in first_span.get("font", "").lower()
                            or (int(first_span.get("flags", 0) or 0) & 16))
    return make_block(
        type="text", page_num=page_num, bbox=(x0, y0, x1, y1),
        lines=model_lines, text_override=text,
        font_size=max_font, font_flags=max_flags, color=max_color,
        starts_with_bold=starts_with_bold,
    )


def extract_text_blocks(page: "fitz.Page", page_num: int = 0) -> List[Block]:
    """Extract paragraph-level text blocks (pdf2html algorithm).

    Single ``get_text("dict")`` source; no ``clip`` re-extraction fallback.
    """
    data = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks: List[Block] = []
    for raw in data.get("blocks", []):
        if raw.get("type") != 0:
            continue
        raw_lines = raw.get("lines", [])
        if not raw_lines:
            continue
        lines_info = []
        for line in sorted(raw_lines, key=lambda l: (round(float(l["bbox"][1]), 3),
                                                     round(float(l["bbox"][0]), 3))):
            spans = [s for s in line.get("spans", []) if s.get("text", "")]
            if not spans:
                continue
            spans = [s for _, s in sorted(enumerate(spans),
                                          key=lambda p: _span_sort_key(p[1], p[0]))]
            sizes = [float(s.get("size", 0) or 0) for s in spans
                     if float(s.get("size", 0) or 0) > 0]
            median_size = statistics.median(sizes) if sizes else 12.0
            dom = min(spans, key=lambda s: abs(float(s.get("size", 0) or median_size) - median_size))
            flags = int(dom.get("flags", 0) or 0)
            if any("bold" in s.get("font", "").lower() or (int(s.get("flags", 0) or 0) & 16)
                   for s in spans):
                flags |= 16
            line_text = _build_line_text(spans)
            if not line_text:
                continue
            lines_info.append({
                "bbox": line["bbox"], "spans": spans, "size": median_size,
                "color": dom.get("color", 0), "flags": flags,
                "quality": _text_quality(line_text),
            })
        if not lines_info:
            continue
        groups = [[lines_info[0]]]
        for cur in lines_info[1:]:
            prev = groups[-1][-1]
            gap = cur["bbox"][1] - prev["bbox"][3]
            h = max(1.0, prev["bbox"][3] - prev["bbox"][1])
            split = (
                abs(cur["size"] - prev["size"]) > 1.2
                or cur["color"] != prev["color"]
                or bool(cur["flags"] & 16) != bool(prev["flags"] & 16)
                or gap > h * 1.65
            )
            if split:
                groups.append([cur])
            else:
                groups[-1].append(cur)
        for group in groups:
            blk = _create_text_block_from_lines(group, page_num=page_num or (page.number + 1))
            if blk:
                blocks.append(blk)
    return blocks


# ── Figure overlap: line-level, caption-protected ─────────────────────────
def is_caption_text(text: str) -> bool:
    return bool(_CAPTION_RE.match(text or ""))


def _intersection_ratio(a: fitz.Rect, b: fitz.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    area_a = max(0.0, a.get_area())
    if area_a <= 0:
        return 0.0
    return inter.get_area() / area_a


def _line_center_inside(line_bbox, figure_rects, margin_x=2, margin_y=2) -> bool:
    cx = (line_bbox[0] + line_bbox[2]) * 0.5
    cy = (line_bbox[1] + line_bbox[3]) * 0.5
    for fr in figure_rects:
        if (fr.x0 - margin_x) <= cx <= (fr.x1 + margin_x) and \
           (fr.y0 - margin_y) <= cy <= (fr.y1 + margin_y):
            return True
    return False


def _line_fully_covered(line_bbox, figure_rects, ratio=0.60) -> bool:
    lr = fitz.Rect(line_bbox)
    for fr in figure_rects:
        if _intersection_ratio(lr, fr) >= ratio:
            return True
    return False


def _rect_chebyshev_gap(line_bbox, fr) -> float:
    dx = max(0.0, max(line_bbox[0], fr.x0) - min(line_bbox[2], fr.x1))
    dy = max(0.0, max(line_bbox[1], fr.y0) - min(line_bbox[3], fr.y1))
    return max(dx, dy)


def _looks_like_axis_label(text: str) -> bool:
    t = (text or "").strip()
    if not t or len(t) > 60 or is_caption_text(t):
        return False
    if re.match(r"^\s*(?:table|fig\.?|figure|рис|схема|scheme)\b", t, re.I):
        return False
    return bool(re.search(r"\d", t))


def _line_near_figure_edge(line_bbox, line_text, figure_rects, max_gap: float = 14.0) -> bool:
    if not _looks_like_axis_label(line_text):
        return False
    w = max(1.0, line_bbox[2] - line_bbox[0])
    for fr in figure_rects:
        if _rect_chebyshev_gap(line_bbox, fr) > max_gap:
            continue
        x_overlap = max(0.0, min(line_bbox[2], fr.x1) - max(line_bbox[0], fr.x0))
        if x_overlap >= 0.5 * w:
            return True
    return False


def _line_should_drop(line_bbox, figure_rects, margin_x=2, margin_y=2, line_text: str = "") -> bool:
    if _line_fully_covered(line_bbox, figure_rects):
        return True
    if _line_center_inside(line_bbox, figure_rects, margin_x, margin_y):
        return True
    if line_text and _line_near_figure_edge(line_bbox, line_text, figure_rects):
        return True
    return False


def _line_spans_text(line) -> str:
    try:
        return _build_line_text(list(getattr(line, "spans", []) or []))
    except Exception:
        return ""


def _rebuild_block_from_lines(orig_block: Block, kept_lines: List[Line]) -> Optional[Block]:
    if not kept_lines:
        return None
    x0 = min(ln.bbox[0] for ln in kept_lines)
    y0 = min(ln.bbox[1] for ln in kept_lines)
    x1 = max(ln.bbox[2] for ln in kept_lines)
    y1 = max(ln.bbox[3] for ln in kept_lines)
    line_texts = [_line_spans_text(ln) for ln in kept_lines]
    new_text = _join_lines([t for t in line_texts if t])
    if not new_text:
        return None
    return make_block(
        type=getattr(orig_block, "type", "text"),
        page_num=getattr(orig_block, "page_num", 0),
        bbox=(x0, y0, x1, y1), lines=list(kept_lines),
        text_override=new_text,
        font_size=getattr(orig_block, "font_size", 0.0),
        font_flags=getattr(orig_block, "font_flags", 0),
        color=getattr(orig_block, "color", 0),
        starts_with_bold=getattr(orig_block, "starts_with_bold", False),
        table_data=getattr(orig_block, "table_data", None),
        caption=getattr(orig_block, "caption", None),
    )


def remove_figure_overlapping_text(text_blocks: List[Block], figure_rects,
                                   margin_x=2, margin_y=2) -> List[Block]:
    """Drop only figure-owned *lines*; captions always survive."""
    if not figure_rects:
        return list(text_blocks)
    result: List[Block] = []
    for b in text_blocks:
        if is_caption_text(getattr(b, "text", "")):
            result.append(b)
            continue
        lines = getattr(b, "lines", None) or []
        if not lines:
            bb = tuple(b.bbox)
            if not _line_should_drop(bb, figure_rects, margin_x, margin_y,
                                     getattr(b, "text", "") or ""):
                result.append(b)
            continue
        kept = []
        for ln in lines:
            bb = tuple(ln.bbox)
            if _line_should_drop(bb, figure_rects, margin_x, margin_y):
                continue
            if _line_should_drop(bb, figure_rects, margin_x, margin_y,
                                 _line_spans_text(ln)):
                continue
            kept.append(ln)
        if len(kept) == len(lines):
            result.append(b)
            continue
        new_block = _rebuild_block_from_lines(b, kept)
        if new_block is not None:
            result.append(new_block)
    return result


# ── Cross-block hyphen merge ──────────────────────────────────────────────
def _try_merge_hyphen_pair(a: Block, b: Block) -> Optional[Block]:
    at = (getattr(a, "text", "") or "").rstrip()
    if not at or at[-1] not in _HYPHEN_CHARS:
        return None
    base = at[:-1].rstrip()
    if not base or not base[-1].isalpha():
        return None
    bt = (getattr(b, "text", "") or "").lstrip()
    if not bt or not bt[0].isalpha() or not bt[0].islower():
        return None
    if is_caption_text(at) or is_caption_text(bt):
        return None
    if bool(_RE_FIG_MARKER.match(bt)) or bool(_RE_TBL_MARKER.match(bt)):
        return None
    if _looks_like_real_hyphen(base[-80:], bt[:80]):
        return None
    try:
        ax0, ay0, ax1, ay1 = tuple(a.bbox)
        bx0, by0, bx1, by1 = tuple(b.bbox)
    except Exception:
        return None
    x_overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    min_w = max(1.0, min(ax1 - ax0, bx1 - bx0))
    a_lines = getattr(a, "lines", None) or []
    line_h = max(1.0, (ay1 - ay0) / max(1, len(a_lines)))
    gap = by0 - ay1
    if x_overlap < 0.3 * min_w:
        if not (by0 < ay0 - 2.0 * line_h):
            return None
    elif not (-2.0 <= gap <= 2.5 * line_h):
        return None
    try:
        if abs(float(getattr(a, "font_size", 0) or 0) - float(getattr(b, "font_size", 0) or 0)) > 1.5:
            return None
    except Exception:
        pass
    a_lines = list(getattr(a, "lines", None) or [])
    b_lines = list(getattr(b, "lines", None) or [])
    if not a_lines or not b_lines:
        return None
    return _rebuild_block_from_lines(a, a_lines + b_lines)


def merge_hyphen_split_blocks(text_blocks: List[Block]) -> List[Block]:
    if len(text_blocks) < 2:
        return list(text_blocks)
    out = [text_blocks[0]]
    for b in text_blocks[1:]:
        merged = _try_merge_hyphen_pair(out[-1], b)
        out[-1] = merged if merged is not None else out[-1]
        if merged is None:
            out.append(b)
    return out
