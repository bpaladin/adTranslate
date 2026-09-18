import re
import string
import html as html_mod
from typing import List, Dict

import markdown as md
from jinja2 import Template

from .models import Block, Page

REF_PATTERN = re.compile(r'\[(?:[0-9][0-9,\s;\-–—]*|[A-Za-z][A-Za-z0-9-]*)\]')

_CSS_TEMPLATE = string.Template("""
<style>
body { font-family: $BODY_FONT; margin: $BODY_MARGIN; background: $BG; color: $TEXT; }
.container { max-width: 1100px; margin: 0 auto; padding: 20px 40px; }
h1 { color: $H1_COLOR; border-bottom: 2px solid $H1_BORDER; padding-bottom: 12px; }
h3 { color: $H3_COLOR; margin-top: 28px; }
.page { background: $PAGE_BG; padding: 15px 20px; margin-bottom: 20px; border-radius: 8px; $PAGE_BOX }
.page > summary { cursor: pointer; font-size: 1.05em; font-weight: bold; color: $SUMMARY_COLOR; padding: 4px 0 8px; user-select: none; }
.page > summary:hover { color: $SUMMARY_HOVER; }
.trans-head { border-left: 3px solid #3b82f6; padding-left: 12px; margin: 12px 0; }
.trans-head h2, .trans-head h3, .trans-head h4 { margin: 4px 0; }
p { line-height: 1.7; margin: 0 0 10px; text-align: justify; }
.orig { color: $ORIG_COLOR; font-size: 0.88em; }
.trans { color: $TEXT; }
.ref-section h2 { color: $H1_COLOR; border-top: 2px solid $TABLEWRAP_BORDER; margin-top: 24px; padding-top: 16px; padding-bottom: 6px; }
.ref { color: $REF_COLOR; font-size: 0.88em; margin-left: 16px; font-family: 'Courier New', monospace; line-height: 1.4; margin-bottom: 8px; }
.meta { color: $META_COLOR; font-size: 0.82em; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 1px solid $TABLEWRAP_BORDER; }
.table-wrap { overflow-x: auto; margin: 12px 0; border-radius: 8px; border: 1px solid $TABLEWRAP_BORDER; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.table-wrap th { background: $TH_BG; border: 1px solid $TABLEWRAP_BORDER; padding: 8px 10px; color: $TH_COLOR; }
.table-wrap td { border: 1px solid $TABLEWRAP_BORDER; padding: 6px 10px; }
.figure { margin: 20px 0; text-align: center; }
.figure img { max-width: 100%; height: auto; border-radius: 6px; cursor: zoom-in; }
.figure-caption { margin-top: 6px; font-style: italic; color: $CAPTION_COLOR; font-size: 0.85em; }
.figure-extra { margin-top: 4px; font-size: 0.8em; color: #999; }
.list-block { margin-left: 20px; line-height: 1.6; }
.ref-link { color: $REFLINK_COLOR; font-weight: bold; }
.table-wrap tbody tr:nth-child(even) td { background: $TBODY_EVEN; }
.table-pre { white-space: pre-wrap; font-family: 'Courier New', monospace; font-size: 0.85em; margin: 8px 0; }
.table-caption { font-style: italic; color: $CAPTION_COLOR; font-size: 0.85em; margin: 8px 0; }
.toc { background: $TOC_BG; border: 1px solid $TOC_BORDER; border-radius: 8px; padding: 14px 20px; margin-bottom: 20px; }
.toc ul { list-style: none; margin: 6px 0 0; padding-left: 18px; }
.toc > ul { padding-left: 0; }
.toc a { color: $TOC_A; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
.btn-orig { background: $BTN_BG; border: 1px solid $BTN_BORDER; color: $BTN_COLOR; border-radius: 6px; padding: 6px 12px; margin: 0 4px 12px 0; cursor: pointer; }
.btn-orig:hover { color: $BTN_HOVER; }
details.original-block { margin: 4px 0; }
details.original-block summary { color: $DETAILS_COLOR; font-size: 0.85em; cursor: pointer; }
details.original-block summary:hover { color: $DETAILS_HOVER; }
#lightbox { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 999; align-items: center; justify-content: center; cursor: zoom-out; }
#lightbox img { max-width: 92%; max-height: 92%; border-radius: 6px; }
</style>
""")

_CSS_PALETTES = {
    True: {
        "BODY_FONT": "'Segoe UI', Arial, sans-serif", "BODY_MARGIN": "0; padding: 0", "BG": "#0f172a", "TEXT": "#e2e8f0",
        "H1_COLOR": "#60a5fa", "H1_BORDER": "#1e3a5f", "H3_COLOR": "#93c5fd", "PAGE_BG": "#1e293b",
        "PAGE_BOX": "border: 1px solid #334155", "SUMMARY_COLOR": "#93c5fd", "SUMMARY_HOVER": "#60a5fa",
        "ORIG_COLOR": "#64748b", "REF_COLOR": "#64748b", "META_COLOR": "#64748b", "TABLEWRAP_BORDER": "#334155",
        "TH_BG": "#0f172a", "TH_COLOR": "#60a5fa", "TBODY_EVEN": "#16213a", "CAPTION_COLOR": "#94a3b8",
        "REFLINK_COLOR": "#60a5fa", "TOC_BG": "#1e293b", "TOC_BORDER": "#334155", "TOC_A": "#93c5fd",
        "BTN_BG": "#1e293b", "BTN_BORDER": "#334155", "BTN_COLOR": "#94a3b8", "BTN_HOVER": "#e2e8f0",
        "DETAILS_COLOR": "#64748b", "DETAILS_HOVER": "#94a3b8",
    },
    False: {
        "BODY_FONT": "Arial, sans-serif", "BODY_MARGIN": "0; padding: 20px", "BG": "#f0f2f5", "TEXT": "#111827",
        "H1_COLOR": "#1f2937", "H1_BORDER": "#d1d5db", "H3_COLOR": "#374151", "PAGE_BG": "white",
        "PAGE_BOX": "box-shadow: 0 2px 4px rgba(0,0,0,0.1);", "SUMMARY_COLOR": "#1f2937", "SUMMARY_HOVER": "#2563eb",
        "ORIG_COLOR": "#9ca3af", "REF_COLOR": "#6b7280", "META_COLOR": "#6b7280", "TABLEWRAP_BORDER": "#d1d5db",
        "TH_BG": "#f3f4f6", "TH_COLOR": "#1f2937", "TBODY_EVEN": "#f9fafb", "CAPTION_COLOR": "#6b7280",
        "REFLINK_COLOR": "#2563eb", "TOC_BG": "#f8fafc", "TOC_BORDER": "#e5e7eb", "TOC_A": "#2563eb",
        "BTN_BG": "#fff", "BTN_BORDER": "#d1d5db", "BTN_COLOR": "#6b7280", "BTN_HOVER": "#111827",
        "DETAILS_COLOR": "#9ca3af", "DETAILS_HOVER": "#6b7280",
    }
}


def build_css(dark: bool = True) -> str:
    return _CSS_TEMPLATE.safe_substitute(_CSS_PALETTES[bool(dark)])


def _reference_entries(block: Block) -> List[str]:
    entries: List[List[str]] = []
    cur: List[str] = []
    for line in block.lines:
        txt = " ".join(s.text for s in line.spans).strip()
        if not txt:
            continue
        if re.match(r'^\d+\.', txt) and cur:
            entries.append(cur)
            cur = []
        cur.append(txt)
    if cur:
        entries.append(cur)
    return [" ".join(e) for e in entries]


def highlight_refs(escaped_text: str) -> str:
    return REF_PATTERN.sub(lambda m: f'<span class="ref-link">{m.group(0)}</span>', escaped_text)


def render_spans(block):
    parts = []
    for line in block.lines:
        for span in line.spans:
            text = html_mod.escape(span.text)
            if span.flags & 2**0:
                text = f"<b>{text}</b>"
            if span.flags & 2**1:
                text = f"<i>{text}</i>"
            parts.append(text)
        parts.append("\n")
    return _repair_render_hyphens(" ".join(parts))


_RENDER_PREFIXES = frozenset(
    "non pre post anti auto co de dis ex extra hyper inter intra macro meta "
    "micro mid mini multi over pseudo re semi sub super trans ultra under uni "
    "well high low long short self cross deep fast real un mis counter "
    "де ре пост пре анти авто со экс гипер интер макро микро мульти сверх суб "
    "транс ультра высоко низко долго само взаимо полу недо пере".split()
)
_RENDER_TAILS = frozenset(
    "of the a an in on and or to "
    "в на с к о у об от до по из за над под".split()
)
_RENDER_BREAK_RE = re.compile(
    r"([^\s<>]+)"
    r"([\-‐‑‒–—―−⁃﹘﹣])"
    r"((?:</[bi]>)?)"
    r"\s*\n\s*"
    r"((?:<[bi]>)?)"
    r"([^\s<>]+)"
)
_WORD_EDGE_RE = re.compile(
    r"[A-Za-zÀ-ÿА-Яа-яЁё]+(?:['’][A-Za-zА-Яа-яЁё]+)?"
)


def _repair_render_hyphens(s: str) -> str:
    """Сшивает переносы, видимые в already-escaped span-потоке.

    render_spans склеивает строки через '\\n', поэтому дефис на конце
    строки + продолжение на следующей чинятся здесь же — иначе raw-HTML
    показывает «How- ever», хотя block.text уже склеен. Теги <b>/<i>
    вокруг стыка сохраняются.
    """
    def _sub(m: re.Match) -> str:
        lw_full, hy, ct, ot, rw_full = m.groups()
        lw_m = list(_WORD_EDGE_RE.finditer(lw_full))
        rw_m = _WORD_EDGE_RE.search(rw_full)
        if not lw_m or not rw_m:
            return f"{lw_full}{hy}{ct} {ot}{rw_full}"
        lw = lw_m[-1].group(0)
        rw = rw_m.group(0)
        lw_pre = lw_full[:lw_m[-1].start()]
        rw_post = rw_full[rw_m.end():]
        # Только буква-буква; цифры/пунктуация — настоящий дефис с пробелом
        if not (lw and rw and lw[-1].isalpha() and rw[0].isalpha()):
            return f"{lw_full}{hy}{ct} {ot}{rw_full}"
        # Составное слово (префикс/хвост/обломок/заглавная): дефис держим.
        # Строчное продолжение клеим без пробела («well-known»),
        # заглавное — через пробел (новое предложение).
        if (len(lw) <= 1 or len(rw) <= 1
                or lw.lower() in _RENDER_PREFIXES
                or rw.lower() in _RENDER_TAILS
                or rw[0].isupper()):
            if rw[0].islower():
                return f"{lw_pre}{lw}{hy}{ct}{ot}{rw}{rw_post}"
            return f"{lw_full}{hy}{ct} {ot}{rw_full}"
        # Перенос: убираем дефис и разрыв строки
        return f"{lw_pre}{lw}{ct}{ot}{rw}{rw_post}"

    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = _RENDER_BREAK_RE.sub(_sub, cur)
    return cur


def _block_html(block: Block) -> str:
    e = html_mod.escape
    bt = block.type
    trans = block.translation if block.translation is not None else "[ПЕРЕВОД НЕ ПОЛУЧЕН]"
    orig = block.text

    def trans_html(text: str) -> str:
        return highlight_refs(e(text))

    def orig_html(spans_text: str) -> str:
        return highlight_refs(e(spans_text))

    if bt == 'reference_heading':
        return f'<div class="ref-section"><h2>{trans_html(trans)}</h2></div>\n'
    if bt == 'empty':
        return ''
    if bt == 'reference':
        entries = _reference_entries(block)
        if len(entries) > 1:
            return ''.join(f'<p class="ref">{trans_html(x)}</p>\n' for x in entries)
        return f'<p class="ref">{trans_html(trans)}</p>\n'
    if bt == 'metadata':
        if orig != trans:
            return (f'<div class="meta">{trans_html(trans)}</div>'
                    f'<details class="original-block"><summary>Оригинал</summary>'
                    f'<p class="orig">{orig_html(orig)}</p></details>\n')
        return f"<div class='meta'>{orig_html(orig)}</div>\n"
    if bt == 'table':
        if block.table_data:
            rows = block.table_data
            out = []
            if block.caption:
                out.append(f'<div class="table-caption">{trans_html(block.caption)}</div>\n')
            out.append('<div class="table-wrap"><table>\n')
            for idx, row in enumerate(rows[:20]):
                tag = 'th' if idx == 0 else 'td'
                if idx == 0:
                    out.append('<thead><tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr></thead><tbody>\n')
                else:
                    out.append('<tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr>\n')
            out.append('</tbody></table>\n</div>\n')
            return ''.join(out)
        if orig.strip():
            return f'<div class="table-wrap"><pre class="table-pre">{e(orig)}</pre></div>\n'
    if bt == 'figure':
        parts = []
        if block.image_data:
            img_src = f"data:image/{block.image_ext or 'png'};base64,{block.image_data}"
            alt = block.caption or f"Рисунок {block.page_num}"
            parts.append(f'<figure class="figure"><img src="{img_src}" alt="{e(alt)}" onclick="zoomImg(this)" />')
            if block.caption:
                parts.append(f'<figcaption class="figure-caption">{e(block.caption)}</figcaption>')
            parts.append('</figure>\n')
            return ''.join(parts)
        return '<p class="orig">(Изображение не извлечено)</p>\n'
    if bt == 'heading':
        spans_text = render_spans(block)
        anchor = f' id="{e(getattr(block, "_anchor", ""))}"' if getattr(block, "_anchor", None) else ''
        if orig != trans:
            return f'<div class="trans-head"{anchor}><p class="orig">{orig_html(spans_text)}</p><h3>{trans_html(trans)}</h3></div>\n'
        return f'<div class="trans-head"{anchor}><h3>{trans_html(trans)}</h3></div>\n'
    if bt == 'list':
        items = orig.split('\n')
        numbered = bool(re.match(r'^\s*\d+[\.\)]', items[0])) if items and items[0].strip() else False
        tag = 'ol' if numbered else 'ul'
        lis = ''.join(f'<li>{e(item.lstrip("•-*►▸‣⁃◦○●▪ 0123456789.)"))}</li>' for item in items if item.strip())
        return f'<{tag} class="list-block">{lis}</{tag}>\n'

    spans_text = render_spans(block)
    if orig != trans:
        return f'<p class="trans">{trans_html(trans)}</p><details class="original-block"><summary>Оригинал</summary><p class="orig">{orig_html(spans_text)}</p></details>\n'
    return f"<p>{trans_html(trans)}</p>\n"


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title | e }}</title>
    {{ css | safe }}
</head>
<body>
<div class="container">
    <h1>{{ title | e }}</h1>
    {% if toc %}
    <div class="toc"><strong>Содержание</strong>
        <ul>
        {% for item in toc %}
            <li><a href="#{{ item.anchor }}">{{ item.text | e }}</a></li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    <div>
        <button class="btn-orig" onclick="toggleAllOriginals(true)">Показать все оригиналы</button>
        <button class="btn-orig" onclick="toggleAllOriginals(false)">Скрыть оригиналы</button>
    </div>
    {% for page in pages %}
    <details class="page" open>
        <summary> Страница {{ page.num }}</summary>
        {% for block in page.blocks %}
            {{ _block_html(block) | safe }}
        {% endfor %}
    </details>
    {% endfor %}
</div>
<div id="lightbox" onclick="closeLightbox()"></div>
<script>
function toggleAllOriginals(open) {
    document.querySelectorAll('details.original-block').forEach(function(d){ d.open = open; });
}
function zoomImg(img) {
    var lb = document.getElementById('lightbox');
    lb.innerHTML = '';
    var c = document.createElement('img');
    c.src = img.src;
    lb.appendChild(c);
    lb.style.display = 'flex';
}
function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
}
</script>
</body>
</html>
"""


def _build_toc(pages: List[Page]) -> List[Dict[str, str]]:
    toc = []
    idx = 0
    for page in pages:
        for block in page.blocks:
            if block.type == "heading":
                block._anchor = f"h{idx}"
                toc.append({"anchor": f"h{idx}", "text": block.translation or block.text})
                idx += 1
    return toc


def generate_html(pages: List[Page], title: str, output_path: str, dark: bool = True):
    css = build_css(dark)
    toc = _build_toc(pages)
    template = Template(HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, css=css, toc=toc, _block_html=_block_html)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)


SUMMARY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>{{ title }}</title>
<style>
body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f2f5; }
.summary-container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); max-width: 95%; margin: 0 auto; }
.summary-block { background: #e7f3ff; padding: 20px; border-radius: 8px; border: 1px solid #b3d7ff; margin-bottom: 20px; }
.summary-block h2 { margin-top: 0; color: #004085; }
.metadata { background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 2em; border: 1px solid #dee2e6; }
.metadata h1 { margin-top: 0; color: #111; font-size: 2em; }
.page > summary { cursor: pointer; font-size: 1.05em; font-weight: bold; color: #1f2937; padding: 4px 0 8px; user-select: none; }
.page > summary:hover { color: #2563eb; }
</style>
</head>
<body>
<div class="summary-container">
<div class="metadata"><h1>{{ title }}</h1></div>
<div class="summary-block">{{ summary_html | safe }}</div>
<details><summary>Показать исходные тексты</summary>
{% for page in pages %}
<details class="page" open><summary>Страница {{ page.num }}</summary>
{% for block in page.blocks %}
    {% if block.type in ('paragraph', 'heading', 'metadata', 'text', 'list') %}
        {% set orig = render_spans(block) %}
        {% if orig | length > 40 %}
        <details><summary>Оригинал (стр. {{ page.num }})</summary><span class="original">{{ orig }}</span></details>
        {% endif %}
    {% endif %}
{% endfor %}</details>
{% endfor %}</details></div></body></html>
"""


def generate_summary_html(pages: List[Page], summary_html: str, title: str, output_path: str):
    template = Template(SUMMARY_TEMPLATE)
    rendered = template.render(pages=pages, title=title, summary_html=summary_html, render_spans=render_spans)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)


def _block_html_raw(block: Block) -> str:
    e = html_mod.escape
    bt = block.type
    orig = block.text

    def orig_html(spans_text: str) -> str:
        return highlight_refs(e(spans_text))

    def spans_html(block) -> str:
        return highlight_refs(render_spans(block))

    if bt == 'empty':
        return ''
    if bt == 'reference_heading':
        return f'<div class="ref-section"><h2>{orig_html(orig)}</h2></div>\n'
    if bt == 'reference':
        entries = _reference_entries(block)
        if len(entries) > 1:
            return ''.join(f'<p class="ref">{orig_html(x)}</p>\n' for x in entries)
        return f'<p class="ref">{orig_html(orig)}</p>\n'
    if bt == 'metadata':
        return f"<div class='meta'>{orig_html(orig)}</div>\n"
    if bt == 'table':
        if block.table_data:
            rows = block.table_data
            out = []
            if block.caption:
                out.append(f'<div class="table-caption">{orig_html(block.caption)}</div>\n')
            out.append('<div class="table-wrap"><table>\n')
            for idx, row in enumerate(rows[:20]):
                tag = 'th' if idx == 0 else 'td'
                if idx == 0:
                    out.append('<thead><tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr></thead><tbody>\n')
                else:
                    out.append('<tr>')
                    out.extend(f'<{tag}>{e(c)}</{tag}>' for c in row)
                    out.append('</tr>\n')
            out.append('</tbody></table>\n</div>\n')
            return ''.join(out)
        if orig.strip():
            return f'<div class="table-wrap"><pre class="table-pre">{e(orig)}</pre></div>\n'
    if bt == 'figure':
        parts = []
        if block.image_data:
            img_src = f"data:image/{block.image_ext or 'png'};base64,{block.image_data}"
            alt = block.caption or f"Рисунок {block.page_num}"
            parts.append(f'<figure class="figure"><img src="{img_src}" alt="{e(alt)}" onclick="zoomImg(this)" />')
            if block.caption:
                parts.append(f'<figcaption class="figure-caption">{e(block.caption)}</figcaption>')
            parts.append('</figure>\n')
            return ''.join(parts)
        return '<p class="orig">(Изображение не извлечено)</p>\n'
    if bt == 'heading':
        anchor = f' id="{e(getattr(block, "_anchor", ""))}"' if getattr(block, "_anchor", None) else ''
        return f'<div class="trans-head"{anchor}><h3>{spans_html(block)}</h3></div>\n'
    if bt == 'list':
        items = orig.split('\n')
        numbered = bool(re.match(r'^\s*\d+[\.\)]', items[0])) if items and items[0].strip() else False
        tag = 'ol' if numbered else 'ul'
        lis = ''.join(f'<li>{e(item.lstrip("•-*►▸‣⁃◦○●▪ 0123456789.)"))}</li>' for item in items if item.strip())
        return f'<{tag} class="list-block">{lis}</{tag}>\n'

    return f'<p>{spans_html(block)}</p>\n'


RAW_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{{ title | e }}</title>
    {{ css | safe }}
</head>
<body>
<div class="container">
    <h1>{{ title | e }}</h1>
    {% if toc %}
    <div class="toc"><strong>Содержание</strong>
        <ul>
        {% for item in toc %}
            <li><a href="#{{ item.anchor }}">{{ item.text | e }}</a></li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    {% for page in pages %}
    <details class="page" open>
        <summary> Страница {{ page.num }}</summary>
        {% for block in page.blocks %}
            {{ _block_html_raw(block) | safe }}
        {% endfor %}
    </details>
    {% endfor %}
</div>
<div id="lightbox" onclick="closeLightbox()"></div>
<script>
function zoomImg(img) {
    var lb = document.getElementById('lightbox');
    lb.innerHTML = '';
    var c = document.createElement('img');
    c.src = img.src;
    lb.appendChild(c);
    lb.style.display = 'flex';
}
function closeLightbox() {
    document.getElementById('lightbox').style.display = 'none';
}
</script>
</body>
</html>
"""


def _build_toc_raw(pages: List[Page]) -> List[Dict[str, str]]:
    toc = []
    idx = 0
    for page in pages:
        for block in page.blocks:
            if block.type == "heading":
                block._anchor = f"h{idx}"
                toc.append({"anchor": f"h{idx}", "text": block.text})
                idx += 1
    return toc


def generate_raw_html(pages: List[Page], title: str, output_path: str, dark: bool = True):
    css = build_css(dark)
    toc = _build_toc_raw(pages)
    template = Template(RAW_HTML_TEMPLATE)
    rendered = template.render(pages=pages, title=title, css=css, toc=toc, _block_html_raw=_block_html_raw)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(rendered)
