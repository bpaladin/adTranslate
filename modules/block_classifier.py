import re
from modules.pdf_extractor import Block


REF_HEADING_RE = re.compile(
    r'^(References?|Bibliography|Литература|Список\s+литературы|'
    r'Список\s+использованных\s+источников)\s*$',
    re.IGNORECASE,
)


def classify_block(block: Block, page_width: float, avg_font_size: float) -> str:
    if not block.lines:
        return "empty"

    text = " ".join(
        span.text for line in block.lines for span in line.spans
    ).strip()

    if not text:
        return "empty"

    total_spans = sum(len(line.spans) for line in block.lines)

    max_font = max(span.size for line in block.lines for span in line.spans)
    avg_font = (
        sum(span.size for line in block.lines for span in line.spans)
        / total_spans
        if total_spans else 0
    )

    all_upper = all(
        span.text.isupper()
        for line in block.lines
        for span in line.spans
        if span.text
    )

    has_bold = any(
        span.flags & 2 ** 0
        for line in block.lines
        for span in line.spans
    )

    bbox = block.bbox
    x0, y0, x1, y1 = bbox

    center_x = (x0 + x1) / 2
    page_center = page_width / 2
    is_centered = abs(center_x - page_center) < page_width * 0.1

    # Заголовок списка литературы
    if REF_HEADING_RE.match(text):
        return "reference_heading"

    score = 0

    if max_font > avg_font_size * 1.2:
        score += 3
    if all_upper:
        score += 2
    if has_bold:
        score += 2
    if is_centered:
        score += 1
    if len(text) < 80:
        score += 1
    if re.match(r'^\d+(\.\d+)*\s+', text):
        score += 2

    # Метаданные
    if re.search(r'[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.', text):
        return "metadata"
    if re.search(
        r'(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright)',
        text,
        re.I,
    ):
        return "metadata"

    # Отдельные элементы библиографии
    if re.match(r'^\[\d+\]', text):
        return "reference"

    if score >= 5:
        return "heading"

    return "paragraph"
