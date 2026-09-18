import re
from typing import Optional

from .models import Block, block_metrics
from .utils import _count_list_lines
from .table_detection import _is_table_caption

_REF_HEADING_ALT = (
    r'References?|Reference\s+list|References?\s+and\s+notes?|'
    r'Bibliography|Библиография|Библиографический\s+список|'
    r'Литература|Список\s+литературы|Список\s+использованной\s+литературы|'
    r'Список\s+использованных\s+источников|Список\s+источников|'
    r'Использованная\s+литература|Источники'
)
REF_HEADING_RE = re.compile(rf'^\s*({_REF_HEADING_ALT})\s*[:.]?\s*$', re.IGNORECASE)


class BlockClassifier:
    LABELS = ["paragraph", "heading", "list", "metadata", "reference", "reference_heading", "empty", "table"]
    METADATA_WORD_LIMIT = 10
    TABLE_COL_RE = re.compile(r'\s{3,}|\t+')

    def classify(self, block: Block, page_width: float, avg_font_size: float) -> str:
        if not block.lines:
            return "empty"
        m = block_metrics(block, page_width)
        if not m.text or m.total_spans == 0:
            return "empty"
        if not re.search(r'[A-Za-zА-Яа-я]', m.text):
            return "empty"
        if re.fullmatch(r'\s*(?:[a-zA-Z_]+\(\)|[\W\d]+)\s*', m.text):
            return "empty"

        word_count = len(m.text.split())
        stripped = m.text.strip()

        # Одно слово заглавными — заголовок (требование: одно слово капсом → <h3>)
        # Примеры: INTRODUCTION, METHODS, RESULTS, ВВЕДЕНИЕ, ABSTRACT
        # Фильтр len>=3 чтобы отсечь фрагменты "NE", "IE" от разбитых блоков
        # Исключаем reference-ключевые слова (REFERENCES, ЛИТЕРАТУРА и т.д.)
        if word_count == 1 and m.all_upper and len(stripped) >= 3:
            if not REF_HEADING_RE.match(m.text):
                core = stripped.rstrip(':.')
                if len(core) >= 3 and re.fullmatch(r'[A-ZА-ЯЁ0-9\-]+', core) and re.search(r'[A-ZА-ЯЁ]', core):
                    return "heading"

        # Эвристика таблицы по тексту (колонки через табуляцию / 3+ пробела)
        if self._is_likely_table(m.text):
            return "table"

        if REF_HEADING_RE.match(m.text):
            return "reference_heading"
        if _is_table_caption(m.text):
            return "table"

        if word_count <= self.METADATA_WORD_LIMIT:
            if re.match(r'^\[\d+\]', m.text):
                return "reference"
            if re.search(r'[A-Z][a-z]+\s+[A-Z]\.[A-Z]\.', m.text):
                return "metadata"
            if re.search(r'(Journal|Volume|Issue|Pages|DOI|ISSN|ISBN|©|Copyright|Email|Corresponding|Received|Accepted|Published)', m.text, re.I):
                return "metadata"
            if re.fullmatch(r'[\w\s\-]+,\s*\d{4}', m.text):
                return "metadata"

        list_lines = _count_list_lines(m.text)
        if list_lines >= 2 or (list_lines >= 1 and len(m.text.split('\n')) >= 2):
            return "list"

        score = 0
        if m.max_font > avg_font_size * 1.25:
            score += 3
        if m.all_upper:
            score += 2
        if m.has_bold:
            score += 2
        if m.is_centered:
            score += 1
        if len(m.text) < 100:
            score += 1
        if re.match(r'^\d+(\.\d+)*\s+', m.text):
            score += 2
        if word_count <= 4 and m.max_font >= avg_font_size * 1.1:
            score += 2
        return "heading" if score >= 5 else "paragraph"

    def _is_likely_table(self, text: str) -> bool:
        lines = [ln for ln in text.split('\n') if ln.strip()]
        if len(lines) < 2:
            return False
        col_counts = set()
        for ln in lines:
            cols = len(re.split(r'\s{3,}|\t+', ln.strip()))
            if cols >= 2:
                col_counts.add(cols)
        if len(col_counts) == 1 and next(iter(col_counts)) >= 2:
            if not any(re.search(r'[.!?;]\s*$', ln) for ln in lines[:-1]):
                return True
        return False


_classifier = BlockClassifier()