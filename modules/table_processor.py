import fitz
from typing import List
from modules.pdf_extractor import Block


def extract_tables(page: fitz.Page, page_num: int) -> List[Block]:
    tables = page.find_tables()
    table_blocks = []

    for tab in tables:
        try:
            rows = tab.extract()
            if not rows:
                continue

            cleaned = []
            for row in rows:
                cleaned.append([cell if cell else "" for cell in row])

            block = Block(
                type="table",
                page_num=page_num,
                bbox=tuple(tab.bbox),
            )
            block.table_data = cleaned
            table_blocks.append(block)
        except Exception:
            continue

    return table_blocks


def intersection_ratio(block_bbox, table_bbox):
    x0 = max(block_bbox[0], table_bbox[0])
    y0 = max(block_bbox[1], table_bbox[1])
    x1 = min(block_bbox[2], table_bbox[2])
    y1 = min(block_bbox[3], table_bbox[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0

    intersection = (x1 - x0) * (y1 - y0)
    block_area = (
        (block_bbox[2] - block_bbox[0])
        * (block_bbox[3] - block_bbox[1])
    )
    if block_area == 0:
        return 0.0

    return intersection / block_area


def mark_table_blocks(text_blocks: List[Block], table_blocks: List[Block]):
    """
    Если текстовый блок более чем на 30% попадает в область таблицы,
    помечаем его как table (не переводится).
    """
    for block in text_blocks:
        if block.type == "table":
            continue
        for table in table_blocks:
            if intersection_ratio(block.bbox, table.bbox) > 0.30:
                block.type = "table"
                break
