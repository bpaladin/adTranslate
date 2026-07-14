import fitz
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Span:
    text: str
    font: str
    size: float
    flags: int          # 2^0 bold, 2^1 italic
    color: int
    origin: tuple       # (x, y)
    bbox: tuple         # (x0, y0, x1, y1)


@dataclass
class Line:
    spans: List[Span] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    y0: float = 0.0


@dataclass
class Block:
    type: str           # 'text', 'image', 'table', 'figure', 'heading',
                        # 'paragraph', 'metadata', 'reference', 'reference_heading'
    lines: List[Line] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    page_num: int = 0
    image_data: Optional[str] = None   # base64
    caption: Optional[str] = None
    table_data: Optional[List[List[str]]] = None
    translation: Optional[str] = None


@dataclass
class Page:
    num: int
    blocks: List[Block] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


class PDFExtractor:
    def __init__(self, path: str):
        self.doc = fitz.open(path)

    def extract(self) -> List[Page]:
        pages = []
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            raw = page.get_text("dict")
            page_blocks = []

            for b in raw.get("blocks", []):
                if b.get("type") == 0:  # текст
                    block = Block(
                        type="text",
                        page_num=page_num + 1,
                        bbox=b.get("bbox", (0, 0, 0, 0)),
                    )
                    for line_dict in b.get("lines", []):
                        bbox = line_dict.get("bbox", (0, 0, 0, 0))
                        line = Line(bbox=bbox, y0=bbox[1])
                        for span_dict in line_dict.get("spans", []):
                            text = span_dict.get("text", "")
                            if not text.strip():
                                continue
                            span = Span(
                                text=text,
                                font=span_dict.get("font", ""),
                                size=span_dict.get("size", 12),
                                flags=span_dict.get("flags", 0),
                                color=span_dict.get("color", 0),
                                origin=span_dict.get("origin", (0, 0)),
                                bbox=span_dict.get("bbox", (0, 0, 0, 0)),
                            )
                            line.spans.append(span)
                        if line.spans:
                            block.lines.append(line)
                    if block.lines:
                        page_blocks.append(block)
                # Изображения и таблицы обрабатываются отдельными модулями

            pages.append(Page(
                num=page_num + 1,
                blocks=page_blocks,
                width=raw.get("width", 612),
                height=raw.get("height", 792),
            ))
        return pages
