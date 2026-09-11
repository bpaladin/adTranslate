from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class Span:
    text: str
    font: str
    size: float
    flags: int
    color: int
    origin: tuple
    bbox: tuple


@dataclass
class Line:
    spans: List[Span] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    y0: float = 0.0


@dataclass
class Block:
    type: str
    lines: List[Line] = field(default_factory=list)
    bbox: tuple = (0, 0, 0, 0)
    page_num: int = 0
    image_data: Optional[str] = None
    image_ext: Optional[str] = None
    caption: Optional[str] = None
    table_data: Optional[List[List[str]]] = None
    translation: Optional[str] = None

    @property
    def text(self) -> str:
        return " ".join(span.text for line in self.lines for span in line.spans).strip()


@dataclass
class Page:
    num: int
    blocks: List[Block] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0


def make_span(text: str, font: str = "", size: float = 12, flags: int = 0,
              color: int = 0, origin: tuple = (0, 0), bbox: tuple = (0, 0, 0, 0)) -> Span:
    return Span(text=text, font=font, size=size, flags=flags, color=color, origin=origin, bbox=bbox)


def make_line(spans: Optional[List[Span]] = None, bbox: tuple = (0, 0, 0, 0), y0: float = 0.0) -> Line:
    return Line(spans=spans or [], bbox=bbox, y0=y0)


def make_block(type: str, lines: Optional[List[Line]] = None, bbox: tuple = (0, 0, 0, 0),
               page_num: int = 0, **kwargs) -> Block:
    b = Block(type=type, lines=lines or [], bbox=bbox, page_num=page_num)
    for k, v in kwargs.items():
        setattr(b, k, v)
    return b


@dataclass
class BlockMetrics:
    text: str
    max_font: float
    all_upper: bool
    has_bold: bool
    is_centered: bool
    total_spans: int
    bbox: tuple


def block_metrics(block: Block, page_width: float = 0.0) -> BlockMetrics:
    text = block.text
    max_font = 0.0
    all_upper = True
    has_bold = False
    total_spans = 0
    for line in block.lines:
        for span in line.spans:
            total_spans += 1
            if span.size > max_font:
                max_font = span.size
            if span.text and not span.text.isupper():
                all_upper = False
            if span.flags & 2**0:
                has_bold = True
    bbox = block.bbox
    center_x = (bbox[0] + bbox[2]) / 2
    is_centered = abs(center_x - page_width / 2) < page_width * 0.1 if page_width else False
    return BlockMetrics(text=text, max_font=max_font, all_upper=all_upper, has_bold=has_bold,
                        is_centered=is_centered, total_spans=total_spans, bbox=bbox)