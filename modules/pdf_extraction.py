import logging
from typing import List, Optional

import fitz

from .models import make_block, make_span, make_line, Page

try:
    from pymupdf_layout import extract_layout
    HAVE_LAYOUT = True
except ImportError:
    HAVE_LAYOUT = False

logger = logging.getLogger(__name__)


class PDFExtractor:
    def __init__(self, path: str, crop_top: float = 40.0, crop_bottom: float = 45.0):
        self.doc = fitz.open(path)
        self.use_layout = HAVE_LAYOUT
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def _crop_rect(self, page: fitz.Page) -> fitz.Rect:
        r = page.rect
        top = min(self.crop_top, r.height * 0.4)
        bottom = min(self.crop_bottom, r.height * 0.4)
        return fitz.Rect(r.x0, r.y0 + top, r.x1, r.y1 - bottom)

    def extract(self) -> List[Page]:
        pages = []
        if self.use_layout:
            try:
                layout = extract_layout(self.doc)
                for page_num, page_data in enumerate(layout.pages):
                    top_y = self.crop_top
                    bottom_y = page_data.height - self.crop_bottom
                    page_blocks = []
                    for item in page_data.items:
                        if item['type'] != 'text':
                            continue
                        bbox = item['bbox']
                        if bbox[3] < top_y or bbox[1] > bottom_y:
                            continue
                        block = make_block(type="text", page_num=page_num + 1, bbox=bbox)
                        for line_text in item['text'].split('\n'):
                            if not line_text.strip():
                                continue
                            span = make_span(text=line_text)
                            line = make_line(spans=[span], bbox=bbox, y0=bbox[1])
                            block.lines.append(line)
                        if block.lines:
                            page_blocks.append(block)
                    pages.append(Page(num=page_num + 1, blocks=page_blocks,
                                      width=page_data.width, height=page_data.height))
                return pages
            except Exception as e:
                logger.warning(f"pymupdf_layout error: {e}, falling back to standard")
                self.use_layout = False

        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            clip = self._crop_rect(page)
            raw = page.get_text("dict", clip=clip)
            page_blocks = []
            for b in raw.get("blocks", []):
                if b.get("type") == 0:
                    block = make_block(type="text", page_num=page_num + 1, bbox=b.get("bbox", (0, 0, 0, 0)))
                    for line_dict in b.get("lines", []):
                        bbox = line_dict.get("bbox", (0, 0, 0, 0))
                        line = make_line(bbox=bbox, y0=bbox[1])
                        for span_dict in line_dict.get("spans", []):
                            text = span_dict.get("text", "")
                            if not text.strip():
                                continue
                            span = make_span(
                                text=text, font=span_dict.get("font", ""),
                                size=span_dict.get("size", 12),
                                flags=span_dict.get("flags", 0),
                                color=span_dict.get("color", 0),
                                origin=span_dict.get("origin", (0, 0)),
                                bbox=span_dict.get("bbox", (0, 0, 0, 0))
                            )
                            line.spans.append(span)
                        if line.spans:
                            block.lines.append(line)
                    if block.lines:
                        page_blocks.append(block)
            pages.append(Page(num=page_num + 1, blocks=page_blocks,
                              width=raw.get("width", 612), height=raw.get("height", 792)))
        return pages