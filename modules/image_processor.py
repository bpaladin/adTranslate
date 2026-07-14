# image_processor.py
import fitz
import base64
import re
from typing import List
from modules.pdf_extractor import Block

def extract_images(page: fitz.Page, page_num: int, text_blocks: List[Block]) -> List[Block]:
    image_list = page.get_images(full=True)
    figure_blocks = []
    for img_idx, img in enumerate(image_list):
        xref = img[0]
        try:
            pix = fitz.Pixmap(page.parent, xref)
            if pix.n - pix.alpha < 4:
                if pix.alpha:
                    pix = fitz.Pixmap(pix, 0)
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            pix = None
        except:
            continue

        # Ищем caption среди текстовых блоков
        caption = None
        # Получаем координаты изображения (приблизительно)
        # В rawdict изображения имеют блок с типом 1, но мы их не обрабатывали – можно получить через get_image_rects
        # Для простоты: ищем текст с "Figure" в радиусе 100 pt по вертикали
        img_bbox = None
        # Для упрощения пропустим точный поиск – можно взять из get_image_info
        # Здесь нужно сохранить координаты изображения, но в PyMuPDF нет прямого метода.
        # Альтернатива: использовать page.get_image_rects(xref) – возвращает список прямоугольников.
        rects = page.get_image_rects(xref)
        if rects:
            img_bbox = rects[0]
        else:
            continue

        # Ищем текстовые блоки рядом
        for block in text_blocks:
            # Проверяем расстояние по вертикали
            bbox = block.bbox
            y0, y1 = bbox[1], bbox[3]
            img_y0, img_y1 = img_bbox[1], img_bbox[3]
            # Если блок выше или ниже изображения в пределах 80 pt
            if abs(y1 - img_y0) < 80 or abs(img_y1 - y0) < 80:
                text = " ".join(span.text for line in block.lines for span in line.spans)
                if re.search(r'Figure|Fig\.|Рис\.|Схема', text, re.I):
                    caption = text
                    # Удаляем этот текстовый блок из списка, чтобы не дублировать
                    text_blocks.remove(block)
                    break

        # Создаём блок figure
        fig_block = Block(type="figure", page_num=page_num, bbox=img_bbox)
        fig_block.image_data = img_b64
        fig_block.caption = caption
        figure_blocks.append(fig_block)
    return figure_blocks