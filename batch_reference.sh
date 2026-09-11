#!/bin/bash
# Аргументы:
#   КАТАЛОГ   — каталог с PDF-файлами (по умолчанию: ./pdfs)

CATALOG="${1:-./refs}"

for pdf_file in "$CATALOG"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file" .pdf)
        python ~/venv/adTranslate/reference.py "$pdf_file"
    fi
done