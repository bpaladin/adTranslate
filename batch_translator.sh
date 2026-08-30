#!/bin/bash
# batch_translator.sh — пакетная генерация рефератов для всех PDF в каталоге
#
# Использование:
#   ./batch_translator.sh [КАТАЛОГ] [РЕЖИМ] [МОДЕЛЬ]
#
# Аргументы:
#   КАТАЛОГ   — каталог с PDF-файлами (по умолчанию: ./pdfs)

CATALOG="${1:-./pdfs}"

for pdf_file in "$CATALOG"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file" .pdf)
        python ~/venv/adTranslate/main.py -t llama "$pdf_file"
    fi
done
rm ./translation_cache.json