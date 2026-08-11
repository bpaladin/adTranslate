#!/bin/bash
for pdf_file in ./pdfs/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file" .pdf)
        python main.py "$pdf_file" "./pdfs/${filename}.html"
    fi
done
