#!/bin/bash
# batch_summary.sh — пакетная генерация рефератов для всех PDF в каталоге
#
# Использование:
#   ./batch_summary.sh [КАТАЛОГ] [РЕЖИМ] [МОДЕЛЬ]
#
# Аргументы:
#   КАТАЛОГ   — каталог с PDF-файлами (по умолчанию: ./pdfs)
#   РЕЖИМ     — summary | summary_translate (по умолчанию: summary)
#   МОДЕЛЬ    — модель llama (по умолчанию: gemma4)
#
# Примеры:
#   ./batch_summary.sh                                    # summary, gemma4
#   ./batch_summary.sh ./pdfs summary_translate qwen3.5   # комбинированный режим
#   ./batch_summary.sh ./pdfs summary smollm              # стандартный режим

CATALOG="${1:-./pdfs}"
MODE="${2:-summary}"
MODEL="${3:-gemma4}"

# Запускаем llama.cpp если ещё не запущен
if ! curl -s http://localhost:8080/v1/models >/dev/null 2>&1; then
    echo "Llama.cpp сервер не запущен. Запуск..."
    ./start_llama.sh "$MODEL" 8080
fi

for pdf_file in "$CATALOG"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        filename=$(basename "$pdf_file" .pdf)
        python main.py -t llama -m "$MODE" --llama-model "$MODEL" "$pdf_file" "./outputs/${filename}.html"
    fi
done
