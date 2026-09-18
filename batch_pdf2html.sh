#!/bin/bash
# batch_html.sh — пакетная конвертация всех PDF каталога в HTML (без перевода).
#
# Использование:
#   ./batch_html.sh [КАТАЛОГ] [--light] [--debug-text]
#
# Аргументы:
#   КАТАЛОГ       — каталог с PDF-файлами (по умолчанию: ./pdfs)
#   --light       — светлая тема HTML (по умолчанию тёмная)
#   --debug-text  — отладка извлечения текста (bbox/quality)
#
# Выход: рядом с каждым PDF создаётся файл *_raw.html.
# Алгоритм — модульный pipeline (main.py --raw-html), портированный из pdf2html.py.

set -euo pipefail

CATALOG="./pdfs"
LIGHT=""
DEBUG=""

for arg in "$@"; do
    case "$arg" in
        --light) LIGHT="--light-html" ;;
        --debug-text) DEBUG="--debug-text" ;;
        -h|--help)
            echo "Использование: $0 [КАТАЛОГ] [--light] [--debug-text]"
            exit 0 ;;
        *) CATALOG="$arg" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"

command -v python >/dev/null 2>&1 || { echo "Ошибка: python не найден" >&2; exit 1; }

if [[ ! -f "$MAIN_PY" ]]; then
    echo "Ошибка: $MAIN_PY не найден" >&2
    exit 1
fi

if [[ ! -d "$CATALOG" ]]; then
    echo "Ошибка: каталог '$CATALOG' не существует" >&2
    exit 1
fi

total=$(find "$CATALOG" -maxdepth 1 -name "*.pdf" | wc -l)
if [[ "$total" -eq 0 ]]; then
    echo "Ошибка: в каталоге '$CATALOG' нет PDF-файлов" >&2
    exit 1
fi
echo "Найдено $total PDF-файлов в '$CATALOG'."

count=0
find "$CATALOG" -maxdepth 1 -name "*.pdf" -print0 | while IFS= read -r -d '' pdf_file; do
    count=$((count + 1))
    filename=$(basename "$pdf_file")
    echo "[$count/$total] Конвертация $filename ..."
    # shellcheck disable=SC2086
    if ! python "$MAIN_PY" --raw-html $LIGHT $DEBUG "$pdf_file"; then
        echo "Ошибка при обработке $filename (код возврата $?)" >&2
    else
        echo "  -> готово"
    fi
done
echo "Все задачи выполнены."
