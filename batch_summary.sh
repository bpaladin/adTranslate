#!/bin/bash
set -euo pipefail

# ----------------------------------------------------------------------
# Функция справки
# ----------------------------------------------------------------------
show_help() {
    cat <<EOF
Использование: $0 [ОПЦИИ] [РЕЖИМ] [МОДЕЛЬ]

Пакетная генерация рефератов для всех PDF в каталоге.

ОПЦИИ:
  -d, --directory КАТАЛОГ   каталог с PDF-файлами (по умолчанию: ./pdfs)
  -h, --help                показать эту справку и выйти

Позиционные аргументы (после всех опций):
  РЕЖИМ    summary | summary_translate (по умолчанию: summary)
  МОДЕЛЬ   модель llama (по умолчанию: translategemma)

Примеры:
  $0                                    # summary, translategemma, каталог ./pdfs
  $0 -d ./my_pdfs                       # каталог ./my_pdfs
  $0 summary_translate qwen3.5          # режим и модель, каталог ./pdfs
  $0 -d ./docs summary_translate        # каталог ./docs, режим summary_translate
EOF
}

# ----------------------------------------------------------------------
# Разбор аргументов
# ----------------------------------------------------------------------
if ! OPTIONS=$(getopt -o d:h --long directory:,help -n "$0" -- "$@"); then
    exit 1
fi
eval set -- "$OPTIONS"

CATALOG=""
MODE=""
MODEL=""

while true; do
    case "$1" in
        -d|--directory)
            CATALOG="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Внутренняя ошибка: неизвестная опция $1" >&2
            exit 1
            ;;
    esac
done

if [[ $# -gt 0 ]]; then
    MODE="$1"
    shift
else
    MODE="summary"
fi
if [[ $# -gt 0 ]]; then
    MODEL="$1"
    shift
else
    MODEL="translategemma"   # изменено на translategemma
fi

CATALOG="${CATALOG:-./pdfs}"

# ----------------------------------------------------------------------
# Проверка зависимостей и подготовка
# ----------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START_LLAMA=""
if [[ -x "$SCRIPT_DIR/start_llama.sh" ]]; then
    START_LLAMA="$SCRIPT_DIR/start_llama.sh"
elif command -v start_llama.sh >/dev/null 2>&1; then
    START_LLAMA="$(command -v start_llama.sh)"
else
    echo "Ошибка: не найден start_llama.sh ни в $SCRIPT_DIR, ни в PATH" >&2
    exit 1
fi

MAIN_PY="$SCRIPT_DIR/main.py"
if [[ ! -f "$MAIN_PY" ]]; then
    echo "Ошибка: $MAIN_PY не найден" >&2
    exit 1
fi

command -v curl >/dev/null 2>&1 || { echo "Ошибка: curl не установлен" >&2; exit 1; }
command -v python >/dev/null 2>&1 || { echo "Ошибка: python не найден" >&2; exit 1; }

if [[ ! -d "$CATALOG" ]]; then
    echo "Ошибка: каталог '$CATALOG' не существует" >&2
    exit 1
fi

# ----------------------------------------------------------------------
# Запуск сервера (с перезапуском, если порт занят, но сервер не отвечает)
# ----------------------------------------------------------------------
if ! curl -s http://localhost:8080/v1/models >/dev/null 2>&1; then
    echo "Сервер не отвечает. Пытаемся запустить..."

    # Проверяем, занят ли порт
    PORT_BUSY=false
    if command -v lsof >/dev/null 2>&1; then
        if lsof -i :8080 -sTCP:LISTEN >/dev/null 2>&1; then
            PORT_BUSY=true
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -lpn "sport = :8080" | grep -q LISTEN; then
            PORT_BUSY=true
        fi
    elif command -v netstat >/dev/null 2>&1; then
        if netstat -tulpn 2>/dev/null | grep -q ":8080 "; then
            PORT_BUSY=true
        fi
    fi

    if $PORT_BUSY; then
        echo "Порт 8080 занят, но сервер не отвечает. Перезапускаем..."
        # Убиваем процесс, занимающий порт
        if command -v lsof >/dev/null 2>&1; then
            kill -9 $(lsof -t -i :8080) 2>/dev/null || true
        else
            # Упрощённо: пытаемся найти PID через fuser
            fuser -k 8080/tcp 2>/dev/null || true
        fi
        sleep 2
    fi

    # Запускаем сервер
    "$START_LLAMA" "$MODEL" 8080 &
    TIMEOUT=30
    while ! curl -s http://localhost:8080/v1/models >/dev/null 2>&1; do
        sleep 1
        ((TIMEOUT--)) || { echo "Ошибка: сервер не запустился за 30 секунд" >&2; exit 1; }
    done
    echo "Сервер готов."
else
    # Сервер уже работает – ничего не выводим
    :
fi

# ----------------------------------------------------------------------
# Обработка PDF-файлов
# ----------------------------------------------------------------------
# Вместо объявления массива и проверки его размера используем find
total=$(find "$CATALOG" -maxdepth 1 -name "*.pdf" | wc -l)
if [[ $total -eq 0 ]]; then
    echo "Ошибка: в каталоге '$CATALOG' нет PDF-файлов" >&2
    exit 1
fi
echo "Найдено $total PDF-файлов."
count=0
find "$CATALOG" -maxdepth 1 -name "*.pdf" -print0 | while IFS= read -r -d '' pdf_file; do
    # ИСПРАВЛЕНИЕ: безопасное увеличение счетчика
    count=$((count + 1))

    filename=$(basename "$pdf_file" .pdf)

    echo "[$count/$total] Обработка $filename ..."
    if ! python "$MAIN_PY" -t llama --summary --summary-translator llama --summary-lang-translator llama "$pdf_file"; then
        echo "Ошибка при обработке $filename (код возврата $?)" >&2
    else
        echo "  -> готово"
    fi
done
echo "Все задачи выполнены. Обработано $total файлов."