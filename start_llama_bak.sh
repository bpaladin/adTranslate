#!/bin/bash
# start_llama.sh — запуск llama.cpp сервера
#
# Использование:
#   ./start_llama.sh [МОДЕЛЬ] [ПОРТ]
#
# Аргументы:
#   МОДЕЛЬ   — имя модели (по умолчанию: gemma4)
#   ПОРТ     — порт сервера (по умолчанию: 8080)
#
# Примеры:
#   ./start_llama.sh              # gemma4 на порту 8080
#   ./start_llama.sh smollm 8081  # smollm на порту 8081

MODEL="${1:-gemma4}"
PORT="${2:-8080}"

LLAMA_SERVER="/opt/llama.cpp/llama-server"
MODELS_DIR="/home/ad/llm_models"
LOG_FILE="/tmp/llama-server.log"

# Проверяем existence бинарника
if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ОШИБКА: llama-server не найден: $LLAMA_SERVER"
    exit 1
fi

# Проверяем, не запущен ли уже сервер на этом порту
if lsof -i :"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Сервер уже запущен на порту $PORT"
    exit 0
fi

# Маппинг имен моделей на файлы
case "$MODEL" in
    translate|translategemma)
        GGUF="$MODELS_DIR/translategemma-4b-it-Q4_K_M_GGUF.gguf"
        ;;
    gemma4|gemma-4|gemma)
        GGUF="$MODELS_DIR/gemma-4-E4B-it-Q4_K_M.gguf"
        ;;
    smollm|smollm3)
        GGUF="$MODELS_DIR/SmolLM3-3B-Q4_K_M.gguf"
        ;;
    qwen-coder|qwen2.5-coder)
        GGUF="$MODELS_DIR/Qwen2.5-Coder-7B-Instruct-Q3_K_L.gguf"
        ;;
    qwen|qwen3|qwen3.5|gwen|gwen3|gwen3.5)
        GGUF="$MODELS_DIR/Qwen3.5-4B-Q4_K_S.gguf"
        ;;
    qwopus|qwopus3.5)
        GGUF="$MODELS_DIR/Qwopus3.5-4B-Coder-MTP-Q8_0.gguf"
        ;;
    *)
        # Если это путь к файлу
        if [ -f "$MODEL" ]; then
            GGUF="$MODEL"
        else
            GGUF="$MODELS_DIR/$MODEL.gguf"
        fi
        ;;
esac

if [ ! -f "$GGUF" ]; then
    echo "ОШИБКА: Модель не найдена: $GGUF"
    echo "Доступные модели в $MODELS_DIR:"
    ls "$MODELS_DIR"/*.gguf 2>/dev/null
    exit 1
fi

echo "Запуск llama.cpp сервера..."
echo "  Модель: $GGUF"
echo "  Порт: $PORT"
echo "  Лог: $LOG_FILE"

# Запускаем сервер в фоне
nohup "$LLAMA_SERVER" \
    --model "$GGUF" \
    --port "$PORT" \
    --ctx-size 8192 \
    --n-gpu-layers 99 \
    --parallel 1 \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Сервер запущен (PID: $SERVER_PID)"

# Ждем пока сервер поднимется (макс 60 сек)
echo -n "Ожидание готовности сервера"
for i in $(seq 1 60); do
    sleep 1
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo " ГОТОВО!"
        echo "Сервер доступен на http://localhost:$PORT"
        exit 0
    fi
    echo -n "."
done

echo " ТАЙМАУТ!"
echo "Сервер не запустился за 60 секунд. Проверьте лог: $LOG_FILE"
exit 1
