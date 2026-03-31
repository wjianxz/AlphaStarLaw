#!/usr/bin/env bash
set -euo pipefail

export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
TIMESTAMP=$(date -d "+8 hours" +"%m%d_%H%M%S")
DATASET_NAME="${DATASET_NAME:-JECQA}"

# API 配置：
# - 默认连接本机 8000（OpenAI 兼容服务）
# - 不填 API_MODEL 时会自动从 /v1/models 探测第一个模型
API_URL="${API_URL:-http://127.0.0.1:8000}"
API_KEY="${API_KEY:-}"
API_MODEL="${API_MODEL:-}"

TASK="${TASK:-all}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
BATCH_SIZE="${BATCH_SIZE:-32}" 
NUM_WORKERS="${NUM_WORKERS:-1}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
# thinking 开关：
# - 0/false: 关闭 thinking（默认）
# - 1/true : 开启 thinking
ENABLE_THINKING="${ENABLE_THINKING:-0}"

CMD=(
  python legalkit/main.py
  --model_mode api8000
  --datasets "${DATASET_NAME}"
  --task "${TASK}"
  --api_url "${API_URL}"
  --max_tokens "${MAX_TOKENS}"
  --num_workers "${NUM_WORKERS}"
  --batch_size "${BATCH_SIZE}"
  --temperature "${TEMPERATURE}"
  --top_p "${TOP_P}"
)

if [[ -n "${API_MODEL}" ]]; then
  CMD+=(--api_model "${API_MODEL}")
fi

if [[ -n "${API_KEY}" ]]; then
  CMD+=(--api_key "${API_KEY}")
fi

case "${ENABLE_THINKING,,}" in
  1|true|yes|on)
    CMD+=(--enable_thinking)
    THINKING_STATUS="on"
    ;;
  *)
    CMD+=(--no_thinking)
    THINKING_STATUS="off"
    ;;
esac

echo "[INFO] dataset=${DATASET_NAME}, task=${TASK}"
echo "[INFO] api_url=${API_URL}, api_model=${API_MODEL:-<auto-detect>}"
echo "[INFO] thinking=${THINKING_STATUS}"
"${CMD[@]}" 2>&1 | tee "${DATASET_NAME}_${TIMESTAMP}.log"

