#!/usr/bin/env bash
set -euo pipefail

# 批量配置：仅需要填 dataset 和 model
PAIRS=(
  "LexEval|/mnt/public/mdl/Qwen/Qwen3-Next-80B-A3B-Instruct"
  "JECQA|/mnt/public/mdl/Qwen/Qwen3-14B"
  "JECQA|/mnt/public/mdl/Qwen/Qwen3-Next-80B-A3B-Instruct"
)

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3

for pair in "${PAIRS[@]}"; do
  DATASET_NAME="${pair%%|*}"
  MODEL_PATH="${pair#*|}"
  TIMESTAMP=$(date -d "+8 hours" +"%m%d_%H%M%S")

  echo "=== Running dataset=${DATASET_NAME}, model=${MODEL_PATH} ==="
  python legalkit/main.py \
    --models "${MODEL_PATH}" \
    --datasets "${DATASET_NAME}" \
    --accelerator vllm \
    --max_tokens 4096 \
    --num_workers 1 \
    --tensor_parallel 4 \
    --batch_size 64 \
    --temperature 0.1 \
    --top_p 0.9 \
    2>&1 | tee "${DATASET_NAME}_${TIMESTAMP}.log"
done
