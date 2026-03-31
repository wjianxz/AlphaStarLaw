export PYTHONPATH="/mnt/public/wjianxz/LegalKit-main:$PYTHONPATH"
law-eval.shTIMESTAMP=$(date -d "+8 hours" +"%m%d_%H%M%S")
DATASET_NAME="LawBench"
DATASET_NAME="LexEval"
# DATASET_NAME="JECQA"
# DATASET_NAME="CAIL2025"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1

# /mnt/public/mdl/law_model/law_model_rl_8b_off_policy_v4
# law_model_rl_450_v3
# /mnt/public/mdl/Qwen/Qwen3-8B
python legalkit/main.py \
  --models /mnt/public/wjianxz/merge_model/law_model_rl/law_model_rl_8b_off_policy_v5 \
  --datasets ${DATASET_NAME} \
  --accelerator vllm \
  --max_tokens 4096 \
  --num_workers 1 \
  --tensor_parallel 2 \
  --batch_size 32 \
  --temperature 0.1 \
  --top_p 0.9 \
  2>&1 | tee ${DATASET_NAME}_${TIMESTAMP}.log
