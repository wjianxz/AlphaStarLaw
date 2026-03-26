# AlphaStarLaw - 中文法律大模型

## 简介

欢迎来到AlphaStarLaw项目！本项目旨在构建一个强大的中文法律大模型，以提供高质量的法律咨询和解答。我们的目标是实现以下功能：

🔥 **Legal-Tuning-Cookbook**：系统梳理和探索法律大模型从**增量预训练（Continue Pre-training）-> 全量指令微调（Full SFT）-> RLHF**的技术方案。深度研究如何平衡通用能力与法律专业知识，避免模型在学习法律知识后出现“灾难性遗忘”、“指令退化”。

🔥 **可信训练范式（Credible-Tuning Paradigm）**：针对法律场景对训练范式进行原创性探索，训练范式具有可移植性(迁移到其他领域)。针对法律领域的特性，我们开发了一套 **“证据增强+逻辑链约束+动态调优”** 的可信训练方式。通过在损失函数中引入法理逻辑一致性惩罚以及在训练过程中进行动态调优，并结合法律知识图谱（KG）实时校验，从根源上抑制了模型的“事实性幻觉”。

🔥 **性能超越基线**：经过多维度多数据集评测，模型在中文法律基准测试（如 LawBench、LexEval、CAIL）中的表现远超同参数规模的基线模型（Baseline）。在多项核心任务上，该模型展现出了跨越量级的竞争力，其实际表现甚至超越了参数量远大于自身的通用大模型。在案情分析、罪名预测及法律文书生成等关键任务上，其准确率与逻辑严密性甚至优于部分闭源商业模型。

## 数据集
    数据集主要来自3方面：
        1. 网络开源数据集，例如：国家法律法规数据库、裁判文书网、法律问答网站、中文书籍等；
        2. 法律问答以及合成数据集，利用模型对query进行改写和扩充；
        3. 英文法律数据集。
        4. 内部标注数据集。
    
## 模型训练

## 微调
### CPT
采用ms-swift进行训练
```bash
python finetune.py
```
### SFT
分别尝试了不同的基线模型：Qwen32B、Qwen8B；
### RLHF
#### on-policy
基线模型：Qwen8B；
#### off-policy
分别尝试了不同的基线模型：Qwen32B、Qwen8B；
## 模型性能

### 如何评估模型的性能
### 评估数据集

```bash

law-eval.shTIMESTAMP=$(date -d "+8 hours" +"%m%d_%H%M%S")
DATASET_NAME="LawBench"
# DATASET_NAME="LexEval"
MODEL_NAME="Qwen8B-off-policy-fintuned"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_DEBUG=INFO

export CUDA_VISIBLE_DEVICES=0,1

# 
python legalkit/main.py \
  --models  ${MODEL_NAME} \
  --datasets ${DATASET_NAME} \
  --accelerator vllm \
  --max_tokens 4096 \
  --num_workers 1 \
  --tensor_parallel 2 \
  --batch_size 32 \
  --temperature 0.1 \
  --top_p 0.9 \
  2>&1 | tee ${DATASET_NAME}_${TIMESTAMP}.log

```
本项目遵循MIT许可证。