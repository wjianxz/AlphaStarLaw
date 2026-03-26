# AlphaStarLaw - 中文法律大模型

## 简介

🪐 AlphaStarLaw | 律启星辰

欢迎来到 AlphaStarLaw。在法律这个严谨的领域，我们相信 AI 不应只是“聊天工具”，而应成为“专业助手”。**以技术普惠法治，以范式定义可信**。

### 🛠 我们在做什么？

🔥 **Legal-Tuning-Cookbook**：系统梳理和探索法律大模型从**增量预训练（Continue Pre-training）-> 全量指令微调（Full SFT）-> RLHF**的技术方案。深度研究如何平衡通用能力与法律专业知识，避免模型在学习法律知识后出现“灾难性遗忘”、“指令退化”。

🔥 **可信训练范式（Credible-Tuning Paradigm）**：针对法律场景对训练范式进行原创性探索，训练范式具有可移植性。输出一套标准化的训练模组，让“可信 AI”在不同垂直领域生根发芽。针对法律领域的特性，我们开发了一套 **“证据增强+逻辑链约束+动态调优”** 的可信训练方式。通过在损失函数中引入法理逻辑一致性惩罚以及在训练过程中进行动态调优，并结合法律知识图谱（KG）实时校验，从根源上抑制了模型的“事实性幻觉”。

🔥 **多维度性能评估**：经过多维度多数据集评测，模型在中文法律基准测试（如 LawBench、LexEval、CAIL）中的表现远超同参数规模的基线模型（Baseline）。在多项核心任务上，该模型展现出了跨越量级的竞争力，其实际表现甚至超越了参数量远大于自身的通用大模型。在案情分析、罪名预测及法律文书生成等关键任务上，其准确率与逻辑严密性甚至优于部分闭源商业模型。

## 模型性能
所有的测试结果参考log日志。
### LawBench 数据集
1. 模型在Zero-Shot上的平均分数排序如下：

2. 展示我们经过调优模型和基线模型以及其他不同尺寸模型的性能。

3. 模型和基线模型的性能对比图如下：

### LexEval 数据集
1. 模型在Zero-Shot上的平均分数排序如下：

2. 展示我们经过调优模型和基线模型以及其他不同尺寸模型的性能。

3. 模型和基线模型的性能对比图如下：


### 如何评估模型的性能

#### 评估脚本

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


## 数据集
    数据集主要来自3方面：
        1. 网络开源数据集，例如：国家法律法规数据库、裁判文书网、法律问答网站、中文书籍等；
        2. 法律问答以及合成数据集，利用模型对query进行改写和扩充；
        3. 英文法律数据集。
        4. 内部标注数据集。
    
## 模型训练

## 微调
### CPT
```bash
训练源码待补充
```
### SFT
分别尝试了不同的基线模型：Qwen32B、Qwen8B；
```bash
训练源码待补充
```
### RLHF
分别尝试了不同的基线模型：Qwen32B、Qwen8B；
```bash
训练源码待补充
```
本项目遵循MIT许可证。