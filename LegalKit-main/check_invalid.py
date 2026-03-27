#!/usr/bin/env python3
"""
检测复读机/未完成回答的独立脚本。

用法:
    # 独立运行
    python check_invalid.py <input_json> [--field prediction] [--output invalid.jsonl]
    
    # 示例
    python check_invalid.py predict.json --field prediction --output invalid.jsonl
    python check_invalid.py data.json --field answer
    
    # 作为模块导入
    from check_invalid import get_invalid_ids, is_invalid

输出:
    - 打印统计信息
    - 生成 invalid.jsonl 文件（每行一条无效记录）
"""

import sys
import json
import re
import argparse
from pathlib import Path
from collections import Counter


def is_repetitive(text: str, threshold: int = 5) -> bool:
    """
    检测是否存在大量重复内容。
    
    优化特点：
    1. O(N) 复杂度，处理长文本极快
    2. 能检测无标点的连续复读（N-gram）
    3. 能检测仅在末尾出现的局部复读
    4. 对正常结尾（句号）的文本使用更严格阈值，减少误判
    """
    if not text or len(text) < 50:
        return False
    s = str(text).strip()
    total_len = len(s)
    
    # 检查是否以句号/问号/感叹号正常结尾 -> 使用更严格的阈值
    ends_normally = s[-1] in '。！？.!?)）」】'
    # 严格模式下，所有阈值提高
    if ends_normally:
        threshold = threshold + 3  # 5 -> 8
        ratio_threshold = 0.5      # 比例阈值从 0.3/0.4 提高到 0.5
    else:
        ratio_threshold = 0.3      # 非正常结尾使用原阈值
    
    # =========================================
    # 策略 1：按行检测（对多行重复有效）
    # =========================================
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    if len(lines) >= threshold:
        # 1.1 连续重复行
        repeat_count = 1
        for i in range(1, len(lines)):
            if lines[i] == lines[i-1]:
                repeat_count += 1
                if repeat_count >= threshold:
                    return True
            else:
                repeat_count = 1
        
        # 1.2 高频重复行
        counter = Counter(lines)
        most_common = counter.most_common(1)[0]
        if most_common[1] >= threshold and most_common[1] / len(lines) >= ratio_threshold:
            return True
        
        # 1.3 周期性重复行
        for period in range(1, min(6, len(lines) // threshold + 1)):
            pattern = lines[:period]
            match_count = 0
            for i in range(0, len(lines) - period + 1, period):
                if lines[i:i+period] == pattern:
                    match_count += 1
            if match_count >= threshold:
                return True
    
    # =========================================
    # 策略 2：按句子检测（粗粒度，需要标点）
    # =========================================
    sentences_coarse = re.split(r'[。！？；.!?;]', s)
    sentences_coarse = [sent.strip() for sent in sentences_coarse if sent.strip() and len(sent.strip()) > 10]
    
    if len(sentences_coarse) >= 3:
        counter = Counter(sentences_coarse)
        most_common = counter.most_common(1)[0]
        if most_common[1] >= 3 and most_common[1] / len(sentences_coarse) >= ratio_threshold + 0.2:
            return True
    
    # =========================================
    # 策略 3：按标点/符号分割，得到完整子句，排除纯数字和短填充词
    # =========================================
    # 用所有标点和符号分割
    segments = re.split(r'[。！？；，、.!?;,：:（）()【】\[\]《》<>""\'\'\"\'\n\r\t*#]+', s)
    # 过滤：排除空白、纯数字、短填充词
    # 短填充词：无、未提及、没有、不详、暂无 等
    SHORT_FILLERS = {'无', '未提及', '没有', '不详', '暂无', '未知', '不明', '空', '略', '否', '是'}
    def is_text_segment(seg):
        seg = seg.strip()
        if not seg:
            return False
        # 排除短填充词
        if seg in SHORT_FILLERS:
            return False
        # 排除纯数字或数字+单位
        if re.match(r'^[\d\s.]+[元万亿年月日号%个条款项]*$', seg):
            return False
        # 要求子句长度>=4，排除过短的碎片
        if len(seg) < 4:
            return False
        return True
    
    segments = [seg.strip() for seg in segments if is_text_segment(seg)]
    
    if len(segments) >= threshold:
        # 3.1 连续重复子句
        repeat_count = 1
        for i in range(1, len(segments)):
            if segments[i] == segments[i-1]:
                repeat_count += 1
                if repeat_count >= threshold:
                    return True
            else:
                repeat_count = 1
        
        # 3.2 高频重复子句（同一个完整子句出现多次）
        counter = Counter(segments)
        most_common = counter.most_common(1)[0]
        if most_common[1] >= 4 and most_common[1] / len(segments) >= ratio_threshold:
            return True
    
    # =========================================
    # 策略 4：N-gram 重复检测（仅用于无标点的纯文本流）
    # 只有当文本几乎没有标点时才启用，避免误判有标点的正常法律文书
    # =========================================
    # 统计标点数量
    punct_count = len(re.findall(r'[。！？；，、.!?;,\n]', s))
    punct_ratio = punct_count / total_len if total_len > 0 else 0
    
    # 只有标点比例很低（<1%）且文本较长时，才使用N-gram检测
    if punct_ratio < 0.01 and total_len > 100:
        ngram_len = 15  # 用更长的N-gram避免碎片匹配
        if total_len > ngram_len * 2:
            grams = [s[i:i+ngram_len] for i in range(total_len - ngram_len + 1)]
            counts = Counter(grams)
            
            if counts:
                most_common_gram, count = counts.most_common(1)[0]
                coverage = (count * ngram_len) / total_len
                
                # 无标点文本的重复检测：重复次数>=5 且 覆盖率>=25%
                if count >= 5 and coverage >= 0.25:
                    return True
                
                # 极端重复
                if count >= 15:
                    return True
    
    # =========================================
    # 策略 5：末尾局部死循环检测（解决 80%好 + 20%坏 的情况）
    # LLM 的复读 90% 发生在结尾
    # =========================================
    check_len = min(500, int(total_len * 0.3))
    if check_len > 50:
        tail = s[-check_len:]
        
        # 检查尾部是否有极高频的短语 (N=6)
        tail_ngram = 6
        tail_grams = [tail[i:i+tail_ngram] for i in range(len(tail) - tail_ngram + 1)]
        tail_counts = Counter(tail_grams)
        
        if tail_counts:
            top_gram, top_count = tail_counts.most_common(1)[0]
            
            # 排除纯下划线、纯空格、纯特殊字符的重复（法律文书签名格式）
            if re.match(r'^[_\-\s*#=\.\,]+$', top_gram):
                pass  # 跳过这种无意义的重复
            else:
                # 在尾部窗口内，如果某段内容占据 50% 以上，或者重复了 12 次以上
                # （提高阈值，避免法律文书中正常引用当事人名称被误判）
                tail_coverage = (top_count * tail_ngram) / len(tail)
                if top_count >= 12 or tail_coverage > 0.50:
                    return True
    
    # =========================================
    # 策略 6：连续标点/特殊字符检测（针对特定坏死模式）
    # =========================================
    # 纯文本中不应出现连续超过 10 个标点
    if re.search(r'[。！？!?.]{10,}', s):
        return True
    
    return False


def is_invalid(text: str) -> bool:
    """
    检测是否为无效回答：
    1. 只出现 <think> 而没有 </think>（未完成）
    2. 存在大量重复内容（复读机）
    """
    if text is None:
        return False
    s = str(text)
    
    # 检查1：<think> 没有闭合
    if "<think>" in s and "</think>" not in s:
        return True
    
    # 检查2：清洗后内容存在大量重复
    cleaned = s
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if is_repetitive(cleaned):
        return True
    
    return False


def get_invalid_ids(data: list, field: str = "prediction") -> tuple:
    """
    检测数据中的无效记录。
    
    Args:
        data: 记录列表
        field: 要检测的字段名
    
    Returns:
        (invalid_ids, invalid_records)
    """
    invalid_ids = []
    invalid_records = []
    
    for rec in data:
        rec_id = rec.get("id")
        text = rec.get(field, "")
        
        if is_invalid(text):
            invalid_ids.append(rec_id)
            invalid_records.append({
                "id": rec_id,
                field: text,
                "gold": rec.get("gold", rec.get("answer", "")),
            })
    
    return invalid_ids, invalid_records


def main():
    parser = argparse.ArgumentParser(description="检测复读机/未完成回答")
    parser.add_argument("input", help="输入JSON文件路径")
    parser.add_argument("--field", "-f", default="prediction", help="要检测的字段名 (默认: prediction)")
    parser.add_argument("--output", "-o", help="输出invalid文件路径 (默认: {input}_invalid.jsonl)")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: 文件不存在: {input_path}")
        sys.exit(1)
    
    # 默认输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(input_path.stem + "_invalid.jsonl")
    
    # 读取数据
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 检测无效记录
    invalid_ids, invalid_records = get_invalid_ids(data, args.field)
    
    total_count = len(data)
    invalid_count = len(invalid_ids)
    valid_count = total_count - invalid_count
    invalid_ratio = (invalid_count / total_count * 100) if total_count > 0 else 0.0
    
    # 保存无效记录
    if invalid_records:
        with output_path.open("w", encoding="utf-8") as f:
            for rec in invalid_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Invalid saved: {output_path}")
    
    # 打印统计
    print(f"Input:   {input_path}")
    print(f"Field:   {args.field}")
    print(f"Total:   {total_count}")
    print(f"Invalid: {invalid_count} ({invalid_ratio:.2f}%)")
    print(f"Valid:   {valid_count}")
    
    if invalid_ids and len(invalid_ids) <= 20:
        print(f"Invalid IDs: {invalid_ids}")
    elif invalid_ids:
        print(f"Invalid IDs: {invalid_ids[:10]}... (共{len(invalid_ids)}个)")


if __name__ == "__main__":
    main()
