#!/usr/bin/env python3
"""
填空式标签任务评分脚本（例如争议焦点分类）

用法:
    python score_slection.py /path/to/predict.json

输出:
    上四级目录下的 {task_id}.json
"""

import sys
import json
import re
from pathlib import Path
from check_invalid import get_invalid_ids


def clean_prediction(text: str) -> str:
    """清洗模型输出，只保留最后一个 </think> 之后的内容。"""
    if text is None:
        return ""
    s = str(text)
    if "</think>" in s:
        s = s.split("</think>")[-1]
    else:
        lower = s.lower()
        idx = lower.rfind("think")
        if idx != -1:
            s = s[idx + len("think"):]
    s = s.strip()
    s = re.sub(r"(?is)<think>.*?</think>", "", s).strip()
    s = re.sub(r"^[^\w\u4e00-\u9fffA-Za-z0-9《]+", "", s).strip()
    return s


def _gold_terms(gold: str) -> list:
    """将 gold 按分隔符拆成词列表。"""
    parts = re.split(r'[/、，,；;。！？\s]+', gold)
    terms = []
    for p in parts:
        p = p.strip()
        m = re.match(r'^《(.+?)》$', p)
        if m:
            p = m.group(1)
        if p:
            terms.append(p)
    return terms


def extract_law_numbers(text: str) -> set:
    """从文本中提取所有法条编号（数字）。
    
    例如：
        "刑法第264、348条" -> {264, 348}
        "刑法第234条、第303条" -> {234, 303}
    """
    if not text:
        return set()
    # 提取所有数字
    numbers = re.findall(r'\d+', str(text))
    return set(int(n) for n in numbers if n)


def score_law_articles(cleaned: str, gold: str) -> tuple:
    """法条预测专用评分：用数字集合比较（忽略顺序）。
    
    返回: (prop_score, exact_score)
        - prop_score: 交集/gold集合大小
        - exact_score: 集合完全相等则1，否则0
    """
    pred_nums = extract_law_numbers(cleaned)
    gold_nums = extract_law_numbers(gold)
    
    if not gold_nums:
        return (0.0, 0.0)
    
    if not pred_nums:
        return (0.0, 0.0)
    
    # 交集
    intersection = pred_nums & gold_nums
    
    # 比例分 = 交集大小 / gold集合大小
    prop_score = len(intersection) / len(gold_nums)
    
    # 精确分 = 集合完全相等
    exact_score = 1.0 if pred_nums == gold_nums else 0.0
    
    return (prop_score, exact_score)


def _remove_punctuation(text: str) -> str:
    """去掉所有标点符号，只保留字母数字和中文。"""
    return re.sub(r'[^\w\u4e00-\u9fff]', '', text)

def score_gold(cleaned: str, gold: str) -> float:
    """按命中词数/总词数给分。
    
    对于 key:value 格式的 gold（如 犯罪嫌疑人:宋某某），
    如果整体匹配失败，则去掉标点后匹配（犯罪嫌疑人宋某某）。
    """
    if not cleaned or not gold:
        return 0.0
    cleaned_s = str(cleaned).replace("：", ":")
    cleaned_no_punct = _remove_punctuation(cleaned_s)
    gold_norm = str(gold).replace("：", ":")
    parts = _gold_terms(gold_norm)
    if not parts:
        return 0.0
    
    hit = 0
    for part in parts:
        if part in cleaned_s:
            hit += 1
        else:
            # 去掉标点后匹配
            part_no_punct = _remove_punctuation(part)
            if part_no_punct and part_no_punct in cleaned_no_punct:
                hit += 1
    
    return hit / len(parts)


def exact_score_gold(cleaned: str, gold: str) -> float:
    """严格 1/0 评分：所有词都命中才记 1。
    
    对于 key:value 格式的 gold，如果整体匹配失败，
    则去掉标点后匹配。
    """
    if not cleaned or not gold:
        return 0.0
    cleaned_s = str(cleaned).replace("：", ":")
    cleaned_no_punct = _remove_punctuation(cleaned_s)
    gold_norm = str(gold).replace("：", ":")
    parts = _gold_terms(gold_norm)
    if not parts:
        return 0.0
    
    hit = 0
    for part in parts:
        if part in cleaned_s:
            hit += 1
        else:
            # 去掉标点后匹配
            part_no_punct = _remove_punctuation(part)
            if part_no_punct and part_no_punct in cleaned_no_punct:
                hit += 1
    
    return 1.0 if hit == len(parts) else 0.0


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_slection.py <input_json_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 判断是否是法条预测任务：检查第一条记录的 prompt 是否以特定内容开头
    is_law_article_task = False
    if data:
        first_prompt = data[0].get("prompt", "")
        if first_prompt.startswith("根据下列事实和罪名给出涉及的刑法法条"):
            is_law_article_task = True
    
    task_id = input_path.parent.name

    # 获取无效ID
    invalid_ids, invalid_records = get_invalid_ids(data, "prediction")
    invalid_set = set(invalid_ids)

    total_score = 0.0
    exact_sum = 0.0
    scored_count = 0
    wrong_records = []  # 错误记录（有效但未完全正确）

    for rec in data:
        rec_id = rec.get("id")
        raw_pred = rec.get("prediction", "")
        gold = rec.get("gold", "")

        if rec_id in invalid_set:
            continue

        cleaned_pred = clean_prediction(raw_pred)
        
        # 3-1 任务使用法条数字集合比较
        if is_law_article_task:
            prop_score, exact = score_law_articles(cleaned_pred, gold)
        else:
            prop_score = score_gold(cleaned_pred, gold)
            exact = exact_score_gold(cleaned_pred, gold)
        
        total_score += prop_score
        exact_sum += exact
        scored_count += 1
        
        # 记录错误（未完全正确的回答）
        if exact < 1.0:
            wrong_records.append({
                "id": rec_id,
                "prediction": cleaned_pred,
                "gold": gold,
                "score": round(prop_score, 4),
            })

    total_count = len(data)
    invalid_count = len(invalid_ids)
    valid_count = scored_count
    correct_count = int(exact_sum)
    wrong_count = valid_count - correct_count
    invalid_ratio = (invalid_count / total_count * 100) if total_count > 0 else 0.0

    valid_accuracy = (total_score / valid_count * 100) if valid_count > 0 else 0.0
    total_accuracy = (total_score / total_count * 100) if total_count > 0 else 0.0
    final_rate_valid = (exact_sum / valid_count * 100) if valid_count > 0 else 0.0
    final_rate_total = (exact_sum / total_count * 100) if total_count > 0 else 0.0

    output = {
        "total_count": total_count,
        "invalid_count": invalid_count,
        "invalid_ratio": round(invalid_ratio, 2),
        "invalid_ids": invalid_ids,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "total_score": round(total_score, 4),
        "valid_accuracy": round(valid_accuracy, 2),
        "total_accuracy": round(total_accuracy, 2),
        "final_exact_count": correct_count,
        "final_rate_valid": round(final_rate_valid, 2),
        "final_rate_total": round(final_rate_total, 2),
        "wrong_details": wrong_records,
    }

    task_id = input_path.parent.name
    output_dir = input_path.parent.parent.parent.parent
    output_path = output_dir / f"{task_id}.json"
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # 保存无效记录
    if invalid_records:
        invalid_path = output_dir / f"{task_id}_invalid.jsonl"
        with invalid_path.open("w", encoding="utf-8") as f:
            for rec in invalid_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Total:      {total_count}")
    print(f"Invalid:    {invalid_count} ({invalid_ratio:.2f}%)")
    print(f"Valid:      {valid_count}")
    print(f"  Correct:  {correct_count}")
    print(f"  Wrong:    {wrong_count}")
    print(f"Exact(1/0): count={correct_count}  valid={final_rate_valid:.2f}%  total={final_rate_total:.2f}%")
    print(f"Weighted:   score_sum={total_score:.4f}  valid={valid_accuracy:.2f}%  total={total_accuracy:.2f}%")


if __name__ == "__main__":
    main()
