#!/usr/bin/env python3
"""
计算题评分脚本：提取数字，按完全相等和接近程度两种方式评分。

用法:
    python score_calculation.py /path/to/predict.json

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
    s = re.sub(r"^[^\w\u4e00-\u9fffA-Za-z0-9.+-]+", "", s).strip()
    return s


def extract_gold_number(text: str):
    """从 gold 中提取数字。"""
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+\.?\d*", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def extract_pred_number(text: str):
    """从 prediction 中提取最后一个数字。"""
    if not text:
        return None
    matches = list(re.finditer(r"-?\d+\.?\d*", str(text)))
    if not matches:
        return None
    try:
        return float(matches[-1].group(0))
    except ValueError:
        return None


def extract_pred_number_months(text: str):
    """从 prediction 中提取最后一个"个月"前面的数字（用于刑期预测 3-4, 3-5）。"""
    if not text:
        return None
    s = str(text)
    # 查找所有 "数字+个月" 的模式
    matches = list(re.finditer(r'(\d+\.?\d*)\s*个月', s))
    if matches:
        # 返回最后一个匹配的数字
        try:
            return float(matches[-1].group(1))
        except ValueError:
            pass
    # 如果没找到"个月"，回退到提取最后一个数字
    return extract_pred_number(text)


def final_score(pred_num, gold_num) -> float:
    """完全相等得 1，否则 0。"""
    if pred_num is None or gold_num is None:
        return 0.0
    return 1.0 if pred_num == gold_num else 0.0


def weighted_score(pred_num, gold_num) -> float:
    """接近分。"""
    if pred_num is None or gold_num is None:
        return 0.0
    if gold_num == 0:
        return 1.0 if pred_num == 0 else 0.0
    ratio = 1.0 - abs(pred_num - gold_num) / abs(gold_num)
    return max(0.0, ratio)


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_calculation.py <input_json_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 判断任务类型
    task_id = input_path.parent.name
    # 刑期预测任务（3-4, 3-5, 3_4, 3_5）需要提取"个月"前的数字
    is_sentence_task = task_id in ("3-4", "3-5", "3_4", "3_5")

    # 获取无效ID
    invalid_ids, invalid_records = get_invalid_ids(data, "prediction")
    invalid_set = set(invalid_ids)

    final_sum = 0.0
    weighted_sum = 0.0
    scored_count = 0
    wrong_details = []

    for rec in data:
        rec_id = rec.get("id")
        raw_pred = rec.get("prediction", "")
        gold_raw = rec.get("gold", "")

        if rec_id in invalid_set:
            continue

        cleaned_pred = clean_prediction(raw_pred)
        # 刑期预测任务使用特殊的数字提取方式
        if is_sentence_task:
            pred_num = extract_pred_number_months(cleaned_pred)
        else:
            pred_num = extract_pred_number(cleaned_pred)
        gold_num = extract_gold_number(gold_raw)

        f_score = final_score(pred_num, gold_num)
        w_score = weighted_score(pred_num, gold_num)
        
        final_sum += f_score
        weighted_sum += w_score
        scored_count += 1
        
        # 记录错误（未完全正确的回答）
        if f_score < 1.0:
            wrong_details.append({
                "id": rec_id,
                "pred_num": pred_num,
                "gold_num": gold_num,
                "weighted_score": round(w_score, 4),
            })

    total_count = len(data)
    invalid_count = len(invalid_ids)
    valid_count = scored_count
    correct_count = int(final_sum)
    wrong_count = valid_count - correct_count
    invalid_ratio = (invalid_count / total_count * 100) if total_count > 0 else 0.0

    final_rate_total = (final_sum / total_count) if total_count > 0 else 0.0
    final_rate_valid = (final_sum / valid_count) if valid_count > 0 else 0.0
    weighted_rate_total = (weighted_sum / total_count) if total_count > 0 else 0.0
    weighted_rate_valid = (weighted_sum / valid_count) if valid_count > 0 else 0.0

    output = {
        "total_count": total_count,
        "invalid_count": invalid_count,
        "invalid_ratio": round(invalid_ratio, 2),
        "invalid_ids": invalid_ids,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "wrong_count": wrong_count,
        "final_exact_count": correct_count,
        "final_rate_total": round(final_rate_total, 4),
        "final_rate_valid": round(final_rate_valid, 4),
        "weighted_sum": round(weighted_sum, 4),
        "weighted_rate_total": round(weighted_rate_total, 4),
        "weighted_rate_valid": round(weighted_rate_valid, 4),
        "wrong_details": wrong_details,
    }

    task_id = input_path.parent.name
    output_dir = input_path.parent.parent.parent.parent
    output_path = output_dir / f"{task_id}.json"
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if invalid_records:
        invalid_path = output_dir / f"{task_id}_invalid.jsonl"
        with invalid_path.open("w", encoding="utf-8") as f:
            for rec in invalid_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Input:           {input_path}")
    print(f"Output:          {output_path}")
    print(f"Total:           {total_count}")
    print(f"Invalid:         {invalid_count} ({invalid_ratio:.2f}%)")
    print(f"Valid:           {valid_count}")
    print(f"  Correct:       {correct_count}")
    print(f"  Wrong:         {wrong_count}")
    print(f"Final(exact):    {correct_count}  rate_total={final_rate_total:.4f}  rate_valid={final_rate_valid:.4f}")
    print(f"Weighted:        sum={weighted_sum:.4f}  rate_total={weighted_rate_total:.4f}  rate_valid={weighted_rate_valid:.4f}")


if __name__ == "__main__":
    main()
