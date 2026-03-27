#!/usr/bin/env python3
"""
选择题评分脚本

用法:
    python score_multiple_choice.py /path/to/predict.json

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
    s = re.sub(r"^[^\w\u4e00-\u9fffA-Za-z]+", "", s).strip()
    return s


def extract_choices(text: str) -> str:
    """从清洗后的文本中提取选项标签字母（A-G）。"""
    if not text:
        return ""
    letters = re.findall(r"[A-Ga-g]", str(text))
    unique = sorted(set(c.upper() for c in letters))
    return "".join(unique)


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_multiple_choice.py <input_json_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取无效ID
    invalid_ids, invalid_records = get_invalid_ids(data, "prediction")
    invalid_set = set(invalid_ids)

    total_score = 0.0
    scored_count = 0
    wrong_details = []

    for rec in data:
        rec_id = rec.get("id")
        raw_pred = rec.get("prediction", "")
        gold = rec.get("gold", "")

        # 跳过无效回答
        if rec_id in invalid_set:
            continue

        cleaned = clean_prediction(raw_pred)
        pred_choices = extract_choices(cleaned)
        gold_choices = extract_choices(gold)
        
        score = 1.0 if pred_choices == gold_choices else 0.0
        total_score += score
        scored_count += 1

        if score == 0.0:
            wrong_details.append({
                "id": rec_id,
                "pred_choices": pred_choices,
                "gold_choices": gold_choices,
            })

    total_count = len(data)
    invalid_count = len(invalid_ids)
    valid_count = scored_count
    correct_count = int(total_score)

    valid_accuracy = (correct_count / valid_count * 100) if valid_count > 0 else 0.0
    total_accuracy = (correct_count / total_count * 100) if total_count > 0 else 0.0
    invalid_ratio = (invalid_count / total_count * 100) if total_count > 0 else 0.0

    output = {
        "total_count": total_count,
        "invalid_count": invalid_count,
        "invalid_ratio": round(invalid_ratio, 2),
        "invalid_ids": invalid_ids,
        "valid_count": valid_count,
        "correct_count": correct_count,
        "valid_accuracy": round(valid_accuracy, 2),
        "total_accuracy": round(total_accuracy, 2),
        "wrong_details": wrong_details,
    }

    # 输出路径
    task_id = input_path.parent.name
    output_dir = input_path.parent.parent.parent.parent
    output_path = output_dir / f"{task_id}.json"
    
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # 保存 invalid.jsonl
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
    print(f"Correct:    {correct_count}")
    print(f"Acc(valid): {valid_accuracy:.2f}%")
    print(f"Acc(total): {total_accuracy:.2f}%")


if __name__ == "__main__":
    main()
