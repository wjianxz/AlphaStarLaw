#!/usr/bin/env python3
"""
自由回答统计脚本：统计 token 数和平均 token 数。

用法:
    python score_free_answer.py /path/to/predict.json

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


def count_tokens(text: str) -> int:
    """简单统计字符数。"""
    if not text:
        return 0
    return len(text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python score_free_answer.py <input_json_path>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 获取无效ID
    invalid_ids, invalid_records = get_invalid_ids(data, "prediction")
    invalid_set = set(invalid_ids)

    valid_token_list = []

    for rec in data:
        rec_id = rec.get("id")
        raw_pred = rec.get("prediction", "")

        if rec_id in invalid_set:
            continue
        
        cleaned_pred = clean_prediction(raw_pred)
        tokens = count_tokens(cleaned_pred)
        valid_token_list.append(tokens)

    total_count = len(data)
    invalid_count = len(invalid_ids)
    valid_count = len(valid_token_list)
    invalid_ratio = (invalid_count / total_count * 100) if total_count > 0 else 0.0
    valid_ratio = (valid_count / total_count * 100) if total_count > 0 else 0.0

    max_valid = max(valid_token_list) if valid_token_list else 0
    min_valid = min(valid_token_list) if valid_token_list else 0
    avg_valid = (sum(valid_token_list) / valid_count) if valid_count > 0 else 0.0

    output = {
        "total_count": total_count,
        "invalid_count": invalid_count,
        "invalid_ratio": round(invalid_ratio, 2),
        "valid_count": valid_count,
        "valid_ratio": round(valid_ratio, 2),
        "invalid_ids": invalid_ids,
        "max_valid": max_valid,
        "min_valid": min_valid,
        "avg_valid": round(avg_valid, 2),
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

    print(f"Input:      {input_path}")
    print(f"Output:     {output_path}")
    print(f"Total:      {total_count}")
    print(f"Invalid:    {invalid_count} ({invalid_ratio:.2f}%)")
    print(f"Valid:      {valid_count}")
    print(f"Tokens:     max={max_valid}  min={min_valid}  avg={avg_valid:.2f}")


if __name__ == "__main__":
    main()
