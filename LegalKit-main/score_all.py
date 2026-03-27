#!/usr/bin/env python3
"""
批量评分脚本：根据任务类型自动调用对应的评分脚本。

用法:
    python score_all.py /path/to/run_output/时间戳/模型目录

示例:
    python /mnt/public/haoduo/code/LegalKit-main/score_all.py
    
"""

import subprocess
import sys
import json
from pathlib import Path

import yaml


def _normalize_model_name(model_spec: str) -> str:
    """将 config.yaml 里的 models 字段转成用于文件名的短名字。
    例如: 'api:Qwen3.5-9B' -> 'Qwen3.5-9B_api'
    其他情况则取路径最后一段。
    """
    if isinstance(model_spec, str) and model_spec.startswith("api:"):
        name = model_spec.split("api:", 1)[1]
        return f"{name}"
    return Path(model_spec).name


def _prompt_model_name_override(model_names: list[str], *, on_conflict: bool = False) -> list[str]:
    """显示当前模型名，允许用户在终端中覆盖。on_conflict 为 True 时提示「或按 Enter 退出」且回车会退出。"""
    current_model_name = "+".join(model_names) if model_names else "unknown_models"
    print()
    print("当前模型名称: ")
    print(current_model_name)
    if on_conflict:
        user_input = input("重新输入被评测的模型名称；或按 Enter 退出：").strip()
        if not user_input:
            print("已退出。")
            sys.exit(0)
    else:
        user_input = input("输入被评测的模型名称；或按 Enter 使用当前名称：").strip()
        if not user_input:
            return model_names
    print(f"[OVERRIDE] 使用新的 model name: {user_input}")
    return [user_input]


def _prompt_input_dir(default_input_path: str | None = None) -> Path:
    """交互式获取结果目录地址。"""
    if default_input_path:
        prompt = (
            "请输入结果目录地址（run_output/时间戳/模型目录），"
            f"直接按 Enter 使用默认值 {default_input_path}："
        )
        input_path = input(prompt).strip() or default_input_path
    else:
        input_path = input("请输入结果目录地址（run_output/时间戳/模型目录）：").strip()

    if not input_path:
        print("未输入结果目录地址，退出。")
        sys.exit(1)
    return Path(input_path)

# 脚本所在目录
SCRIPT_DIR = Path(__file__).parent

# 任务类型映射（按数据集切换）
ALL_TASK_TYPES = {
    "LawBench": {
        "multiple_choice": ["1-2", "2-8", "3-6"],
        "selection": ["2-1", "2-2", "2-3", "2-4", "2-5", "2-6", "2-9", "2-10", "3-1", "3-3"],
        "calculation": ["3-4", "3-5", "3-7"],
        "free_answer": ["1-1", "2-7", "3-2", "3-8"],
    },
    "LexEval": {
        "multiple_choice": [
            "1_1", "1_2", "1_3",
            "2_1", "2_2", "2_3", "2_4", "2_5",
            "3_1", "3_2", "3_3", "3_4", "3_5", "3_6",
            "4_1", "4_2",
            "6_1", "6_2", "6_3",
        ],
        "free_answer": ["5_1", "5_2", "5_3", "5_4"],
    },
    "JECQA": {
        "multiple_choice": ["0_test", "1_test"],
    },
    "CaseGen": {
        "free_answer": ["defense", "fact", "judgement", "reasoning"],
    },
}

# 默认数据集（兼容旧用法：直接传参时不交互）
TASK_TYPES = ALL_TASK_TYPES["LawBench"]

LAW_BENCH_TASK_NAME_MAP = {
    "1-1": "法条背诵",
    "1-2": "知识问答",
    "2-1": "文件校对",
    "2-2": "纠纷焦点识别",
    "2-3": "婚姻纠纷鉴定",
    "2-4": "问题主题识别",
    "2-5": "阅读理解",
    "2-6": "命名实体识别",
    "2-7": "舆情摘要",
    "2-8": "论点挖掘",
    "2-9": "事件检测",
    "2-10": "触发词提取",
    "3-1": "法条预测",
    "3-2": "法律预测",
    "3-3": "罪名预测",
    "3-4": "刑期预测（无法条内容）",
    "3-5": "刑期预测（给定法条内容）",
    "3-6": "案例分析",
    "3-7": "犯罪金额计算",
    "3-8": "法律咨询",
}

LEX_EVAL_TASK_NAME_MAP = {
    "1_1": "法律概念",
    "1_2": "法规查询",
    "1_3": "法律演进",
    "2_1": "要素识别",
    "2_2": "事实验证",
    "2_3": "阅读理解",
    "2_4": "关系提取",
    "2_5": "实体识别",
    "3_1": "案由预测",
    "3_2": "法条预测",
    "3_3": "刑期预测",
    "3_4": "多层推理",
    "3_5": "法律计算",
    "3_6": "论辩挖掘",
    "4_1": "类案辨别",
    "4_2": "文书质量",
    "5_1": "摘要生成",
    "5_2": "裁判分析",
    "5_3": "法律翻译",
    "5_4": "开放问题",
    "6_1": "偏见歧视",
    "6_2": "道德",
    "6_3": "隐私",
}

JECQA_TASK_NAME_MAP = {
    "0_test": "法律概念",
    "1_test": "案例分析",
}


def _format_task_id_for_output(dataset_key: str, task_id: str) -> str:
    if (dataset_key or "").lower() == "lawbench":
        name = LAW_BENCH_TASK_NAME_MAP.get(task_id)
        if name:
            return f"{task_id} {name}"
    if (dataset_key or "").lower() == "lexeval":
        name = LEX_EVAL_TASK_NAME_MAP.get(task_id)
        if name:
            return f"{task_id} {name}"
    if (dataset_key or "").lower() == "jecqa":
        name = JECQA_TASK_NAME_MAP.get(task_id)
        if name:
            if task_id == "0_test":
                return f"test0 {name}"
            if task_id == "1_test":
                return f"test1 {name}"
            return f"{task_id} {name}"
    return task_id


def _choose_dataset_interactively() -> str:
    options = [
        ("1", "LawBench"),
        ("2", "LexEval"),
        ("3", "JECQA"),
        ("4", "CaseGen"),
    ]
    print("请选择数据集：")
    for num, name in options:
        print(f"  [{num}] {name}")
    while True:
        choice = input("请输入编号或名称：").strip().lower()
        if not choice:
            continue
        for num, name in options:
            if choice == num or choice == name.lower():
                return name
        print("输入无效，请重新输入。")

# 评分脚本映射
SCORE_SCRIPTS = {
    "multiple_choice": SCRIPT_DIR / "score_multiple_choice.py",
    "selection": SCRIPT_DIR / "score_slection.py",
    "calculation": SCRIPT_DIR / "score_calculation.py",
    "free_answer": SCRIPT_DIR / "score_free_answer.py",
}


def get_task_type(task_id: str) -> str:
    """根据任务ID获取任务类型"""
    for task_type, task_ids in TASK_TYPES.items():
        if task_id in task_ids:
            return task_type
    return None


def _infer_dataset_from_config(config_file: Path) -> str | None:
    """从 run_output/时间戳/config.yaml 推断数据集键（ALL_TASK_TYPES 的 key）。

    规则：
    - 读取 args.datasets（可能是 list 或 str）
    - 找到第一个能匹配到 ALL_TASK_TYPES 的名称（忽略大小写）
    - 若不存在或无法匹配，返回 None
    """
    try:
        if not config_file.exists():
            return None
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        args_cfg = cfg.get("args", {}) if isinstance(cfg, dict) else {}
        ds = args_cfg.get("datasets")
        if ds is None:
            return None
        if isinstance(ds, str):
            ds_list = [ds]
        elif isinstance(ds, (list, tuple)):
            ds_list = list(ds)
        else:
            return None

        # build case-insensitive lookup
        key_by_lower = {k.lower(): k for k in ALL_TASK_TYPES.keys()}
        for item in ds_list:
            if item is None:
                continue
            k = str(item).strip().lower()
            if k in key_by_lower:
                return key_by_lower[k]
        return None
    except Exception:
        return None


def _rename_timestamp_dir(timestamp_dir: Path, datasets_part: str, models_part: str) -> None:
    """将 run_output 下的时间戳目录重命名为 datasets__models。"""
    target_name = f"{datasets_part}__{models_part}"
    if timestamp_dir.name == target_name:
        print(f"run_output 目录已是目标名称，跳过重命名：{timestamp_dir}")
        return

    target_dir = timestamp_dir.parent / target_name
    if target_dir.exists():
        print(f"[WARN] 目标目录已存在，跳过重命名：{target_dir}")
        return

    timestamp_dir.rename(target_dir)
    print(f"Renamed run_output directory -> {target_dir}")


def main():
    global TASK_TYPES
    # 先交互式获取输入路径；若命令行传入则作为默认值
    default_input_path = sys.argv[1] if len(sys.argv) >= 2 else None
    input_dir = _prompt_input_dir(default_input_path)

    # 支持三种输入：
    # 1) .../predict
    # 2) .../run_output/时间戳/模型目录
    # 3) .../run_output/时间戳  （自动在下一级寻找包含 predict 的模型目录，并读取同级 config.yaml）
    timestamp_dir: Path
    model_dir: Path
    predict_dir: Path

    if input_dir.name == "predict":
        model_dir = input_dir.parent
        predict_dir = input_dir
        timestamp_dir = model_dir.parent
    elif (input_dir / "config.yaml").exists() and not (input_dir / "predict").exists():
        # 看起来是 run_output/时间戳/
        timestamp_dir = input_dir
        candidates = sorted([d for d in timestamp_dir.iterdir() if d.is_dir() and (d / "predict").exists()])
        if not candidates:
            print(f"Error: no model subdir with predict/ found under: {timestamp_dir}")
            sys.exit(1)
        if len(candidates) > 1:
            print(f"[WARN] 在 {timestamp_dir} 下发现多个模型目录，默认使用第一个：{candidates[0].name}")
        model_dir = candidates[0]
        predict_dir = model_dir / "predict"
    else:
        # 看起来是模型目录
        model_dir = input_dir
        predict_dir = model_dir / "predict"
        timestamp_dir = model_dir.parent

    if not predict_dir.exists():
        print(f"Error: predict directory not found: {predict_dir}")
        sys.exit(1)

    # 推断 run_output/时间戳 目录并从 config.yaml 自动选择数据集
    output_dir = timestamp_dir  # run_output/时间戳/
    inferred_dataset = _infer_dataset_from_config(output_dir / "config.yaml")
    if inferred_dataset and inferred_dataset in ALL_TASK_TYPES:
        dataset_key = inferred_dataset
        print(f"[AUTO] 从 config.yaml 识别到数据集：{dataset_key}")
    else:
        dataset_key = _choose_dataset_interactively()
    TASK_TYPES = ALL_TASK_TYPES[dataset_key]

    # 收集所有任务目录
    task_dirs = sorted([d for d in predict_dir.iterdir() if d.is_dir()])
    
    print(f"Model dir: {model_dir}")
    print(f"Found {len(task_dirs)} task directories")
    print("=" * 60)

    success_count = 0
    skip_count = 0
    error_count = 0

    for task_dir in task_dirs:
        task_id = task_dir.name
        task_type = get_task_type(task_id)

        # 查找预测文件 (格式: {task_id}_0.json)
        pred_file = task_dir / f"{task_id}_0.json"
        if not pred_file.exists():
            print(f"[SKIP] {task_id}: prediction file not found")
            skip_count += 1
            continue

        if task_type is None:
            print(f"[SKIP] {task_id}: unknown task type")
            skip_count += 1
            continue

        script = SCORE_SCRIPTS[task_type]
        if not script.exists():
            print(f"[ERROR] {task_id}: score script not found: {script}")
            error_count += 1
            continue

        print(f"[{task_type.upper()}] {task_id} -> {script.name}")
        
        try:
            result = subprocess.run(
                ["python", str(script), str(pred_file)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                # 只打印关键输出行
                for line in result.stdout.strip().split("\n"):
                    if line.startswith("Output:") or "Acc" in line or "rate" in line or "Avg" in line:
                        print(f"    {line}")
                success_count += 1
            else:
                print(f"    [ERROR] {result.stderr.strip()}")
                error_count += 1
        except Exception as e:
            print(f"    [ERROR] {e}")
            error_count += 1

    print("=" * 60)
    print(f"Summary: {success_count} success, {skip_count} skipped, {error_count} errors")

    # 合并所有 invalid 记录到一个文件
    invalid_files = sorted(output_dir.glob("*_invalid.jsonl"))
    
    if invalid_files:
        merged_invalid = []
        for inv_file in invalid_files:
            # 从文件名提取 task_id (如 1-1_invalid.jsonl -> 1-1)
            task_id = inv_file.stem.replace("_invalid", "")
            with open(inv_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            # 添加 task_id 字段，放在最前面
                            new_record = {"task_id": task_id}
                            new_record.update(record)
                            merged_invalid.append(new_record)
                        except json.JSONDecodeError:
                            pass
            # 删除原来的单独文件
            inv_file.unlink()
        
        # 保存合并后的文件
        merged_file = output_dir / "invalid.jsonl"
        with open(merged_file, "w", encoding="utf-8") as f:
            for record in merged_invalid:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"\nMerged {len(merged_invalid)} invalid records -> {merged_file}")

    # 生成汇总 score.json
    score_summary = {}
    task_json_files = sorted(output_dir.glob("*.json"))
    
    for task_file in task_json_files:
        if task_file.name in ("score.json", "invalid.json"):
            continue
        task_id = task_file.stem  # 如 "1-1", "2-2"
        task_type = get_task_type(task_id)
        
        try:
            with open(task_file, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            
            # 提取关键指标
            task_score = {}
            
            # invalid_ratio (已是百分数) - 所有任务都有
            if "invalid_ratio" in task_data:
                task_score["invalid_ratio"] = round(task_data["invalid_ratio"], 2)
            
            # 根据任务类型使用不同字段名
            if task_type == "multiple_choice":
                # 选择题: valid_accuracy -> score
                if "valid_accuracy" in task_data:
                    task_score["score"] = round(task_data["valid_accuracy"], 2)
            
            elif task_type == "selection":
                # 填空题: valid_accuracy -> accuracy, final_rate_valid -> score
                if "valid_accuracy" in task_data:
                    task_score["accuracy"] = round(task_data["valid_accuracy"], 2)
                if "final_rate_valid" in task_data:
                    val = task_data["final_rate_valid"]
                    if val < 1:
                        val = val * 100
                    task_score["score"] = round(val, 2)
            
            elif task_type == "calculation":
                # 计算题: weighted_rate_valid -> accuracy, final_rate_valid -> score
                if "weighted_rate_valid" in task_data:
                    val = task_data["weighted_rate_valid"]
                    if val < 1:
                        val = val * 100
                    task_score["accuracy"] = round(val, 2)
                if "final_rate_valid" in task_data:
                    val = task_data["final_rate_valid"]
                    if val < 1:
                        val = val * 100
                    task_score["score"] = round(val, 2)
            
            # free_answer 只有 invalid_ratio，不需要额外处理
            
            if task_score:
                score_summary[task_id] = task_score
                
        except (json.JSONDecodeError, KeyError):
            pass
    
    # 按任务ID排序（支持 "1-2" 和 "0_test" 等多种格式）
    def sort_key(item):
        task_id = item[0]
        # 尝试按 "数字-数字" 格式排序
        if "-" in task_id:
            parts = task_id.split("-")
            try:
                return (0, int(parts[0]), int(parts[1]))
            except ValueError:
                pass
        # 尝试按 "数字_xxx" 格式排序
        if "_" in task_id:
            parts = task_id.split("_")
            try:
                return (1, int(parts[0]), parts[1])
            except ValueError:
                pass
        # 默认按字符串排序
        return (2, task_id, "")
    
    score_summary = dict(sorted(score_summary.items(), key=sort_key))
    
    # 保存 score.json
    score_file = output_dir / "score.json"
    with open(score_file, "w", encoding="utf-8") as f:
        json.dump(score_summary, f, ensure_ascii=False, indent=2)
        f.write("\n")
    
    print(f"Generated score summary -> {score_file}")

    # 读取 config.yaml 中的 datasets 和 models，生成带头信息的 jsonl 文件
    try:
        config_file = output_dir / "config.yaml"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            args_cfg = config.get("args", {})
            datasets = args_cfg.get("datasets") or []
            models = args_cfg.get("models") or []

            # 提取模型名，特别处理 api: 前缀
            model_names = [_normalize_model_name(m) for m in models]
            model_names = _prompt_model_name_override(model_names)

            # 目标目录（提前用于冲突检测）
            model_scores_dir = Path("/mnt/public/haoduo/code/LegalKit-main/model_scores")
            model_scores_dir.mkdir(parents=True, exist_ok=True)

            # 检测 model_scores 下是否已有相同 datasets + model name 的文件，有则提示更换名称并重新输入
            while True:
                datasets_part = "+".join(str(d) for d in datasets) if datasets else "unknown_datasets"
                models_part = "+".join(model_names) if model_names else "unknown_models"
                jsonl_name = f"{datasets_part}__{models_part}.jsonl"
                jsonl_path = model_scores_dir / jsonl_name
                if not jsonl_path.exists():
                    print("名称检测通过")
                    break
                print("已存在相同 datasets 与 model name 的评分文件，请更换名称。")
                model_names = _prompt_model_name_override(model_names, on_conflict=True)

            # 读取刚刚生成的 score.json
            with open(score_file, "r", encoding="utf-8") as f:
                score_data = json.load(f)

            # datasets_part, models_part, jsonl_path 已在上面冲突检测循环中确定

            # 计算总平均 score 和总平均 invalid_ratio
            scores = [v["score"] for v in score_data.values() if "score" in v and v["score"] is not None]
            invalid_ratios = [v["invalid_ratio"] for v in score_data.values() if "invalid_ratio" in v and v["invalid_ratio"] is not None]
            avg_score = round(sum(scores) / len(scores), 2) if scores else None
            avg_invalid_ratio = round(sum(invalid_ratios) / len(invalid_ratios), 2) if invalid_ratios else None

            # 写 jsonl：第一行是 meta（含 avg_score、invalid_ratio），其余每行一个任务（不含 invalid_ratio）
            with open(jsonl_path, "w", encoding="utf-8") as f:
                meta = {
                    "datasets": datasets,
                    "models": model_names,
                    "avg_score": avg_score,
                    "invalid_ratio": avg_invalid_ratio,
                }
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")

                # score_data 是 {task_id: 指标字典}，每条任务行不写 invalid_ratio
                for task_id, task_score in score_data.items():
                    record = {"task_id": _format_task_id_for_output(dataset_key, task_id)}
                    for k, v in task_score.items():
                        if k != "invalid_ratio":
                            record[k] = v
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"Generated score jsonl with datasets/models -> {jsonl_path}")
            _rename_timestamp_dir(timestamp_dir, datasets_part, models_part)
        else:
            print(f"config.yaml not found in {output_dir}, skip generating score jsonl.")
    except Exception as e:
        print(f"Failed to generate score jsonl: {e}")


if __name__ == "__main__":
    main()
