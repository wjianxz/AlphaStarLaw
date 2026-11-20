import argparse
import json
import os
import re
import tempfile
from pathlib import Path


# 支持前缀匹配（忽略大小写），同时匹配文件名后缀与 JSONL 内的 model_name
LAW_MODEL_FILE_SUFFIXES = {
    "alpha_star_law_32B",
    "alpha_star_law_8B",
}

CATEGORY_COLORS = {
    "法律模型": "#7c3aed",
    "通用模型": "#2a9d8f",
}

# 同数据集出现同名模型时的处理策略：
# - "highest_score": 去重，仅保留 avg_score 最高的记录（同分时保留最新修改文件），并打印冲突文件名
# - "latest_mtime":  去重，仅保留文件修改时间最新的记录，并打印冲突文件名
# - "keep_all_with_filename": 保留全部同名记录，并在模型名后附加文件名以区分
DUPLICATE_MODEL_STRATEGY = "highest_score"

OUTPUT_FILE = "dashboard.html"
HIDDEN_TASK_IDS_BY_DATASET = {
    "LawBench": {"2-7 舆情摘要", "3-2 法律预测", "3-8 法律咨询"},
    "LexEval": set(),
}


def format_number(value):
    if value is None:
        return None
    return round(float(value), 2)


def parse_task_line(raw_line):
    try:
        return json.loads(raw_line)
    except json.JSONDecodeError:
        task_match = re.search(r'"task_id"\s*:\s*"([^"]+)"', raw_line)
        if not task_match:
            return None

        score_match = re.search(r'"score"\s*:\s*(-?\d+(?:\.\d+)?)', raw_line)
        accuracy_match = re.search(r'"accuracy"\s*:\s*(-?\d+(?:\.\d+)?)', raw_line)
        repaired = {"task_id": task_match.group(1)}
        if score_match:
            repaired["score"] = float(score_match.group(1))
        if accuracy_match:
            repaired["accuracy"] = float(accuracy_match.group(1))
        return repaired


def get_model_category(filepath: Path, model_name: str = "") -> str:
    """
    前缀匹配（忽略大小写），同时兼容：
    1. alpha_star_law_8B_v2 等带版本号后缀的模型名/文件名；
    2. 多段 '__' 分隔的文件名（如 LexEval__20261119__alpha_star_law_8B_v2.jsonl）；
    3. 文件名未包含模型前缀但 JSONL 内 model_name 包含前缀的场景。
    """
    stem_parts = filepath.stem.split("__")
    candidates = [stem_parts[-1], filepath.stem]
    if len(stem_parts) > 1:
        candidates.append(filepath.stem.split("__", 1)[-1])
    if model_name:
        candidates.append(model_name)

    normalized_prefixes = tuple(
        prefix.strip().lower() for prefix in LAW_MODEL_FILE_SUFFIXES if prefix.strip()
    )
    for candidate in candidates:
        c_lower = candidate.strip().lower()
        if any(c_lower.startswith(prefix) for prefix in normalized_prefixes):
            return "法律模型"
    return "通用模型"


def _choose_preferred_record(existing: dict, candidate: dict, strategy: str) -> dict:
    existing_score = float("-inf") if existing["avg_score"] is None else existing["avg_score"]
    cand_score = float("-inf") if candidate["avg_score"] is None else candidate["avg_score"]

    if strategy == "latest_mtime":
        existing_key = (existing["_mtime"], existing_score, existing["file_name"])
        cand_key = (candidate["_mtime"], cand_score, candidate["file_name"])
        return candidate if cand_key >= existing_key else existing

    # 默认策略：highest_score（同分时优先保留更新修改的文件）
    existing_key = (existing_score, existing["_mtime"], existing["file_name"])
    cand_key = (cand_score, candidate["_mtime"], candidate["file_name"])
    return candidate if cand_key >= existing_key else existing


def load_records(directory: Path, duplicate_strategy: str = DUPLICATE_MODEL_STRATEGY):
    raw_by_dataset = {}

    for filepath in sorted(directory.glob("*.jsonl")):
        lines = [
            line.strip()
            for line in filepath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            continue

        summary = json.loads(lines[0])
        dataset_list = summary.get("datasets") or []
        model_list = summary.get("models") or []
        if not dataset_list or not model_list:
            continue

        dataset_name = dataset_list[0]
        model_name = model_list[0]
        raw_by_dataset.setdefault(dataset_name, [])

        hidden_task_ids = HIDDEN_TASK_IDS_BY_DATASET.get(dataset_name, set())
        tasks = {}
        task_ids_in_file = []
        for raw_line in lines[1:]:
            item = parse_task_line(raw_line)
            if not item:
                continue
            task_id = item.get("task_id")
            if not task_id or task_id in hidden_task_ids:
                continue
            tasks[task_id] = {
                "score": format_number(item.get("score")),
                "accuracy": format_number(item.get("accuracy")),
            }
            if task_id not in task_ids_in_file:
                task_ids_in_file.append(task_id)

        raw_by_dataset[dataset_name].append(
            {
                "file_name": filepath.name,
                "model_name": model_name,
                "avg_score": format_number(summary.get("avg_score")),
                "invalid_ratio": format_number(summary.get("invalid_ratio")),
                "category": get_model_category(filepath, model_name),
                "tasks": tasks,
                "_task_order": task_ids_in_file,
                "_mtime": filepath.stat().st_mtime,
            }
        )

    datasets = {}
    for dataset_name, records in raw_by_dataset.items():
        if duplicate_strategy == "keep_all_with_filename":
            counts = {}
            for rec in records:
                counts[rec["model_name"]] = counts.get(rec["model_name"], 0) + 1
            kept_records = []
            for rec in records:
                if counts[rec["model_name"]] > 1:
                    suffix = Path(rec["file_name"]).stem
                    rec["model_name"] = f'{rec["model_name"]} ({suffix})'
                kept_records.append(rec)
        else:
            deduped = {}
            for rec in records:
                m_name = rec["model_name"]
                if m_name not in deduped:
                    deduped[m_name] = rec
                else:
                    existing = deduped[m_name]
                    chosen = _choose_preferred_record(existing, rec, duplicate_strategy)
                    dropped = rec if chosen is existing else existing
                    print(
                        f"[去重提示] 数据集 {dataset_name} 发现同名模型 '{m_name}': "
                        f"保留 {chosen['file_name']} (avg_score={chosen['avg_score']}), "
                        f"忽略 {dropped['file_name']} (avg_score={dropped['avg_score']})"
                    )
                    deduped[m_name] = chosen
            kept_records = list(deduped.values())

        # 基于去重后的最终模型集合构建 taskOrder，防止被忽略文件污染任务列表
        task_order = []
        seen_tasks = set()
        for rec in kept_records:
            for tid in rec["_task_order"]:
                if tid not in seen_tasks:
                    seen_tasks.add(tid)
                    task_order.append(tid)

        kept_records.sort(
            key=lambda item: float("-inf") if item["avg_score"] is None else item["avg_score"],
            reverse=True,
        )

        datasets[dataset_name] = {
            "models": [
                {
                    "file_name": rec["file_name"],
                    "model_name": rec["model_name"],
                    "avg_score": rec["avg_score"],
                    "invalid_ratio": rec["invalid_ratio"],
                    "category": rec["category"],
                    "tasks": rec["tasks"],
                }
                for rec in kept_records
            ],
            "taskOrder": task_order,
        }

    return datasets


def build_html(datasets):
    data_json = json.dumps(
        {
            "datasets": datasets,
            "categoryColors": CATEGORY_COLORS,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>法律评测模型分数对比</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --law: #7c3aed;
      --general: #2a9d8f;
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .container {
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }
    .card {
      background: var(--card);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      border: 1px solid rgba(229, 231, 235, 0.9);
    }
    .section {
      padding: 20px;
      margin-bottom: 20px;
    }
    .section h2 {
      margin: 0 0 8px;
      font-size: 22px;
    }
    .section p {
      margin: 0 0 16px;
      color: var(--muted);
    }
    .legend {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 16px;
      color: var(--muted);
      font-size: 14px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }
    .legend-color {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }
    .toolbar {
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }
    .toolbar select {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      min-width: 280px;
    }
    .task-bars {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .task-row {
      display: grid;
      grid-template-columns: minmax(180px, 240px) minmax(320px, 1fr) max-content;
      gap: 10px;
      align-items: center;
    }
    .task-model {
      font-size: 16px;
      font-weight: 600;
      word-break: break-word;
    }
    .task-track {
      width: 100%;
      min-width: 320px;
      height: 18px;
      background: #edf2f7;
      border-radius: 999px;
      overflow: hidden;
    }
    .task-fill {
      height: 100%;
      border-radius: 999px;
    }
    .task-meta {
      font-size: 16px;
      color: var(--muted);
      text-align: right;
      white-space: nowrap;
      justify-self: end;
    }
    .task-meta strong {
      color: var(--text);
      font-weight: 700;
    }
    .matrix-wrap {
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }
    .matrix-table {
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 12px 10px;
      text-align: center;
      vertical-align: middle;
      width: 120px;
      word-break: break-word;
    }
    th {
      position: sticky;
      top: 0;
      background: #fbfcfe;
      z-index: 1;
      font-size: 13px;
    }
    td {
      font-size: 14px;
    }
    th:first-child, td:first-child {
      text-align: left;
      position: sticky;
      left: 0;
      background: #fff;
      z-index: 2;
      width: 220px;
    }
    th:first-child {
      background: #fbfcfe;
      z-index: 3;
    }
    .pill {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      color: #fff;
    }
    .muted {
      color: var(--muted);
    }
    @media (max-width: 900px) {
      .task-row {
        grid-template-columns: 1fr;
      }
      .task-meta {
        text-align: left;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card section">
      <h2>分数对比</h2>
      <div class="legend" id="legend"></div>
      <div class="toolbar">
        <label for="datasetSelect">数据集选择</label>
        <select id="datasetSelect"></select>
        <label for="taskSelect">任务选择</label>
        <select id="taskSelect"></select>
      </div>
      <div class="task-bars" id="taskBars"></div>
    </div>

    <div class="card section">
      <h2>矩阵对比</h2>
      <div class="matrix-wrap">
        <table id="matrixTable"></table>
      </div>
    </div>
  </div>

  <script>
    const DATA = __DATA_JSON__;

    function formatNum(value) {
      return value === null || value === undefined ? "-" : Number(value).toFixed(2);
    }

    function getCellText(taskData) {
      if (!taskData || taskData.score === null || taskData.score === undefined) {
        return "-";
      }
      return formatNum(taskData.score);
    }

    function getCurrentDatasetName() {
      return document.getElementById("datasetSelect").value;
    }

    function getCurrentDataset() {
      return DATA.datasets[getCurrentDatasetName()];
    }

    function renderLegend() {
      const legend = document.getElementById("legend");
      legend.innerHTML = Object.entries(DATA.categoryColors).map(([label, color]) => `
        <span class="legend-item">
          <span class="legend-color" style="background:${color}"></span>
          <span>${label}</span>
        </span>
      `).join("");
    }

    function renderDatasetOptions() {
      const select = document.getElementById("datasetSelect");
      const datasetNames = Object.keys(DATA.datasets);
      select.innerHTML = datasetNames.map(name => `<option value="${name}">${name}</option>`).join("");
      select.value = datasetNames.includes("LexEval") ? "LexEval" : (datasetNames[0] || "");
      select.onchange = () => {
        renderTaskOptions();
        renderMatrixTable();
      };
    }

    function renderTaskOptions() {
      const dataset = getCurrentDataset();
      const select = document.getElementById("taskSelect");
      if (!dataset) {
        select.innerHTML = "";
        document.getElementById("taskBars").innerHTML = "";
        return;
      }
      const options = [
        `<option value="__avg_score__">平均分对比</option>`,
        ...dataset.taskOrder.map(taskId => `<option value="${taskId}">${taskId}</option>`),
      ];
      select.innerHTML = options.join("");
      select.onchange = () => renderTaskBars(select.value);
      renderTaskBars("__avg_score__");
    }

    function renderTaskBars(taskId) {
      const dataset = getCurrentDataset();
      const container = document.getElementById("taskBars");
      if (!dataset) {
        container.innerHTML = "";
        return;
      }
      const rows = dataset.models.map(model => {
        const isAvgMode = taskId === "__avg_score__";
        const taskData = isAvgMode ? null : model.tasks[taskId];
        const score = isAvgMode
          ? model.avg_score
          : taskData && taskData.score !== null && taskData.score !== undefined
            ? taskData.score
            : null;
        return {
          model_name: model.model_name,
          category: model.category,
          score,
          accuracy: taskData ? taskData.accuracy : null,
        };
      }).sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));

      const maxScore = Math.max(...rows.map(item => item.score ?? 0), 1);

      container.innerHTML = rows.map(item => {
        const fillWidth = item.score === null ? 0 : (item.score / maxScore) * 100;
        const color = DATA.categoryColors[item.category] || DATA.categoryColors["通用模型"];
        let meta = "无数据";
        if (item.score !== null) {
          meta = taskId === "__avg_score__"
            ? `<strong>平均分: ${formatNum(item.score)}</strong>`
            : item.accuracy === null || item.accuracy === undefined
              ? `<strong>正确率: ${formatNum(item.score)}</strong>`
              : `<strong>正确率: ${formatNum(item.score)} | 准确率: ${formatNum(item.accuracy)}</strong>`;
        }

        return `
          <div class="task-row">
            <div class="task-model">${item.model_name}</div>
            <div class="task-track">
              <div class="task-fill" style="width:${fillWidth}%;background:${color}"></div>
            </div>
            <div class="task-meta">${meta}</div>
          </div>
        `;
      }).join("");
    }

    function renderMatrixTable() {
      const dataset = getCurrentDataset();
      const table = document.getElementById("matrixTable");
      if (!dataset) {
        table.innerHTML = "";
        return;
      }
      table.className = "matrix-table";
      const colgroup = `
        <colgroup>
          <col style="width:220px">
          ${dataset.models.map(() => `<col style="width:90px">`).join("")}
        </colgroup>
      `;
      const header = `
        <thead>
          <tr>
            <th>Task</th>
            ${dataset.models.map(model => `
              <th>
                <div>${model.model_name}</div>
                ${model.category === "法律模型" ? `
                  <div style="margin-top:6px;">
                    <span class="pill" style="background:${DATA.categoryColors[model.category]}">${model.category}</span>
                  </div>
                ` : ""}
              </th>
            `).join("")}
          </tr>
        </thead>
      `;

      const body = `
        <tbody>
          ${dataset.taskOrder.map(taskId => `
            <tr>
              <td>${taskId}</td>
              ${dataset.models.map(model => `<td>${getCellText(model.tasks[taskId])}</td>`).join("")}
            </tr>
          `).join("")}
        </tbody>
      `;

      table.innerHTML = colgroup + header + body;
    }

    renderLegend();
    renderDatasetOptions();
    renderTaskOptions();
    renderMatrixTable();
  </script>
</body>
</html>
"""
    return html.replace("__DATA_JSON__", data_json)


def run_sandbox_tests():
    """在临时隔离目录中构造全量回归用例，校验去重、前缀着色、异常修复与 HTML 输出"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 1. alpha_star_law_32B（正常法律模型）
        (tmp / "LexEval__alpha_star_law_32B.jsonl").write_text(
            json.dumps({"datasets": ["LexEval"], "models": ["alpha_star_law_32B"], "avg_score": 80.35}) + "\n"
            + json.dumps({"task_id": "1-1 法律记忆", "score": 82.1, "accuracy": 81.0}) + "\n",
            encoding="utf-8",
        )
        # 2. 模拟截图中的两个同名 alpha_star_law_8B_v2（不同文件、不同分数、不同时间戳）
        f_old = tmp / "LexEval__20261118__alpha_star_law_8B_v2.jsonl"
        f_old.write_text(
            json.dumps({"datasets": ["LexEval"], "models": ["alpha_star_law_8B_v2"], "avg_score": 70.22}) + "\n"
            + json.dumps({"task_id": "1-1 法律记忆", "score": 70.0}) + "\n",
            encoding="utf-8",
        )
        os.utime(f_old, (1700000100, 1700000100))

        f_new = tmp / "LexEval__20261119__alpha_star_law_8B_v2.jsonl"
        f_new.write_text(
            json.dumps({"datasets": ["LexEval"], "models": ["alpha_star_law_8B_v2"], "avg_score": 73.82}) + "\n"
            + '{"task_id": "1-1 法律记忆", "score": 74.5, broken_json\n',
            encoding="utf-8",
        )
        os.utime(f_new, (1700000200, 1700000200))

        # 3. 文件名不含前缀，但 model_name 含小写前缀（测试兜底与忽略大小写）
        (tmp / "custom_eval_run.jsonl").write_text(
            json.dumps({"datasets": ["LexEval"], "models": ["alpha_star_law_8b_v3"], "avg_score": 75.0}) + "\n"
            + json.dumps({"task_id": "1-1 法律记忆", "score": 75.0}) + "\n",
            encoding="utf-8",
        )
        # 4. 通用模型 + avg_score 为 None 的边界模型
        (tmp / "LexEval__Kimi-K2.5.jsonl").write_text(
            json.dumps({"datasets": ["LexEval"], "models": ["Kimi-K2.5"], "avg_score": 74.12}) + "\n"
            + json.dumps({"task_id": "1-1 法律记忆", "score": 74.12}) + "\n",
            encoding="utf-8",
        )
        (tmp / "LexEval__NoScoreModel.jsonl").write_text(
            json.dumps({"datasets": ["LexEval"], "models": ["NoScoreModel"], "avg_score": None}) + "\n",
            encoding="utf-8",
        )
        # 5. LawBench 隐藏任务过滤校验
        (tmp / "LawBench__alpha_star_law_8B_v2.jsonl").write_text(
            json.dumps({"datasets": ["LawBench"], "models": ["alpha_star_law_8B_v2"], "avg_score": 71.5}) + "\n"
            + json.dumps({"task_id": "2-7 舆情摘要", "score": 60.0}) + "\n"
            + json.dumps({"task_id": "1-1 法条背诵", "score": 72.0}) + "\n",
            encoding="utf-8",
        )

        # 断言 1：highest_score 去重仅保留 1 个 73.82 的 alpha_star_law_8B_v2，且归类为“法律模型”
        ds = load_records(tmp, duplicate_strategy="highest_score")
        lex_models = ds["LexEval"]["models"]
        v2_items = [m for m in lex_models if m["model_name"] == "alpha_star_law_8B_v2"]
        assert len(v2_items) == 1, f"去重失败: {v2_items}"
        assert v2_items[0]["avg_score"] == 73.82
        assert v2_items[0]["category"] == "法律模型"
        assert v2_items[0]["tasks"]["1-1 法律记忆"]["score"] == 74.5

        # 断言 2：前缀与忽略大小写匹配准确
        by_name = {m["model_name"]: m for m in lex_models}
        assert by_name["alpha_star_law_32B"]["category"] == "法律模型"
        assert by_name["alpha_star_law_8b_v3"]["category"] == "法律模型"
        assert by_name["Kimi-K2.5"]["category"] == "通用模型"
        assert lex_models[-1]["model_name"] == "NoScoreModel"

        # 断言 3：LawBench 隐藏任务与分类
        assert ds["LawBench"]["taskOrder"] == ["1-1 法条背诵"]
        assert ds["LawBench"]["models"][0]["category"] == "法律模型"

        # 断言 4：latest_mtime 策略生效
        os.utime(f_old, (1700000999, 1700000999))
        ds_mtime = load_records(tmp, duplicate_strategy="latest_mtime")
        v2_mtime = [m for m in ds_mtime["LexEval"]["models"] if m["model_name"] == "alpha_star_law_8B_v2"][0]
        assert v2_mtime["avg_score"] == 70.22

        # 断言 5：keep_all_with_filename 策略自动附加文件名区分
        ds_keep = load_records(tmp, duplicate_strategy="keep_all_with_filename")
        keep_names = [m["model_name"] for m in ds_keep["LexEval"]["models"]]
        assert any("20261118" in n for n in keep_names) and any("20261119" in n for n in keep_names)

        # 断言 6：HTML 模板生成与内嵌 JSON 完整性校验
        html = build_html(ds)
        assert "{{" not in html and "}}" not in html and "__DATA_JSON__" not in html
        match = re.search(r"const DATA = (\{.*?\});", html, flags=re.DOTALL)
        assert match is not None
        embedded = json.loads(match.group(1))
        assert embedded["categoryColors"]["法律模型"] == "#7c3aed"
        assert len([m for m in embedded["datasets"]["LexEval"]["models"] if m["model_name"] == "alpha_star_law_8B_v2"]) == 1

    print("[PASS] 全部 6 组沙箱回归校验通过！")


def main():
    parser = argparse.ArgumentParser(description="生成法律评测模型分数对比仪表板")
    parser.add_argument("--test", action="store_true", help="运行隔离沙箱自测套件")
    parser.add_argument(
        "--dedup",
        choices=("highest_score", "latest_mtime", "keep_all_with_filename"),
        default=DUPLICATE_MODEL_STRATEGY,
        help="同名模型处理策略",
    )
    args = parser.parse_args()

    if args.test:
        run_sandbox_tests()
        return

    directory = Path(__file__).resolve().parent
    datasets = load_records(directory, duplicate_strategy=args.dedup)
    output_path = directory / OUTPUT_FILE
    output_path.write_text(build_html(datasets), encoding="utf-8")

    dataset_count = len(datasets)
    model_count = sum(len(dataset["models"]) for dataset in datasets.values())
    print(f"共读取 {dataset_count} 个数据集，{model_count} 个模型。")
    print(f"网页已生成: {output_path}")


if __name__ == "__main__":
    main()