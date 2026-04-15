import json
import re
from pathlib import Path


LAW_MODEL_FILE_SUFFIXES = {
    "alpha_star_law_32B",
    "alpha_star_law_8B",
}

CATEGORY_COLORS = {
    "法律模型": "#7c3aed",
    "通用模型": "#2a9d8f",
}

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


def get_model_category(filepath: Path):
    file_suffix = filepath.stem.split("__", 1)[-1]
    return "法律模型" if file_suffix in LAW_MODEL_FILE_SUFFIXES else "通用模型"


def load_records(directory: Path):
    datasets = {}

    for filepath in sorted(directory.glob("*.jsonl")):
        lines = [
            line.strip()
            for line in filepath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            continue

        summary = json.loads(lines[0])
        dataset_name = summary["datasets"][0]
        if dataset_name not in datasets:
            datasets[dataset_name] = {
                "models": [],
                "taskOrder": [],
                "_seenTasks": set(),
            }

        hidden_task_ids = HIDDEN_TASK_IDS_BY_DATASET.get(dataset_name, set())
        tasks = {}
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
            if task_id not in datasets[dataset_name]["_seenTasks"]:
                datasets[dataset_name]["_seenTasks"].add(task_id)
                datasets[dataset_name]["taskOrder"].append(task_id)

        datasets[dataset_name]["models"].append(
            {
                "file_name": filepath.name,
                "model_name": summary["models"][0],
                "avg_score": format_number(summary.get("avg_score")),
                "invalid_ratio": format_number(summary.get("invalid_ratio")),
                "category": get_model_category(filepath),
                "tasks": tasks,
            }
        )

    for dataset in datasets.values():
        dataset["models"].sort(key=lambda item: item["avg_score"], reverse=True)
        del dataset["_seenTasks"]

    return datasets


def build_html(datasets):
    data_json = json.dumps(
        {
            "datasets": datasets,
            "categoryColors": CATEGORY_COLORS,
        },
        ensure_ascii=False,
    )

    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>法律评测模型分数对比</title>
  <style>
    :root {{
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1f2937;
      --muted: #6b7280;
      --line: #e5e7eb;
      --law: #7c3aed;
      --general: #2a9d8f;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
      color: var(--text);
      background: var(--bg);
    }}
    .container {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }}
    .card {{
      background: var(--card);
      border-radius: 14px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      border: 1px solid rgba(229, 231, 235, 0.9);
    }}
    .section {{
      padding: 20px;
      margin-bottom: 20px;
    }}
    .section h2 {{
      margin: 0 0 8px;
      font-size: 22px;
    }}
    .section p {{
      margin: 0 0 16px;
      color: var(--muted);
    }}
    .legend {{
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
      margin-bottom: 16px;
      color: var(--muted);
      font-size: 14px;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-color {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .avg-chart {{
      display: flex;
      align-items: flex-end;
      gap: 10px;
      min-height: 430px;
      overflow-x: auto;
      padding: 8px 4px 0;
    }}
    .avg-bar-wrap {{
      flex: 0 0 72px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-end;
      gap: 8px;
    }}
    .avg-value {{
      font-size: 12px;
      font-weight: 700;
      color: var(--text);
    }}
    .avg-bar {{
      width: 100%;
      border-radius: 10px 10px 0 0;
      min-height: 4px;
      position: relative;
    }}
    .avg-label {{
      width: 100%;
      font-size: 12px;
      text-align: center;
      line-height: 1.35;
      word-break: break-word;
      color: var(--text);
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .toolbar select {{
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      min-width: 280px;
    }}
    .task-bars {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .task-row {{
      display: grid;
      grid-template-columns: minmax(180px, 240px) minmax(320px, 1fr) max-content;
      gap: 10px;
      align-items: center;
    }}
    .task-model {{
      font-size: 16px;
      font-weight: 600;
      word-break: break-word;
    }}
    .task-track {{
      width: 100%;
      min-width: 320px;
      height: 18px;
      background: #edf2f7;
      border-radius: 999px;
      overflow: hidden;
    }}
    .task-fill {{
      height: 100%;
      border-radius: 999px;
    }}
    .task-meta {{
      font-size: 16px;
      color: var(--muted);
      text-align: right;
      white-space: nowrap;
      justify-self: end;
    }}
    .task-meta strong {{
      color: var(--text);
      font-weight: 700;
    }}
    .matrix-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    .matrix-table {{
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 12px 10px;
      text-align: center;
      vertical-align: middle;
      width: 120px;
      word-break: break-word;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #fbfcfe;
      z-index: 1;
      font-size: 13px;
    }}
    td {{
      font-size: 14px;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      position: sticky;
      left: 0;
      background: #fff;
      z-index: 2;
      width: 220px;
    }}
    th:first-child {{
      background: #fbfcfe;
      z-index: 3;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      color: #fff;
    }}
    .muted {{
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      .task-row {{
        grid-template-columns: 1fr;
      }}
      .task-meta {{
        text-align: left;
      }}
    }}
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

    function formatNum(value) {{
      return value === null || value === undefined ? "-" : Number(value).toFixed(2);
    }}

    function getCellText(taskData) {{
      if (!taskData || taskData.score === null || taskData.score === undefined) {{
        return "-";
      }}
      return formatNum(taskData.score);
    }}

    function getCurrentDatasetName() {{
      return document.getElementById("datasetSelect").value;
    }}

    function getCurrentDataset() {{
      return DATA.datasets[getCurrentDatasetName()];
    }}

    function renderLegend() {{
      const legend = document.getElementById("legend");
      legend.innerHTML = Object.entries(DATA.categoryColors).map(([label, color]) => `
        <span class="legend-item">
          <span class="legend-color" style="background:${{color}}"></span>
          <span>${{label}}</span>
        </span>
      `).join("");
    }}

    function renderDatasetOptions() {{
      const select = document.getElementById("datasetSelect");
      const datasetNames = Object.keys(DATA.datasets);
      select.innerHTML = datasetNames.map(name => `<option value="${{name}}">${{name}}</option>`).join("");
      select.value = "LexEval"; // 默认选择 LexEval 数据集
      select.onchange = () => {{
        renderTaskOptions();
        renderMatrixTable();
      }};
    }}

    function renderTaskOptions() {{
      const dataset = getCurrentDataset();
      const select = document.getElementById("taskSelect");
      const options = [
        `<option value="__avg_score__">平均分对比</option>`,
        ...dataset.taskOrder.map(taskId => `<option value="${{taskId}}">${{taskId}}</option>`),
      ];
      select.innerHTML = options.join("");
      select.onchange = () => renderTaskBars(select.value);
      renderTaskBars("__avg_score__");
    }}

    function renderTaskBars(taskId) {{
      const dataset = getCurrentDataset();
      const rows = dataset.models.map(model => {{
        const isAvgMode = taskId === "__avg_score__";
        const taskData = isAvgMode ? null : model.tasks[taskId];
        const score = isAvgMode
          ? model.avg_score
          : taskData && taskData.score !== null && taskData.score !== undefined
            ? taskData.score
            : null;
        return {{
          model_name: model.model_name,
          category: model.category,
          score,
          accuracy: taskData ? taskData.accuracy : null,
        }};
      }}).sort((a, b) => (b.score ?? -1) - (a.score ?? -1));

      const maxScore = Math.max(...rows.map(item => item.score ?? 0), 1);
      const container = document.getElementById("taskBars");

      container.innerHTML = rows.map(item => {{
        const fillWidth = item.score === null ? 0 : (item.score / maxScore) * 100;
        const color = DATA.categoryColors[item.category];
        let meta = "无数据";
        if (item.score !== null) {{
          meta = taskId === "__avg_score__"
            ? `<strong>平均分: ${{formatNum(item.score)}}</strong>`
            : item.accuracy === null || item.accuracy === undefined
              ? `<strong>正确率: ${{formatNum(item.score)}}</strong>`
              : `<strong>正确率: ${{formatNum(item.score)}} | 准确率: ${{formatNum(item.accuracy)}}</strong>`;
        }}

        return `
          <div class="task-row">
            <div class="task-model">${{item.model_name}}</div>
            <div class="task-track">
              <div class="task-fill" style="width:${{fillWidth}}%;background:${{color}}"></div>
            </div>
            <div class="task-meta">${{meta}}</div>
          </div>
        `;
      }}).join("");
    }}

    function renderMatrixTable() {{
      const dataset = getCurrentDataset();
      const table = document.getElementById("matrixTable");
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
                ` : ``}
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
    }}

    renderLegend();
    renderDatasetOptions();
    renderTaskOptions();
    renderMatrixTable();
  </script>
</body>
</html>
"""
    return html.replace("{{", "{").replace("}}", "}").replace("__DATA_JSON__", data_json)


def main():
    directory = Path(__file__).resolve().parent
    datasets = load_records(directory)
    output_path = directory / OUTPUT_FILE
    output_path.write_text(build_html(datasets), encoding="utf-8")

    dataset_count = len(datasets)
    model_count = sum(len(dataset["models"]) for dataset in datasets.values())
    print(f"共读取 {dataset_count} 个数据集，{model_count} 个模型。")
    print(f"网页已生成: {output_path}")


if __name__ == "__main__":
    main()
