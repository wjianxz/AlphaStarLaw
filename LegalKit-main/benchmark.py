import argparse
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

#  python /mnt/public/haoduo/code/LegalKit-main/benchmark.py --port 8777

HTML_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>法律模型测评结果</title>
  <style>
    :root {
      --bg: #f5f7fb;
      --card: #ffffff;
      --line: #d7dce5;
      --text: #1f2937;
      --muted: #6b7280;
      --primary: #2563eb;
      --good-bg: #dcfce7;
      --good-text: #166534;
      --bad-bg: #fee2e2;
      --bad-text: #991b1b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    }
    .container {
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1, h2 {
      margin: 0;
    }
    .subtitle {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      padding: 10px 14px;
      background: #eaf2ff;
      border-radius: 10px;
    }
    .section {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 18px;
      margin-bottom: 18px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.05);
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 14px;
    }
    .controls {
      display: grid;
      grid-template-columns: 220px minmax(220px, 0.8fr) auto;
      gap: 12px;
      align-items: start;
      margin-bottom: 14px;
    }
    label {
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }
    label.field-label {
      font-size: 15px;
      font-weight: 650;
      color: var(--text);
    }
    select, button {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      font: inherit;
    }
    button {
      background: var(--primary);
      color: #fff;
      border: none;
      cursor: pointer;
      font-weight: 600;
      padding: 11px 14px;
      margin-top: 0;
    }
.action-col {
  align-self: start;
  display: flex;
  flex-direction: column;
  align-items: flex-end;   /* 贴右边 */
}

.action-col button {
  width: auto;             /* 不再撑满整列 */
  min-width: 192px;         /* 想更窄可以改小一点，比如 88px */
  padding: 10px 14px;
  margin-top: 0;
}
    .hint {
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .panel-title {
      margin-bottom: 12px;
      font-size: 14px;
      color: var(--muted);
    }
    .checkbox-panel {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: #fff;
      min-height: 170px;
      max-height: 260px;
      overflow: auto;
      padding: 10px;
    }
    .checkbox-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .mini-btn {
      width: auto;
      margin-top: 0;
      padding: 6px 10px;
      font-size: 12px;
      border-radius: 8px;
      background: #eef2ff;
      color: #1e3a8a;
      border: 1px solid #c7d2fe;
    }
    .check-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 6px 4px;
      font-size: 14px;
    }
    .check-item input {
      width: 16px;
      height: 16px;
      margin: 0;
    }
    .table-wrap {
      overflow: auto;
      max-height: 620px;
      border: 1px solid var(--line);
      border-radius: 12px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #fff;
      table-layout: fixed;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 12px 14px;
      text-align: left;
      vertical-align: middle;
      font-size: 14px;
    }
    th {
      background: #f7f9fc;
      font-weight: 700;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .task-name {
      width: 220px;
      min-width: 220px;
      max-width: 220px;
      font-size: 15px;
      word-break: break-word;
    }
    .metric-cell {
      width: 140px;
      min-width: 140px;
      max-width: 140px;
      color: #374151;
      word-break: break-word;
    }
    .model-col {
      min-width: 160px;
    }
    .value-box {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-variant-numeric: tabular-nums;
    }
    .value-box.up {
      color: var(--good-text);
      font-weight: 600;
    }
    .value-box.down {
      color: var(--bad-text);
      font-weight: 600;
    }
    .empty {
      padding: 24px;
      text-align: center;
      color: var(--muted);
      font-size: 14px;
    }
    .summary-line {
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      margin-bottom: 16px;
      color: var(--muted);
      font-size: 15px;
    }
    #datasetSummary {
      color: var(--text);
      font-size: 16px;
      font-weight: 650;
    }
    #modelOrderHint {
      font-size: 14px;
      margin-top: 0;
      margin-bottom: 16px;
    }
    .model-order {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
      padding: 0 8px;
      overflow: visible;
    }
    .model-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      cursor: grab;
      user-select: none;
    }
    .model-chip.dragging {
      opacity: 0.55;
    }
    .model-chip.drop-before,
    .model-chip.drop-after {
      position: relative;
    }
    .model-chip.drop-before::before,
    .model-chip.drop-after::after {
      content: "";
      position: absolute;
      top: -4px;
      bottom: -4px;
      width: 3px;
      background: var(--primary);
      border-radius: 2px;
      pointer-events: none;
      z-index: 2;
    }
    .model-chip.drop-before::before {
      left: -7px;
    }
    .model-chip.drop-after::after {
      right: -7px;
    }
    .model-chip > span {
      pointer-events: none;
    }
    .model-chip.baseline {
      border-color: var(--primary);
      background: #eff6ff;
    }
    .rank-wrap {
      overflow: auto;
      max-height: 360px;
      border: 1px solid var(--line);
      border-radius: 12px;
    }
    @media (max-width: 980px) {
      .header {
        display: block;
      }
      .meta {
        margin-top: 12px;
      }
      .controls,
      .controls {
        grid-template-columns: 1fr;
      }
      button {
        margin-top: 0;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <h1>模型评估看板</h1>
      </div>
      <div class="meta" id="refreshTime">正在加载数据...</div>
    </div>

    <div class="section">
      <div class="controls">
        <div>
          <label for="datasetSelect" class="field-label">数据集</label>
          <select id="datasetSelect"></select>
        </div>
        <div>
          <label class="field-label">模型（可多选）</label>
          <div id="modelCheckboxList" class="checkbox-panel"></div>
          <div class="checkbox-actions">
            <button type="button" class="mini-btn" id="selectAllModelsBtn">全选</button>
            <button type="button" class="mini-btn" id="clearModelsBtn">清空</button>
          </div>
          <div class="hint">通过上方模型列表勾选，下方模型标签可拖拽更改baseline，双击取消选择。</div>
        </div>
        <div class="action-col">
          <label class="field-label ghost-label">&nbsp;</label>
          <button id="datasetApplyBtn">查看评测</button>
          <button id="exportCsvBtn" style="margin-top: 12px; background: #10b981;">导出表格</button>
        </div>
      </div>
      <div id="datasetSummary" class="summary-line"></div>
      <div id="modelOrder" class="model-order"></div>
      <div class="table-wrap">
        <div id="datasetTable" class="empty">请选择数据集和模型后查看。</div>
      </div>
    </div>

    <div class="section">
      <div class="section-head">
        <h2>模型总榜</h2>
      </div>
      <div class="rank-wrap">
        <table>
          <thead>
            <tr>
              <th>排名</th>
              <th>模型</th>
              <th>平均分</th>
              <th>评测文件</th>
            </tr>
          </thead>
          <tbody id="rankingBody">
            <tr><td colspan="4" class="empty">请选择数据集。</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let rawData = null;
    let selectedModelOrder = [];
    let draggingModel = null;
    let dropIndicatorChip = null;

    function clearDropIndicator() {
      if (dropIndicatorChip) {
        dropIndicatorChip.classList.remove("drop-before", "drop-after");
        dropIndicatorChip = null;
      }
    }

    function isNumber(value) {
      return value !== null && value !== undefined && !Number.isNaN(Number(value));
    }

    function formatNumber(value) {
      return isNumber(value) ? Number(value).toFixed(2) : "N/A";
    }

    function formatInvalidRatio(value) {
      return isNumber(value) ? `${(Number(value) * 100).toFixed(2)}%` : "N/A";
    }

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function getSelectedModelsFromCheckboxes() {
      return Array.from(document.querySelectorAll('.model-checkbox:checked')).map((el) => el.value);
    }

    function computeDelta(value, baselineValue) {
      if (!isNumber(value) || !isNumber(baselineValue)) {
        return null;
      }
      const current = Number(value);
      const baseline = Number(baselineValue);
      if (baseline === 0) {
        return current === 0 ? 0 : null;
      }
      return ((current - baseline) / Math.abs(baseline)) * 100;
    }

    function formatDeltaPercent(value, baselineValue) {
      const delta = computeDelta(value, baselineValue);
      if (delta === null) {
        return "";
      }
      const sign = delta >= 0 ? "+" : "";
      return `(${sign}${delta.toFixed(1)}%)`;
    }

    function deltaBgStyle(value, baselineValue) {
      const delta = computeDelta(value, baselineValue);
      if (delta === null || delta === 0) {
        return "";
      }
      const absDelta = Math.abs(delta);
      const t = Math.min(absDelta / 80, 1);
      const alpha = (0.06 + Math.sqrt(t) * 0.44).toFixed(2);
      if (delta > 0) {
        return `background:rgba(22,163,74,${alpha})`;
      }
      return `background:rgba(220,38,38,${alpha})`;
    }

    function formatCompareValue(value, baselineValue, isBaseline, metric) {
      const text = metric === 'invalid_ratio' ? formatInvalidRatio(value) : formatNumber(value);
      if (isBaseline || text === "N/A") {
        return text;
      }
      const deltaText = formatDeltaPercent(value, baselineValue);
      return deltaText ? `${text} ${deltaText}` : text;
    }

    function buildTaskTable(records) {
      if (!records.length) {
        return '<div class="empty">没有可展示的数据。</div>';
      }

      const taskOrder = ['__avg_score__'];
      const taskMetrics = new Map([['__avg_score__', ['avg_score']]]);
      const taskMaps = records.map((record) => {
        const map = new Map();
        map.set('__avg_score__', { task_id: '平均分', avg_score: record.avg_score });
        record.tasks.forEach((task) => {
          const taskId = task.task_id || "未命名任务";
          const metrics = Object.keys(task).filter((key) => key !== "task_id" && task[key] !== null && task[key] !== undefined);
          if (metrics.length === 0) return;
          if (!taskMetrics.has(taskId)) {
            taskOrder.push(taskId);
            taskMetrics.set(taskId, []);
          }
          const currentMetrics = taskMetrics.get(taskId);
          metrics.forEach((metric) => {
            if (!currentMetrics.includes(metric)) {
              currentMetrics.push(metric);
            }
          });
          map.set(taskId, task);
        });
        return map;
      });

      let thead = `
        <thead>
          <tr>
            <th class="task-name">任务名称</th>
            <th class="metric-cell">指标</th>
            ${records.map((record) => `<th class="model-col">${escapeHtml(record.model)}</th>`).join("")}
          </tr>
        </thead>
      `;

      let bodyRows = "";
      taskOrder.forEach((taskId) => {
        const metrics = taskMetrics.get(taskId) || ["score"];
        metrics.forEach((metric, metricIndex) => {
          let row = "<tr>";
          if (metricIndex === 0) {
            row += `<td class="task-name" rowspan="${metrics.length}">${escapeHtml(taskId === '__avg_score__' ? '平均分' : taskId)}</td>`;
          }
          row += `<td class="metric-cell">${escapeHtml(metric)}</td>`;

          const baselineTask = taskMaps[0].get(taskId);
          const baselineValue = baselineTask ? baselineTask[metric] : null;

          records.forEach((record, recordIndex) => {
            const task = taskMaps[recordIndex].get(taskId);
            const value = task ? task[metric] : null;
            let cls = "";
            let style = "";
            if (recordIndex > 0 && isNumber(value) && isNumber(baselineValue)) {
              if (Number(value) > Number(baselineValue)) {
                cls = "up";
              } else if (Number(value) < Number(baselineValue)) {
                cls = "down";
              }
              style = deltaBgStyle(value, baselineValue);
            }
            row += `<td><span class="value-box ${cls}" style="${style}">${escapeHtml(formatCompareValue(value, baselineValue, recordIndex === 0, metric))}</span></td>`;
          });

          row += "</tr>";
          bodyRows += row;
        });
      });

      return `<table>${thead}<tbody>${bodyRows || '<tr><td colspan="' + (records.length + 2) + '" class="empty">没有任务数据。</td></tr>'}</tbody></table>`;
    }

    function renderDatasetOptions() {
      const datasetSelect = document.getElementById("datasetSelect");
      datasetSelect.innerHTML = rawData.datasets
        .map((dataset) => `<option value="${escapeHtml(dataset)}">${escapeHtml(dataset)}</option>`)
        .join("");
      if (rawData.datasets.length) {
        datasetSelect.value = rawData.datasets[0];
      }
      updateModelOptions();
    }

    function updateModelOptions() {
      const dataset = document.getElementById("datasetSelect").value;
      const modelList = document.getElementById('modelCheckboxList');
      const models = rawData.records
        .filter((item) => item.dataset === dataset)
        .map((item) => item.model)
        .sort((a, b) => a.localeCompare(b, "zh-CN"));

      selectedModelOrder = models.slice();

      modelList.innerHTML = models.map((model) => `
        <label class="check-item">
          <input type="checkbox" class="model-checkbox" value="${escapeHtml(model)}">
          <span>${escapeHtml(model)}</span>
        </label>
      `).join('');

      modelList.querySelectorAll('.model-checkbox').forEach((el) => {
        el.addEventListener('change', () => {
          syncSelectedModelOrder();
          renderDatasetView();
        });
      });
    }

    function syncSelectedModelOrder() {
      const selectedModels = getSelectedModelsFromCheckboxes();
      const kept = selectedModelOrder.filter((model) => selectedModels.includes(model));
      const appended = selectedModels.filter((model) => !kept.includes(model));
      selectedModelOrder = kept.concat(appended);
    }

    function unselectModel(model) {
      const checkbox = document.querySelector(`.model-checkbox[value="${CSS.escape(model)}"]`);
      if (!checkbox) {
        return;
      }
      checkbox.checked = false;
      syncSelectedModelOrder();
      renderDatasetView();
    }

    function renderModelOrder(records) {
      const container = document.getElementById("modelOrder");
      if (!records.length) {
        container.innerHTML = "";
        return;
      }
      container.innerHTML = records.map((record, index) => `
        <div class="model-chip ${index === 0 ? "baseline" : ""}" draggable="true" data-model="${escapeHtml(record.model)}">
          <span>${escapeHtml(record.model)}</span>
        </div>
      `).join("");

      container.querySelectorAll(".model-chip").forEach((chip) => {
        chip.addEventListener("dragstart", () => {
          draggingModel = chip.dataset.model;
          chip.classList.add("dragging");
        });
        chip.addEventListener("dragend", () => {
          draggingModel = null;
          chip.classList.remove("dragging");
          clearDropIndicator();
        });
        chip.addEventListener("dragover", (event) => {
          event.preventDefault();
          if (!draggingModel || chip.dataset.model === draggingModel) {
            return;
          }
          const rect = chip.getBoundingClientRect();
          const isLeft = event.clientX < rect.left + rect.width / 2;
          clearDropIndicator();
          chip.classList.add(isLeft ? "drop-before" : "drop-after");
          dropIndicatorChip = chip;
        });
        chip.addEventListener("dragleave", (event) => {
          if (dropIndicatorChip === chip && !chip.contains(event.relatedTarget)) {
            clearDropIndicator();
          }
        });
        chip.addEventListener("drop", (event) => {
          event.preventDefault();
          const targetModel = chip.dataset.model;
          const rect = chip.getBoundingClientRect();
          const isLeft = event.clientX < rect.left + rect.width / 2;
          clearDropIndicator();
          if (!draggingModel || draggingModel === targetModel) {
            return;
          }
          const nextOrder = selectedModelOrder.filter((model) => model !== draggingModel);
          const targetIndex = nextOrder.indexOf(targetModel);
          nextOrder.splice(isLeft ? targetIndex : targetIndex + 1, 0, draggingModel);
          selectedModelOrder = nextOrder;
          renderDatasetView();
        });
        chip.addEventListener("dblclick", () => {
          unselectModel(chip.dataset.model);
        });
      });
    }

    function renderDatasetView() {
      const dataset = document.getElementById("datasetSelect").value;
      syncSelectedModelOrder();
      const selectedRecords = selectedModelOrder
        .map((model) => rawData.records.find((item) => item.dataset === dataset && item.model === model))
        .filter(Boolean);

      const summary = document.getElementById("datasetSummary");
      if (!selectedRecords.length) {
        summary.innerHTML = "";
        document.getElementById("modelOrder").innerHTML = "";
        document.getElementById("datasetTable").innerHTML = '<div class="empty">请选择至少一个模型。</div>';
      } else {
        const baseline = selectedRecords[0];
        summary.innerHTML = `<span>${escapeHtml(baseline.model)}：总分 ${escapeHtml(formatNumber(baseline.avg_score))}，坍塌率 ${escapeHtml(formatNumber(baseline.invalid_ratio))}%</span>`;
        renderModelOrder(selectedRecords);
        document.getElementById("datasetTable").innerHTML = buildTaskTable(selectedRecords);
      }

      renderRanking(dataset);
    }

    function renderRanking(dataset) {
      const rankingBody = document.getElementById("rankingBody");
      const rankingTitle = document.getElementById("rankingTitle");
      if (!dataset) {
        rankingTitle.textContent = "选择数据集后显示该数据集下的模型平均分排名。";
        rankingBody.innerHTML = '<tr><td colspan="4" class="empty">请选择数据集。</td></tr>';
        return;
      }

      const rows = rawData.records
        .filter((item) => item.dataset === dataset)
        .slice()
        .sort((a, b) => {
          const av = isNumber(a.avg_score) ? Number(a.avg_score) : -Infinity;
          const bv = isNumber(b.avg_score) ? Number(b.avg_score) : -Infinity;
          return bv - av;
        });
      if (!rows.length) {
        rankingBody.innerHTML = '<tr><td colspan="4" class="empty">当前数据集没有评分文件。</td></tr>';
        return;
      }

      rankingBody.innerHTML = rows.map((item, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(item.model)}</td>
          <td>${escapeHtml(formatNumber(item.avg_score))}</td>
          <td>${escapeHtml(item.file_name)}</td>
        </tr>
      `).join("");
    }

    async function loadData() {
      const resp = await fetch("/api/scores");
      rawData = await resp.json();
      renderDatasetOptions();
      renderDatasetView();
      const ts = new Date(rawData.generated_at);
      document.getElementById("refreshTime").textContent = `最近刷新：${ts.toLocaleString("zh-CN")}`;
    }

    document.getElementById("datasetSelect").addEventListener("change", () => {
      updateModelOptions();
      renderDatasetView();
    });
    document.getElementById("datasetApplyBtn").addEventListener("click", renderDatasetView);
    document.getElementById('selectAllModelsBtn').addEventListener('click', () => {
      document.querySelectorAll('.model-checkbox').forEach((el) => { el.checked = true; });
      syncSelectedModelOrder();
      renderDatasetView();
    });
    document.getElementById('clearModelsBtn').addEventListener('click', () => {
      document.querySelectorAll('.model-checkbox').forEach((el) => { el.checked = false; });
      syncSelectedModelOrder();
      renderDatasetView();
    });

    function exportTableToCSV(filename) {
      const table = document.querySelector("#datasetTable table");
      if (!table) {
        alert("没有可导出的表格数据");
        return;
      }
      let csv = [];
      const rows = table.querySelectorAll("tr");
      for (let i = 0; i < rows.length; i++) {
        let row = [], cols = rows[i].querySelectorAll("td, th");
        for (let j = 0; j < cols.length; j++) {
          let data = cols[j].innerText.replace(/(\\r\\n|\\n|\\r)/gm, " ").trim();
          data = data.replace(/"/g, '""');
          row.push('"' + data + '"');
        }
        csv.push(row.join(","));
      }
      const csvFile = new Blob(["\\uFEFF" + csv.join("\\n")], {type: "text/csv;charset=utf-8;"});
      const downloadLink = document.createElement("a");
      downloadLink.download = filename;
      downloadLink.href = window.URL.createObjectURL(csvFile);
      downloadLink.style.display = "none";
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
    }

    document.getElementById("exportCsvBtn").addEventListener("click", () => {
      const dataset = document.getElementById("datasetSelect").value;
      exportTableToCSV(`${dataset}_评测结果.csv`);
    });

    loadData().catch((err) => {
      document.getElementById("datasetTable").innerHTML = `<div class="empty">加载失败：${escapeHtml(err)}</div>`;
    });
  </script>
</body>
</html>
"""


def load_jsonl_scores(scores_dir: Path) -> dict:
    records = []

    for file_path in sorted(scores_dir.glob("*.jsonl")):
        payloads = []
        for line in file_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        if not payloads:
            continue

        summary = payloads[0]
        tasks = []
        for task in payloads[1:]:
            item = {"task_id": task.get("task_id")}
            for key, value in task.items():
                if key != "task_id":
                    item[key] = value
            tasks.append(item)

        dataset = (summary.get("datasets") or [file_path.stem.split("__")[0]])[0]
        model = (summary.get("models") or [file_path.stem.split("__")[-1]])[0]

        records.append(
            {
                "id": file_path.stem,
                "file_name": file_path.name,
                "dataset": dataset,
                "model": model,
                "avg_score": summary.get("avg_score"),
                "invalid_ratio": summary.get("invalid_ratio"),
                "tasks": tasks,
            }
        )

    records.sort(key=lambda item: (item["dataset"], item["model"], item["file_name"]))

    default_dataset_order = ["LawBench", "LexEval", "JecQA", "CaseGen"]
    datasets = sorted({item["dataset"] for item in records})
    datasets.sort(key=lambda name: (default_dataset_order.index(name) if name in default_dataset_order else 10_000, name))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "datasets": datasets,
        "records": records,
    }


class ScoreHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, scores_dir: Path, **kwargs):
        self.scores_dir = scores_dir
        super().__init__(*args, **kwargs)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_html(self, html: str, status: int = 200) -> None:
        data = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(HTML_PAGE)
            return

        if parsed.path == "/api/scores":
            self._send_json(load_jsonl_scores(self.scores_dir))
            return

        if parsed.path == "/health":
            self._send_json({"ok": True, "scores_dir": str(self.scores_dir)})
            return

        self._send_json({"error": "not found", "path": parsed.path}, status=404)

    def log_message(self, fmt: str, *args) -> None:
        return


def build_handler(scores_dir: Path):
    def handler(*args, **kwargs):
        ScoreHandler(*args, scores_dir=scores_dir, **kwargs)
    return handler


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="启动 model_scores 网页评分看板")
    parser.add_argument(
        "--scores-dir",
        default=str(base_dir / "model_scores"),
        help="jsonl 评分目录，默认是脚本同级的 model_scores",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", default=8000, type=int, help="监听端口")
    args = parser.parse_args()

    scores_dir = Path(args.scores_dir).resolve()
    if not scores_dir.exists():
        raise SystemExit(f"评分目录不存在: {scores_dir}")

    server = ThreadingHTTPServer((args.host, args.port), build_handler(scores_dir))
    url = f"http://{args.host}:{args.port}"
    print(f"评分看板已启动: {url}")
    print(f"读取目录: {scores_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
