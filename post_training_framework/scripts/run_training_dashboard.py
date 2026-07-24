"""本地训练监视看板。

用法示例:
python post_training_framework/scripts/run_training_dashboard.py ^
  --run-dir models/grpo/qwen3_0d6b_grpo_v5_rollout8_len256_lr2e-6_eval100 ^
  --port 7860
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 只在读取历史 CSV 时接受旧字段；对外统一暴露语义明确的新名称。
LEGACY_CSV_FIELD_ALIASES = {
    "rollout_hit_max_rate": "rollout_max_tokens_reached_without_eos_rate",
    "val_hit_max_rate": "val_max_tokens_reached_without_eos_rate",
    "val_sample_hit_max_rate": "val_sample_max_tokens_reached_without_eos_rate",
}


def _to_number(value: str) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer() and "." not in text and "e" not in text.lower():
        return int(number)
    return number


def _read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                normalized = {key: _to_number(value) for key, value in row.items()}
                for legacy_name, current_name in LEGACY_CSV_FIELD_ALIASES.items():
                    if current_name not in normalized and legacy_name in normalized:
                        normalized[current_name] = normalized.pop(legacy_name)
                rows.append(normalized)
    except OSError:
        return []
    return rows


def _dedupe_step_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = row.get("step")
        if isinstance(step, int):
            by_step[step] = row
        elif isinstance(step, float) and step.is_integer():
            by_step[int(step)] = row
    return [by_step[step] for step in sorted(by_step)]


def _latest_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    if not checkpoints:
        return None
    step, path = sorted(checkpoints, key=lambda item: item[0])[-1]
    return {
        "name": path.name,
        "step": step,
        "path": str(path),
        "mtime": path.stat().st_mtime,
        "has_optimizer": (path / "optimizer.pt").exists(),
        "has_rng_state": (path / "rng_state.pt").exists(),
        "has_trainer_state": (path / "trainer_state.json").exists(),
    }


def _all_checkpoints(run_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in run_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match:
            continue
        items.append(
            {
                "name": path.name,
                "step": int(match.group(1)),
                "mtime": path.stat().st_mtime,
                "has_optimizer": (path / "optimizer.pt").exists(),
                "has_rng_state": (path / "rng_state.pt").exists(),
            }
        )
    return sorted(items, key=lambda item: item["step"])


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _latest_log(run_dir: Path) -> Path | None:
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        return None
    logs = sorted(log_dir.glob("*.log"), key=lambda item: item.stat().st_mtime)
    return logs[-1] if logs else None


def _tail_text(path: Path | None, max_lines: int = 80, max_bytes: int = 64 * 1024) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        with path.open("rb") as f:
            size = f.seek(0, os.SEEK_END)
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()[-max_lines:]


def _infer_total_steps(state: dict[str, Any], log_lines: list[str]) -> int | None:
    value = state.get("total_training_steps")
    if isinstance(value, int):
        return value
    for line in log_lines:
        match = re.search(r"最大训练步数:\s*(\d+)", line)
        if match:
            return int(match.group(1))
    return None


def _state_from_log_age(log_path: Path | None, latest_step_time: float | None) -> dict[str, Any]:
    if log_path is None or not log_path.exists():
        return {"label": "unknown", "reason": "未找到日志文件", "log_age_seconds": None}
    age = time.time() - log_path.stat().st_mtime
    # 训练 step 可能很慢，用最近 step_time 的 3 倍作为动态宽限。
    threshold = 300.0
    if latest_step_time:
        threshold = max(threshold, float(latest_step_time) * 3.0)
    if age <= threshold:
        return {"label": "running_or_active", "reason": "日志仍在更新", "log_age_seconds": age}
    return {"label": "stale_or_finished", "reason": "日志较久未更新", "log_age_seconds": age}


def _format_ts(ts: float | None) -> str | None:
    if not ts:
        return None
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


@dataclass(frozen=True)
class DashboardConfig:
    run_dir: Path
    refresh_seconds: int
    tail_lines: int


def build_status(config: DashboardConfig) -> dict[str, Any]:
    run_dir = config.run_dir.resolve()
    train_rows = _dedupe_step_rows(_read_csv_rows(run_dir / "plots" / "train_metrics.csv"))
    val_rows = _dedupe_step_rows(_read_csv_rows(run_dir / "plots" / "val_metrics.csv"))
    group_rows = _dedupe_step_rows(_read_csv_rows(run_dir / "plots" / "group_diagnostics.csv"))
    run_config = _load_json(run_dir / "run_config.json")
    stop_summary = _load_json(run_dir / "training_stop.json")
    saved_config = run_config.get("config", {})
    if not isinstance(saved_config, dict):
        saved_config = {}
    latest_ckpt = _latest_checkpoint(run_dir)
    ckpts = _all_checkpoints(run_dir)
    state = {}
    if latest_ckpt:
        state = _load_json(Path(latest_ckpt["path"]) / "trainer_state.json")

    log_path = _latest_log(run_dir)
    log_lines = _tail_text(log_path, max_lines=config.tail_lines)
    latest_train = train_rows[-1] if train_rows else {}
    latest_val = val_rows[-1] if val_rows else {}
    latest_group = group_rows[-1] if group_rows else {}
    total_steps = _infer_total_steps(state, log_lines)
    if total_steps is None and isinstance(saved_config.get("total_training_steps"), int):
        total_steps = saved_config["total_training_steps"]
    latest_step = latest_train.get("step")
    if not isinstance(latest_step, int):
        latest_step = state.get("step") if isinstance(state.get("step"), int) else None
    progress = None
    if isinstance(latest_step, int) and isinstance(total_steps, int) and total_steps > 0:
        progress = min(1.0, max(0.0, (latest_step + 1) / total_steps))

    step_time = latest_train.get("step_time")
    if not isinstance(step_time, (int, float)):
        step_time = None

    stop_decision = stop_summary.get("decision", {})
    if not isinstance(stop_decision, dict):
        stop_decision = {}
    state_session = state.get("training_session_id")
    summary_session = stop_summary.get("session_id")
    if (
        state.get("training_status") == "running"
        and state_session
        and state_session != summary_session
    ):
        training_status = "running"
        stop_reason = None
        stop_category = None
    else:
        training_status = stop_summary.get("status") or state.get("training_status")
        stop_reason = stop_decision.get("reason") or state.get("stop_reason")
        stop_category = stop_decision.get("category")
    return {
        "now": _format_ts(time.time()),
        "run_dir": str(run_dir),
        "run_name": state.get("run_name") or saved_config.get("run_name") or run_dir.name,
        "refresh_seconds": config.refresh_seconds,
        "total_steps": total_steps,
        "latest_step": latest_step,
        "progress": progress,
        "best_val_em": state.get("best_val_em") if isinstance(state.get("best_val_em"), (int, float)) else latest_train.get("best_val_em"),
        "best_step": state.get("best_step"),
        "steps_no_improve": state.get("steps_no_improve") if isinstance(state.get("steps_no_improve"), int) else latest_train.get("steps_no_improve"),
        "training_status": training_status,
        "stop_reason": stop_reason,
        "stop_category": stop_category,
        "latest_train": latest_train,
        "latest_val": latest_val,
        "latest_group": latest_group,
        "train_rows": train_rows,
        "val_rows": val_rows,
        "group_rows": group_rows,
        "checkpoints": ckpts,
        "latest_checkpoint": latest_ckpt,
        "latest_checkpoint_time": _format_ts(latest_ckpt["mtime"]) if latest_ckpt else None,
        "log_file": str(log_path) if log_path else None,
        "log_mtime": _format_ts(log_path.stat().st_mtime) if log_path and log_path.exists() else None,
        "activity": _state_from_log_age(log_path, step_time),
        "tail_lines": log_lines,
    }


HTML_PAGE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>训练监视看板</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #687385;
      --line: #d9dee8;
      --blue: #2563eb;
      --green: #059669;
      --orange: #d97706;
      --red: #dc2626;
      --violet: #7c3aed;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Microsoft YaHei", Arial, sans-serif;
    }
    header {
      padding: 18px 22px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 5;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    main {
      max-width: 1440px;
      margin: 0 auto;
      padding: 16px;
    }
    .grid {
      display: grid;
      gap: 12px;
    }
    .cards {
      grid-template-columns: repeat(6, minmax(120px, 1fr));
    }
    .charts {
      grid-template-columns: repeat(2, minmax(280px, 1fr));
      margin-top: 12px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }
    .metric-value {
      font-size: 22px;
      font-weight: 700;
      line-height: 1.1;
    }
    .metric-note {
      color: var(--muted);
      font-size: 12px;
      margin-top: 6px;
    }
    .progress-wrap {
      margin-top: 10px;
      height: 8px;
      background: #e8edf5;
      border-radius: 99px;
      overflow: hidden;
    }
    .progress-bar {
      height: 100%;
      width: 0%;
      background: var(--blue);
      transition: width 220ms ease;
    }
    canvas {
      display: block;
      width: 100%;
      height: 300px;
    }
    .chart-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 8px;
      font-size: 14px;
      font-weight: 650;
    }
    .legend {
      color: var(--muted);
      font-size: 12px;
      font-weight: 400;
    }
    .legend-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 14px;
      margin: 2px 0 8px;
      color: var(--muted);
      font-size: 12px;
    }
    .legend-item {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      white-space: nowrap;
    }
    .legend-swatch {
      width: 18px;
      height: 3px;
      border-radius: 99px;
      background: var(--line);
    }
    .chart-box {
      position: relative;
    }
    .chart-tooltip {
      position: absolute;
      min-width: 190px;
      max-width: min(320px, calc(100% - 24px));
      pointer-events: none;
      display: none;
      z-index: 3;
      padding: 9px 10px;
      border: 1px solid #c7d0de;
      border-radius: 7px;
      background: rgba(255, 255, 255, 0.96);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.14);
      font-size: 12px;
      line-height: 1.45;
    }
    .tooltip-title {
      font-weight: 650;
      margin-bottom: 4px;
    }
    .tooltip-row {
      display: flex;
      justify-content: space-between;
      gap: 14px;
    }
    .tooltip-name {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: var(--muted);
    }
    .tooltip-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--line);
    }
    .wide {
      grid-column: 1 / -1;
    }
    pre {
      margin: 0;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      color: #172033;
      background: #f9fafb;
      border-radius: 6px;
      padding: 10px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
    }
    th {
      color: var(--muted);
      font-weight: 600;
    }
    .ok { color: var(--green); }
    .warn { color: var(--orange); }
    .bad { color: var(--red); }
    @media (max-width: 980px) {
      .cards, .charts { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
    }
    @media (max-width: 640px) {
      .cards, .charts { grid-template-columns: 1fr; }
      canvas { height: 260px; }
    }
  </style>
</head>
<body>
  <header>
    <h1 id="title">训练监视看板</h1>
    <div class="sub" id="runDir"></div>
  </header>
  <main>
    <section class="grid cards">
      <div class="panel">
        <div class="metric-label">当前 Step</div>
        <div class="metric-value" id="step">-</div>
        <div class="progress-wrap"><div class="progress-bar" id="progress"></div></div>
      </div>
      <div class="panel">
        <div class="metric-label">Best EM</div>
        <div class="metric-value" id="bestEm">-</div>
        <div class="metric-note" id="bestStep">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">No Improve</div>
        <div class="metric-value" id="noImprove">-</div>
        <div class="metric-note">早停耐心按训练配置判断</div>
      </div>
      <div class="panel">
        <div class="metric-label">Train Reward</div>
        <div class="metric-value" id="reward">-</div>
        <div class="metric-note" id="rewardStd">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">Reference KL / Update KL</div>
        <div class="metric-value" id="kl">-</div>
        <div class="metric-note" id="clip">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">组内有效信号</div>
        <div class="metric-value" id="effectiveGroup">-</div>
        <div class="metric-note" id="mixedGroup">-</div>
      </div>
      <div class="panel">
        <div class="metric-label">状态</div>
        <div class="metric-value" id="activity">-</div>
        <div class="metric-note" id="logTime">-</div>
      </div>
    </section>

    <section class="grid charts">
      <div class="panel">
        <div class="chart-title">训练 Reward <span class="legend">按 step 查看训练奖励波动</span></div>
        <div class="legend-row" id="rewardLegend"></div>
        <div class="chart-box"><canvas id="rewardChart"></canvas><div class="chart-tooltip" id="rewardTooltip"></div></div>
      </div>
      <div class="panel">
        <div class="chart-title">验证指标 <span class="legend">固定 eval 集上的 EM / reward / format</span></div>
        <div class="legend-row" id="valLegend"></div>
        <div class="chart-box"><canvas id="valChart"></canvas><div class="chart-tooltip" id="valTooltip"></div></div>
      </div>
      <div class="panel">
        <div class="chart-title">PPO 稳定性 <span class="legend">区分累计 reference KL 与单次 update KL</span></div>
        <div class="legend-row" id="ppoLegend"></div>
        <div class="chart-box"><canvas id="ppoChart"></canvas><div class="chart-tooltip" id="ppoTooltip"></div></div>
      </div>
      <div class="panel">
        <div class="chart-title">格式与截断 <span class="legend">训练采样与随机验证的尾部退化监控</span></div>
        <div class="legend-row" id="formatLegend"></div>
        <div class="chart-box"><canvas id="formatChart"></canvas><div class="chart-tooltip" id="formatTooltip"></div></div>
      </div>
      <div class="panel">
        <div class="chart-title">长度与耗时 <span class="legend">响应长度和每步耗时</span></div>
        <div class="legend-row" id="timeLegend"></div>
        <div class="chart-box"><canvas id="timeChart"></canvas><div class="chart-tooltip" id="timeTooltip"></div></div>
      </div>
      <div class="panel">
        <div class="chart-title">组内训练信号 <span class="legend">有效组、mixed 组和零 advantage</span></div>
        <div class="legend-row" id="groupLegend"></div>
        <div class="chart-box"><canvas id="groupChart"></canvas><div class="chart-tooltip" id="groupTooltip"></div></div>
      </div>
      <div class="panel">
        <div class="chart-title">最近 Checkpoint <span class="legend" id="ckptSummary"></span></div>
        <table>
          <thead><tr><th>名称</th><th>Step</th><th>优化器</th><th>RNG</th></tr></thead>
          <tbody id="ckptTable"></tbody>
        </table>
      </div>
      <div class="panel">
        <div class="chart-title">最新验证 <span class="legend" id="valSummary"></span></div>
        <table>
          <tbody id="latestTable"></tbody>
        </table>
      </div>
      <div class="panel wide">
        <div class="chart-title">日志尾部 <span class="legend" id="updatedAt"></span></div>
        <pre id="logTail"></pre>
      </div>
    </section>
  </main>

  <script>
    const COLORS = {
      blue: "#2563eb",
      green: "#059669",
      orange: "#d97706",
      red: "#dc2626",
      violet: "#7c3aed",
      gray: "#687385",
      cyan: "#0891b2"
    };
    const chartStore = {};

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || value === "") return "-";
      if (typeof value === "number") return value.toFixed(digits).replace(/\.?0+$/, "");
      return String(value);
    }

    function metric(id, value) {
      document.getElementById(id).textContent = value;
    }

    function getSeries(rows, field) {
      return rows
        .filter(row => Number.isFinite(row.step) && Number.isFinite(row[field]))
        .map(row => ({ x: row.step, y: row[field] }));
    }

    function renderLegend(containerId, seriesList) {
      const el = document.getElementById(containerId);
      if (!el) return;
      el.innerHTML = seriesList.map(series => `
        <span class="legend-item">
          <span class="legend-swatch" style="background:${series.color}"></span>
          <span>${series.name}</span>
          <strong style="color:${series.color}">${fmt(series.latest, series.digits ?? 3)}</strong>
        </span>
      `).join("");
    }

    function niceTicks(minValue, maxValue, count = 5) {
      if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) return [];
      if (minValue === maxValue) return [minValue];
      const ticks = [];
      const step = (maxValue - minValue) / (count - 1);
      for (let i = 0; i < count; i += 1) ticks.push(minValue + step * i);
      return ticks;
    }

    function buildChartModel(canvas, seriesList) {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      const w = rect.width;
      const h = rect.height;
      const pad = { left: 58, right: 22, top: 16, bottom: 40 };
      const points = seriesList.flatMap(s => s.points);
      if (!points.length) {
        return { empty: true, dpr, w, h, pad, seriesList, points };
      }
      let minX = Math.min(...points.map(p => p.x));
      let maxX = Math.max(...points.map(p => p.x));
      let minY = Math.min(...points.map(p => p.y));
      let maxY = Math.max(...points.map(p => p.y));
      if (minX === maxX) maxX = minX + 1;
      if (minY === maxY) {
        minY -= 1;
        maxY += 1;
      }
      const yPad = (maxY - minY) * 0.08 || 1;
      minY -= yPad;
      maxY += yPad;
      const sx = x => pad.left + (x - minX) / (maxX - minX) * (w - pad.left - pad.right);
      const sy = y => h - pad.bottom - (y - minY) / (maxY - minY) * (h - pad.top - pad.bottom);
      return {
        empty: false,
        dpr,
        w,
        h,
        pad,
        seriesList,
        points,
        minX,
        maxX,
        minY,
        maxY,
        sx,
        sy,
        xTicks: niceTicks(minX, maxX, 6),
        yTicks: niceTicks(minY, maxY, 6),
      };
    }

    function nearestStep(model, mouseX) {
      let best = null;
      let bestDist = Infinity;
      const steps = [...new Set(model.points.map(point => point.x))];
      for (const step of steps) {
        const dist = Math.abs(model.sx(step) - mouseX);
        if (dist < bestDist) {
          best = step;
          bestDist = dist;
        }
      }
      return best;
    }

    function drawChart(canvas, seriesList, hoverStep = null) {
      const model = buildChartModel(canvas, seriesList);
      canvas.width = Math.max(1, Math.floor(model.w * model.dpr));
      canvas.height = Math.max(1, Math.floor(model.h * model.dpr));
      const ctx = canvas.getContext("2d");
      ctx.scale(model.dpr, model.dpr);
      ctx.clearRect(0, 0, model.w, model.h);
      if (model.empty) {
        ctx.fillStyle = COLORS.gray;
        ctx.font = "13px Segoe UI";
        ctx.fillText("暂无数据", 16, 28);
        chartStore[canvas.id] = model;
        return model;
      }

      ctx.strokeStyle = "#e5eaf2";
      ctx.lineWidth = 1;
      ctx.fillStyle = COLORS.gray;
      ctx.font = "11px Segoe UI";
      ctx.textBaseline = "middle";
      for (const tick of model.yTicks) {
        const y = model.sy(tick);
        ctx.beginPath();
        ctx.moveTo(model.pad.left, y);
        ctx.lineTo(model.w - model.pad.right, y);
        ctx.stroke();
        ctx.fillText(fmt(tick, 3), 8, y);
      }
      ctx.textBaseline = "alphabetic";
      for (const tick of model.xTicks) {
        const x = model.sx(tick);
        ctx.beginPath();
        ctx.moveTo(x, model.pad.top);
        ctx.lineTo(x, model.h - model.pad.bottom);
        ctx.stroke();
        ctx.fillText(String(Math.round(tick)), x - 10, model.h - 16);
      }

      ctx.strokeStyle = "#b7c1d1";
      ctx.beginPath();
      ctx.moveTo(model.pad.left, model.pad.top);
      ctx.lineTo(model.pad.left, model.h - model.pad.bottom);
      ctx.lineTo(model.w - model.pad.right, model.h - model.pad.bottom);
      ctx.stroke();
      ctx.fillStyle = COLORS.gray;
      ctx.fillText("step", model.w - model.pad.right - 28, model.h - 4);

      for (const series of seriesList) {
        const pts = series.points;
        if (!pts.length) continue;
        ctx.strokeStyle = series.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        pts.forEach((p, i) => {
          const x = model.sx(p.x);
          const y = model.sy(p.y);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.fillStyle = series.color;
        for (const p of pts) {
          ctx.beginPath();
          ctx.arc(model.sx(p.x), model.sy(p.y), 2.4, 0, Math.PI * 2);
          ctx.fill();
        }
        const last = pts[pts.length - 1];
        ctx.fillStyle = series.color;
        ctx.beginPath();
        ctx.arc(model.sx(last.x), model.sy(last.y), 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.font = "11px Segoe UI";
        ctx.fillText(series.name, Math.min(model.sx(last.x) + 6, model.w - model.pad.right - 60), model.sy(last.y) - 4);
      }

      if (hoverStep !== null && hoverStep !== undefined) {
        const x = model.sx(hoverStep);
        ctx.strokeStyle = "#475569";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, model.pad.top);
        ctx.lineTo(x, model.h - model.pad.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        for (const series of seriesList) {
          const point = series.points.find(p => p.x === hoverStep);
          if (!point) continue;
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = series.color;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(model.sx(point.x), model.sy(point.y), 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      }
      chartStore[canvas.id] = model;
      return model;
    }

    function attachChartHover(canvasId, tooltipId) {
      const canvas = document.getElementById(canvasId);
      const tooltip = document.getElementById(tooltipId);
      if (canvas.dataset.hoverReady === "1") return;
      canvas.dataset.hoverReady = "1";
      canvas.addEventListener("mousemove", event => {
        const model = chartStore[canvasId];
        if (!model || model.empty) return;
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const step = nearestStep(model, mouseX);
        drawChart(canvas, model.seriesList, step);
        const rows = model.seriesList
          .map(series => ({ series, point: series.points.find(p => p.x === step) }))
          .filter(item => item.point);
        tooltip.innerHTML = `
          <div class="tooltip-title">step = ${step}</div>
          ${rows.map(({ series, point }) => `
            <div class="tooltip-row">
              <span class="tooltip-name"><span class="tooltip-dot" style="background:${series.color}"></span>${series.name}</span>
              <strong>${fmt(point.y, series.digits ?? 4)}</strong>
            </div>
          `).join("")}
        `;
        const tooltipWidth = tooltip.offsetWidth || 220;
        const left = Math.min(Math.max(8, event.clientX - rect.left + 14), rect.width - tooltipWidth - 8);
        const top = Math.max(8, event.clientY - rect.top - 24);
        tooltip.style.left = `${left}px`;
        tooltip.style.top = `${top}px`;
        tooltip.style.display = "block";
      });
      canvas.addEventListener("mouseleave", () => {
        const model = chartStore[canvasId];
        if (model) drawChart(canvas, model.seriesList, null);
        tooltip.style.display = "none";
      });
    }

    function renderTables(data) {
      const ckpts = (data.checkpoints || []).slice(-8).reverse();
      document.getElementById("ckptTable").innerHTML = ckpts.map(item => `
        <tr>
          <td>${item.name}</td>
          <td>${item.step}</td>
          <td class="${item.has_optimizer ? "ok" : "bad"}">${item.has_optimizer ? "有" : "无"}</td>
          <td class="${item.has_rng_state ? "ok" : "bad"}">${item.has_rng_state ? "有" : "无"}</td>
        </tr>
      `).join("");
      const latest = data.latest_val || {};
      document.getElementById("latestTable").innerHTML = `
        <tr><th>val_reward_mean</th><td>${fmt(latest.val_reward_mean)}</td></tr>
        <tr><th>val_exact_match</th><td>${fmt(latest.val_exact_match)}</td></tr>
        <tr><th>val_format_rate</th><td>${fmt(latest.val_format_rate)}</td></tr>
        <tr><th>val_response_len_mean</th><td>${fmt(latest.val_response_len_mean, 1)}</td></tr>
        <tr><th>val_max_tokens_reached_without_eos_rate</th><td>${fmt(latest.val_max_tokens_reached_without_eos_rate)}</td></tr>
        <tr><th>val_eos_rate</th><td>${fmt(latest.val_eos_rate)}</td></tr>
        <tr><th>val_sample_exact_match</th><td>${fmt(latest.val_sample_exact_match)}</td></tr>
        <tr><th>val_sample_format_rate</th><td>${fmt(latest.val_sample_format_rate)}</td></tr>
        <tr><th>val_sample_max_tokens_reached_without_eos_rate</th><td>${fmt(latest.val_sample_max_tokens_reached_without_eos_rate)}</td></tr>
        <tr><th>best_val_em_so_far</th><td>${fmt(latest.best_val_em_so_far)}</td></tr>
      `;
      document.getElementById("ckptSummary").textContent =
        data.latest_checkpoint ? `${data.latest_checkpoint.name} | ${data.latest_checkpoint_time}` : "暂无";
      document.getElementById("valSummary").textContent =
        Number.isFinite(latest.step) ? `step ${latest.step}` : "暂无";
    }

    async function refresh() {
      const response = await fetch("/api/status", { cache: "no-store" });
      const data = await response.json();
      document.getElementById("title").textContent = data.run_name || "训练监视看板";
      document.getElementById("runDir").textContent = data.run_dir || "";
      metric("step", `${data.latest_step ?? "-"} / ${data.total_steps ?? "-"}`);
      metric("bestEm", fmt(data.best_val_em));
      metric("bestStep", data.best_step === null || data.best_step === undefined ? "best step 未保存" : `best step ${data.best_step}`);
      metric("noImprove", data.steps_no_improve ?? "-");
      const train = data.latest_train || {};
      metric("reward", fmt(train.reward_mean));
      metric("rewardStd", `std ${fmt(train.reward_std)} | len ${fmt(train.response_len_mean, 1)}`);
      metric("kl", `${fmt(train.kl_loss, 5)} / ${fmt(train.approx_kl, 6)}`);
      metric("clip", `coef ${fmt(train.kl_loss_coef, 5)} | clip ${fmt(train.clip_frac, 6)} | grad ${fmt(train.grad_norm, 3)}`);
      const group = data.latest_group || {};
      metric("effectiveGroup", fmt(group.effective_group_rate, 3));
      metric("mixedGroup", `mixed ${fmt(group.mixed_group_rate, 3)} | zero_adv ${fmt(group.zero_advantage_rate, 3)}`);
      const activity = data.activity || {};
      const active = activity.label === "running_or_active";
      const activityEl = document.getElementById("activity");
      activityEl.textContent = active ? "更新中" : (data.training_status || "未更新");
      activityEl.className = `metric-value ${active ? "ok" : "warn"}`;
      const stopText = !active && data.stop_reason ? ` | ${data.stop_reason}` : "";
      metric("logTime", `${data.log_mtime || "-"} | age ${fmt(activity.log_age_seconds, 0)}s${stopText}`);
      document.getElementById("progress").style.width = `${Math.round((data.progress || 0) * 100)}%`;
      document.getElementById("updatedAt").textContent = `刷新时间 ${data.now || "-"}`;
      document.getElementById("logTail").textContent = (data.tail_lines || []).join("\n");

      const trainRows = data.train_rows || [];
      const valRows = data.val_rows || [];
      const groupRows = data.group_rows || [];
      const rewardSeries = [
        { name: "reward_mean", color: COLORS.green, points: getSeries(trainRows, "reward_mean"), latest: train.reward_mean, digits: 4 },
        { name: "reward_std", color: COLORS.orange, points: getSeries(trainRows, "reward_std"), latest: train.reward_std, digits: 4 },
      ];
      const valSeries = [
        { name: "val_exact_match", color: COLORS.blue, points: getSeries(valRows, "val_exact_match"), latest: (data.latest_val || {}).val_exact_match, digits: 4 },
        { name: "val_reward_mean", color: COLORS.green, points: getSeries(valRows, "val_reward_mean"), latest: (data.latest_val || {}).val_reward_mean, digits: 4 },
        { name: "val_format_rate", color: COLORS.violet, points: getSeries(valRows, "val_format_rate"), latest: (data.latest_val || {}).val_format_rate, digits: 4 },
      ];
      const ppoSeries = [
        { name: "update_kl", color: COLORS.blue, points: getSeries(trainRows, "approx_kl"), latest: train.approx_kl, digits: 6 },
        { name: "clip_frac", color: COLORS.red, points: getSeries(trainRows, "clip_frac"), latest: train.clip_frac, digits: 6 },
        { name: "reference_kl", color: COLORS.cyan, points: getSeries(trainRows, "kl_loss"), latest: train.kl_loss, digits: 6 },
        { name: "kl_coef", color: COLORS.orange, points: getSeries(trainRows, "kl_loss_coef"), latest: train.kl_loss_coef, digits: 6 },
      ];
      const formatSeries = [
        { name: "rollout_format_rate", color: COLORS.green, points: getSeries(groupRows, "rollout_format_rate"), latest: group.rollout_format_rate, digits: 4 },
        { name: "rollout_max_tokens_reached_without_eos_rate", color: COLORS.red, points: getSeries(groupRows, "rollout_max_tokens_reached_without_eos_rate"), latest: group.rollout_max_tokens_reached_without_eos_rate, digits: 4 },
        { name: "rollout_eos_rate", color: COLORS.cyan, points: getSeries(groupRows, "rollout_eos_rate"), latest: group.rollout_eos_rate, digits: 4 },
        { name: "val_sample_format_rate", color: COLORS.violet, points: getSeries(valRows, "val_sample_format_rate"), latest: (data.latest_val || {}).val_sample_format_rate, digits: 4 },
      ];
      const timeSeries = [
        { name: "response_len_mean", color: COLORS.violet, points: getSeries(trainRows, "response_len_mean"), latest: train.response_len_mean, digits: 2 },
        { name: "step_time_sec", color: COLORS.orange, points: getSeries(trainRows, "step_time"), latest: train.step_time, digits: 2 },
      ];
      const groupSeries = [
        { name: "effective_group_rate", color: COLORS.green, points: getSeries(groupRows, "effective_group_rate"), latest: group.effective_group_rate, digits: 4 },
        { name: "mixed_group_rate", color: COLORS.blue, points: getSeries(groupRows, "mixed_group_rate"), latest: group.mixed_group_rate, digits: 4 },
        { name: "zero_advantage_rate", color: COLORS.red, points: getSeries(groupRows, "zero_advantage_rate"), latest: group.zero_advantage_rate, digits: 4 },
      ];
      renderLegend("rewardLegend", rewardSeries);
      renderLegend("valLegend", valSeries);
      renderLegend("ppoLegend", ppoSeries);
      renderLegend("formatLegend", formatSeries);
      renderLegend("timeLegend", timeSeries);
      renderLegend("groupLegend", groupSeries);
      drawChart(document.getElementById("rewardChart"), rewardSeries);
      drawChart(document.getElementById("valChart"), valSeries);
      drawChart(document.getElementById("ppoChart"), ppoSeries);
      drawChart(document.getElementById("formatChart"), formatSeries);
      drawChart(document.getElementById("timeChart"), timeSeries);
      drawChart(document.getElementById("groupChart"), groupSeries);
      attachChartHover("rewardChart", "rewardTooltip");
      attachChartHover("valChart", "valTooltip");
      attachChartHover("ppoChart", "ppoTooltip");
      attachChartHover("formatChart", "formatTooltip");
      attachChartHover("timeChart", "timeTooltip");
      attachChartHover("groupChart", "groupTooltip");
      renderTables(data);
      window.__refreshSeconds = data.refresh_seconds || window.__refreshSeconds || 5;
    }

    async function loop() {
      try {
        await refresh();
      } catch (error) {
        document.getElementById("logTail").textContent = String(error);
      }
      setTimeout(loop, (window.__refreshSeconds || 5) * 1000);
    }

    window.addEventListener("resize", () => refresh());
    loop();
  </script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    config: DashboardConfig

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self) -> None:
        body = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            query = parse_qs(parsed.query)
            config = self.config
            if "run_dir" in query:
                run_dir = Path(query["run_dir"][0])
                if not run_dir.is_absolute():
                    run_dir = PROJECT_ROOT / run_dir
                config = DashboardConfig(
                    run_dir=run_dir,
                    refresh_seconds=config.refresh_seconds,
                    tail_lines=config.tail_lines,
                )
            self._send_json(build_status(config))
            return
        if parsed.path in {"/", "/index.html"}:
            self._send_html()
            return
        self.send_error(404, "Not Found")

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动本地训练监视看板")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="训练输出目录，例如 models/grpo/xxx",
    )
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口")
    parser.add_argument("--refresh-seconds", type=int, default=5, help="页面刷新间隔")
    parser.add_argument("--tail-lines", type=int, default=80, help="日志尾部显示行数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    config = DashboardConfig(
        run_dir=run_dir,
        refresh_seconds=max(1, args.refresh_seconds),
        tail_lines=max(1, args.tail_lines),
    )
    DashboardHandler.config = config
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"训练监视看板已启动: {url}", flush=True)
    print(f"监视目录: {config.run_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n训练监视看板已停止", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
