"""串行执行并汇总C0/L1多seed扩大sample评估。"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
SINGLE_EVAL_ENTRY = FRAMEWORK_ROOT / "scripts" / "run_grpo_sample_eval.py"
sys.path.insert(0, str(FRAMEWORK_ROOT / "src"))

from ptf.sample_eval import BINARY_METRICS, paired_prompt_bootstrap


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON顶层必须是object: {path}")
    return data


def _workspace_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_event(path: Path, event: dict[str, Any]) -> None:
    record = {"time": datetime.now().astimezone().isoformat(timespec="seconds"), **event}
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                data = json.loads(line)
                if isinstance(data, dict):
                    rows.append(data)
    return rows


def validate_manifest(manifest: dict[str, Any]) -> None:
    """检查评估协议、路径和C0/L1配对完整性。"""
    required = (
        "suite_name",
        "config",
        "output_root",
        "python_executable",
        "sampling",
        "thresholds",
        "trials",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"评估矩阵缺少字段: {missing}")
    if not _workspace_path(manifest["config"]).exists():
        raise FileNotFoundError(_workspace_path(manifest["config"]))
    trial_ids: set[str] = set()
    pairs: set[tuple[int, str]] = set()
    for trial in manifest["trials"]:
        trial_id = str(trial["id"])
        if trial_id in trial_ids:
            raise ValueError(f"重复trial id: {trial_id}")
        trial_ids.add(trial_id)
        variant = str(trial["variant"])
        training_seed = int(trial["training_seed"])
        pairs.add((training_seed, variant))
        checkpoint = _workspace_path(trial["checkpoint"])
        for filename in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
            if not (checkpoint / filename).exists():
                raise FileNotFoundError(checkpoint / filename)
    seeds = sorted({seed for seed, _ in pairs})
    for seed in seeds:
        for variant in ("c0", "l1"):
            if (seed, variant) not in pairs:
                raise ValueError(f"seed={seed}缺少{variant}配对")


def _expected_responses(manifest: dict[str, Any]) -> int:
    sampling = manifest["sampling"]
    return (
        int(sampling["max_items"])
        * int(sampling["return_sequences"])
        * len(sampling["eval_seeds"])
    )


def _trial_dir(manifest: dict[str, Any], trial: dict[str, Any]) -> Path:
    return _workspace_path(manifest["output_root"]) / str(trial["id"])


def _trial_complete(manifest: dict[str, Any], trial: dict[str, Any]) -> bool:
    summary_path = _trial_dir(manifest, trial) / "summary.json"
    rows_path = _trial_dir(manifest, trial) / "rows.jsonl"
    if not summary_path.exists() or not rows_path.exists():
        return False
    try:
        summary = _load_json(summary_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        summary.get("status") == "completed"
        and int(summary.get("full", {}).get("responses", 0)) == _expected_responses(manifest)
        and Path(str(summary.get("checkpoint", ""))).resolve()
        == _workspace_path(trial["checkpoint"])
    )


def write_protocol(manifest: dict[str, Any]) -> Path:
    """在查看新结果前写出不可回溯修改的评估协议。"""
    output_root = _workspace_path(manifest["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "protocol.md"
    if path.exists():
        return path
    sampling = manifest["sampling"]
    thresholds = manifest["thresholds"]
    expected = _expected_responses(manifest)
    lines = [
        f"# {manifest['suite_name']} 扩大sample评估协议",
        "",
        f"> 登记时间：{datetime.now().astimezone().isoformat(timespec='seconds')}。本协议在读取任何新sample结果前写入。",
        "",
        "## 目标",
        "",
        str(manifest["description"]),
        "",
        "本轮冻结全部checkpoint，只做前向生成。它不能反向修改上一轮Confirm50不晋级结论，只为下一轮协议提供更稳定的格式、截顶和sample EM证据。",
        "",
        "## 固定设计",
        "",
        f"- 每模型固定评估前{sampling['max_items']}道eval题。",
        f"- 每题每个评估seed生成{sampling['return_sequences']}条，共{len(sampling['eval_seeds'])}个评估seed。",
        f"- 每模型总回答数：{expected}；6个模型总回答数：{expected * len(manifest['trials'])}。",
        f"- 评估seeds：{sampling['eval_seeds']}；所有C0/L1使用相同seed、题目、batch和return index。",
        f"- temperature/top-p/top-k：{sampling['temperature']}/{sampling['top_p']}/{sampling['top_k']}。",
        f"- max response tokens：{sampling['max_response_tokens']}；batch size：{sampling['eval_batch_size']}。",
        "- 主分析使用完整eval100；前10题只作为旧80条sample的复现切片，不参与晋级计数。",
        "",
        "## 统计与门槛",
        "",
        f"- 单模型二项比例使用{float(thresholds['confidence']) * 100:.0f}% Wilson区间。",
        f"- C0/L1差值按prompt聚类bootstrap {thresholds['bootstrap_samples']}次。",
        f"- L1格式安全：Wilson下界 >= {thresholds['format_min']}。",
        f"- L1截顶安全：Wilson上界 <= {thresholds['hit_max_max']}。",
        f"- L1相对C0格式非劣：差值区间下界 >= {thresholds['format_delta_min']}。",
        f"- L1相对C0截顶非劣：差值区间上界 <= {thresholds['hit_max_delta_max']}。",
        f"- L1相对C0 sample EM非劣：差值区间下界 >= {thresholds['em_delta_min']}。",
        f"- 至少{thresholds['min_support_seeds']}/3训练seed全部满足上述条件，才恢复L2/K3设计。",
        "- 区间跨越门槛记为“不确定”，不按通过处理；不得只看点估计挑选seed。",
        "",
        "## 输出",
        "",
        "- `<trial>/rows.jsonl`：逐回答原文和格式/截顶/EM证据。",
        "- `<trial>/summary.json|md`：单模型比例与Wilson区间。",
        "- `trials.csv`、`pairs.csv`、`decision.json|md`：统一汇总与晋级判定。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _build_command(manifest: dict[str, Any], trial: dict[str, Any]) -> list[str]:
    sampling = manifest["sampling"]
    command = [
        str(Path(manifest["python_executable"])),
        "-B",
        str(SINGLE_EVAL_ENTRY),
        "--config",
        str(_workspace_path(manifest["config"])),
        "--checkpoint",
        str(_workspace_path(trial["checkpoint"])),
        "--output-dir",
        str(_trial_dir(manifest, trial)),
        "--trial-id",
        str(trial["id"]),
        "--variant",
        str(trial["variant"]),
        "--training-seed",
        str(trial["training_seed"]),
        "--max-items",
        str(sampling["max_items"]),
        "--max-response-tokens",
        str(sampling["max_response_tokens"]),
        "--eval-batch-size",
        str(sampling["eval_batch_size"]),
        "--return-sequences",
        str(sampling["return_sequences"]),
        "--eval-seeds",
        *[str(seed) for seed in sampling["eval_seeds"]],
        "--temperature",
        str(sampling["temperature"]),
        "--top-p",
        str(sampling["top_p"]),
        "--top-k",
        str(sampling["top_k"]),
        "--max-prompt-length",
        str(sampling["max_prompt_length"]),
        "--format-instruction",
        str(sampling["format_instruction"]),
    ]
    if bool(sampling.get("enable_thinking", False)):
        command.append("--enable-thinking")
    return command


def run_matrix(
    manifest: dict[str, Any], selected: set[str] | None, force: bool
) -> int:
    """串行启动所有未完成评估，并记录编排事件。"""
    output_root = _workspace_path(manifest["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    events_path = output_root / "events.jsonl"
    failures = 0
    for trial in manifest["trials"]:
        trial_id = str(trial["id"])
        if selected and trial_id not in selected:
            continue
        if not force and _trial_complete(manifest, trial):
            print(f"[跳过完成项] {trial_id}")
            _append_event(events_path, {"event": "skip_complete", "trial_id": trial_id})
            continue
        trial_dir = _trial_dir(manifest, trial)
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "run.log"
        command = _build_command(manifest, trial)
        _append_event(
            events_path,
            {
                "event": "start",
                "trial_id": trial_id,
                "checkpoint": str(_workspace_path(trial["checkpoint"])),
                "command": command,
            },
        )
        print(f"[启动] {trial_id} -> {trial_dir}", flush=True)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as log_file:
            result = subprocess.run(
                command,
                cwd=WORKSPACE_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=environment,
                check=False,
            )
        complete = _trial_complete(manifest, trial)
        _append_event(
            events_path,
            {
                "event": "finish",
                "trial_id": trial_id,
                "return_code": result.returncode,
                "complete": complete,
                "log": str(log_path),
            },
        )
        print(
            f"[结束] {trial_id} return_code={result.returncode} complete={complete}",
            flush=True,
        )
        if result.returncode != 0 or not complete:
            failures += 1
            break
    return 1 if failures else 0


def _absolute_min_status(metric: dict[str, Any], threshold: float) -> str:
    if float(metric["wilson_low"]) >= threshold:
        return "pass"
    if float(metric["wilson_high"]) < threshold:
        return "fail"
    return "uncertain"


def _absolute_max_status(metric: dict[str, Any], threshold: float) -> str:
    if float(metric["wilson_high"]) <= threshold:
        return "pass"
    if float(metric["wilson_low"]) > threshold:
        return "fail"
    return "uncertain"


def _delta_min_status(metric: dict[str, Any], threshold: float) -> str:
    if float(metric["ci_low"]) >= threshold:
        return "pass"
    if float(metric["ci_high"]) < threshold:
        return "fail"
    return "uncertain"


def _delta_max_status(metric: dict[str, Any], threshold: float) -> str:
    if float(metric["ci_high"]) <= threshold:
        return "pass"
    if float(metric["ci_low"]) > threshold:
        return "fail"
    return "uncertain"


def summarize_matrix(manifest: dict[str, Any]) -> dict[str, Any]:
    """汇总六个trial，生成配对区间和新协议判定。"""
    output_root = _workspace_path(manifest["output_root"])
    incomplete = [
        str(trial["id"])
        for trial in manifest["trials"]
        if not _trial_complete(manifest, trial)
    ]
    if incomplete:
        raise RuntimeError(f"仍有未完成trial，不能汇总: {incomplete}")

    trial_summaries: dict[str, dict[str, Any]] = {}
    trial_rows: dict[str, list[dict[str, Any]]] = {}
    table_rows: list[dict[str, Any]] = []
    for trial in manifest["trials"]:
        trial_id = str(trial["id"])
        summary = _load_json(_trial_dir(manifest, trial) / "summary.json")
        rows = _read_rows(_trial_dir(manifest, trial) / "rows.jsonl")
        trial_summaries[trial_id] = summary
        trial_rows[trial_id] = rows
        full = summary["full"]["metrics"]
        first10 = summary["first10"]["metrics"]
        table_rows.append(
            {
                "trial_id": trial_id,
                "variant": trial["variant"],
                "training_seed": trial["training_seed"],
                "responses": summary["full"]["responses"],
                "sample_em": full["exact_match"]["rate"],
                "sample_em_low": full["exact_match"]["wilson_low"],
                "sample_em_high": full["exact_match"]["wilson_high"],
                "format_rate": full["format_ok"]["rate"],
                "format_low": full["format_ok"]["wilson_low"],
                "format_high": full["format_ok"]["wilson_high"],
                "hit_max_rate": full["reached_max_tokens_without_eos"]["rate"],
                "hit_max_low": full["reached_max_tokens_without_eos"]["wilson_low"],
                "hit_max_high": full["reached_max_tokens_without_eos"]["wilson_high"],
                "eos_rate": full["terminated_by_eos"]["rate"],
                "first10_format_rate": first10["format_ok"]["rate"],
                "first10_hit_max_rate": first10["reached_max_tokens_without_eos"]["rate"],
                "seconds": summary["seconds"],
                "checkpoint": summary["checkpoint"],
                "adapter_sha256": summary["adapter_sha256"],
            }
        )
    trials_csv = output_root / "trials.csv"
    with trials_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    thresholds = manifest["thresholds"]
    paired_rows: list[dict[str, Any]] = []
    seeds = sorted({int(trial["training_seed"]) for trial in manifest["trials"]})
    by_seed_variant = {
        (int(trial["training_seed"]), str(trial["variant"])): str(trial["id"])
        for trial in manifest["trials"]
    }
    for training_seed in seeds:
        control_id = by_seed_variant[(training_seed, "c0")]
        candidate_id = by_seed_variant[(training_seed, "l1")]
        bootstrap: dict[str, dict[str, Any]] = {}
        for offset, metric in enumerate(
            ("exact_match", "format_ok", "reached_max_tokens_without_eos")
        ):
            bootstrap[metric] = paired_prompt_bootstrap(
                trial_rows[control_id],
                trial_rows[candidate_id],
                metric=metric,
                samples=int(thresholds["bootstrap_samples"]),
                seed=int(thresholds["bootstrap_seed"]) + training_seed + offset,
                confidence=float(thresholds["confidence"]),
            )
        candidate_metrics = trial_summaries[candidate_id]["full"]["metrics"]
        format_absolute = _absolute_min_status(
            candidate_metrics["format_ok"], float(thresholds["format_min"])
        )
        hit_max_absolute = _absolute_max_status(
            candidate_metrics["reached_max_tokens_without_eos"],
            float(thresholds["hit_max_max"]),
        )
        format_relative = _delta_min_status(
            bootstrap["format_ok"], float(thresholds["format_delta_min"])
        )
        hit_max_relative = _delta_max_status(
            bootstrap["reached_max_tokens_without_eos"],
            float(thresholds["hit_max_delta_max"]),
        )
        em_relative = _delta_min_status(
            bootstrap["exact_match"], float(thresholds["em_delta_min"])
        )
        statuses = (
            format_absolute,
            hit_max_absolute,
            format_relative,
            hit_max_relative,
            em_relative,
        )
        overall = "pass" if all(item == "pass" for item in statuses) else (
            "fail" if any(item == "fail" for item in statuses) else "uncertain"
        )
        paired_rows.append(
            {
                "training_seed": training_seed,
                "control": control_id,
                "candidate": candidate_id,
                "candidate_format_rate": candidate_metrics["format_ok"]["rate"],
                "candidate_format_low": candidate_metrics["format_ok"]["wilson_low"],
                "candidate_format_high": candidate_metrics["format_ok"]["wilson_high"],
                "candidate_hit_max_rate": candidate_metrics["reached_max_tokens_without_eos"]["rate"],
                "candidate_hit_max_low": candidate_metrics["reached_max_tokens_without_eos"]["wilson_low"],
                "candidate_hit_max_high": candidate_metrics["reached_max_tokens_without_eos"]["wilson_high"],
                "delta_sample_em": bootstrap["exact_match"]["delta"],
                "delta_sample_em_low": bootstrap["exact_match"]["ci_low"],
                "delta_sample_em_high": bootstrap["exact_match"]["ci_high"],
                "delta_format": bootstrap["format_ok"]["delta"],
                "delta_format_low": bootstrap["format_ok"]["ci_low"],
                "delta_format_high": bootstrap["format_ok"]["ci_high"],
                "delta_hit_max": bootstrap["reached_max_tokens_without_eos"]["delta"],
                "delta_hit_max_low": bootstrap["reached_max_tokens_without_eos"]["ci_low"],
                "delta_hit_max_high": bootstrap["reached_max_tokens_without_eos"]["ci_high"],
                "format_absolute": format_absolute,
                "hit_max_absolute": hit_max_absolute,
                "format_relative": format_relative,
                "hit_max_relative": hit_max_relative,
                "sample_em_relative": em_relative,
                "overall": overall,
            }
        )
    pairs_csv = output_root / "pairs.csv"
    with pairs_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    support = sum(1 for row in paired_rows if row["overall"] == "pass")
    suite_pass = support >= int(thresholds["min_support_seeds"])
    decision = {
        "schema_version": 1,
        "suite_name": manifest["suite_name"],
        "status": "pass" if suite_pass else "not_pass",
        "supporting_seeds": support,
        "total_training_seeds": len(paired_rows),
        "required_supporting_seeds": int(thresholds["min_support_seeds"]),
        "paired_results": paired_rows,
        "next_action": (
            "可登记L2/K3单变量实验，但不得同时修改两个主变量。"
            if suite_pass
            else "不得启动L2/K3；先根据fail/uncertain项继续格式或评估方差诊断。"
        ),
        "protocol": str((output_root / "protocol.md").resolve()),
        "trials_csv": str(trials_csv.resolve()),
        "pairs_csv": str(pairs_csv.resolve()),
    }
    _write_json(output_root / "decision.json", decision)
    _write_decision_markdown(output_root / "decision.md", decision)
    return decision


def _write_decision_markdown(path: Path, decision: dict[str, Any]) -> None:
    """写出便于复盘的最终判定。"""
    lines = [
        "# 扩大sample评估判定",
        "",
        f"总体状态：`{decision['status']}`；通过seed数="
        f"{decision['supporting_seeds']}/{decision['total_training_seeds']}"
        f"（最低要求{decision['required_supporting_seeds']}）。",
        "",
        "| seed | L1格式率 [CI] | L1截顶率 [CI] | Δsample EM [CI] | Δ格式 [CI] | Δ截顶 [CI] | 绝对格式/截顶 | 相对EM/格式/截顶 | 总判定 |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for row in decision["paired_results"]:
        lines.append(
            f"| {row['training_seed']} | {row['candidate_format_rate']:.4f} "
            f"[{row['candidate_format_low']:.4f}, {row['candidate_format_high']:.4f}] | "
            f"{row['candidate_hit_max_rate']:.4f} "
            f"[{row['candidate_hit_max_low']:.4f}, {row['candidate_hit_max_high']:.4f}] | "
            f"{row['delta_sample_em']:+.4f} [{row['delta_sample_em_low']:+.4f}, {row['delta_sample_em_high']:+.4f}] | "
            f"{row['delta_format']:+.4f} [{row['delta_format_low']:+.4f}, {row['delta_format_high']:+.4f}] | "
            f"{row['delta_hit_max']:+.4f} [{row['delta_hit_max_low']:+.4f}, {row['delta_hit_max_high']:+.4f}] | "
            f"{row['format_absolute']}/{row['hit_max_absolute']} | "
            f"{row['sample_em_relative']}/{row['format_relative']}/{row['hit_max_relative']} | "
            f"{row['overall']} |"
        )
    lines.extend(
        [
            "",
            f"下一步：{decision['next_action']}",
            "",
            "判定必须与`protocol.md`一起阅读。`uncertain`表示置信区间跨越预注册边界，不等同于通过或确定失败。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO扩大sample矩阵评估")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="真正串行执行GPU评估")
    parser.add_argument("--force", action="store_true", help="覆盖已完成trial")
    parser.add_argument("--only", nargs="*", help="只执行指定trial id")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.manifest.resolve())
    validate_manifest(manifest)
    protocol = write_protocol(manifest)
    print(f"协议: {protocol}")
    if args.summarize_only:
        decision = summarize_matrix(manifest)
        print(f"判定: {decision['status']} -> {_workspace_path(manifest['output_root']) / 'decision.md'}")
        return 0
    selected = set(args.only or []) or None
    for trial in manifest["trials"]:
        if selected and str(trial["id"]) not in selected:
            continue
        status = "complete" if _trial_complete(manifest, trial) else "pending"
        print(f"{trial['id']}: {status} -> {_trial_dir(manifest, trial)}")
    if not args.execute:
        print("未传入--execute，只输出计划。")
        return 0
    return_code = run_matrix(manifest, selected=selected, force=args.force)
    if return_code != 0:
        return return_code
    remaining = [trial for trial in manifest["trials"] if not _trial_complete(manifest, trial)]
    if not remaining:
        decision = summarize_matrix(manifest)
        print(f"判定: {decision['status']} -> {_workspace_path(manifest['output_root']) / 'decision.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
