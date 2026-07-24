"""GRPO 控制变量实验的计划、串行运行与汇总入口。

默认只打印计划。真正启动 GPU 训练必须显式传入 ``--execute``，并且同一张 GPU
上的实验始终串行执行，避免多个 actor/reference 同时占用显存。
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = FRAMEWORK_ROOT.parent
DEFAULT_MANIFEST = (
    FRAMEWORK_ROOT
    / "configs"
    / "gsm8k_qwen3_0d6b_grpo_v7_causal_matrix.json"
)
TRAIN_ENTRY = FRAMEWORK_ROOT / "scripts" / "run_grpo_train.py"


@dataclass
class TrialPlan:
    """一次 variant × seed × phase 的完整执行计划。"""

    variant_id: str
    seed: int
    phase_id: str
    target_steps: int
    trial_dir: Path
    config_path: Path
    config: dict[str, Any]
    resume_checkpoint: Path
    expected_checkpoint: Path
    status: str
    detail: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是 object: {path}")
    return data


def _workspace_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _split_values(values: Iterable[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or []:
        result.extend(item.strip() for item in value.split(",") if item.strip())
    return result


def _phase_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in manifest["phases"]}


def _variant_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["id"]): item for item in manifest["variants"]}


def validate_manifest(manifest: dict[str, Any]) -> None:
    """在生成命令前检查矩阵结构与关键实验卫生条件。"""
    required = (
        "suite_name",
        "base_train_config",
        "source_checkpoint",
        "output_root",
        "phases",
        "variants",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(f"实验矩阵缺少字段: {missing}")

    base_path = _workspace_path(manifest["base_train_config"])
    source_checkpoint = _workspace_path(manifest["source_checkpoint"])
    if not base_path.exists():
        raise FileNotFoundError(f"基础训练配置不存在: {base_path}")
    if not source_checkpoint.exists():
        raise FileNotFoundError(f"源 checkpoint 不存在: {source_checkpoint}")
    for filename in ("adapter_config.json", "adapter_model.safetensors", "trainer_state.json"):
        if not (source_checkpoint / filename).exists():
            raise FileNotFoundError(f"源 checkpoint 缺少 {filename}: {source_checkpoint}")

    base = _load_json(base_path)
    grpo = base.get("grpo", {})
    configured_source = _workspace_path(grpo.get("resume_from_checkpoint", ""))
    if configured_source != source_checkpoint:
        raise ValueError(
            "基础配置与矩阵的 source_checkpoint 不一致: "
            f"{configured_source} != {source_checkpoint}"
        )
    required_paths = {
        "base model": base.get("model", {}).get("base_model_dir"),
        "SFT adapter": base.get("model", {}).get("sft_adapter_dir"),
        "train parquet": base.get("dataset", {}).get("train_file"),
        "eval parquet": base.get("dataset", {}).get("eval_file"),
    }
    for label, raw_path in required_paths.items():
        if not raw_path or not _workspace_path(raw_path).exists():
            raise FileNotFoundError(f"{label} 不存在: {raw_path}")
    hygiene = {
        "resume_state_mode": "weights_only",
        "deterministic_prompt_sampling": True,
        "val_before_train": True,
        "gradient_accumulation_steps": 1,
    }
    for key, expected in hygiene.items():
        if grpo.get(key) != expected:
            raise ValueError(f"基础配置实验卫生条件不满足: grpo.{key} 应为 {expected!r}")

    phases = manifest["phases"]
    phase_ids = [str(item["id"]) for item in phases]
    if len(phase_ids) != len(set(phase_ids)):
        raise ValueError("phase id 不能重复")
    seen: set[str] = set()
    previous_target = 0
    for phase in phases:
        phase_id = str(phase["id"])
        target = int(phase["target_steps"])
        parent = phase.get("parent")
        if target <= previous_target:
            raise ValueError("phase target_steps 必须严格递增")
        if parent is not None and str(parent) not in seen:
            raise ValueError(f"phase {phase_id} 的 parent 必须出现在它之前")
        previous_target = target
        seen.add(phase_id)

    variants = manifest["variants"]
    variant_ids = [str(item["id"]) for item in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("variant id 不能重复")
    control_variant_id = str(manifest.get("control_variant_id", "c0_fresh_control"))
    if not variants or str(variants[0].get("id")) != control_variant_id:
        raise ValueError(f"第一个 variant 必须是控制组 {control_variant_id}")
    for variant in variants:
        changed = list(variant.get("changed_keys", []))
        override_keys = list(variant.get("overrides", {}).keys())
        if changed != override_keys:
            raise ValueError(
                f"{variant['id']} 的 changed_keys 必须与 overrides 按相同顺序完全一致"
            )
        unknown_keys = set(override_keys) - set(grpo)
        if unknown_keys:
            raise ValueError(f"{variant['id']} 覆盖了基础 GRPO 配置中不存在的字段: {unknown_keys}")
        if (
            variant.get("overrides", {}).get("resume_state_mode") == "weights_and_optimizer"
            and not (source_checkpoint / "optimizer.pt").exists()
        ):
            raise FileNotFoundError("O1 需要源 checkpoint 含 optimizer.pt")


def select_variants(
    manifest: dict[str, Any],
    variant_values: Iterable[str] | None,
    tier_values: Iterable[str] | None,
    default_tier: str | None,
) -> list[dict[str, Any]]:
    requested_ids = set(_split_values(variant_values))
    requested_tiers = set(_split_values(tier_values))
    if not requested_ids and not requested_tiers and default_tier:
        requested_tiers = {default_tier}

    all_variants = manifest["variants"]
    known_ids = {str(item["id"]) for item in all_variants}
    unknown = requested_ids - known_ids
    if unknown:
        raise ValueError(f"未知 variant: {sorted(unknown)}")
    if not requested_ids and not requested_tiers:
        return list(all_variants)
    return [
        item
        for item in all_variants
        if str(item["id"]) in requested_ids or str(item.get("tier")) in requested_tiers
    ]


def planned_updates_per_step(grpo: dict[str, Any]) -> int:
    """按配置计算每个 GRPO step 计划执行的 optimizer 更新次数。"""
    rollout_count = int(grpo["train_batch_size"]) * int(grpo["rollout_n"])
    mini_batches = math.ceil(rollout_count / int(grpo["ppo_mini_batch_size"]))
    updates_per_epoch = math.ceil(
        mini_batches / int(grpo.get("gradient_accumulation_steps", 1))
    )
    return int(grpo["ppo_epochs"]) * updates_per_epoch


def _checkpoint_completed(checkpoint: Path, target_steps: int) -> bool:
    state_path = checkpoint / "trainer_state.json"
    if not state_path.exists():
        return False
    try:
        state = _load_json(state_path)
        return int(state.get("next_step", 0)) >= target_steps
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _checkpoint_steps(trial_dir: Path) -> list[int]:
    steps: list[int] = []
    for path in trial_dir.glob("checkpoint-*") if trial_dir.exists() else []:
        try:
            steps.append(int(path.name.rsplit("-", 1)[1]))
        except ValueError:
            continue
    return sorted(steps)


def build_trial_plan(
    manifest: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    phase_id: str,
) -> TrialPlan:
    phases = _phase_map(manifest)
    if phase_id not in phases:
        raise ValueError(f"未知 phase: {phase_id}")
    phase = phases[phase_id]
    target_steps = int(phase["target_steps"])

    base = _load_json(_workspace_path(manifest["base_train_config"]))
    config = deepcopy(base)
    grpo = config.setdefault("grpo", {})
    grpo.update(deepcopy(variant.get("overrides", {})))

    suite_name = str(manifest["suite_name"])
    variant_id = str(variant["id"])
    output_root = _workspace_path(manifest["output_root"])
    trial_dir = output_root / variant_id / f"seed-{seed}"
    run_name = f"{suite_name}__{variant_id}__seed{seed}"
    config["experiment_name"] = run_name
    grpo["run_name"] = run_name
    grpo["output_dir"] = _display_path(trial_dir)
    grpo["seed"] = int(seed)
    grpo["prompt_sampling_seed"] = int(seed)
    grpo["total_training_steps"] = target_steps

    parent_id = phase.get("parent")
    if parent_id is None:
        resume_checkpoint = _workspace_path(manifest["source_checkpoint"])
        grpo["resume_from_checkpoint"] = _display_path(resume_checkpoint)
        grpo["allow_resume_objective_change"] = True
    else:
        parent = phases[str(parent_id)]
        resume_checkpoint = trial_dir / f"checkpoint-{int(parent['target_steps']) - 1}"
        grpo["resume_from_checkpoint"] = _display_path(resume_checkpoint)
        grpo["resume_state_mode"] = "full"
        grpo["allow_resume_objective_change"] = False

    config_path = (
        output_root
        / "_orchestration"
        / "configs"
        / f"{variant_id}__seed{seed}__{phase_id}.json"
    )
    expected_checkpoint = trial_dir / f"checkpoint-{target_steps - 1}"
    existing_steps = _checkpoint_steps(trial_dir)

    parent_checkpoint_step = None
    if parent_id is not None:
        parent_checkpoint_step = int(phases[str(parent_id)]["target_steps"]) - 1

    if _checkpoint_completed(expected_checkpoint, target_steps):
        status, detail = "complete", f"已达到 {target_steps} steps"
    elif parent_id is not None and not resume_checkpoint.exists():
        status, detail = "blocked", f"缺少父阶段 checkpoint: {resume_checkpoint.name}"
    elif existing_steps and (
        parent_checkpoint_step is None or max(existing_steps) > parent_checkpoint_step
    ):
        status, detail = "partial", f"已有 checkpoint steps={existing_steps}，拒绝隐式覆盖"
    elif parent_id is None and trial_dir.exists() and any(trial_dir.iterdir()):
        status, detail = "dirty", "目录已有非 checkpoint 产物，需人工核对"
    else:
        status, detail = "ready", "可以启动"

    return TrialPlan(
        variant_id=variant_id,
        seed=seed,
        phase_id=phase_id,
        target_steps=target_steps,
        trial_dir=trial_dir,
        config_path=config_path,
        config=config,
        resume_checkpoint=resume_checkpoint,
        expected_checkpoint=expected_checkpoint,
        status=status,
        detail=detail,
    )


def _write_trial_config(plan: TrialPlan) -> None:
    plan.config_path.parent.mkdir(parents=True, exist_ok=True)
    with plan.config_path.open("w", encoding="utf-8") as file:
        json.dump(plan.config, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _command_for_plan(manifest: dict[str, Any], plan: TrialPlan) -> list[str]:
    python_executable = str(_workspace_path(manifest.get("python_executable", sys.executable)))
    return [
        python_executable,
        str(TRAIN_ENTRY),
        "--config",
        str(plan.config_path),
    ]


def print_plan(manifest: dict[str, Any], plans: list[TrialPlan]) -> None:
    print(f"suite: {manifest['suite_name']}")
    print(f"source: {_display_path(_workspace_path(manifest['source_checkpoint']))}")
    print("variant | seed | phase | mode | prompts | ppo | lr | KL coef/interval | updates | status")
    for plan in plans:
        grpo = plan.config["grpo"]
        print(
            f"{plan.variant_id} | {plan.seed} | {plan.phase_id} | "
            f"{grpo['resume_state_mode']} | {grpo['train_batch_size']} | "
            f"{grpo['ppo_epochs']} | {float(grpo['learning_rate']):.1e} | "
            f"{float(grpo['kl_loss_coef']):.3g}/{grpo['adaptive_kl_interval']} | "
            f"{planned_updates_per_step(grpo)} | {plan.status}: {plan.detail}"
        )


def _append_event(output_root: Path, payload: dict[str, Any]) -> None:
    path = output_root / "_orchestration" / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_plans(manifest: dict[str, Any], plans: list[TrialPlan], execute: bool) -> int:
    output_root = _workspace_path(manifest["output_root"])
    failures = 0
    for plan in plans:
        command = _command_for_plan(manifest, plan)
        if plan.status == "complete":
            print(f"[跳过] {plan.variant_id} seed={plan.seed}: {plan.detail}")
            continue
        if plan.status != "ready":
            print(f"[阻止] {plan.variant_id} seed={plan.seed}: {plan.status} - {plan.detail}")
            failures += 1
            continue
        if not execute:
            print("[dry-run] " + subprocess.list2cmdline(command))
            continue

        _write_trial_config(plan)
        log_path = plan.trial_dir / "logs" / f"{plan.phase_id}.orchestrator.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_event(
            output_root,
            {
                "event": "start",
                "time": started_at,
                "variant": plan.variant_id,
                "seed": plan.seed,
                "phase": plan.phase_id,
                "config": _display_path(plan.config_path),
            },
        )
        print(f"[启动] {plan.variant_id} seed={plan.seed} phase={plan.phase_id}")
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(WORKSPACE_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()

        completed = _checkpoint_completed(plan.expected_checkpoint, plan.target_steps)
        result = "complete" if return_code == 0 and completed else "failed_or_guarded"
        _append_event(
            output_root,
            {
                "event": "finish",
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "variant": plan.variant_id,
                "seed": plan.seed,
                "phase": plan.phase_id,
                "return_code": return_code,
                "target_checkpoint_exists": completed,
                "result": result,
            },
        )
        if result != "complete":
            print(
                f"[未达标] {plan.variant_id} seed={plan.seed}: "
                f"return_code={return_code}, target_checkpoint={completed}"
            )
            failures += 1
    return 1 if failures else 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    """计算中位数，空列表返回0。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _slope(rows: list[dict[str, str]], key: str, tail: int = 10) -> float:
    points = [(int(_number(row, "step")), _number(row, key)) for row in rows[-tail:]]
    if len(points) < 2:
        return 0.0
    x_mean = _mean([float(x) for x, _ in points])
    y_mean = _mean([y for _, y in points])
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in points) / denominator


def _first_crossing(rows: list[dict[str, str]], key: str, threshold: float) -> str:
    for row in rows:
        if _number(row, key) > threshold:
            return str(int(_number(row, "step")))
    return ""


def _extract_stop_reason(trial_dir: Path) -> str:
    stop_summary_path = trial_dir / "training_stop.json"
    if stop_summary_path.exists():
        structured = _load_json(stop_summary_path)
        decision = structured.get("decision", {})
        if isinstance(decision, dict) and decision.get("reason"):
            return str(decision["reason"])
    reasons: list[str] = []
    for path in sorted((trial_dir / "logs").glob("*.log")) if (trial_dir / "logs").exists() else []:
        try:
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
                if "训练结束, 原因:" in line:
                    reasons.append(line.split("训练结束, 原因:", 1)[1].strip())
        except OSError:
            continue
    return reasons[-1] if reasons else ""


def summarize_trial(
    manifest: dict[str, Any],
    variant: dict[str, Any],
    seed: int,
    target_steps: int | None = None,
) -> dict[str, Any]:
    output_root = _workspace_path(manifest["output_root"])
    trial_dir = output_root / str(variant["id"]) / f"seed-{seed}"
    train_rows = _read_csv(trial_dir / "plots" / "train_metrics.csv")
    val_rows = _read_csv(trial_dir / "plots" / "val_metrics.csv")
    group_rows = _read_csv(trial_dir / "plots" / "group_diagnostics.csv")
    if target_steps is not None:
        train_rows = [row for row in train_rows if int(_number(row, "step", -1)) < target_steps]
        group_rows = [row for row in group_rows if int(_number(row, "step", -1)) < target_steps]
        val_rows = [row for row in val_rows if int(_number(row, "step", -1)) < target_steps]

    base = _load_json(_workspace_path(manifest["base_train_config"]))
    grpo = deepcopy(base["grpo"])
    grpo.update(variant.get("overrides", {}))
    warning = float(grpo["kl_warning_threshold"])
    hard = float(grpo["kl_threshold"])
    baseline_row = min(val_rows, key=lambda row: _number(row, "step")) if val_rows else {}
    final_val = max(val_rows, key=lambda row: _number(row, "step")) if val_rows else {}
    kl_values = [_number(row, "kl_loss") for row in train_rows]
    rollout_values = [_number(row, "rollout_exact_rate") for row in group_rows]
    optimizer_updates = sum(_number(row, "optimizer_update_count") for row in train_rows)
    if optimizer_updates == 0 and train_rows:
        optimizer_updates = len(train_rows) * planned_updates_per_step(grpo)
    existing_steps = _checkpoint_steps(trial_dir)
    if target_steps is not None:
        existing_steps = [step for step in existing_steps if step < target_steps]

    last_step_kl_coef = (
        round(_number(train_rows[-1], "kl_loss_coef"), 8) if train_rows else 0.0
    )
    post_step_kl_coef = last_step_kl_coef
    if existing_steps:
        trainer_state_path = (
            trial_dir / f"checkpoint-{max(existing_steps)}" / "trainer_state.json"
        )
        if trainer_state_path.exists():
            trainer_state = _load_json(trainer_state_path)
            try:
                post_step_kl_coef = round(
                    float(trainer_state.get("current_kl_loss_coef", last_step_kl_coef)),
                    8,
                )
            except (TypeError, ValueError):
                post_step_kl_coef = last_step_kl_coef

    return {
        "variant": variant["id"],
        "tier": variant.get("tier", ""),
        "factor": variant.get("factor", ""),
        "seed": seed,
        "completed_grpo_steps": len(train_rows),
        "last_step": int(_number(train_rows[-1], "step", -1)) if train_rows else -1,
        "max_checkpoint_step": max(existing_steps) if existing_steps else -1,
        "optimizer_updates": int(optimizer_updates),
        "prompt_exposures": int(sum(_number(row, "prompt_count") for row in train_rows)),
        "rollout_exposures": int(sum(_number(row, "rollout_count") for row in train_rows)),
        "reference_kl_mean": round(_mean(kl_values), 6),
        "reference_kl_last10_mean": round(_mean(kl_values[-10:]), 6),
        "reference_kl_last10_slope": round(_slope(train_rows, "kl_loss"), 6),
        "warning_cross_step": _first_crossing(train_rows, "kl_loss", warning),
        "warning_cross_count": sum(1 for value in kl_values if value > warning),
        "hard_cross_step": _first_crossing(train_rows, "kl_loss", hard),
        "hard_cross_count": sum(1 for value in kl_values if value > hard),
        "update_kl_max": round(max((_number(row, "approx_kl") for row in train_rows), default=0.0), 6),
        "last_step_kl_coef": last_step_kl_coef,
        "final_kl_coef": post_step_kl_coef,
        "reference_kl_final": round(kl_values[-1], 6) if kl_values else 0.0,
        "rollout_em_first10": round(_mean(rollout_values[:10]), 4),
        "rollout_em_last10": round(_mean(rollout_values[-10:]), 4),
        "baseline_greedy_em": round(_number(baseline_row, "val_exact_match"), 4),
        "baseline_sample_em": round(_number(baseline_row, "val_sample_exact_match"), 4),
        "final_greedy_em": round(_number(final_val, "val_exact_match"), 4),
        "best_greedy_em": round(
            max((_number(row, "val_exact_match") for row in val_rows), default=0.0), 4
        ),
        "final_sample_em": round(_number(final_val, "val_sample_exact_match"), 4),
        "final_format_rate": round(_number(final_val, "val_format_rate"), 4),
        "final_eos_rate": round(_number(final_val, "val_eos_rate"), 4),
        "final_hit_max_rate": round(
            _number(final_val, "val_max_tokens_reached_without_eos_rate"), 4
        ),
        "final_sample_format_rate": round(
            _number(final_val, "val_sample_format_rate"), 4
        ),
        "final_sample_eos_rate": round(_number(final_val, "val_sample_eos_rate"), 4),
        "final_sample_hit_max_rate": round(
            _number(final_val, "val_sample_max_tokens_reached_without_eos_rate"), 4
        ),
        "stop_reason": _extract_stop_reason(trial_dir),
    }


CONFIG_LABELS = {
    "resume_state_mode": "checkpoint 状态加载模式",
    "ppo_epochs": "PPO epochs",
    "learning_rate": "学习率",
    "kl_loss_coef": "初始 KL 系数",
    "adaptive_kl_interval": "自适应 KL 检查间隔",
    "adaptive_kl_min_coef": "自适应 KL 系数下限",
    "train_batch_size": "每步独立 prompt 数",
    "gradient_accumulation_steps": "梯度累积 mini-batch 数",
}


def _display_config_value(key: str, value: Any) -> str:
    """把配置值转换为适合 Markdown 阅读的短文本。"""
    if key == "learning_rate":
        return f"{float(value):.1e}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _changed_parameter_text(base_grpo: dict[str, Any], variant: dict[str, Any]) -> str:
    """生成单变量分支相对公共基线的参数变化说明。"""
    changed_keys = list(variant.get("changed_keys", []))
    overrides = variant.get("overrides", {})
    if not changed_keys:
        return "无；全部使用公共基线"
    parts = []
    for key in changed_keys:
        label = CONFIG_LABELS.get(key, key)
        before = _display_config_value(key, base_grpo.get(key))
        after = _display_config_value(key, overrides.get(key))
        parts.append(f"`{label}`：`{before}` → `{after}`")
    return "<br>".join(parts)


def _signed(value: Any, digits: int = 4) -> str:
    """显示带正负号的对照差值。"""
    number = float(value)
    return f"{number:+.{digits}f}"


def _cell(value: Any) -> str:
    """把空值显示为短横线，避免与数值0混淆。"""
    return "-" if value in (None, "") else str(value)


def _variant_reading_note(variant_id: str) -> str:
    """返回每个控制变量分支固定不变的阅读边界。"""
    notes = {
        "c0_fresh_control": (
            "它定义同 seed 下的比较基准。其他分支的差值必须先与 C0 比，"
            "不能只按各组绝对最高值排名。"
        ),
        "o1_old_optimizer": (
            "这是污染诊断分支，不是正式训练候选。若它相对 C0 同时表现为 KL 漂移更快、"
            "能力更差，才支持旧 optimizer 有负面贡献；10步仍不能证明它是唯一主因。"
        ),
        "p1_ppo_epoch1": (
            "同一 GRPO step 下 prompt/rollout 暴露量与 C0 相同，但 optimizer update 减半。"
            "因此既要按相同步数比较，也要补做相同 update 数比较。"
        ),
        "l1_lr3e6": (
            "它检验单次参数步幅是否过大。学习率降低后若 KL 更稳且 EM 不降，才支持降低学习率；"
            "只看到 update KL 变小并不等于 reference KL 已被拉回。"
        ),
        "l2": (
            "它以已确认的L1为基线，只继续降低学习率。若KL更低但EM、rollout明显变差，"
            "说明步幅过小而不是更优稳定点。"
        ),
        "l1": (
            "它是三seed与扩大sample确认后的新阶段控制组。L2和K3都必须先与同seed L1比较，"
            "不能再把5e-6 C0当作当前剂量/控制器实验的直接基线。"
        ),
        "k3": (
            "它以已确认的L1为基线，只提高自适应KL系数下限。必须观察实际reference KL，"
            "不能把系数维持在0.005本身当作收益。"
        ),
        "k1_kl_coef2e2": (
            "KL 系数是总 loss 中 reference KL 项的权重，不是 reference KL 的硬上限。"
            "系数变大后仍需观察实际 KL 轨迹和能力保持情况。"
        ),
        "k2_kl_interval2": (
            "缩短检查间隔只代表控制器更频繁响应，不代表每次都增强约束。"
            "当观测 KL 低于目标区间时，控制器也可能下调 KL 系数。"
        ),
        "b1_prompt8_accum2": (
            "它把每步独立 prompt 和 rollout 总量翻倍，同时用梯度累积把 optimizer update 保持为4次。"
            "计算量和样本暴露量并不与 C0 相同，需同时按 step、prompt 暴露量和墙钟成本比较。"
        ),
    }
    return notes.get(variant_id, "按 changed_keys 确认变量边界，并只与相同 seed 的 C0 比较。")


def _summary_output_dir(manifest: dict[str, Any], output_group: str | None) -> Path:
    """返回受控的汇总目录，禁止标签穿越到编排目录之外。"""
    output_dir = _workspace_path(manifest["output_root"]) / "_orchestration"
    if output_group:
        if not all(char.isalnum() or char in "_-" for char in output_group):
            raise ValueError("summary-group 只能包含字母、数字、下划线和连字符")
        output_dir = output_dir / output_group
    return output_dir


def write_paired_summary(
    rows: list[dict[str, Any]],
    output_dir: Path,
    phase_id: str,
    target_steps: int,
    control_id: str,
    candidate_id: str,
) -> tuple[Path, Path]:
    """生成同seed配对差值与跨seed方向汇总，并排除未达阶段终点的删失配对。"""
    indexed = {(str(row["variant"]), int(row["seed"])): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    paired_rows: list[dict[str, Any]] = []
    for seed in seeds:
        control = indexed.get((control_id, seed))
        candidate = indexed.get((candidate_id, seed))
        if not control or not candidate:
            continue
        control_greedy_retention = float(control["final_greedy_em"]) - float(
            control["baseline_greedy_em"]
        )
        candidate_greedy_retention = float(candidate["final_greedy_em"]) - float(
            candidate["baseline_greedy_em"]
        )
        control_sample_retention = float(control["final_sample_em"]) - float(
            control["baseline_sample_em"]
        )
        candidate_sample_retention = float(candidate["final_sample_em"]) - float(
            candidate["baseline_sample_em"]
        )
        control_steps = int(control["completed_grpo_steps"])
        candidate_steps = int(candidate["completed_grpo_steps"])
        eligible_for_phase_aggregate = (
            control_steps >= target_steps and candidate_steps >= target_steps
        )
        censor_reason = ""
        if not eligible_for_phase_aggregate:
            censor_reason = (
                f"阶段目标{target_steps}步未同时达到: "
                f"control={control_steps}, candidate={candidate_steps}"
            )
        paired_rows.append(
            {
                "seed": seed,
                "control": control_id,
                "candidate": candidate_id,
                "phase_target_steps": target_steps,
                "control_steps": control_steps,
                "candidate_steps": candidate_steps,
                "eligible_for_phase_aggregate": eligible_for_phase_aggregate,
                "censor_reason": censor_reason,
                "control_tail_kl_mean": control["reference_kl_last10_mean"],
                "candidate_tail_kl_mean": candidate["reference_kl_last10_mean"],
                "delta_tail_kl_mean": round(
                    float(candidate["reference_kl_last10_mean"])
                    - float(control["reference_kl_last10_mean"]),
                    6,
                ),
                "control_tail_kl_slope": control["reference_kl_last10_slope"],
                "candidate_tail_kl_slope": candidate["reference_kl_last10_slope"],
                "delta_tail_kl_slope": round(
                    float(candidate["reference_kl_last10_slope"])
                    - float(control["reference_kl_last10_slope"]),
                    6,
                ),
                "control_final_kl": control["reference_kl_final"],
                "candidate_final_kl": candidate["reference_kl_final"],
                "delta_final_kl": round(
                    float(candidate["reference_kl_final"])
                    - float(control["reference_kl_final"]),
                    6,
                ),
                "control_warning_count": control["warning_cross_count"],
                "candidate_warning_count": candidate["warning_cross_count"],
                "control_hard_count": control["hard_cross_count"],
                "candidate_hard_count": candidate["hard_cross_count"],
                "control_greedy_retention": round(control_greedy_retention, 4),
                "candidate_greedy_retention": round(candidate_greedy_retention, 4),
                "delta_greedy_retention": round(
                    candidate_greedy_retention - control_greedy_retention, 4
                ),
                "control_sample_retention": round(control_sample_retention, 4),
                "candidate_sample_retention": round(candidate_sample_retention, 4),
                "delta_sample_retention": round(
                    candidate_sample_retention - control_sample_retention, 4
                ),
                "control_final_greedy_em": control["final_greedy_em"],
                "candidate_final_greedy_em": candidate["final_greedy_em"],
                "control_final_sample_em": control["final_sample_em"],
                "candidate_final_sample_em": candidate["final_sample_em"],
                "control_rollout_em_last10": control["rollout_em_last10"],
                "candidate_rollout_em_last10": candidate["rollout_em_last10"],
                "delta_rollout_em_last10": round(
                    float(candidate["rollout_em_last10"])
                    - float(control["rollout_em_last10"]),
                    4,
                ),
                "candidate_sample_format_rate": candidate["final_sample_format_rate"],
                "candidate_sample_hit_max_rate": candidate["final_sample_hit_max_rate"],
                "candidate_final_kl_coef": candidate["final_kl_coef"],
                "control_stop_reason": control["stop_reason"],
                "candidate_stop_reason": candidate["stop_reason"],
            }
        )

    if not paired_rows:
        raise ValueError("没有找到完整的control/candidate同seed配对")

    csv_name = "paired_deltas.csv" if phase_id == "confirm50" else f"{phase_id}_paired_deltas.csv"
    md_name = "aggregate_summary.md" if phase_id == "confirm50" else f"{phase_id}_aggregate_summary.md"
    csv_path = output_dir / csv_name
    md_path = output_dir / md_name
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(paired_rows[0].keys()))
        writer.writeheader()
        writer.writerows(paired_rows)

    delta_keys = [
        "delta_tail_kl_mean",
        "delta_tail_kl_slope",
        "delta_final_kl",
        "delta_greedy_retention",
        "delta_sample_retention",
        "delta_rollout_em_last10",
    ]
    direction_rules = {
        "delta_tail_kl_mean": lambda value: value < 0,
        "delta_tail_kl_slope": lambda value: value <= 0,
        "delta_final_kl": lambda value: value < 0,
        "delta_greedy_retention": lambda value: value >= 0,
        "delta_sample_retention": lambda value: value >= 0,
        "delta_rollout_em_last10": lambda value: value >= 0,
    }
    eligible_rows = [row for row in paired_rows if row["eligible_for_phase_aggregate"]]
    censored_rows = [row for row in paired_rows if not row["eligible_for_phase_aggregate"]]
    lines = [
        f"# {phase_id} 配对多seed汇总",
        "",
        f"control=`{control_id}`，candidate=`{candidate_id}`。所有delta均为candidate-control。",
        f"阶段目标为{target_steps}步；未同时达到目标的配对只留作删失审计，不进入跨seed聚合。",
        "",
        "| seed | 步数 C0/L1 | 聚合资格 | Δtail KL mean | Δtail KL slope | Δfinal KL | Δgreedy retention | Δsample retention | Δrollout EM last10 | sample fmt/hit-max |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in paired_rows:
        eligibility = "纳入" if row["eligible_for_phase_aggregate"] else "删失"
        lines.append(
            f"| {row['seed']} | {row['control_steps']}/{row['candidate_steps']} | {eligibility} | "
            f"{row['delta_tail_kl_mean']} | {row['delta_tail_kl_slope']} | "
            f"{row['delta_final_kl']} | {row['delta_greedy_retention']} | "
            f"{row['delta_sample_retention']} | {row['delta_rollout_em_last10']} | "
            f"{row['candidate_sample_format_rate']}/{row['candidate_sample_hit_max_rate']} |"
        )
    lines.extend(["", "## 跨seed聚合", "", "| 指标 | 均值 | 中位数 | 支持方向seed数 |", "|---|---:|---:|---:|"])
    if eligible_rows:
        for key in delta_keys:
            values = [float(row[key]) for row in eligible_rows]
            direction_count = sum(1 for value in values if direction_rules[key](value))
            lines.append(
                f"| `{key}` | {_mean(values):.6f} | {_median(values):.6f} | "
                f"{direction_count}/{len(values)} |"
            )
    else:
        lines.append("| - | - | - | 0/0 |")
    if censored_rows:
        lines.extend(["", "## 删失配对", ""])
        for row in censored_rows:
            lines.append(
                f"- seed={row['seed']}：{row['censor_reason']}；"
                f"C0停止原因=`{row['control_stop_reason']}`，"
                f"L1停止原因=`{row['candidate_stop_reason']}`。"
            )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 先看同seed配对方向，再看均值和中位数，不选取单个最高EM。",
            "- 跨seed均值、中位数和方向计数只使用达到当前阶段目标步数的完整配对。",
            "- 只有KL、能力、rollout和格式安全同时满足预注册条件，才支持候选。",
            "- 3个seed只支持方向复验，不把小样本p值当作主要证据。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_summary(
    manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    phase_id: str = "gate10",
    output_group: str | None = None,
) -> tuple[Path, Path]:
    output_dir = _summary_output_dir(manifest, output_group)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / (f"{phase_id}_trials.csv" if output_group else "summary.csv")
    md_path = output_dir / (f"{phase_id}_summary.md" if output_group else "summary.md")

    control_variant_id = str(manifest.get("control_variant_id", "c0_fresh_control"))
    controls = {
        int(row["seed"]): row for row in rows if row["variant"] == control_variant_id
    }
    for row in rows:
        control = controls.get(int(row["seed"]))
        row["final_em_delta_vs_control"] = (
            round(float(row["final_greedy_em"]) - float(control["final_greedy_em"]), 4)
            if control and row is not control
            else 0.0 if control else ""
        )
        row["kl_slope_delta_vs_control"] = (
            round(
                float(row["reference_kl_last10_slope"])
                - float(control["reference_kl_last10_slope"]),
                6,
            )
            if control and row is not control
            else 0.0 if control else ""
        )

    fieldnames = list(rows[0].keys()) if rows else ["variant", "seed"]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    variants = _variant_map(manifest)
    base_config = _load_json(_workspace_path(manifest["base_train_config"]))
    base_grpo = base_config["grpo"]
    phase_map = _phase_map(manifest)
    if phase_id not in phase_map:
        raise ValueError(f"未知汇总阶段: {phase_id}")
    phase = phase_map[phase_id]
    phase_target = int(phase["target_steps"])
    warning = float(base_grpo["kl_warning_threshold"])
    hard = float(base_grpo["kl_threshold"])
    approx_threshold = float(base_grpo["approx_kl_threshold"])
    sample_answers = int(base_grpo["val_stochastic_n"]) * int(
        base_grpo["val_stochastic_max_items"]
    )

    lines = [
        f"# {manifest['suite_name']} {phase_id}实验汇总与阅读说明",
        "",
        "> 本文由 `run_grpo_causal_experiments.py summarize` 自动生成。",
        f"> 本轮汇总阶段为 `{phase_id}`，目标累计{phase_target}步；不把单次最高EM自动判定为根因。",
        "",
        "## 1. 这轮实验在回答什么问题",
        "",
        manifest["description"],
        "",
        f"所有首轮分支都从 `{manifest['source_checkpoint']}` 的同一份 GRPO LoRA 权重出发，"
        "使用同一个 seed 和确定性 prompt 调度。C0 只加载 LoRA 权重并重置 optimizer；"
        "其他分支在 C0 上只改变指定因素。这样才能把结果差异对应到明确变量。",
        "",
        f"本阶段目的：{phase['purpose']} 即使达到目标步数，也必须结合KL、能力、格式和多seed方向再判断。",
        "",
        "### 1.1 公共基线参数",
        "",
        "| 类别 | 参数 | 公共值 | 含义 |",
        "|---|---|---:|---|",
        f"| 数据 | train/eval | `{base_config['dataset']['train_file']}` / `{base_config['dataset']['eval_file']}` | 项目自有 `messages` parquet；固定验证集不参与训练 |",
        f"| rollout | `rollout_n` | {base_grpo['rollout_n']} | 每道 prompt 采样8条回答 |",
        f"| rollout | 采样参数 | temperature={base_grpo['temperature']}, top-p={base_grpo['top_p']}, top-k={base_grpo['top_k']} | 训练 rollout 的随机采样分布 |",
        f"| batch | `train_batch_size` | {base_grpo['train_batch_size']} | 每个 GRPO step 的独立 prompt 数 |",
        f"| PPO | `ppo_epochs` | {base_grpo['ppo_epochs']} | 同一批 rollout 被重复用于 PPO 优化的轮数 |",
        f"| PPO | `ppo_mini_batch_size` | {base_grpo['ppo_mini_batch_size']} | 每个 PPO mini-batch 包含的 rollout 轨迹数 |",
        f"| 优化 | `gradient_accumulation_steps` | {base_grpo['gradient_accumulation_steps']} | 累积多少个 mini-batch 后执行一次 optimizer step |",
        f"| 优化 | `learning_rate` | {float(base_grpo['learning_rate']):.1e} | AdamW 参数更新步幅 |",
        f"| KL | `kl_loss_coef` | {base_grpo['kl_loss_coef']} | 总 loss 中 reference KL 项的初始权重 |",
        f"| KL | target/warning/hard | {base_grpo['adaptive_kl_target']}/{warning}/{hard} | 自适应目标、预警阈值和硬停止阈值 |",
        f"| KL | `adaptive_kl_interval` | {base_grpo['adaptive_kl_interval']} steps | 每隔多少个 GRPO step 调整一次 KL 系数 |",
        f"| 验证 | greedy | {base_grpo['val_max_items']}题 × 每题1条 | `do_sample=False`，主要能力指标 |",
        f"| 验证 | sample | {base_grpo['val_stochastic_max_items']}题 × 每题{base_grpo['val_stochastic_n']}条 = {sample_answers}条 | 固定随机种子的采样鲁棒性诊断，不是 pass@8 |",
        "",
        "公共基线每步的更新次数为：",
        "",
        "```text",
        "4 prompt × 8 rollout = 32条轨迹",
        "32 ÷ mini-batch 16 = 每个PPO epoch有2个mini-batch",
        "2 mini-batch × PPO epochs 2 ÷ 梯度累积1 = 每步4次optimizer update",
        "```",
        "",
        "## 2. 本阶段每个实验具体改了什么",
        "",
        "### 2.1 参数矩阵",
        "",
        "| 实验 | 状态模式 | prompt/step | rollout/step | PPO epochs | mini-batch | 梯度累积 | update/step | LR | 初始KL系数 | KL间隔 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        variant = variants[str(row["variant"])]
        grpo = deepcopy(base_grpo)
        grpo.update(variant.get("overrides", {}))
        rollout_per_step = int(grpo["train_batch_size"]) * int(grpo["rollout_n"])
        lines.append(
            f"| {row['variant']} | {grpo['resume_state_mode']} | {grpo['train_batch_size']} | "
            f"{rollout_per_step} | {grpo['ppo_epochs']} | {grpo['ppo_mini_batch_size']} | "
            f"{grpo['gradient_accumulation_steps']} | {planned_updates_per_step(grpo)} | "
            f"{float(grpo['learning_rate']):.1e} | {grpo['kl_loss_coef']} | "
            f"{grpo['adaptive_kl_interval']} |"
        )

    lines.extend(["", "### 2.2 逐实验目的与变量边界", ""])
    for row in rows:
        variant_id = str(row["variant"])
        variant = variants[variant_id]
        lines.extend(
            [
                f"#### {variant_id}",
                "",
                f"- 改动：{_changed_parameter_text(base_grpo, variant)}。",
                f"- 要检验的问题：{variant['description']}",
                f"- 阅读边界：{_variant_reading_note(variant_id)}",
                "",
            ]
        )

    lines.extend(
        [
            "## 3. 本轮结果总表",
            "",
            "### 3.1 训练完成度与 KL 轨迹",
            "",
        "| variant | seed | steps/last/ckpt | updates | prompt/rollout暴露 | ref KL mean/last10/slope | warning/hard首穿 | update KL max | KL coef步内→步后 | stop |",
            "|---|---:|---|---:|---:|---|---|---:|---:|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['completed_grpo_steps']}/"
            f"{row['last_step']}/{row['max_checkpoint_step']} | {row['optimizer_updates']} | "
            f"{row['prompt_exposures']}/{row['rollout_exposures']} | "
            f"{row['reference_kl_mean']}/{row['reference_kl_last10_mean']}/"
            f"{row['reference_kl_last10_slope']} | {_cell(row['warning_cross_step'])}/"
            f"{_cell(row['hard_cross_step'])} | {row['update_kl_max']} | "
            f"{row['last_step_kl_coef']}→{row['final_kl_coef']} | {row['stop_reason'] or '-'} |"
        )

    lines.extend(
        [
            "",
            "### 3.2 能力、格式与生成终止",
            "",
            "| variant | rollout EM first10→last10 | greedy EM baseline→final/best | final ΔC0 | sample EM | greedy fmt/eos/hit-max | sample fmt/eos/hit-max |",
            "|---|---|---|---:|---:|---|---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['rollout_em_first10']}→{row['rollout_em_last10']} | "
            f"{row['baseline_greedy_em']}→{row['final_greedy_em']}/{row['best_greedy_em']} | "
            f"{_signed(row['final_em_delta_vs_control'])} | {row['final_sample_em']} | "
            f"{row['final_format_rate']}/{row['final_eos_rate']}/{row['final_hit_max_rate']} | "
            f"{row['final_sample_format_rate']}/{row['final_sample_eos_rate']}/"
            f"{row['final_sample_hit_max_rate']} |"
        )

    lines.extend(
        [
            "",
            "## 4. summary.csv 每一个指标的含义",
            "",
            "### 4.1 实验身份与完成度",
            "",
            "| 指标 | 具体含义 | 本轮应怎样读 |",
            "|---|---|---|",
            "| `variant` | 实验分支ID | 先找到 C0，再将同 seed 的其他分支与它比较 |",
            f"| `tier` | 实验层级：single、sensitivity 或 interaction | 本轮汇总{len(rows)}条轨迹；层级不表示优劣 |",
            "| `factor` | 该分支隔离的因果因素 | 如 optimizer_history、rollout_reuse、learning_rate |",
            "| `seed` | 模型采样、prompt调度等使用的随机种子 | 只有相同 seed 才能做首轮直接对照 |",
            f"| `completed_grpo_steps` | `train_metrics.csv` 的有效训练行数 | {phase_target}表示完成本阶段累计{phase_target}个GRPO step |",
            f"| `last_step` | 最后一行训练指标的step编号 | step从0开始，因此本阶段最后编号是{phase_target - 1} |",
            f"| `max_checkpoint_step` | 本阶段不超过目标的最大checkpoint编号 | {phase_target - 1}与last_step一致时证明目标checkpoint已落盘 |",
            f"| `stop_reason` | 训练日志记录的最终退出原因 | `达到最大步数`表示{phase_id}正常完成，不是异常早停 |",
            "",
            "### 4.2 训练工作量",
            "",
            "| 指标 | 具体含义 | 本轮应怎样读 |",
            "|---|---|---|",
            "| `optimizer_updates` | 所有step实际执行的optimizer.step总次数 | C0为40，P1为20；相同步数不等于相同参数更新次数 |",
            "| `prompt_exposures` | 所有step独立训练题目的累计暴露数 | C0为40，B1为80；同一题的8条rollout不算8个独立prompt |",
            "| `rollout_exposures` | 所有step生成并参与训练的轨迹总数 | C0为320，B1为640，用于衡量实际采样工作量 |",
            "",
            "### 4.3 两种不同的 KL",
            "",
            "| 指标 | 具体含义 | 本轮应怎样读 |",
            "|---|---|---|",
            "| `reference_kl_mean` | 全部已完成step中，当前policy相对固定reference模型的KL代理均值 | 衡量策略总体离reference多远；不是越接近0越好，当前自适应目标约为0.04 |",
            "| `reference_kl_last10_mean` | 最后最多10个step的reference KL均值 | Screen30/Confirm50看尾部状态；Gate10恰好10步，所以它等于全程均值 |",
            "| `reference_kl_last10_slope` | 对最后最多10个`(step, reference KL)`点做线性回归后的每step斜率 | 正值表示向外漂移，负值表示回落；越接近0越稳定，但需结合EM判断是否学不动 |",
            f"| `warning_cross_step` | reference KL首次严格大于warning={warning}的step | 空白表示尚未越过，不表示未来长训不会越过 |",
            "| `warning_cross_count` | 已汇总step中raw reference KL严格大于warning的次数 | 区分单点越界与反复越界；仍需结合guard窗口 |",
            f"| `hard_cross_step` | reference KL首次严格大于hard={hard}的step | 空白表示未触发硬阈值；出现值时需核对guard窗口和停止原因 |",
            "| `hard_cross_count` | 已汇总step中raw reference KL严格大于hard的次数 | 候选轨迹应为0；非0时必须审计停止链路 |",
            f"| `update_kl_max` | PPO更新期间，新policy相对采样时旧policy的`approx_kl`最大值 | 它反映单轮更新幅度，不是reference KL；当前保护阈值为{approx_threshold} |",
            "| `last_step_kl_coef` | 最后一个已完成step计算loss时实际使用的reference KL系数 | 它属于本步优化信号；控制器在本步结束后的调整不会追溯改变本步loss |",
            "| `final_kl_coef` | 最终checkpoint保存的`current_kl_loss_coef`，即续跑下一步将使用的控制器状态 | 与`last_step_kl_coef`不同表示控制器在终点已经响应；系数更大不保证观测KL立即下降 |",
            "| `reference_kl_final` | 最后一个训练step的raw reference KL | 单点波动较大，必须与last10均值和斜率共同看 |",
            "| `kl_slope_delta_vs_control` | 该分支KL尾部斜率减去同seed C0斜率 | 负值表示比C0更稳，正值表示比C0漂移更快；只解释相对方向 |",
            "",
            "### 4.4 rollout训练信号",
            "",
            "| 指标 | 具体含义 | 本轮应怎样读 |",
            "|---|---|---|",
            "| `rollout_em_first10` | 最前最多10个训练step中，全部训练rollout答案exact match率的均值 | 它来自随机训练rollout，不等同于固定验证集EM |",
            "| `rollout_em_last10` | 最后最多10个训练step的rollout exact match率均值 | 长于10步时用来判断训练分布上的正确率是否改善或退化 |",
            "",
            ("> 本阶段只有10步，first10和last10使用同一批step，不能据此声称稳定。"
             if phase_target == 10 else
             "> 本阶段首10和尾10窗口不重叠，可用于观察训练分布能力变化，但仍受prompt采样波动影响。"),
            "",
            "### 4.5 固定验证集能力指标",
            "",
            "| 指标 | 具体含义 | 本轮应怎样读 |",
            "|---|---|---|",
            f"| `baseline_greedy_em` | 加载checkpoint-169权重后、开始本分支训练前，在{base_grpo['val_max_items']}题上的greedy exact match | 它是本轮零步共同起点，不是原始SFT模型指标 |",
            "| `baseline_sample_em` | 开始训练前固定随机sample验证的逐回答exact match率 | 计算sample retention时使用；不同seed间不能直接混作同一起点 |",
            f"| `final_greedy_em` | 最后一次验证的greedy exact match | 本轮是step {phase_target - 1}后的100题准确率，分辨率为0.01 |",
            "| `best_greedy_em` | 包含零步基线在内，所有验证点的最高greedy EM | 当前0.67主要来自零步基线，不能误读为训练后曾恢复到0.67 |",
            f"| `final_sample_em` | 固定{base_grpo['val_stochastic_max_items']}题、每题{base_grpo['val_stochastic_n']}次随机采样，共{sample_answers}条回答的逐回答exact match率 | 分辨率为1/{sample_answers}={1/sample_answers:.4f}；不是pass@8/oracle@8 |",
            "| `final_em_delta_vs_control` | 该分支final greedy EM减去同seed C0 final greedy EM | 正值只表示本次终点高于C0，不能单独证明因果或统计显著 |",
            "",
            "### 4.6 格式与生成终止指标",
            "",
            "| 指标 | 具体含义 | 理想方向与注意事项 |",
            "|---|---|---|",
            "| `final_format_rate` | 最后一次100题greedy验证中，回答包含合法`#### 数字`格式的比例 | 越高越好；高格式率不等于答案正确 |",
            "| `final_eos_rate` | greedy回答在长度上限前生成EOS的比例 | 通常越高越健康，但仍需结合长度和答案完整性 |",
            "| `final_hit_max_rate` | greedy回答达到`max_response_length`且未生成EOS的比例 | 越低越好；升高意味着截断/失控风险 |",
            "| `final_sample_format_rate` | 80条随机采样回答的合法格式比例 | 衡量随机生成时的格式鲁棒性，通常比greedy更敏感 |",
            "| `final_sample_eos_rate` | 80条随机采样回答中正常生成EOS的比例 | 与sample hit-max互补，但二者不一定严格相加为1 |",
            "| `final_sample_hit_max_rate` | 80条随机采样回答中达到长度上限且无EOS的比例 | 越低越好；sample EM上升但该值同步上升时不能直接判好 |",
            "",
            "所有rate/EM列均在0到1之间，例如0.67代表67%。KL和loss不是准确率，不受0到1范围约束。",
            "",
            "## 5. 本轮逐实验结果应该怎样解释",
            "",
        ]
    )

    control_by_seed = {
        int(row["seed"]): row for row in rows if row["variant"] == control_variant_id
    }
    for row in rows:
        control = control_by_seed.get(int(row["seed"]))
        sample_delta = (
            float(row["final_sample_em"]) - float(control["final_sample_em"])
            if control
            else 0.0
        )
        lines.extend(
            [
                f"### 5.{len([x for x in lines if x.startswith('### 5.')]) + 1} {row['variant']}",
                "",
                f"- 工作量：{row['prompt_exposures']}个prompt、{row['rollout_exposures']}条rollout、"
                f"{row['optimizer_updates']}次optimizer update。",
                f"- KL：均值{row['reference_kl_mean']}，尾部斜率{row['reference_kl_last10_slope']}，"
                f"相对C0斜率差{_signed(row['kl_slope_delta_vs_control'], 6)}；"
                f"warning/hard首穿为{_cell(row['warning_cross_step'])}/{_cell(row['hard_cross_step'])}。",
                f"- 能力：greedy EM从{row['baseline_greedy_em']}到{row['final_greedy_em']}，"
                f"相对C0终点差{_signed(row['final_em_delta_vs_control'])}；sample EM为"
                f"{row['final_sample_em']}，相对C0差{sample_delta:+.4f}。",
                f"- 鲁棒性：greedy格式/截顶={row['final_format_rate']}/{row['final_hit_max_rate']}；"
                f"sample格式/截顶={row['final_sample_format_rate']}/{row['final_sample_hit_max_rate']}。",
                f"- 结论边界：{_variant_reading_note(str(row['variant']))}",
                "",
            ]
        )

    all_warning_blank = all(row["warning_cross_step"] in (None, "") for row in rows)
    all_hard_blank = all(row["hard_cross_step"] in (None, "") for row in rows)
    completed_phase_count = sum(
        int(row["completed_grpo_steps"]) >= phase_target
        and int(row["last_step"]) >= phase_target - 1
        and int(row["max_checkpoint_step"]) >= phase_target - 1
        for row in rows
    )
    lowest_kl = min(rows, key=lambda row: float(row["reference_kl_mean"])) if rows else None
    highest_sample = max(rows, key=lambda row: float(row["final_sample_em"])) if rows else None
    lines.extend(
        [
            f"## 6. 当前{phase_id}可以下什么结论",
            "",
            f"- 工程链路：{completed_phase_count}/{len(rows)}条已汇总轨迹达到{phase_target}步，"
            f"并具有对应的step {phase_target - 1}指标和checkpoint。",
            f"- KL保护：warning阈值全部未穿越={'是' if all_warning_blank else '否'}，"
            f"hard阈值全部未穿越={'是' if all_hard_blank else '否'}。"
            f"这只说明累计{phase_target}步内的越界情况。",
            f"- 最低reference KL均值是`{lowest_kl['variant']}`的{lowest_kl['reference_kl_mean']}。"
            "最低KL不自动等于最优，还要同时看EM和样本/计算暴露量。" if lowest_kl else "",
            f"- 最高sample EM是`{highest_sample['variant']}`的{highest_sample['final_sample_em']}。"
            "sample单点必须与greedy EM、sample格式率和hit-max率共同判断。" if highest_sample else "",
            f"- {phase_id}单阶段不能独立确定根因；必须结合配对多seed方向和完整轨迹。",
            "",
            "## 7. 推荐阅读与决策顺序",
            "",
            "1. 先确认`completed_grpo_steps`、`last_step`、`max_checkpoint_step`和`stop_reason`，排除未完成或异常退出。",
            "2. 再看`update_kl_max`，排除单次PPO更新过猛；随后看reference KL均值、斜率和阈值穿越。",
            f"3. 再看`rollout_em_first10/last10`，判断训练分布上的信号是否改善；{phase_id}按窗口定义解读。",
            "4. 再看greedy EM，确认固定验证集能力；同时看sample EM评估随机生成鲁棒性。",
            "5. 最后联合检查format、EOS和hit-max，避免把格式退化或截断换来的EM误判为收益。",
            "6. 只有同一变量同时改善KL轨迹和能力/鲁棒性，才值得进入Screen30；最优1至2组还需多seed复验。",
            "",
            "## 8. 原始证据入口",
            "",
            f"- 实验矩阵：`post_training_framework/configs/gsm8k_qwen3_0d6b_grpo_v7_causal_matrix.json`",
            f"- 公共基线：`{manifest['base_train_config']}`",
            f"- 机器可读汇总：`{_display_path(csv_path)}`",
            "- 每组原始指标：`<variant>/seed-<seed>/plots/train_metrics.csv`、`val_metrics.csv`、`group_diagnostics.csv`",
            "- 每组实际配置：`_orchestration/configs/<variant>__seed<seed>__<phase>.json`",
            "- 编排事件：`_orchestration/events.jsonl`",
            "",
            "## 9. 解读门槛",
            "",
            "- 先比较同一 seed、同一 prompt 暴露量下的 KL 斜率和 rollout EM，再看单点验证 EM。",
            "- 单变量至少同时改善 KL 漂移和能力指标，才进入组合实验；只延后 guard 不算解决。",
            "- 最优1至2组必须用 confirmation_seeds 复验，方向在多数 seed 一致后才支持因果结论。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO 控制变量实验编排器")
    parser.add_argument("action", choices=["plan", "run", "summarize"])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", default="gate10")
    parser.add_argument("--variants", nargs="*", help="variant id，可用空格或逗号分隔")
    parser.add_argument("--tiers", nargs="*", help="single/sensitivity/interaction")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument(
        "--summary-group",
        help="将阶段汇总写入_orchestration下的独立目录，避免覆盖默认summary",
    )
    parser.add_argument("--paired-control", default="c0_fresh_control")
    parser.add_argument("--paired-candidate", help="生成同seed配对差值和跨seed聚合")
    parser.add_argument("--write-configs", action="store_true", help="plan 时写出运行配置")
    parser.add_argument("--execute", action="store_true", help="run 时真正启动串行 GPU 训练")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = WORKSPACE_ROOT / manifest_path
    manifest = _load_json(manifest_path.resolve())
    validate_manifest(manifest)

    default_tier = "single" if args.action in {"plan", "run"} else None
    variants = select_variants(manifest, args.variants, args.tiers, default_tier)
    seeds = args.seeds or [int(manifest.get("default_seed", 42))]

    if args.action == "summarize":
        phases = _phase_map(manifest)
        if args.phase not in phases:
            raise ValueError(f"未知汇总阶段: {args.phase}")
        target_steps = int(phases[args.phase]["target_steps"])
        rows = [
            summarize_trial(manifest, variant, seed, target_steps=target_steps)
            for variant in variants
            for seed in seeds
        ]
        csv_path, md_path = write_summary(
            manifest,
            rows,
            phase_id=args.phase,
            output_group=args.summary_group,
        )
        print(f"CSV 汇总: {csv_path}")
        print(f"Markdown 汇总: {md_path}")
        if args.paired_candidate:
            if not args.summary_group:
                raise ValueError("生成配对汇总时必须提供--summary-group，避免覆盖历史结果")
            paired_csv, paired_md = write_paired_summary(
                rows,
                _summary_output_dir(manifest, args.summary_group),
                args.phase,
                target_steps,
                args.paired_control,
                args.paired_candidate,
            )
            print(f"配对CSV: {paired_csv}")
            print(f"配对Markdown: {paired_md}")
        return 0

    plans = [
        build_trial_plan(manifest, variant, seed, args.phase)
        for variant in variants
        for seed in seeds
    ]
    print_plan(manifest, plans)
    if args.write_configs:
        for plan in plans:
            _write_trial_config(plan)
        print(f"已写出 {len(plans)} 份 phase 配置。")
    if args.action == "run":
        if args.execute and not args.variants and not args.tiers:
            raise ValueError("正式执行必须显式指定 --variants 或 --tiers，避免误启动整套矩阵。")
        return run_plans(manifest, plans, execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
