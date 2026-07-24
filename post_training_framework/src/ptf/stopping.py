"""训练停止决策、统一调度和结构化留档。"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


STOP_ARCHIVE_SCHEMA_VERSION = 1


class StopCategory:
    """停止类别常量，避免训练器之间使用不一致的字符串。"""

    KL_GUARD = "kl_guard"
    SIGNAL_GUARD = "signal_guard"
    FORMAT_GUARD = "format_guard"
    EARLY_STOPPING = "early_stopping"
    MAX_STEPS = "max_steps"
    OUT_OF_MEMORY = "out_of_memory"
    INTERRUPTED = "interrupted"
    RUNTIME_ERROR = "runtime_error"


class StopSeverity:
    """停止严重程度常量。"""

    NORMAL = "normal"
    CONTROLLED = "controlled"
    FATAL = "fatal"


STOP_PRIORITY = {
    StopCategory.OUT_OF_MEMORY: 1000,
    StopCategory.INTERRUPTED: 950,
    StopCategory.RUNTIME_ERROR: 900,
    StopCategory.KL_GUARD: 800,
    StopCategory.SIGNAL_GUARD: 700,
    StopCategory.FORMAT_GUARD: 600,
    StopCategory.EARLY_STOPPING: 500,
    StopCategory.MAX_STEPS: 100,
}


def _now_text() -> str:
    """返回便于人工阅读且带时区信息的时间。"""
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json_safe(value: Any) -> Any:
    """把常见路径、标量和容器转换成可写入JSON的值。"""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


@dataclass(frozen=True)
class StopDecision:
    """某个独立判定器提交的结构化停止决定。"""

    category: str
    source: str
    reason: str
    step: int
    priority: int
    severity: str = StopSeverity.CONTROLLED
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """返回稳定、可JSON序列化的字典。"""
        return _json_safe(asdict(self))


def build_stop_decision(
    category: str,
    source: str,
    reason: str,
    step: int,
    *,
    severity: str = StopSeverity.CONTROLLED,
    details: dict[str, Any] | None = None,
) -> StopDecision:
    """按统一优先级表构造停止决定。"""
    if category not in STOP_PRIORITY:
        raise ValueError(f"未知停止类别: {category}")
    return StopDecision(
        category=category,
        source=source,
        reason=reason,
        step=int(step),
        priority=STOP_PRIORITY[category],
        severity=severity,
        details=details or {},
    )


class TrainingStopController:
    """汇总独立判定器，选择主停止原因并统一写出停止档案。"""

    def __init__(self, output_dir: str | Path, run_name: str) -> None:
        self.output_dir = Path(output_dir)
        self.run_name = run_name
        self.session_id = uuid.uuid4().hex
        self.started_at = _now_text()
        self.logs_dir = self.output_dir / "logs"
        self.events_path = self.logs_dir / "stop_events.jsonl"
        self.summary_path = self.output_dir / "training_stop.json"
        self.selected_decision: StopDecision | None = None

    def start(self, context: dict[str, Any] | None = None) -> None:
        """记录本次训练会话开始，续训时追加而不覆盖历史事件。"""
        self._append_event(
            {
                "event": "session_started",
                "schema_version": STOP_ARCHIVE_SCHEMA_VERSION,
                "session_id": self.session_id,
                "run_name": self.run_name,
                "timestamp": self.started_at,
                "context": context or {},
            }
        )

    def select(
        self,
        stage: str,
        candidates: Iterable[StopDecision | None],
    ) -> StopDecision | None:
        """在一个调度阶段选择优先级最高的停止决定，并保留全部候选。"""
        valid = [candidate for candidate in candidates if candidate is not None]
        if not valid:
            return None
        selected = sorted(
            valid,
            key=lambda item: (-item.priority, item.source, item.reason),
        )[0]
        self.selected_decision = selected
        self._append_event(
            {
                "event": "stop_selected",
                "schema_version": STOP_ARCHIVE_SCHEMA_VERSION,
                "session_id": self.session_id,
                "run_name": self.run_name,
                "timestamp": _now_text(),
                "stage": stage,
                "selected": selected.to_dict(),
                "candidates": [item.to_dict() for item in valid],
            }
        )
        return selected

    def finalize(
        self,
        *,
        status: str,
        decision: StopDecision,
        last_completed_step: int,
        checkpoint_path: str | Path | None,
        finalization_errors: list[str] | None = None,
    ) -> None:
        """写入本次会话的最终结构化摘要和追加式结束事件。"""
        finished_at = _now_text()
        payload = {
            "schema_version": STOP_ARCHIVE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "run_name": self.run_name,
            "status": status,
            "started_at": self.started_at,
            "finished_at": finished_at,
            "last_completed_step": int(last_completed_step),
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "decision": decision.to_dict(),
            "finalization_errors": list(finalization_errors or []),
            "events_path": str(self.events_path),
        }
        self._append_event({"event": "session_finished", **payload})
        self._write_summary_atomically(payload)

    def _append_event(self, payload: dict[str, Any]) -> None:
        """追加单行JSON事件；每次打开关闭，保证异常时尽量落盘。"""
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")
            f.flush()

    def _write_summary_atomically(self, payload: dict[str, Any]) -> None:
        """用临时文件替换最终摘要，避免留下半截JSON。"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.summary_path.with_suffix(".json.tmp")
        with temp_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
            f.flush()
        temp_path.replace(self.summary_path)
