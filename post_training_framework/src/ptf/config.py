"""实验配置读取与路径解析。"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _parse_override_value(raw: str) -> Any:
    """把命令行覆盖值尽量解析成 bool/int/float/list/dict。"""
    text = raw.strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    if text.lower() == "null":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


def apply_dot_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """应用形如 sft.learning_rate=3e-5 的配置覆盖。"""
    result = deepcopy(data)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"配置覆盖必须是 key=value 形式: {item}")
        dotted_key, raw_value = item.split("=", 1)
        keys = [part for part in dotted_key.split(".") if part]
        if not keys:
            raise ValueError(f"无效配置键: {item}")

        cursor: dict[str, Any] = result
        for key in keys[:-1]:
            next_value = cursor.setdefault(key, {})
            if not isinstance(next_value, dict):
                raise ValueError(f"无法覆盖非 dict 节点: {dotted_key}")
            cursor = next_value
        cursor[keys[-1]] = _parse_override_value(raw_value)
    return result


@dataclass
class ExperimentConfig:
    """包装原始配置，统一处理相对路径。"""

    data: dict[str, Any]
    config_path: Path
    framework_root: Path
    workspace_root: Path

    @classmethod
    def load(cls, config_path: str | Path, overrides: list[str] | None = None) -> "ExperimentConfig":
        path = Path(config_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() == ".json":
                data = json.load(f)
            elif path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml  # type: ignore
                except ImportError as exc:
                    raise ImportError("读取 YAML 配置需要安装 PyYAML；也可以改用 JSON 配置。") from exc
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"只支持 .json/.yaml/.yml 配置文件: {path}")

        if not isinstance(data, dict):
            raise ValueError("配置文件顶层必须是 JSON/YAML object。")

        data = apply_dot_overrides(data, overrides)
        framework_root = path.parents[1]

        raw_workspace_root = data.get("workspace_root")
        if raw_workspace_root:
            workspace_root = Path(str(raw_workspace_root))
            if not workspace_root.is_absolute():
                workspace_root = framework_root / workspace_root
            workspace_root = workspace_root.resolve()
        else:
            workspace_root = framework_root.parent.resolve()

        return cls(
            data=data,
            config_path=path,
            framework_root=framework_root,
            workspace_root=workspace_root,
        )

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """按 a.b.c 读取配置字段。"""
        cursor: Any = self.data
        for key in dotted_key.split("."):
            if not isinstance(cursor, dict) or key not in cursor:
                return default
            cursor = cursor[key]
        return cursor

    def require(self, dotted_key: str) -> Any:
        """读取必填配置字段。"""
        value = self.get(dotted_key)
        if value is None:
            raise KeyError(f"缺少必填配置: {dotted_key}")
        return value

    def path(self, dotted_key: str, default: Any = None) -> Path:
        """读取路径字段；相对路径默认以 workspace_root 为基准。"""
        raw_value = self.get(dotted_key, default)
        if raw_value is None:
            raise KeyError(f"缺少路径配置: {dotted_key}")
        path = Path(str(raw_value)).expanduser()
        if not path.is_absolute():
            path = self.workspace_root / path
        return path.resolve()

    def output_root(self) -> Path:
        """返回本实验输出根目录。"""
        return self.path("output_root", "post_training_framework/runs")

    def experiment_dir(self) -> Path:
        """返回当前实验专属输出目录。"""
        return self.output_root() / str(self.get("experiment_name", "default_experiment"))

    def ensure_experiment_dir(self) -> Path:
        """创建并返回当前实验专属输出目录。"""
        path = self.experiment_dir()
        path.mkdir(parents=True, exist_ok=True)
        return path
