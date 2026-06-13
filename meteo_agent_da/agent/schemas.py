from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4


class ToolStatus(str, Enum):
    OK = "ok"
    ERROR = "error"
    SKIPPED = "skipped"


class StepKind(str, Enum):
    PLAN = "plan"
    TOOL = "tool"
    REFLECT = "reflect"
    REPORT = "report"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


class Serializable:
    def model_dump(self) -> Dict[str, Any]:
        return _jsonable(asdict(self))

    def model_dump_json(self) -> str:
        return json.dumps(self.model_dump(), ensure_ascii=False)


@dataclass
class ProjectConfig(Serializable):
    project_root: Path = Path("/home/lrx/Unet/satellite_assimilation_v2")
    train_script: Path = Path("train_ddp/train_ddp.py")
    default_data_root: Path = Path("/data/lrx_true/era_obs/npz")
    default_output_dir: Path = Path("/home/lrx/Unet/satellite_assimilation_v2/train_ddp/outputs/meteo_agent_da")
    default_stats_file: Path = Path("/data/lrx_true/era_obs/npz/stats.npz")
    default_increment_stats: Path = Path("/data/lrx_true/era_obs/npz/increment_stats.npz")
    default_split_dir: Path = Path("/home/lrx/Unet/satellite_assimilation_v2/train_ddp/splits/data_efficiency_64")
    dry_run: bool = True
    planner_backend: str = "heuristic"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"
    llm_api_key_env: str = "OPENAI_API_KEY"
    command_timeout_seconds: int = 120
    allow_risky_tools: bool = False
    execute_commands: bool = False

    def resolve_project_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path

    @classmethod
    def from_file(cls, path: Path) -> "ProjectConfig":
        raw = _read_simple_yaml(path)
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "ProjectConfig":
        values = dict(raw)
        path_fields = {
            "project_root",
            "train_script",
            "default_data_root",
            "default_output_dir",
            "default_stats_file",
            "default_increment_stats",
            "default_split_dir",
        }
        bool_fields = {"dry_run", "allow_risky_tools", "execute_commands"}
        int_fields = {"command_timeout_seconds"}

        kwargs: Dict[str, Any] = {}
        for key, value in values.items():
            if key in path_fields:
                kwargs[key] = Path(str(value))
            elif key in bool_fields:
                kwargs[key] = _as_bool(value)
            elif key in int_fields:
                kwargs[key] = int(value)
            elif hasattr(cls, key):
                kwargs[key] = value
        return cls(**kwargs)

    def apply_env(self, prefix: str = "METEO_AGENT_DA_") -> "ProjectConfig":
        updates: Dict[str, Any] = {}
        for key in self.model_dump():
            env_name = prefix + key.upper()
            if env_name not in os.environ:
                continue
            current = getattr(self, key)
            raw = os.environ[env_name]
            if isinstance(current, Path):
                updates[key] = Path(raw)
            elif isinstance(current, bool):
                updates[key] = _as_bool(raw)
            elif isinstance(current, int):
                updates[key] = int(raw)
            else:
                updates[key] = raw
        merged = self.model_dump()
        merged.update(updates)
        return ProjectConfig.from_mapping(merged)


@dataclass
class AgentTask(Serializable):
    request: str
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    dry_run: bool = True
    max_steps: int = 8


@dataclass
class ToolCall(Serializable):
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class AgentPlan(Serializable):
    objective: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass
class ToolResult(Serializable):
    name: str
    status: ToolStatus
    summary: str
    data: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TraceEvent(Serializable):
    run_id: str
    step: int
    kind: StepKind
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class AgentReport(Serializable):
    run_id: str
    request: str
    status: ToolStatus
    summary: str
    tool_results: List[ToolResult] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


def _read_simple_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip()
        if "#" in value:
            value = value.split("#", 1)[0].strip()
        data[key.strip()] = value.strip("'\"")
    return data


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
