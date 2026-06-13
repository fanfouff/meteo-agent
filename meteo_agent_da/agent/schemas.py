from __future__ import annotations

import json
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

    def resolve_project_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return self.project_root / path


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
