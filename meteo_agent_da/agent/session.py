from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .schemas import AgentReport


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]


@dataclass
class ConversationSession:
    """Stateful chat session for the domain research agent.

    The session deliberately stores compact evidence rather than full tool payloads.
    Full reports and traces remain in the run directory.
    """

    session_id: str
    project_root: str
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    history: List[Dict[str, Any]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, project_root: Path, session_id: Optional[str] = None) -> "ConversationSession":
        return cls(
            session_id=session_id or _new_session_id(),
            project_root=str(project_root),
            memory={
                "task_summary": "",
                "observations": [],
                "recent_tools": [],
                "artifacts": [],
                "open_questions": [],
            },
        )

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ConversationSession":
        return cls(
            session_id=raw["session_id"],
            project_root=raw.get("project_root", ""),
            created_at=raw.get("created_at", _now()),
            updated_at=raw.get("updated_at", _now()),
            history=list(raw.get("history", [])),
            memory=dict(raw.get("memory", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def add_user_turn(self, content: str) -> None:
        self.history.append({"role": "user", "content": content, "created_at": _now()})
        self.memory["task_summary"] = content
        self._trim()

    def add_report(self, original_request: str, contextual_request: str, report: AgentReport) -> None:
        used_tools = [result.name for result in report.tool_results]
        artifacts = list(report.artifacts)
        observations = [result.summary for result in report.tool_results if result.summary]
        errors = [result.summary for result in report.tool_results if result.error]

        self.history.append(
            {
                "role": "assistant",
                "content": report.summary,
                "status": str(report.status),
                "run_id": report.run_id,
                "original_request": original_request,
                "contextual_request": contextual_request,
                "used_tools": used_tools,
                "artifacts": artifacts,
                "created_at": _now(),
            }
        )
        self._remember_many("recent_tools", used_tools, limit=30)
        self._remember_many("artifacts", artifacts, limit=30)
        self._remember_many("observations", observations, limit=30)
        self._remember_many("open_questions", errors, limit=12)
        self._trim()

    def context_text(self, max_chars: int = 1800) -> str:
        lines: List[str] = []
        memory = self.memory
        lines.append("会话记忆：")
        if memory.get("task_summary"):
            lines.append(f"- 最近任务：{memory['task_summary']}")
        if memory.get("recent_tools"):
            lines.append(f"- 最近工具：{', '.join(memory['recent_tools'][-8:])}")
        if memory.get("artifacts"):
            lines.append("- 已产生 artifact：")
            lines.extend(f"  - {item}" for item in memory["artifacts"][-5:])
        if memory.get("observations"):
            lines.append("- 最近观测：")
            lines.extend(f"  - {item}" for item in memory["observations"][-8:])
        if memory.get("open_questions"):
            lines.append("- 待确认问题：")
            lines.extend(f"  - {item}" for item in memory["open_questions"][-4:])

        recent_turns = self.history[-6:]
        if recent_turns:
            lines.append("最近对话：")
            for turn in recent_turns:
                role = turn.get("role", "unknown")
                content = str(turn.get("content", "")).replace("\n", " ")
                lines.append(f"- {role}: {content[:240]}")

        text = "\n".join(lines)
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

    def contextualize(self, user_message: str) -> str:
        context = self.context_text()
        return (
            "你正在一个多轮卫星资料同化科研 Agent 会话中工作。"
            "请优先依据会话记忆和工具证据推进任务。\n\n"
            f"{context}\n\n当前用户请求：\n{user_message}"
        )

    def _remember_many(self, key: str, items: List[str], limit: int) -> None:
        bucket = list(self.memory.get(key, []))
        for item in items:
            if not item:
                continue
            if item in bucket:
                bucket.remove(item)
            bucket.append(item)
        self.memory[key] = bucket[-limit:]
        self.updated_at = _now()

    def _trim(self, max_history: int = 24) -> None:
        self.history = self.history[-max_history:]
        self.updated_at = _now()


class SessionStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, session_id: str) -> Path:
        return self.root / f"{session_id}.json"

    def create(self, project_root: Path, session_id: Optional[str] = None) -> ConversationSession:
        session = ConversationSession.create(project_root=project_root, session_id=session_id)
        self.save(session)
        return session

    def load(self, session_id: str) -> ConversationSession:
        path = self.path_for(session_id)
        return ConversationSession.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, session: ConversationSession) -> Path:
        path = self.path_for(session.session_id)
        path.write_text(json.dumps(session.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def exists(self, session_id: str) -> bool:
        return self.path_for(session_id).exists()
