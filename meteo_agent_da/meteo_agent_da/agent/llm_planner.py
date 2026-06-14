from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Iterable, Optional

from .planner import HeuristicPlanner
from .schemas import AgentPlan, AgentTask, ToolCall
from .tool_registry import ToolSpec


@dataclass
class QwenPlannerConfig:
    model: str = "qwen-plus"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = ""
    temperature: float = 0.1
    timeout_seconds: int = 60


class QwenPlanner:
    """OpenAI-compatible planner intended for Qwen deployments.

    The planner asks the LLM only to choose tool calls. Tool execution, safety
    checks, artifacts, and trace writing remain inside the local harness.
    """

    def __init__(
        self,
        tool_specs: Iterable[ToolSpec],
        config: Optional[QwenPlannerConfig] = None,
        fallback: Optional[HeuristicPlanner] = None,
    ) -> None:
        self.tool_specs = list(tool_specs)
        self.config = config or QwenPlannerConfig()
        self.fallback = fallback or HeuristicPlanner()

    @classmethod
    def from_env(
        cls,
        tool_specs: Iterable[ToolSpec],
        model: str = "",
        base_url: str = "",
        api_key_env: str = "QWEN_API_KEY",
    ) -> "QwenPlanner":
        api_key = os.getenv(api_key_env) or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        config = QwenPlannerConfig(
            model=model or os.getenv("QWEN_MODEL") or QwenPlannerConfig.model,
            base_url=base_url or os.getenv("QWEN_BASE_URL") or os.getenv("OPENAI_BASE_URL") or QwenPlannerConfig.base_url,
            api_key=api_key,
        )
        return cls(tool_specs=tool_specs, config=config)

    def plan(self, task: AgentTask) -> AgentPlan:
        try:
            raw = self._chat(task)
            return self._parse_plan(raw, task)
        except Exception as exc:  # pragma: no cover - network failures are environment-specific.
            fallback_plan = self.fallback.plan(task)
            fallback_plan.notes.append(f"Qwen planner fallback: {type(exc).__name__}: {exc}")
            return fallback_plan

    def _chat(self, task: AgentTask) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": task.request},
            ],
        }
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"planner HTTP {exc.code}: {detail[:500]}") from exc

        return payload["choices"][0]["message"]["content"]

    def _system_prompt(self) -> str:
        tool_lines = []
        for spec in self.tool_specs:
            risky = "true" if spec.risky else "false"
            tool_lines.append(f"- {spec.name}: {spec.description} risky={risky}")
        tools = "\n".join(tool_lines)
        return (
            "You are the planning module for MeteoAgent-DA, a data-assimilation research agent.\n"
            "Your only job is to select a short, valid sequence of available tools for the current user request.\n"
            "Do not invent tools. Do not claim that commands, paths, metrics, or artifacts were verified.\n"
            "Prefer evidence-producing tools over text-only reasoning. Start with sanity_check when available.\n"
            "For requests beyond the current PASNet/FY-3F tool coverage, use the closest safe tools and explain the gap in notes.\n\n"
            "Return strict JSON only, with this shape:\n"
            "{\n"
            '  "objective": "short objective",\n'
            '  "tool_calls": [\n'
            '    {"name": "tool_name", "arguments": {}, "reason": "why this tool is needed"}\n'
            "  ],\n"
            '  "notes": ["short caveat or missing capability"]\n'
            "}\n\n"
            f"Available tools:\n{tools}"
        )

    def _parse_plan(self, content: str, task: AgentTask) -> AgentPlan:
        raw = json.loads(_extract_json_object(content))
        known = {spec.name for spec in self.tool_specs}
        calls: list[ToolCall] = []
        for item in raw.get("tool_calls", []):
            name = str(item.get("name", "")).strip()
            if name not in known:
                continue
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            calls.append(
                ToolCall(
                    name=name,
                    arguments=arguments,
                    reason=str(item.get("reason", "")),
                )
            )

        if "sanity_check" in known and not any(call.name == "sanity_check" for call in calls):
            calls.insert(
                0,
                ToolCall(
                    name="sanity_check",
                    arguments={},
                    reason="Verify project paths and resources before domain workflow planning.",
                ),
            )

        if not calls:
            return self.fallback.plan(task)

        notes = [str(item) for item in raw.get("notes", []) if str(item).strip()]
        notes.append("Planned by QwenPlanner through an OpenAI-compatible chat-completions endpoint.")
        return AgentPlan(
            objective=str(raw.get("objective") or task.request),
            tool_calls=calls,
            notes=notes,
        )


def _extract_json_object(content: str) -> str:
    stripped = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence:
        return fence.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("planner response did not contain a JSON object")
    return stripped[start : end + 1]
