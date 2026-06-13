from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
import os
from typing import Iterable

from .schemas import AgentPlan, AgentTask, ProjectConfig, ToolCall
from .tool_registry import ToolSpec


class HeuristicPlanner:
    """A deterministic starter planner.

    Later this can be replaced by an LLM planner while keeping the same ToolCall
    contract.
    """

    def plan(self, task: AgentTask) -> AgentPlan:
        request = task.request
        lower = request.lower()
        calls: list[ToolCall] = [
            ToolCall(
                name="sanity_check",
                arguments={},
                reason="Check PASNet project paths and default resources before planning.",
            )
        ]

        if any(token in lower for token in ["data", "dataset", "split", "npz", "数据", "划分", "样本"]):
            calls.append(
                ToolCall(
                    name="data_indexer",
                    arguments={"split_hint": self._extract_split_hint(lower)},
                    reason="Inspect data and split resources for the requested experiment.",
                )
            )

        if any(token in lower for token in ["train", "训练", "pasnet", "swin", "fuxi", "mamba", "实验", "compare", "比较"]):
            calls.append(
                ToolCall(
                    name="pasnet_runner",
                    arguments={
                        "models": self._extract_models(lower),
                        "split_hint": self._extract_split_hint(lower),
                        "epochs": self._extract_int(lower, "epoch", default=200),
                    },
                    reason="Build controlled PASNet-DA experiment commands.",
                )
            )

        if any(token in lower for token in ["rmse", "mae", "metric", "评估", "误差", "指标"]):
            calls.append(
                ToolCall(
                    name="evaluator",
                    arguments={},
                    reason="Summarize available metric artifacts.",
                )
            )

        if any(token in lower for token in ["plot", "figure", "case", "图", "绘图", "可视化"]):
            calls.append(
                ToolCall(
                    name="plotter",
                    arguments={},
                    reason="Prepare figure generation commands or register expected figures.",
                )
            )

        if any(token in lower for token in ["paper", "latex", "table", "论文", "表格", "图注", "caption"]):
            calls.append(
                ToolCall(
                    name="paper_writer",
                    arguments={"style": "ieee"},
                    reason="Generate paper-ready tables and result wording.",
                )
            )

        return AgentPlan(
            objective=request,
            tool_calls=calls,
            notes=[
                "The first scaffold uses a deterministic planner; replace it with an LLM planner after tools stabilize.",
                "Expensive commands are dry-run by default.",
            ],
        )

    @staticmethod
    def _extract_split_hint(text: str) -> str:
        match = re.search(r"(\d+)\s*%|(\d+)pct|split[_ -]?(\d+)", text)
        if not match:
            return "100pct"
        value = next(group for group in match.groups() if group)
        return f"{value}pct"

    @staticmethod
    def _extract_models(text: str) -> list[str]:
        candidates = {
            "pasnet": "pasnet",
            "physics": "physics_unet",
            "swin": "swin_unet",
            "fuxi": "fuxi_da",
            "mamba": "mamba",
            "vanilla": "vanilla_unet",
        }
        found = [model for key, model in candidates.items() if key in text]
        return found or ["pasnet"]

    @staticmethod
    def _extract_int(text: str, key: str, default: int) -> int:
        match = re.search(rf"{key}s?\s*[=: ]\s*(\d+)", text)
        return int(match.group(1)) if match else default


class OpenAICompatiblePlanner:
    """LLM planner for OpenAI-compatible chat completion APIs.

    The planner only decides tool calls. Tool execution remains inside the local
    registry, so risky commands and verifier checks still go through the harness.
    """

    def __init__(self, config: ProjectConfig, tool_specs: Iterable[ToolSpec]) -> None:
        self.config = config
        self.tool_specs = list(tool_specs)

    def plan(self, task: AgentTask) -> AgentPlan:
        api_key = os.environ.get(self.config.llm_api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key env: {self.config.llm_api_key_env}")

        payload = {
            "model": self.config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the planner of MeteoAgent-DA, a satellite data-assimilation research agent. "
                        "Return only JSON with keys objective, tool_calls, notes. "
                        "tool_calls must be a list of objects: {name, arguments, reason}. "
                        "Use only the provided tools and prefer verifiable tool evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "request": task.request,
                            "max_steps": task.max_steps,
                            "available_tools": [
                                {
                                    "name": spec.name,
                                    "description": spec.description,
                                    "risky": spec.risky,
                                }
                                for spec in self.tool_specs
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            url=self.config.llm_base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM planner request failed: {exc}") from exc

        content = raw["choices"][0]["message"]["content"]
        plan_raw = json.loads(content)
        legal_names = {spec.name for spec in self.tool_specs}
        calls = []
        for item in plan_raw.get("tool_calls", [])[: task.max_steps]:
            name = str(item.get("name", "")).strip()
            if name not in legal_names:
                continue
            args = item.get("arguments") or item.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            calls.append(
                ToolCall(
                    name=name,
                    arguments=args,
                    reason=str(item.get("reason", "LLM planner selected this tool.")),
                )
            )
        if not calls:
            calls.append(ToolCall(name="sanity_check", arguments={}, reason="Fallback safety check when LLM returned no valid tools."))
        return AgentPlan(
            objective=str(plan_raw.get("objective") or task.request),
            tool_calls=calls,
            notes=[str(item) for item in plan_raw.get("notes", [])],
        )
