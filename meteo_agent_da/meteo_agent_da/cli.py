from __future__ import annotations

import argparse
from pathlib import Path

from .agent.llm_planner import QwenPlanner
from .agent.planner import Planner
from .agent.interactive import InteractiveMeteoAgent
from .agent.runtime import MeteoAgentRuntime
from .agent.schemas import AgentTask, ProjectConfig
from .tools.registry import build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MeteoAgent-DA local harness.")
    parser.add_argument("--task", help="Research workflow request.")
    parser.add_argument("--chat", action="store_true", default=False, help="Start an interactive research-agent session.")
    parser.add_argument("--session-id", default="", help="Resume or create a named interactive session.")
    parser.add_argument("--project-root", default="/home/lrx/Unet/satellite_assimilation_v2")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Do not execute expensive commands.")
    parser.add_argument("--execute", action="store_true", default=False, help="Allow command execution for non-risky tools.")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--planner", choices=["heuristic", "qwen"], default="heuristic")
    parser.add_argument("--llm-base-url", default="", help="OpenAI-compatible base URL for --planner qwen.")
    parser.add_argument("--llm-model", default="", help="Qwen model name for --planner qwen.")
    parser.add_argument("--llm-api-key-env", default="QWEN_API_KEY", help="Environment variable containing the API key.")
    return parser


def build_planner(args: argparse.Namespace, registry) -> Planner | None:
    if args.planner == "qwen":
        return QwenPlanner.from_env(
            tool_specs=registry.specs(),
            model=args.llm_model,
            base_url=args.llm_base_url,
            api_key_env=args.llm_api_key_env,
        )
    return None


def main() -> None:
    args = build_parser().parse_args()
    dry_run = True if args.dry_run or not args.execute else False
    config = ProjectConfig(project_root=Path(args.project_root), dry_run=dry_run)
    registry = build_default_registry()
    planner = build_planner(args, registry)

    if args.chat:
        agent = InteractiveMeteoAgent(
            config=config,
            registry=registry,
            planner=planner,
            run_root=Path(args.run_root),
            session_root=Path(args.session_root),
            session_id=args.session_id or None,
        )
        agent.run_repl(max_steps=args.max_steps)
        return

    if not args.task:
        raise SystemExit("--task is required unless --chat is set.")

    runtime = MeteoAgentRuntime(config=config, registry=registry, planner=planner, run_root=Path(args.run_root))
    report = runtime.run(AgentTask(request=args.task, dry_run=dry_run, max_steps=args.max_steps))
    print(report.summary)
    if report.artifacts:
        print("\nArtifacts:")
        for artifact in report.artifacts:
            print(f"- {artifact}")


if __name__ == "__main__":
    main()
