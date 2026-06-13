from __future__ import annotations

import argparse
from pathlib import Path

from .agent.interactive import InteractiveMeteoAgent
from .agent.runtime import MeteoAgentRuntime
from .agent.schemas import AgentTask, ProjectConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the MeteoAgent-DA local harness.")
    parser.add_argument("--task", help="Research workflow request.")
    parser.add_argument("--chat", action="store_true", default=False, help="Start an interactive research-agent session.")
    parser.add_argument("--session-id", default="", help="Resume or create a named interactive session.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to a simple YAML config file.")
    parser.add_argument("--project-root", default="", help="Override project_root from config.")
    parser.add_argument("--planner", choices=["heuristic", "llm"], default="", help="Override planner backend.")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Do not execute expensive commands.")
    parser.add_argument("--execute", action="store_true", default=False, help="Allow command execution for non-risky tools.")
    parser.add_argument("--allow-risky", action="store_true", default=False, help="Allow risky domain tools when --execute is set.")
    parser.add_argument("--run-commands", action="store_true", default=False, help="Actually run generated PASNet commands.")
    parser.add_argument("--max-steps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dry_run = True if args.dry_run or not args.execute else False
    config = ProjectConfig.from_file(Path(args.config)).apply_env()
    if args.project_root:
        config.project_root = Path(args.project_root)
    if args.planner:
        config.planner_backend = args.planner
    config.dry_run = dry_run
    config.allow_risky_tools = args.allow_risky
    config.execute_commands = args.run_commands

    if args.chat:
        agent = InteractiveMeteoAgent(
            config=config,
            run_root=Path(args.run_root),
            session_root=Path(args.session_root),
            session_id=args.session_id or None,
        )
        agent.run_repl(max_steps=args.max_steps)
        return

    if not args.task:
        raise SystemExit("--task is required unless --chat is set.")

    runtime = MeteoAgentRuntime(config=config, run_root=Path(args.run_root))
    report = runtime.run(AgentTask(request=args.task, dry_run=dry_run, max_steps=args.max_steps))
    print(report.summary)
    if report.artifacts:
        print("\nArtifacts:")
        for artifact in report.artifacts:
            print(f"- {artifact}")


if __name__ == "__main__":
    main()
