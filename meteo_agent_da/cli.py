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
    parser.add_argument("--project-root", default="/home/lrx/Unet/satellite_assimilation_v2")
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Do not execute expensive commands.")
    parser.add_argument("--execute", action="store_true", default=False, help="Allow command execution for non-risky tools.")
    parser.add_argument("--max-steps", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dry_run = True if args.dry_run or not args.execute else False
    config = ProjectConfig(project_root=Path(args.project_root), dry_run=dry_run)

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
