from __future__ import annotations

from pathlib import Path
from typing import Optional

from .runtime import MeteoAgentRuntime
from .schemas import AgentTask, ProjectConfig
from .session import SessionStore


HELP_TEXT = """可用命令：
  /help                  显示帮助
  /memory                查看当前会话记忆
  /session               查看 session id 和存储路径
  /dry-run on|off         切换 dry-run
  /exit                  退出
直接输入科研请求即可，例如：
  检查 50pct split，并生成 PASNet 和 Swin-UNet 的 dry-run 训练命令
"""


class InteractiveMeteoAgent:
    """Conversation wrapper around the deterministic MeteoAgent runtime."""

    def __init__(
        self,
        config: Optional[ProjectConfig] = None,
        run_root: Optional[Path] = None,
        session_root: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> None:
        self.config = config or ProjectConfig()
        self.run_root = run_root or Path.cwd() / "runs"
        self.session_store = SessionStore(session_root or Path.cwd() / "sessions")
        if session_id and self.session_store.exists(session_id):
            self.session = self.session_store.load(session_id)
        else:
            self.session = self.session_store.create(project_root=self.config.project_root, session_id=session_id)

    def handle(self, user_message: str, max_steps: int = 8) -> str:
        stripped = user_message.strip()
        if not stripped:
            return ""
        if stripped.startswith("/"):
            return self._handle_command(stripped)

        self.session.add_user_turn(stripped)
        contextual_request = self.session.contextualize(stripped)
        runtime = MeteoAgentRuntime(config=self.config, run_root=self.run_root)
        report = runtime.run(
            AgentTask(
                request=contextual_request,
                dry_run=self.config.dry_run,
                max_steps=max_steps,
            )
        )
        self.session.add_report(
            original_request=stripped,
            contextual_request=contextual_request,
            report=report,
        )
        self.session_store.save(self.session)

        lines = [
            report.summary,
            "",
            f"run_id: {report.run_id}",
            f"session_id: {self.session.session_id}",
        ]
        if report.artifacts:
            lines.append("artifacts:")
            lines.extend(f"- {item}" for item in report.artifacts)
        return "\n".join(lines).strip()

    def run_repl(self, max_steps: int = 8) -> None:
        print("MeteoAgent-DA 交互式科研 Agent")
        print(f"session_id: {self.session.session_id}")
        print("输入 /help 查看命令，输入 /exit 退出。")
        while True:
            try:
                user_message = input("\nmeteo-da> ")
            except EOFError:
                print()
                break
            if user_message.strip() == "/exit":
                break
            response = self.handle(user_message, max_steps=max_steps)
            if response:
                print(response)

    def _handle_command(self, command: str) -> str:
        parts = command.split()
        name = parts[0].lower()
        if name == "/help":
            return HELP_TEXT.strip()
        if name == "/memory":
            return self.session.context_text(max_chars=4000)
        if name == "/session":
            path = self.session_store.path_for(self.session.session_id)
            return f"session_id: {self.session.session_id}\nsession_path: {path}"
        if name == "/dry-run":
            if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
                return "用法：/dry-run on 或 /dry-run off"
            self.config.dry_run = parts[1].lower() == "on"
            return f"dry_run={self.config.dry_run}"
        return "未知命令。输入 /help 查看可用命令。"
