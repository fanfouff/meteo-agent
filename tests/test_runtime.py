from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from meteo_agent_da.agent.interactive import InteractiveMeteoAgent
from meteo_agent_da.agent.runtime import MeteoAgentRuntime
from meteo_agent_da.agent.schemas import AgentTask, ProjectConfig, ToolStatus
from meteo_agent_da.baselines.text_only import build_text_only_report
from meteo_agent_da.bench.pasbench import BenchTask, score_report


class RuntimeTest(unittest.TestCase):
    def test_runtime_dry_run_generates_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = ProjectConfig(
                project_root=Path("/home/lrx/Unet/satellite_assimilation_v2"),
                dry_run=True,
            )
            runtime = MeteoAgentRuntime(config=config, run_root=tmp_path)
            report = runtime.run(
                AgentTask(
                    request="Compare PASNet and Swin-UNet on the 50pct split and generate a paper table.",
                    dry_run=True,
                )
            )

            self.assertIn(report.status, {ToolStatus.OK, ToolStatus.ERROR})
            self.assertTrue(any(result.name == "pasnet_runner" for result in report.tool_results))
            self.assertTrue(any(result.name == "paper_writer" for result in report.tool_results))
            self.assertTrue((tmp_path / report.run_id / "report.json").exists())

    def test_interactive_agent_persists_session_memory(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = ProjectConfig(project_root=tmp_path, dry_run=True)
            agent = InteractiveMeteoAgent(
                config=config,
                run_root=tmp_path / "runs",
                session_root=tmp_path / "sessions",
                session_id="test-session",
            )
            response = agent.handle("检查 50pct split 的数据和 stats", max_steps=4)

            self.assertIn("session_id: test-session", response)
            self.assertTrue((tmp_path / "sessions" / "test-session.json").exists())
            self.assertTrue(agent.session.memory["observations"])

    def test_text_only_baseline_scores_lower_tool_recall(self) -> None:
        task = AgentTask(request="Compare PASNet and Swin-UNet on the 50pct split and generate a paper table.")
        report = build_text_only_report(task)
        bench_task = BenchTask(
            task_id="pasbench_da_002",
            category="experiment_planning",
            prompt=task.request,
            required_tools=["sanity_check", "pasnet_runner", "paper_writer"],
            verifier="required_tools",
            expected_artifacts=[],
            max_tool_steps=8,
            rubric={},
        )
        score = score_report(bench_task, report.model_dump())

        self.assertEqual(score["tool_recall"], 0.0)
        self.assertEqual(score["pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
