from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from meteo_agent_da.agent.interactive import InteractiveMeteoAgent
from meteo_agent_da.agent.runtime import MeteoAgentRuntime
from meteo_agent_da.agent.schemas import AgentTask, ProjectConfig, ToolStatus
from meteo_agent_da.baselines.text_only import build_text_only_report
from meteo_agent_da.bench.pasbench import BenchTask, score_report
from meteo_agent_da.bench.verifiers import verify_report
from meteo_agent_da.post_training.build_preference_data import reports_to_preference


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

    def test_config_loads_simple_yaml(self) -> None:
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"project_root: {tmpdir}",
                        "dry_run: false",
                        "planner_backend: heuristic",
                        "command_timeout_seconds: 9",
                    ]
                ),
                encoding="utf-8",
            )
            config = ProjectConfig.from_file(config_path)

            self.assertEqual(config.project_root, Path(tmpdir))
            self.assertFalse(config.dry_run)
            self.assertEqual(config.command_timeout_seconds, 9)

    def test_verifier_flags_missing_split_for_command_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            config = ProjectConfig(project_root=tmp_path, dry_run=True)
            runtime = MeteoAgentRuntime(config=config, run_root=tmp_path)
            report = runtime.run(
                AgentTask(
                    request="Compare PASNet and Swin-UNet on the 50pct split and generate a paper table.",
                    dry_run=True,
                )
            )
            verified = verify_report(report.model_dump(), required=["command"])

            self.assertEqual(verified["verifier_pass_rate"], 0.0)
            self.assertFalse(verified["verifier_results"]["command"]["ok"])

    def test_preference_builder_rejects_error_chosen(self) -> None:
        with TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            chosen_path = tmp_path / "chosen.json"
            rejected_path = tmp_path / "rejected.json"
            task = AgentTask(request="test")
            error_report = build_text_only_report(task).model_dump()
            error_report["status"] = "error"
            ok_report = build_text_only_report(task).model_dump()
            chosen_path.write_text(json.dumps(error_report), encoding="utf-8")
            rejected_path.write_text(json.dumps(ok_report), encoding="utf-8")

            with self.assertRaises(ValueError):
                reports_to_preference(chosen_path, rejected_path)


if __name__ == "__main__":
    unittest.main()
