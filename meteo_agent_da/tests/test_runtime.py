from __future__ import annotations

import json
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from meteo_agent_da.agent.interactive import InteractiveMeteoAgent
from meteo_agent_da.agent.runtime import MeteoAgentRuntime
from meteo_agent_da.agent.schemas import AgentTask, ProjectConfig, ToolStatus
from meteo_agent_da.baselines.text_only import build_text_only_report
from meteo_agent_da.bench.pasbench import BenchTask, load_jsonl, score_report
from meteo_agent_da.post_training.build_preference_data import reports_to_preference
from meteo_agent_da.post_training.filter_traces import evaluate_trace_filter
from meteo_agent_da.post_training.build_sft_data import trace_to_sft
from meteo_agent_da.post_training.rewards import executable_workflow_reward
from meteo_agent_da.tools.data_indexer import _resolve_split_file


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

    def test_split_hint_resolves_zero_padded_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            split_dir = Path(tmpdir) / "splits"
            split_dir.mkdir()
            expected = split_dir / "split_050pct.json"
            expected.write_text('{"train": [], "val": [], "test": []}', encoding="utf-8")
            config = ProjectConfig(project_root=Path(tmpdir), default_split_dir=split_dir)

            self.assertEqual(_resolve_split_file(config, "50pct"), expected)

    def test_artifact_recall_accepts_filename_match(self) -> None:
        with TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "pasnet_da_agent_table.tex"
            artifact.write_text("\\begin{table}\\end{table}", encoding="utf-8")
            bench_task = BenchTask(
                task_id="artifact_case",
                category="paper_writing",
                prompt="Generate a paper table.",
                required_tools=["paper_writer"],
                verifier="executable",
                expected_artifacts=["pasnet_da_agent_table.tex"],
                max_tool_steps=2,
                rubric={"paper_artifact": 1},
                verifier_checks=["artifact", "cost"],
            )
            report = {
                "status": "ok",
                "artifacts": [str(artifact)],
                "tool_results": [
                    {
                        "name": "paper_writer",
                        "status": "ok",
                        "summary": "Generated table.",
                        "artifacts": [str(artifact)],
                    }
                ],
            }

            score = score_report(bench_task, report)

            self.assertEqual(score["artifact_recall"], 1.0)
            self.assertEqual(score["verifier_pass"], 1.0)

    def test_pasbench_50_loads(self) -> None:
        tasks = load_jsonl(Path("examples/pasbench_da_50.jsonl"))

        self.assertEqual(len(tasks), 50)
        self.assertTrue(any(task.domain_profile == "wrf_3dvar" for task in tasks))

    def test_trace_to_sft_preserves_trace_events(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            report_path = run_dir / "report.json"
            trace_path = run_dir / "trace.jsonl"
            report_path.write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "request": "Inspect data.",
                        "status": "ok",
                        "summary": "Done.",
                        "tool_results": [
                            {
                                "name": "data_indexer",
                                "status": "ok",
                                "summary": "Indexed data.",
                                "data": {"split_file_exists": True},
                                "artifacts": [],
                            }
                        ],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            events = [
                {"kind": "plan", "payload": {"objective": "Inspect data", "tool_calls": [{"name": "data_indexer"}]}},
                {"kind": "tool", "payload": {"name": "data_indexer", "arguments": {"split_hint": "50pct"}, "reason": "Inspect split."}},
                {"kind": "reflect", "message": "Observed data_indexer: ok", "payload": {"name": "data_indexer", "status": "ok", "summary": "Indexed data."}},
            ]
            trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

            sample = trace_to_sft(report_path)
            assistant = json.loads(sample["messages"][2]["content"])

            self.assertEqual(assistant["plan"]["objective"], "Inspect data")
            self.assertEqual(assistant["tool_trajectory"][0]["tool_call"]["arguments"]["split_hint"], "50pct")
            self.assertEqual(sample["metadata"]["trace_event_count"], 3)

    def test_reward_returns_interpretable_breakdown(self) -> None:
        breakdown = executable_workflow_reward(
            [
                {
                    "name": "paper_writer",
                    "status": "ok",
                    "artifacts": [],
                    "data": {},
                }
            ],
            scientific_checks={"dry_run_honest": True},
            report={"summary": "Generated artifact metadata.", "status": "ok", "next_steps": ["Verify output."]},
        )

        self.assertIn("total_reward", breakdown)
        self.assertIn("tool_success_reward", breakdown)
        self.assertGreaterEqual(breakdown["total_reward"], 0.0)

    def test_filter_accepts_clean_tool_trace(self) -> None:
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            artifact = run_dir / "table.tex"
            artifact.write_text("\\begin{table}\\end{table}", encoding="utf-8")
            report_path = run_dir / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "run_id": "clean-1",
                        "request": "Generate a paper table.",
                        "status": "ok",
                        "summary": "Tool calls completed: 1 ok, 0 error.",
                        "tool_results": [
                            {
                                "name": "paper_writer",
                                "status": "ok",
                                "summary": "Generated table.",
                                "data": {},
                                "artifacts": [str(artifact)],
                            }
                        ],
                        "artifacts": [str(artifact)],
                        "next_steps": ["Use the artifact in the paper."],
                    }
                ),
                encoding="utf-8",
            )
            trace_path = run_dir / "trace.jsonl"
            events = [
                {"kind": "plan", "payload": {"objective": "Write table"}},
                {"kind": "tool", "payload": {"name": "paper_writer", "arguments": {}, "reason": "Need table."}},
                {"kind": "reflect", "payload": {"name": "paper_writer", "status": "ok"}},
            ]
            trace_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

            decision = evaluate_trace_filter(report_path)

            self.assertTrue(decision["accepted"], decision["reasons"])

    def test_preference_metadata_records_case_type(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chosen = root / "chosen.json"
            rejected = root / "rejected.json"
            chosen.write_text(
                json.dumps(
                    {
                        "run_id": "chosen",
                        "request": "Inspect data.",
                        "status": "ok",
                        "summary": "Inspected with tools.",
                        "tool_results": [{"name": "data_indexer", "status": "ok", "summary": "ok"}],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )
            rejected.write_text(
                json.dumps(
                    {
                        "run_id": "rejected",
                        "request": "Inspect data.",
                        "status": "ok",
                        "summary": "Text-only answer.",
                        "tool_results": [],
                        "artifacts": [],
                    }
                ),
                encoding="utf-8",
            )

            sample = reports_to_preference(chosen, rejected)

            self.assertEqual(sample["metadata"]["preference_type"], "tool_grounded_vs_text_only")


if __name__ == "__main__":
    unittest.main()
