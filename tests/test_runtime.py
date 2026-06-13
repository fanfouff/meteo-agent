from __future__ import annotations

import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from meteo_agent_da.agent.runtime import MeteoAgentRuntime
from meteo_agent_da.agent.schemas import AgentTask, ProjectConfig, ToolStatus


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


if __name__ == "__main__":
    unittest.main()
