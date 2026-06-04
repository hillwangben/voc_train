import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speech_tasks import TASK_TYPES


class IndependentCpuFlowScriptsTest(unittest.TestCase):
    def test_each_task_has_independent_script_and_report(self):
        root = Path(__file__).resolve().parents[4]
        scripts_dir = root / "scripts" / "cpu_flows"

        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            for task_type in TASK_TYPES:
                script = scripts_dir / f"run_{task_type}.py"
                self.assertTrue(script.exists(), f"missing script {script}")

                subprocess.run(
                    [sys.executable, str(script), "--output-root", str(output_root)],
                    cwd=root,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                report_path = output_root / task_type / "summary.json"
                self.assertTrue(report_path.exists(), f"missing report {report_path}")
                report = json.loads(report_path.read_text(encoding="utf-8"))
                self.assertEqual(task_type, report["task_type"])
                self.assertEqual("trained", report["train"]["status"])
                self.assertEqual("evaluated", report["evaluate"]["status"])


if __name__ == "__main__":
    unittest.main()
