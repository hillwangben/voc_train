import json
import unittest
from pathlib import Path


class RealRemainingTaskReportsTest(unittest.TestCase):
    def test_remaining_nine_real_reports_exist_and_are_small(self):
        root = Path(__file__).resolve().parents[4]
        expected_tasks = {
            "noise_enhancement",
            "cross_channel_asr",
            "low_resource_asr",
            "code_switch_asr",
            "speaker_verification",
            "speech_interaction",
            "speech_translation",
            "transcription",
            "speech_correction",
        }
        report_path = root.parent / "runs" / "real_remaining_tasks" / "summary.json"
        summary = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(expected_tasks, {item["task_type"] for item in summary["tasks"]})
        for item in summary["tasks"]:
            self.assertLess(item["sample_count"], 50)
            self.assertEqual("cpu", item["device"])
            self.assertIn(item["status"], {"completed", "attempted"})


if __name__ == "__main__":
    unittest.main()
