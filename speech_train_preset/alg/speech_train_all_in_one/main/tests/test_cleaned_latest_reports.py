import json
import unittest
from pathlib import Path


class CleanedLatestReportsTest(unittest.TestCase):
    def test_only_latest_run_directories_are_required(self):
        root = Path(__file__).resolve().parents[4]
        runs_root = root.parent / "runs"

        self.assertTrue((runs_root / "all_cpu_tasks" / "summary.json").exists())
        self.assertTrue((runs_root / "cpu_flows").exists())
        self.assertFalse((runs_root / "real_remaining_tasks").exists())
        self.assertFalse((runs_root / "real_vosk_asr").exists())
        self.assertFalse((runs_root / "cross_channel_asr").exists())

    def test_latest_independent_reports_cover_ten_tasks_with_fifty_samples(self):
        root = Path(__file__).resolve().parents[4]
        expected_tasks = {
            "asr",
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
        reports_root = root.parent / "runs" / "cpu_flows"
        found = set()
        for report_path in reports_root.glob("*/summary.json"):
            report = json.loads(report_path.read_text(encoding="utf-8"))
            found.add(report["task_type"])
            self.assertEqual(50, report["train"]["sample_count"])
            self.assertEqual(50, report["evaluate"]["sample_count"])
        self.assertEqual(expected_tasks, found)


if __name__ == "__main__":
    unittest.main()
