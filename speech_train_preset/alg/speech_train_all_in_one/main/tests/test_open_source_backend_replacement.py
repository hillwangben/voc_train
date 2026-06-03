import json
import unittest
from pathlib import Path


class OpenSourceBackendReplacementTest(unittest.TestCase):
    def test_reports_do_not_use_placeholder_backends(self):
        root = Path(__file__).resolve().parents[4]
        summary_path = root.parent / "runs" / "real_remaining_tasks" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        forbidden_prefixes = (
            "stdlib_",
            "small_dictionary_",
            "small_lexicon_",
            "cpu_reference",
        )
        for item in summary["tasks"]:
            backend = item["backend"]
            self.assertFalse(
                backend.startswith(forbidden_prefixes),
                f"{item['task_type']} still uses placeholder backend {backend}",
            )
            self.assertTrue(item.get("open_source", False), item["task_type"])


if __name__ == "__main__":
    unittest.main()
