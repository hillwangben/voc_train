import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speech_tasks import evaluate_task, train_task


class CrossChannelAsrTest(unittest.TestCase):
    def test_train_and_evaluate_cross_channel_asr_writes_reports(self):
        root = Path(__file__).resolve().parents[4]
        data_dir = root / "dataset" / "speech_train_sample-dataset"
        model_dir = root / "model" / "speech_train_base-model"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            train_result = train_task(
                task_type="cross_channel_asr",
                data_dir=data_dir,
                model_dir=model_dir,
                save_dir=tmp_dir / "train",
            )
            eval_result = evaluate_task(
                task_type="cross_channel_asr",
                data_dir=data_dir,
                model_dir=tmp_dir / "train",
                save_dir=tmp_dir / "evaluate",
            )

            self.assertEqual(train_result["task_type"], "cross_channel_asr")
            self.assertEqual(train_result["sample_count"], 1)
            self.assertEqual(train_result["channel_count"], 1)
            self.assertTrue((tmp_dir / "train" / "checkpoint.json").exists())

            self.assertEqual(eval_result["task_type"], "cross_channel_asr")
            self.assertEqual(eval_result["wer"], 0.0)
            self.assertEqual(eval_result["cer"], 0.0)
            self.assertTrue((tmp_dir / "evaluate" / "metrics.json").exists())

            metrics = json.loads((tmp_dir / "evaluate" / "metrics.json").read_text())
            self.assertIn("channel_metrics", metrics)
            self.assertEqual(metrics["channel_metrics"]["mic_array_2"]["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
