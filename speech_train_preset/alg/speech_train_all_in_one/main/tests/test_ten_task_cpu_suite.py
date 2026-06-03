import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speech_tasks import TASK_TYPES, evaluate_task, load_samples, train_task


class TenTaskCpuSuiteTest(unittest.TestCase):
    def test_dataset_covers_all_required_tasks(self):
        root = Path(__file__).resolve().parents[4]
        data_dir = root / "dataset" / "speech_train_sample-dataset"
        samples = load_samples(data_dir)
        task_types = {sample["task_type"] for sample in samples}

        self.assertEqual(set(TASK_TYPES), task_types)
        self.assertEqual(10, len(task_types))

    def test_each_task_trains_and_evaluates_on_cpu(self):
        root = Path(__file__).resolve().parents[4]
        data_dir = root / "dataset" / "speech_train_sample-dataset"
        model_dir = root / "model" / "speech_train_base-model"

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            for task_type in TASK_TYPES:
                train_result = train_task(
                    task_type=task_type,
                    data_dir=data_dir,
                    model_dir=model_dir,
                    save_dir=tmp_dir / task_type / "train",
                    backend="cpu_reference",
                )
                eval_result = evaluate_task(
                    task_type=task_type,
                    data_dir=data_dir,
                    model_dir=tmp_dir / task_type / "train",
                    save_dir=tmp_dir / task_type / "evaluate",
                    backend="cpu_reference",
                )

                self.assertEqual(task_type, train_result["task_type"])
                self.assertEqual("trained", train_result["status"])
                self.assertGreaterEqual(train_result["sample_count"], 1)
                self.assertEqual("cpu_reference", train_result["backend"])
                self.assertTrue((tmp_dir / task_type / "train" / "checkpoint.json").exists())

                self.assertEqual(task_type, eval_result["task_type"])
                self.assertEqual("evaluated", eval_result["status"])
                self.assertEqual("cpu_reference", eval_result["backend"])
                self.assertTrue((tmp_dir / task_type / "evaluate" / "metrics.json").exists())

    def test_open_source_cpu_backend_catalog_covers_tasks(self):
        root = Path(__file__).resolve().parents[4]
        catalog_path = root / "docs" / "open_source_cpu_models.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        catalog_tasks = set()
        for backend in catalog["backends"]:
            catalog_tasks.update(backend["task_types"])

        self.assertTrue(set(TASK_TYPES).issubset(catalog_tasks))
        self.assertTrue(all(backend["device"] == "cpu" for backend in catalog["backends"]))


if __name__ == "__main__":
    unittest.main()
