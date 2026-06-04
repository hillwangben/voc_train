import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.speech_tasks import TASK_TYPES, load_samples


class DatasetFiftyPerTaskTest(unittest.TestCase):
    def test_each_task_has_fifty_samples(self):
        root = Path(__file__).resolve().parents[4]
        data_dir = root / "dataset" / "speech_train_sample-dataset"
        counts = Counter(sample["task_type"] for sample in load_samples(data_dir))

        self.assertEqual(set(TASK_TYPES), set(counts))
        for task_type in TASK_TYPES:
            self.assertEqual(50, counts[task_type], task_type)


if __name__ == "__main__":
    unittest.main()
