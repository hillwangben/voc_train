import tarfile
import unittest
import zipfile
from pathlib import Path


class DeliveryArtifactsTest(unittest.TestCase):
    def test_required_delivery_archives_exist_and_are_openable(self):
        root = Path(__file__).resolve().parents[4]
        env_tar = root / "env" / "speech_train_all-env" / "speech_train_all-env.tar.gz"
        dataset_zip = root / "dataset" / "speech_train_sample-dataset" / "speech_train_sample-dataset.zip"
        model_zip = root / "model" / "speech_train_base-model" / "model.zip"

        self.assertTrue(env_tar.exists(), "missing environment delivery tar.gz")
        self.assertTrue(dataset_zip.exists(), "missing dataset zip")
        self.assertTrue(model_zip.exists(), "missing model zip")

        with tarfile.open(env_tar, "r:gz") as package:
            names = set(package.getnames())
            self.assertIn("speech_train_all-env/Dockerfile", names)
            self.assertIn("speech_train_all-env/ENVIRONMENT_MANIFEST.json", names)

        with zipfile.ZipFile(dataset_zip) as package:
            self.assertIsNone(package.testzip())
            self.assertIn("labels/train.jsonl", package.namelist())

        with zipfile.ZipFile(model_zip) as package:
            self.assertIsNone(package.testzip())
            names = set(package.namelist())
            self.assertIn("config/model_info.json", names)
            self.assertIn("model/cpu_reference_model.json", names)
            self.assertIn("real_cpu_models/offline_cache_summary.json", names)
            manifests = [name for name in names if name.startswith("open_source_cpu_models/") and name.endswith("/manifest.json")]
            self.assertEqual(9, len(manifests))


if __name__ == "__main__":
    unittest.main()
