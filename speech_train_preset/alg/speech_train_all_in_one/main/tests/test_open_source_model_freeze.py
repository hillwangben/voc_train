import json
import unittest
from pathlib import Path


class OpenSourceModelFreezeTest(unittest.TestCase):
    def test_each_catalog_backend_has_frozen_manifest_and_usage_marker(self):
        root = Path(__file__).resolve().parents[4]
        catalog = json.loads((root / "docs" / "open_source_cpu_models.json").read_text(encoding="utf-8"))
        usage = (root / "docs" / "训练平台集成使用说明.md").read_text(encoding="utf-8")
        frozen_root = root / "model" / "open_source_cpu_models"

        for backend in catalog["backends"]:
            name = backend["name"]
            with self.subTest(backend=name):
                manifest_path = frozen_root / name / "manifest.json"
                self.assertTrue(manifest_path.exists(), f"missing frozen manifest for {name}")

                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(name, manifest["backend_name"])
                self.assertEqual(backend["project"], manifest["project"])
                self.assertEqual("cpu", manifest["device"])
                self.assertIn(manifest["freeze_status"], {"bundled", "interface_frozen", "dependency_frozen"})
                self.assertTrue(manifest["engineering_locations"])
                self.assertTrue(manifest.get("offline_cached"), f"{name} must be offline cached")
                cache_path = root / manifest["offline_cache_path"]
                self.assertTrue(cache_path.exists(), f"missing offline cache for {name}")
                self.assertFalse(cache_path.is_symlink(), f"offline cache must not be a symlink for {name}")
                self.assertIn(f"`{name}`", usage)
                self.assertIn(str(manifest_path.relative_to(root)), usage)

        symlinks = list((root / "model" / "real_cpu_models").rglob("*"))
        self.assertFalse([path for path in symlinks if path.is_symlink()])


if __name__ == "__main__":
    unittest.main()
