import json
from pathlib import Path


def load_open_source_cpu_catalog(root_dir):
    catalog_path = Path(root_dir) / "docs" / "open_source_cpu_models.json"
    return json.loads(catalog_path.read_text(encoding="utf-8"))


def recommended_backends_for_task(root_dir, task_type):
    catalog = load_open_source_cpu_catalog(root_dir)
    return [
        backend["name"]
        for backend in catalog["backends"]
        if task_type in backend["task_types"]
    ]

