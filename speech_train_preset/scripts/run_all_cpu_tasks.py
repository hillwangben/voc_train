#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "alg" / "speech_train_all_in_one" / "main"
FLOWS_DIR = ROOT / "scripts" / "cpu_flows"
sys.path.insert(0, str(FLOWS_DIR))
sys.path.insert(0, str(MAIN_DIR))

from run_cpu_flow import run_flow
from src.speech_tasks import TASK_TYPES


def main():
    output_root = ROOT.parent / "runs" / "all_cpu_tasks"
    summary = {"backend": "cpu_reference", "device": "cpu", "tasks": []}

    for task_type in TASK_TYPES:
        report_path = run_flow(task_type, output_root)
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        summary["tasks"].append(report)

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
