#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_DIR = ROOT / "alg" / "speech_train_all_in_one" / "main"
sys.path.insert(0, str(MAIN_DIR))

from src.speech_tasks import TASK_TYPES, evaluate_task, train_task


def main():
    data_dir = ROOT / "dataset" / "speech_train_sample-dataset"
    model_dir = ROOT / "model" / "speech_train_base-model"
    output_root = ROOT.parent / "runs" / "all_cpu_tasks"
    summary = {"backend": "cpu_reference", "device": "cpu", "tasks": []}

    for task_type in TASK_TYPES:
        task_root = output_root / task_type
        train_result = train_task(
            task_type=task_type,
            data_dir=data_dir,
            model_dir=model_dir,
            save_dir=task_root / "train",
            backend="cpu_reference",
        )
        eval_result = evaluate_task(
            task_type=task_type,
            data_dir=data_dir,
            model_dir=task_root / "train",
            save_dir=task_root / "evaluate",
            backend="cpu_reference",
        )
        summary["tasks"].append(
            {
                "task_type": task_type,
                "train": train_result,
                "evaluate": eval_result,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()

