#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "alg" / "speech_train_all_in_one" / "main"
sys.path.insert(0, str(MAIN_DIR))

from src.speech_tasks import TASK_TYPES, evaluate_task, train_task


def run_flow(task_type, output_root=None):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task_type {task_type!r}")

    data_dir = ROOT / "dataset" / "speech_train_sample-dataset"
    model_dir = ROOT / "model" / "speech_train_base-model"
    root = Path(output_root) if output_root else ROOT.parent / "runs" / "cpu_flows"
    task_root = root / task_type

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

    summary = {
        "task_type": task_type,
        "backend": "cpu_reference",
        "device": "cpu",
        "train": train_result,
        "evaluate": eval_result,
        "paths": {
            "train_checkpoint": str(task_root / "train" / "checkpoint.json"),
            "train_metrics": str(task_root / "train" / "train_metrics.json"),
            "evaluate_metrics": str(task_root / "evaluate" / "metrics.json"),
        },
    }
    task_root.mkdir(parents=True, exist_ok=True)
    summary_path = task_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task_type", choices=TASK_TYPES)
    parser.add_argument("--output-root")
    args = parser.parse_args()
    print(run_flow(args.task_type, args.output_root))


if __name__ == "__main__":
    main()

