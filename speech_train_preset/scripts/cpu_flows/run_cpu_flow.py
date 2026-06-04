#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_DIR = ROOT / "alg" / "speech_train_all-alg" / "main"
sys.path.insert(0, str(MAIN_DIR))

from src.speech_tasks import TASK_TYPES, evaluate_task, train_task  # noqa: E402
from src.backends import default_backend_for_task, recommended_backends_for_task  # noqa: E402


def _pick_backend(task_type: str) -> str:
    """选择真实后端(优先有真实权重的),可通过环境变量 CPU_FLOW_BACKEND 覆盖。"""
    override = os.environ.get("CPU_FLOW_BACKEND")
    if override:
        return override
    return default_backend_for_task(task_type)


def run_flow(task_type, output_root=None):
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported task_type {task_type!r}")

    data_dir = ROOT / "dataset" / "speech_train_sample-dataset"
    model_dir = ROOT / "model" / "speech_train_base-model"
    root = Path(output_root) if output_root else ROOT.parent / "runs" / "cpu_flows"
    task_root = root / task_type

    backend = _pick_backend(task_type)
    candidates = recommended_backends_for_task(task_type)

    train_result = train_task(
        task_type=task_type,
        data_dir=data_dir,
        model_dir=model_dir,
        save_dir=task_root / "train",
        backend=backend,
    )
    eval_result = evaluate_task(
        task_type=task_type,
        data_dir=data_dir,
        model_dir=task_root / "train",
        save_dir=task_root / "evaluate",
        backend=backend,
    )

    summary = {
        "task_type": task_type,
        "backend": backend,
        "backend_candidates": candidates,
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
    parser.add_argument(
        "--backend",
        help="强制指定 backend(默认按 default_backend_for_task 选真实后端)",
    )
    args = parser.parse_args()
    if args.backend:
        os.environ["CPU_FLOW_BACKEND"] = args.backend
    print(run_flow(args.task_type, args.output_root))


if __name__ == "__main__":
    main()

