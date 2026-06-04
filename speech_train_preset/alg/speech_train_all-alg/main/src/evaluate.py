"""
评估入口(算法侧)。

与 train.py 同样集成平台 haikit SDK,
负责: 解析参数 → 拉取平台上下文 → 调用评估逻辑 → 上报 evaluate 阶段指标。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# === 路径注入(让 `import haikit` 能找到 main/haikit/ 下的子包) ===
MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

# === 集成 haikit SDK(降级桩) ===
try:
    from haikit import HaikitContext, get_logger, report_metrics
    HAIKIT_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001
    HAIKIT_AVAILABLE = False
    _HAIKIT_IMPORT_ERROR = _exc

    def get_logger(name):  # type: ignore[no-redef]
        import logging
        return logging.getLogger(name)

    class HaikitContext:  # type: ignore[no-redef]
        @classmethod
        def from_env(cls):
            return cls()

        def __init__(self):
            self.task_type = os.environ.get("TASK_TYPE", "asr")
            self.backend = os.environ.get("BACKEND", "cpu_reference")
            self.data_dir = os.environ.get("DATA_DIR", "/workspace/input")
            self.model_dir = os.environ.get("MODEL_DIR", "/workspace/model")
            self.save_dir = os.environ.get("SAVE_DIR", "/workspace/output/evaluate")

    def report_metrics(stage, metrics, extra=None, report_path=None):  # type: ignore[no-redef]
        pass


from speech_tasks import evaluate_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--backend", default="cpu_reference")
    args = parser.parse_args()

    ctx = HaikitContext.from_env()
    log = get_logger("speech-evaluate")
    log.info(
        "haikit context loaded: task_type=%s backend=%s",
        ctx.task_type, ctx.backend,
    )
    if not HAIKIT_AVAILABLE:
        log.warning("haikit SDK not available, using offline stub: %s", _HAIKIT_IMPORT_ERROR)

    result = evaluate_task(args.task_type, args.data_dir, args.model_dir, args.save_dir, args.backend)

    # evaluate 阶段指标上报
    eval_metrics = {
        k: v for k, v in result.items() if k.startswith("eval_") or k in {"wer", "cer", "bleu", "accuracy"}
    }
    if eval_metrics:
        report_metrics(stage="evaluate", metrics=eval_metrics, extra={
            "task_type": args.task_type,
            "backend": args.backend,
        })

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
