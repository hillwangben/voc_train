"""
训练入口(算法侧)。

按照《智能资产改造集成规范》,本入口集成平台 haikit SDK,
负责:
  1. 解析命令行参数(与 train.sh 透传的环境变量/CLI 保持一致)
  2. 通过 haikit.HaikitContext.from_env() 接收平台注入的运行时上下文
  3. 调用真实训练逻辑(speech_tasks.train_task)
  4. 通过 haikit.report_metrics() 将阶段指标上报平台
  5. 通过 haikit.report_artifact() 将产物路径登记到平台

平台 SDK 缺失时,本目录下的 haikit/ 桩会自动接管,行为等价,
保证本地/CI 调试可独立完成。
"""

import argparse
import json
import os
import sys
from pathlib import Path

# === 1. 路径注入:让 `import haikit` 能找到 main/haikit/ 下的 haikit 子包 ===
MAIN_DIR = Path(__file__).resolve().parents[1]
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

# === 2. 集成平台 haikit SDK(本地降级桩在 main/haikit/) ===
try:
    from haikit import HaikitContext, get_logger, report_metrics, report_artifact
    HAIKIT_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001
    # 真离线、且连本地桩都没有时的最终降级:无日志/无上报
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
            self.save_dir = os.environ.get("SAVE_DIR", "/workspace/output/train")

    def report_metrics(stage, metrics, extra=None, report_path=None):  # type: ignore[no-redef]
        pass

    def report_artifact(name, path, kind="model", report_path=None):  # type: ignore[no-redef]
        pass


# === 3. 业务逻辑 ===
from speech_tasks import train_task  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-type", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--backend", default="cpu_reference")
    args = parser.parse_args()

    # 平台上下文(优先使用平台注入,失败时回退到环境变量)
    ctx = HaikitContext.from_env()
    log = get_logger("speech-train")
    log.info(
        "haikit context loaded: task_type=%s backend=%s",
        ctx.task_type, ctx.backend,
    )
    if not HAIKIT_AVAILABLE:
        log.warning("haikit SDK not available, using offline stub: %s", _HAIKIT_IMPORT_ERROR)

    result = train_task(args.task_type, args.data_dir, args.model_dir, args.save_dir, args.backend)

    # 阶段指标上报(train)
    train_metrics = {
        k: v for k, v in result.items() if k.startswith("train_") or k in {"loss", "lr", "epochs"}
    }
    if train_metrics:
        report_metrics(stage="train", metrics=train_metrics, extra={
            "task_type": args.task_type,
            "backend": args.backend,
        })

    # 产物路径登记
    report_artifact(name="train_result", path=args.save_dir, kind="checkpoint")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
