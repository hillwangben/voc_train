"""
平台指标与产物上报接口(本地降级实现)。

真实平台模式下,会通过 SDK 调用平台 metric / artifact 上报通道;
本地降级模式下,把上报内容以 JSON 行追加到 save_dir 下的 haikit_report.jsonl,
便于在评测/回归脚本中读取。
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .context import get_logger

_LOG = get_logger("reporting")


def _default_report_path() -> Path:
    save_dir = os.environ.get("SAVE_DIR", "/workspace/output")
    return Path(save_dir) / "haikit_report.jsonl"


def report_metrics(
    stage: str,
    metrics: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
    report_path: Optional[Path] = None,
) -> None:
    """上报某阶段(train/evaluate/adapt/service)的指标。

    平台模式下: 走 SDK 上传通道(由覆盖同名包的 site-packages 接管)。
    本地降级:   追加到 save_dir/haikit_report.jsonl。
    """
    payload = {
        "kind": "metrics",
        "stage": stage,
        "ts": int(time.time() * 1000),
        "metrics": metrics,
        "extra": extra or {},
    }
    target = report_path or _default_report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _LOG.info("reported metrics stage=%s keys=%s", stage, list(metrics.keys()))


def report_artifact(
    name: str,
    path: str,
    *,
    kind: str = "model",
    report_path: Optional[Path] = None,
) -> None:
    """上报某产物(模型权重、报告文件等)的路径。"""
    payload = {
        "kind": "artifact",
        "artifact_name": name,
        "artifact_kind": kind,
        "artifact_path": path,
        "ts": int(time.time() * 1000),
    }
    target = report_path or _default_report_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    _LOG.info("reported artifact name=%s path=%s", name, path)
