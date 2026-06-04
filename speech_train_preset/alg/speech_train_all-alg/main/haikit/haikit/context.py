"""
平台运行时上下文(本地降级实现)。

集中注入平台注入的环境变量与目录变量,统一算法入口参数。
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class HaikitContext:
    """平台注入的运行时上下文。"""

    task_type: str = "asr"
    backend: str = "cpu_reference"
    data_dir: str = "/workspace/input"
    model_dir: str = "/workspace/model"
    save_dir: str = "/workspace/output/train"
    parameters: Dict[str, Any] = field(default_factory=dict)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resources: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "HaikitContext":
        """从环境变量构造上下文。

        平台会注入以下环境变量:
          - TASK_TYPE / BACKEND / DATA_DIR / MODEL_DIR / SAVE_DIR
        本地调试时使用默认值。
        """
        ctx = cls(
            task_type=os.environ.get("TASK_TYPE", "asr"),
            backend=os.environ.get("BACKEND", "cpu_reference"),
            data_dir=os.environ.get("DATA_DIR", "/workspace/input"),
            model_dir=os.environ.get("MODEL_DIR", "/workspace/model"),
            save_dir=os.environ.get("SAVE_DIR", "/workspace/output/train"),
        )
        # 透传平台时区与 Python 缓冲设置
        ctx.environment_variables.setdefault("TZ", os.environ.get("TZ", "Asia/Shanghai"))
        ctx.environment_variables.setdefault(
            "PYTHONUNBUFFERED", os.environ.get("PYTHONUNBUFFERED", "1")
        )
        # 资源占用由平台注入,本地仅做记录
        ctx.resources = {
            "cpu": float(os.environ.get("HAIKIT_CPU", "0")),
            "gpu": float(os.environ.get("HAIKIT_GPU", "0")),
            "memory": float(os.environ.get("HAIKIT_MEMORY", "0")),
        }
        return ctx


def get_logger(name: str) -> logging.Logger:
    """获取统一 logger,自动附加平台标准字段。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] haikit.%(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
