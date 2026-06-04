"""
Haikit SDK 入口(本地降级桩)。

真实平台模式下,会以 site-packages 里的 haikit 包为优先;
本地调试时使用本目录的桩实现。
"""

from .context import HaikitContext, get_logger
from .reporting import report_metrics, report_artifact

__all__ = [
    "HaikitContext",
    "get_logger",
    "report_metrics",
    "report_artifact",
]
