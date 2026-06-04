"""
Haikit SDK 重导出层。

`main/haikit/` 是规范要求的平台 SDK 目录(目录名固定),
`main/haikit/haikit/` 是与之同名的 Python 子包。
本文件让 `from haikit import ...` 也能正常解析到子包。
"""

from .haikit import (  # noqa: F401
    HaikitContext,
    get_logger,
    report_metrics,
    report_artifact,
)

__all__ = [
    "HaikitContext",
    "get_logger",
    "report_metrics",
    "report_artifact",
]
