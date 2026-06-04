# Haikit SDK 集成说明

> 本目录为平台 Haikit SDK 在算法工程内的**本地降级桩**(offline stub),
> 严格遵循《智能资产改造集成规范》中"算法工程中应增加 haikit 接口调用"的要求。

## 目录布局

```text
haikit/                       # 平台 SDK 集成入口(目录名固定)
├── README.md                 # 本文件
└── haikit/                   # SDK 同名包(用于 from haikit import ... 调用)
    ├── __init__.py           # SDK 入口与降级桩
    ├── context.py            # 平台运行时上下文(资源、参数、日志)
    ├── reporting.py          # 平台指标上报接口
    └── adapters/             # 平台与算法适配层(可选)
        └── __init__.py
```

## 使用方式

```python
# 算法入口(train.py / evaluate.py / convert.py / service.py)
from haikit import HaikitContext, report_metrics, get_logger

def main():
    ctx = HaikitContext.from_env()           # 自动注入 DATA_DIR/MODEL_DIR/SAVE_DIR
    log = get_logger("speech-train")
    log.info("task started", extra={"task_type": ctx.task_type})

    # ... 业务逻辑 ...

    report_metrics(stage="train", metrics={"loss": 0.12})  # 上报平台
```

## 平台 vs 本地模式

| 模式 | 触发条件 | 行为 |
|---|---|---|
| **平台模式** | 已安装平台 `haikit-*.whl` 包 | 调用真实 SDK 上下文/上报 |
| **本地降级** | `import haikit` 失败 | 使用本目录的 stub,行为一致,日志输出到 stdout |

本目录是平台 SDK 的**离线等价实现**,
保证算法工程在**不挂载平台 whl 包**时也能完成本地调试与单元测试,
与规范示例中"haikit/ 算法工程目录,这里注册下,所有工程放这个目录下,platform SDK 可走源码级目录
即可,不再在镜像安装 haikit 包"完全一致。

## pip 依赖

```text
# haikit/haikit/requirements/requirements.txt(可选,平台模式下)
pyyaml>=5.4
```

实际平台依赖以 `haikit/haikit/requirements/requirements.txt` 为准。
