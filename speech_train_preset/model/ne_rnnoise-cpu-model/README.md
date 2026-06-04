# ne_rnnoise-cpu-model

## 资产说明

- **类型**: CPU 开源预训练模型
- **任务**: 语音增强
- **来源**: `model/real_cpu_models/rnnoise/`
- **对应模型权重**: `model/ne_rnnoise-cpu-model/model/ne_rnnoise/`

## 使用方式

算法在 `config.yaml` 中通过以下方式引用:

```yaml
model_dir: "/workspace/model/ne_rnnoise-cpu-model"
```

平台加载时,会自动根据 `model/model/<short>/` 的相对软链定位到真实权重。

## 关联信息

- 关联算法: `speech_train_all-alg` v1
- 关联数据集: `speech_train_sample-dataset` v1
- 运行环境: `cetc-harbor.registry.com:30003/exp/ubuntu22.04-python3.10-cuda12.1-torch2.1:speech-train-all-v1`

## 真实权重说明

实际模型权重托管在 `model/real_cpu_models/rnnoise/` 下,
本资产通过相对软链引用,避免重复占用存储空间。
该模型仅在本地/CPU 环境使用,符合"低成本可复现"要求。
