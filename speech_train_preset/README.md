# 语音类垂直训练资产包

本资产包参考《智能资产改造集成规范》和《资产包准备说明》,
采用方案 2:单一统一训练容器提供语音类模型训练、评估、适配和服务能力。

## 覆盖能力

1. 语音识别垂直训练
2. 复杂噪声下语音增强垂直训练
3. 跨信道连续语音识别垂直训练
4. 小语种连续语音识别垂直训练
5. 混合语种连续语音识别垂直训练
6. 声纹识别垂直训练
7. 语音智能交互垂直训练
8. 多语言语音翻译垂直训练
9. 语音转写垂直训练
10. 语音纠错垂直训练

## 顶层目录结构(符合 preset 规范)

```text
speech_train_preset/                 # preset 根(顶层)
├── alg/                             # 算法工程(规范固定目录)
│   └── speech_train_all-alg/        # 单一统一训练算法(10 类任务入口)
├── env/                             # 环境镜像(规范固定目录)
│   └── speech_train_all-env/        # 离线构建包模式(非完整 Docker 镜像)
├── dataset/                         # 数据集(规范固定目录)
│   └── speech_train_sample-dataset/ # 示例数据集(已划分 train/val/test)
├── model/                           # 模型(规范固定目录)
│   ├── speech_train_base-model/     # 聚合 manifest(指向 9 个独立后端模型)
│   ├── asr_whisper_tiny-cpu-model/  # ── 9 个独立 CPU 开源 model 资产 ──
│   ├── asr_vosk_small-cpu-model/
│   ├── asr_sherpa_paraformer-cpu-model/
│   ├── mt_opus_mt_en_zh-cpu-model/
│   ├── si_minilm_intent-cpu-model/
│   ├── sv_ecapa_voxceleb-cpu-model/
│   ├── ne_deepfilternet3-cpu-model/
│   ├── ne_rnnoise-cpu-model/
│   ├── sc_pycorrector-cpu-model/
│   └── real_cpu_models/             # 真实权重存储(被 9 个资产通过软链引用)
├── docs/                            # ⚠ 附赠文档(非平台规范要求)
├── scripts/                         # ⚠ 附赠脚本(非平台规范要求)
├── README.md
└── PROFILE.jpg
```

> 4 个核心子目录(`alg/env/dataset/model`)为《资产包准备说明》要求;
> `docs/` 与 `scripts/` 为本资产包附赠,非平台规范要求项,接收方可按需删除。

## 资产命名规范

每个资产子文件夹 = `任务前缀 + 标识 + 类型后缀`,
与 `INFO.ini` 的 `name` 字段一致:

| 资产 | 命名 |
|---|---|
| 算法 | `speech_train_all-alg` |
| 环境 | `speech_train_all-env` |
| 数据集 | `speech_train_sample-dataset` |
| 模型 | `speech_train_base-model` / `asr_whisper_tiny-cpu-model` 等 |

## 交付说明

- `alg/speech_train_all-alg/`:统一语音训练算法工程,含 `main/config.yaml`、
  train/eval/convert/service.sh、`main/src/*.py`、`main/haikit/`(平台 SDK 集成桩)、`main/tests/`。
- `env/speech_train_all-env/`:统一语音训练环境,**离线构建包模式**
  (`delivery_type: offline_container_build_package`),含 Dockerfile、requirements、ENVIRONMENT_MANIFEST。
- `dataset/speech_train_sample-dataset/`:示例数据集(500 条 / 10 类任务),
  zip 内已划分 `audio/{train,val,test}/` 与 `labels/{train,val,test}.jsonl`。
- `model/`:
  - `speech_train_base-model/`:基础预训练入口(轻量 manifest),不持有具体权重。
  - 9 个 `<task>_<name>-cpu-model/`:独立 CPU 开源 model 资产,每个含 INFO/README/PROFILE,
    通过 `model/<asset>/model/<name>/` 相对软链访问 `model/real_cpu_models/<src>/` 下的实际权重。
- `docs/`:交付清单、镜像构建说明、数据规范和验收说明(附赠)。
- `scripts/`:本地运行与校验脚本(附赠)。

## 验收路径

```bash
# 1. 解压后查看资产清单
unzip -l speech_train_preset.zip | head -30

# 2. 校验资产结构
python3 scripts/validate_asset_package.py

# 3. 查看 9 个独立 model 资产是否齐全
ls model/*-cpu-model/

# 4. 算法 haikit 桩自检
python3 -c "import sys; sys.path.insert(0, 'alg/speech_train_all-alg/main'); from haikit import HaikitContext, report_metrics; print('haikit OK')"
```
