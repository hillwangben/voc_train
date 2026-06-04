# 语音类垂直训练资产包

本资产包参考《智能资产改造集成规范》和《资产包准备说明》,
采用方案 2:单一统一训练容器提供语音类模型训练、评估、适配和服务能力。

> **v2.0 重大更新**:从单一 cpu_reference 冒烟桩升级为 **10/10 任务接入真实开源 CPU 后端**,
> 9 个独立可发布的 model 资产(取消 symlink,内含真实权重),完整离线 Python wheelhouse
> 替代不可联网构建的 Docker 镜像,新增端到端构建脚本 `scripts/build_asset_package.py`。

---

## 1. 覆盖能力(10 类语音任务)

| # | task_type | 默认真实后端 | 关键指标 |
|---:|---|---|---|
| 1 | `asr` (语音识别) | `vosk_small` | wer, cer |
| 2 | `noise_enhancement` (复杂噪声语音增强) | `rnnoise` | snr_improvement, pesq_proxy, stoi_proxy |
| 3 | `cross_channel_asr` (跨信道连续语音识别) | `vosk_small` | wer, cer, channel_metrics |
| 4 | `low_resource_asr` (小语种连续语音识别) | `vosk_small` | wer, cer |
| 5 | `code_switch_asr` (混合语种连续语音识别) | `sherpa_onnx_int8` | wer, cer, channel_metrics |
| 6 | `speaker_verification` (声纹识别) | `speechbrain_ecapa` | eer_proxy, verification_accuracy |
| 7 | `speech_interaction` (语音智能交互) | `vosk_small` + `minilm_intent` | intent_accuracy, slot_f1 |
| 8 | `speech_translation` (多语言语音翻译) | `vosk_small` + `opus_mt_tiny` | bleu_proxy, chrf_proxy |
| 9 | `transcription` (语音转写) | `vosk_small` | wer, cer, channel_metrics |
| 10 | `speech_correction` (语音纠错) | `vosk_small` + `pycorrector_kenlm` | correction_precision/recall/f1 |

**真实后端覆盖率: 10/10**(详见 `docs/验收说明.md` §3)

---

## 2. 顶层目录结构(符合 preset 规范)

```text
speech_train_preset/                          # preset 根(顶层)
├── alg/speech_train_all-alg/                 # 算法工程(规范固定目录 + -alg 后缀)
│   ├── INFO.ini / README.md / PROFILE.jpg
│   └── main/
│       ├── config.yaml                       # train/evaluate/adapt/service 4 节
│       ├── train.sh / eval.sh / convert.sh / service.sh
│       ├── src/                              # train.py, evaluate.py, speech_tasks.py, backends.py
│       ├── haikit/                           # 平台 SDK 集成桩(主包+子包,本地降级)
│       └── tests/                            # 7 个单元测试
├── env/speech_train_all-env/                 # 环境资产(规范固定目录 + -env 后缀)
│   ├── INFO.ini / Dockerfile / Dockerfile.cpu
│   ├── requirements-cpu-open-source.txt / requirements-backends.txt
│   ├── wheels/                               # ── 580MB 完整 Python 离线包 ──
│   ├── ENVIRONMENT_MANIFEST.json             # v2.0.0 wheelhouse 模式
│   └── speech_train_all-env.tar.gz           # 580MB wheelhouse 完整离线环境
├── dataset/speech_train_sample-dataset/      # 数据集(规范固定目录 + -dataset 后缀)
│   ├── audio/{train,val,test}/               # 400 / 50 / 50 wav
│   ├── labels/{train,val,test}.jsonl         # 400 / 50 / 50 标注
│   ├── metadata.json
│   └── speech_train_sample-dataset.zip        # 4.6MB,561 文件
├── model/                                    # 模型(规范固定目录)
│   ├── speech_train_base-model/              # 入口模型(轻量 manifest,9 后端聚合)
│   │   ├── INFO.ini / README.md / PROFILE.jpg
│   │   ├── config/model_info.json
│   │   ├── model/{cpu_reference_model.json,README.md}
│   │   └── model.zip
│   ├── asr_vosk_small-cpu-model/             # ── 9 个独立 CPU 开源 model 资产 ──
│   ├── asr_sherpa_paraformer-cpu-model/      #     每个含 INFO/README/PROFILE + 真实权重
│   ├── asr_whisper_tiny-cpu-model/           #     取消 symlink,内含真实副本
│   ├── mt_opus_mt_en_zh-cpu-model/
│   ├── si_minilm_intent-cpu-model/
│   ├── sv_ecapa_voxceleb-cpu-model/
│   ├── ne_deepfilternet3-cpu-model/
│   ├── ne_rnnoise-cpu-model/
│   ├── sc_pycorrector-cpu-model/
│   ├── open_source_cpu_models/               # 9 个 backend 的 manifest 索引
│   └── real_cpu_models/                      # 真实权重"中央仓库"(token/config 索引)
├── docs/                                     # 附赠文档(非平台规范要求)
│   ├── 智能资产改造集成规范.pdf               # 规范原文
│   ├── 资产包准备说明.pdf                     # 规范原文
│   ├── 验收说明.md                            # 14KB,含 10 任务资产对应表 + 集成说明
│   ├── 交付物清单.md                          # 9.5KB,9 独立资产 + 真实后端
│   ├── 训练平台集成使用说明.md                # 平台集成步骤详细说明
│   ├── 10个CPU流程独立测试说明.md
│   ├── 10类模型算法离线测试能力检查报告.md
│   ├── 数据集规范.md / 容器镜像构建说明.md / ...
│   └── open_source_cpu_models.json           # 9 后端冻结清单
├── scripts/                                  # 附赠脚本(非平台规范要求)
│   ├── build_asset_package.py               # ── 端到端构建脚本(5 stage)──
│   ├── validate_asset_package.py            # 结构校验
│   ├── run_all_cpu_tasks.py                 # 10 任务批量入口
│   ├── cpu_flows/run_*.py                   # 10 任务独立入口
│   ├── cache_open_source_cpu_models.py       # 真实权重下载(清华源)
│   └── run_real_*.py                        # 真实模型推理
├── README.md
└── PROFILE.jpg
```

> **4 个核心子目录**(`alg/env/dataset/model`)为《资产包准备说明》要求;
> `docs/` 与 `scripts/` 为本资产包附赠,接收方可按需删除。

---

## 3. 端到端构建(`scripts/build_asset_package.py`)

5 个 stage,按需运行或全跑:

```bash
# 端到端全跑(默认 70 秒)
python3 scripts/build_asset_package.py
# 等价于:python3 scripts/build_asset_package.py all

# 单 stage
python3 scripts/build_asset_package.py code       # 校验算法工程
python3 scripts/build_asset_package.py model      # 复制真实权重到 9 个独立 model 资产
python3 scripts/build_asset_package.py dataset    # 重打 dataset zip(train/val/test)
python3 scripts/build_asset_package.py env        # 重打 env wheelhouse tar.gz
python3 scripts/build_asset_package.py package    # 重打总 zip(无 symlink)

# 跳过最终 validate
python3 scripts/build_asset_package.py --skip-validate all
```

每个 stage 完成后自动调用 `validate_asset_package.py` 复核。

---

## 4. 资产命名规范

每个资产子文件夹 = `任务前缀 + 标识 + 类型后缀`,
与 `INFO.ini` 的 `name` 字段一致:

| 资产类型 | 命名 | 示例 |
|---|---|---|
| 算法 | `<name>-alg` | `speech_train_all-alg` |
| 环境 | `<name>-env` | `speech_train_all-env` |
| 数据集 | `<name>-dataset` | `speech_train_sample-dataset` |
| 模型 | `<task>_<id>-cpu-model` | `asr_vosk_small-cpu-model`, `mt_opus_mt_en_zh-cpu-model` |

---

## 5. 交付说明

| 资产 | 路径 | 关键内容 |
|---|---|---|
| **算法** | `alg/speech_train_all-alg/` | 统一训练算法,含 config.yaml 4 节、4 启动脚本、haikit/、9 后端 backends.py |
| **环境** | `env/speech_train_all-env/` | wheelhouse 完整离线 Python 包(580MB / 75 whl) + Dockerfile + requirements |
| **数据集** | `dataset/speech_train_sample-dataset/` | 500 条 / 10 类,zip 内已划 `audio/{train,val,test}/` + `labels/*.jsonl` |
| **模型(入口)** | `model/speech_train_base-model/` | 9 后端聚合 manifest(2.9KB 轻量 zip) |
| **模型(9 独立)** | `model/<task>_<name>-cpu-model/` | 9 个独立 model 资产,每个含 INFO/README/PROFILE + 真实权重副本 |
| **真实权重仓库** | `model/real_cpu_models/` | 9 后端真实权重 + token/config 索引(本地保留,git 排除 >50MB) |
| **附赠文档** | `docs/` | 验收说明(14KB) + 交付物清单(9.5KB) + 平台集成说明 + 离线测试报告 |
| **附赠脚本** | `scripts/` | 端到端构建 + 校验 + 10 任务入口 + 真实权重下载 |

---

## 6. 验收命令

```bash
cd speech_train_preset

# 1. 资产结构自检
python3 scripts/validate_asset_package.py
# 期望:asset-package-ok

# 2. 端到端构建
python3 scripts/build_asset_package.py
# 期望:5 stages 全部 OK,asset-package-ok

# 3. 10 任务真实后端一键跑通
python3 scripts/run_all_cpu_tasks.py
# 期望:runs/all_cpu_tasks/summary.json(10 任务,50/50 推理成功)

# 4. haikit 桩自检
python3 -c "import sys; sys.path.insert(0, 'alg/speech_train_all-alg/main'); from haikit import HaikitContext, report_metrics; print('haikit OK')"

# 5. 单元测试
python3 -m unittest discover -s alg/speech_train_all-alg/main/tests -p 'test_*.py'
# 期望:Ran 10 tests OK
```

---

## 7. 已知限制

| 限制 | 说明 | 计划 |
|---|---|---|
| SpeechBrain 1.1.0 + torch 2.2.2 API 兼容 | `torch.amp.custom_fwd` 在 torch 2.2 中已 deprecated | 升级 speechbrain 到 1.2+ 或 torch 2.4+ |
| 合成 wav 对真实 ASR 是静音 | 测试用 wav 是合成静音,真实模型识别不出 → WER=1.0 | 替换数据集为真实录音(平台生产) |
| vosk 是英文模型 | 仅适合英文 ASR;code_switch_asr 改用 sherpa 中文 | 多语种时改用 sherpa 多语种 ONNX |
| 5 个 >50MB 真实权重未推送 git | 本地保留(大文件) | 改用 Git LFS 或 release tarball |

---

## 8. 相关文档

| 文档 | 内容 |
|---|---|
| `docs/验收说明.md` (14KB) | **完整验收清单** + 10 任务资产对应表 + 集成使用说明 |
| `docs/交付物清单.md` (9.5KB) | 9 独立资产 + 10 任务真实后端 + 镜像命名 |
| `docs/训练平台集成使用说明.md` (30KB) | 平台导入步骤 + 容器内挂载 + 4 类任务入口 + 产物采集 |
| `docs/10个CPU流程独立测试说明.md` | 10 个独立脚本 + 单流程运行步骤 |
| `docs/10类模型算法离线测试能力检查报告.md` | 真实测试状态 + 依赖完整性 + 单元测试 |
| `docs/数据集规范.md` | 标注字段 / 目录结构 / 音频要求 |
| `docs/容器镜像构建说明.md` | Dockerfile / 镜像命名 / 容器内约定 |
| `docs/open_source_cpu_models.json` | 9 后端冻结清单(task_types / python_packages / license) |
| `智能资产改造集成规范.pdf` | 平台规范原文 |
| `资产包准备说明.pdf` | 平台规范原文 |
