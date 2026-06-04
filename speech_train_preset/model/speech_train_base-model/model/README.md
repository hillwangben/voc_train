# speech_train_base-model

## 资产说明

- **类型**: 语音类 CPU 基础预训练入口(聚合 manifest)
- **定位**: **不存放具体模型权重**,而是作为 9 个独立 CPU 开源 model 资产的统一入口
- **真实权重**: 见 `model/<task>_<name>-cpu-model/` 下的 `model/<name>/` 软链目标

## backend 资产清单

| 任务 | 推荐 backend 资产 |
|---|---|
| asr / cross_channel_asr / low_resource_asr / code_switch_asr / transcription | asr_whisper_tiny-cpu-model, asr_vosk_small-cpu-model, asr_sherpa_paraformer-cpu-model |
| speech_translation | mt_opus_mt_en_zh-cpu-model |
| speech_interaction | si_minilm_intent-cpu-model |
| speaker_verification | sv_ecapa_voxceleb-cpu-model |
| noise_enhancement | ne_deepfilternet3-cpu-model, ne_rnnoise-cpu-model |
| speech_correction | sc_pycorrector-cpu-model |

## 加载方式

算法 `config.yaml` 中以 `model_dir: "/workspace/model/<具体资产名>"` 引用,
平台加载时根据 `model/<具体资产>/model/<name>/` 相对软链定位到真实权重
(`model/real_cpu_models/<src>/`)。
