# 统一语音垂直训练算法

本算法工程采用统一入口支持多类语音训练任务，平台通过 `config.yaml` 的 `parameters.task_type` 选择具体能力。

## 支持任务

| task_type | 说明 |
| --- | --- |
| asr | 语音识别和语音转写 |
| noise_enhancement | 复杂噪声语音增强 |
| cross_channel_asr | 跨信道连续语音识别 |
| low_resource_asr | 小语种连续语音识别 |
| code_switch_asr | 混合语种连续语音识别 |
| speaker_verification | 声纹识别 |
| speech_interaction | 语音智能交互 |
| speech_translation | 多语言语音翻译 |
| speech_correction | 语音纠错 |

## 入口脚本

- `main/train.sh`
- `main/eval.sh`
- `main/convert.sh`
- `main/service.sh`

正式交付时，将真实训练代码放入 `main/src/`，并在上述脚本中调用。

