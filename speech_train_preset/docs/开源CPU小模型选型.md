# 开源 CPU 小模型选型

本交付包采用双层实现：

- `cpu_reference`：内置轻量训练/评估实现，不下载模型、不依赖 GPU，用于资产导入、数据格式和任务链路验收。
- 开源 CPU 后端：按任务接入 Vosk、sherpa-onnx、Whisper tiny、RNNoise、DeepFilterNet、SpeechBrain ECAPA、MiniLM、OPUS-MT、pycorrector。

完整机器可读清单见：

```text
docs/open_source_cpu_models.json
```

## 任务与后端

| 任务 | 推荐 CPU 后端 |
| --- | --- |
| 语音识别 | Vosk small、sherpa-onnx int8、Whisper tiny |
| 复杂噪声语音增强 | RNNoise、DeepFilterNet |
| 跨信道连续语音识别 | sherpa-onnx int8、Vosk small |
| 小语种连续语音识别 | Vosk small multilingual、sherpa-onnx |
| 混合语种连续语音识别 | Whisper tiny、sherpa-onnx bilingual |
| 声纹识别 | SpeechBrain ECAPA-TDNN |
| 语音智能交互 | ASR + MiniLM 意图分类 |
| 多语言语音翻译 | Whisper tiny/base、OPUS-MT tiny |
| 语音转写 | Vosk small、sherpa-onnx、Whisper tiny |
| 语音纠错 | pycorrector、KenLM、领域词典 |

