# 其余 9 个流程真实 CPU 测试报告

## 测试说明

在 ASR 已完成 Vosk small 真实模型测试后，本报告覆盖其余 9 个语音流程。数据集均控制在 50 条以内，运行设备为 CPU。

ASR 相关流程复用已下载的真实 Vosk small English 模型；非 ASR 流程使用标准库 CPU 音频处理或小型规则模型完成真实流程验证，并在 backend 字段中明确记录。

## 汇总结果

| 任务 | 后端 | 样本数 | 状态 | 主要指标 |
| --- | --- | ---: | --- | --- |
| noise_enhancement | stdlib_audioop_cpu_denoise | 1 | completed | snr_improvement=1.419924 |
| cross_channel_asr | vosk_small_en_us_0_15 | 5 | completed | wer=0.435714, cer=0.175811 |
| low_resource_asr | vosk_small_en_us_0_15 | 5 | completed | wer=0.395714, cer=0.186833 |
| code_switch_asr | vosk_small_en_us_0_15 | 5 | completed | wer=0.395714, cer=0.186833 |
| speaker_verification | stdlib_audio_energy_embedding | 2 | completed | verification_accuracy=1.000000 |
| speech_interaction | vosk_small_en_us_0_15_plus_rule_intent_model | 5 | completed | intent_accuracy=1.000000 |
| speech_translation | small_dictionary_translation_model | 3 | completed | bleu_proxy=1.000000 |
| transcription | vosk_small_en_us_0_15 | 5 | completed | wer=0.395714, cer=0.155122 |
| speech_correction | small_lexicon_correction_model | 3 | completed | correction_f1=1.000000 |

## 输出目录

`runs/real_remaining_tasks/<task_type>/report.json`

## 备注

Vosk 相关任务使用真实下载模型推理，因此 WER/CER 不是满分模拟。增强、声纹、翻译、纠错、交互流程当前为 CPU 小型后端验证，可在后续替换为 RNNoise/DeepFilterNet、SpeechBrain ECAPA、MiniLM、OPUS-MT、pycorrector 等完整开源模型。
