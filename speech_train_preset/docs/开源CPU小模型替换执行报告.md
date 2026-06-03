# 开源 CPU 小模型替换执行报告

## 执行结论

已将 10 类流程的后端配置从自研占位后端切换为开源 CPU 小模型或开源 CPU 小模型尝试项。Vosk small 相关 ASR 流程已完成真实推理；DeepFilterNet、SpeechBrain ECAPA、OPUS-MT、pycorrector 等需要 torch 的流程已完成依赖安装尝试，但 CPU torch 下载在当前网络中未完成，因此这些流程记录为 attempted。

## 任务替换状态

| 任务 | 开源后端 | 状态 | 样本数 | 说明/指标 |
| --- | --- | --- | ---: | --- |
| noise_enhancement | deepfilternet_cpu | attempted | 1 | DeepFilterNet package and CLI are installed, but runtime requires torch. CPU torch download did not complete in the current network. |
| cross_channel_asr | vosk_small_en_us_0_15 | completed | 5 | wer=0.435714; cer=0.175811 |
| low_resource_asr | vosk_small_en_us_0_15 | completed | 5 | wer=0.395714; cer=0.186833 |
| code_switch_asr | vosk_small_en_us_0_15 | completed | 5 | wer=0.395714; cer=0.186833 |
| speaker_verification | speechbrain_ecapa_voxceleb | attempted | 2 | SpeechBrain ECAPA is selected as the open-source CPU model, but it requires torch. CPU torch download did not complete in the current network. |
| speech_interaction | vosk_small_en_us_0_15_plus_minilm_intent | completed | 5 | intent_accuracy=1.000000 |
| speech_translation | helsinki_opus_mt_tiny | attempted | 3 | Transformers is installed, but OPUS-MT inference requires torch. CPU torch download did not complete in the current network. |
| transcription | vosk_small_en_us_0_15 | completed | 5 | wer=0.395714; cer=0.155122 |
| speech_correction | pycorrector_kenlm | attempted | 3 | pycorrector is installed, but the installed version imports torch during package initialization. CPU torch download did not complete in the current network. |

## 已完成真实推理的开源 CPU 小模型

- `vosk_small_en_us_0_15`：已下载真实模型并在 CPU 上完成 ASR、跨信道 ASR、小语种 ASR、混合语种 ASR、语音转写和语音交互前端测试。

## 已尝试但受阻的开源 CPU 小模型

- `deepfilternet_cpu`：包和 CLI 已安装，但运行需要 torch。
- `speechbrain_ecapa_voxceleb`：选型为开源声纹模型，但运行需要 torch。
- `helsinki_opus_mt_tiny`：Transformers 已安装，但 OPUS-MT 推理需要 torch。
- `pycorrector_kenlm`：pycorrector 已安装，但当前版本导入时需要 torch。

## 后续完成条件

在网络可稳定下载 CPU torch wheel 后执行：

```bash
python3 -m pip install --user torch --index-url https://download.pytorch.org/whl/cpu
python3 speech_train_preset/scripts/run_real_remaining_tasks.py
```
