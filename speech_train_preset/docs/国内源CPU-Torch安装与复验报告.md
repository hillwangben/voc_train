# 国内源安装 CPU Torch 与开源模型替换复验报告

## 安装结果

- `torch==2.2.2+cpu`：已通过上海交大 PyTorch wheels 镜像安装。
- `torchaudio==2.2.2+cpu`：已通过上海交大 PyTorch wheels 镜像安装。
- `transformers==4.40.2`、`sentencepiece`：已通过清华 PyPI 镜像安装。
- `speechbrain`、`deepfilternet`、`pycorrector`：已安装。

验证：`torch.cuda.is_available()` 为 `False`，当前为 CPU-only。

## 复验结果

| 任务 | 后端 | 状态 | 样本数 | 说明/指标 |
| --- | --- | --- | ---: | --- |
| noise_enhancement | deepfilternet_cpu | attempted | 1 | DeepFilterNet and torchaudio CPU dependencies are installed and CLI help works, but full enhancement inference is skipped in bounded platform integration runs because the CLI call did not return reliably in this environment. |
| cross_channel_asr | vosk_small_en_us_0_15 | completed | 1 | wer=0.428571; cer=0.260870 |
| low_resource_asr | vosk_small_en_us_0_15 | completed | 1 | wer=0.428571; cer=0.260870 |
| code_switch_asr | vosk_small_en_us_0_15 | completed | 1 | wer=0.428571; cer=0.260870 |
| speaker_verification | speechbrain_ecapa_voxceleb | attempted | 2 | SpeechBrain and CPU torch are installed. ECAPA model loading is skipped in the bounded integration script because partial HuggingFace cache can block; run a dedicated speaker verification script after caching model weights. |
| speech_interaction | vosk_small_en_us_0_15_plus_minilm_intent | completed | 1 | intent_accuracy=1.000000 |
| speech_translation | helsinki_opus_mt_tiny | attempted | 3 | Transformers and sentencepiece are installed, but OPUS-MT weights are not cached locally. Online HuggingFace model download is skipped to keep integration runs bounded. |
| transcription | vosk_small_en_us_0_15 | completed | 1 | wer=0.428571; cer=0.260870 |
| speech_correction | pycorrector_kenlm | attempted | 3 | pycorrector imports successfully after CPU torch installation, but full corrector initialization is skipped in bounded integration runs because it may load additional language resources. |

## 结论

国内源可成功安装 CPU torch/torchaudio，解决了此前 torch 不可用的问题。Vosk 相关流程继续完成真实 CPU 推理。DeepFilterNet、SpeechBrain、OPUS-MT、pycorrector 的依赖已经具备或部分具备，但为了避免平台集成主脚本在模型下载或长时间推理中阻塞，当前主脚本仍将部分流程记录为 `attempted`，并建议在模型权重预缓存后执行专用验证脚本。
