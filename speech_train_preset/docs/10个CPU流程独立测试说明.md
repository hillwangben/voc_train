# 10 个 CPU 流程独立测试说明

## 说明

为便于已有训练平台逐流程集成和验收，当前资产包已将 10 个 CPU 测试流程拆分为 10 个独立入口脚本。每个脚本只执行一个 `task_type`，并只在自身任务目录下写入训练、评估和汇总报告。

公共执行逻辑位于：

```text
scripts/cpu_flows/run_cpu_flow.py
```

独立流程输出默认位于：

```text
runs/cpu_flows/<task_type>/
```

## 独立流程清单

| 序号 | task_type | 能力 | 脚本 | 指标 |
| ---: | --- | --- | --- | --- |
| 1 | `asr` | 语音识别 | `scripts/cpu_flows/run_asr.py` | WER、CER |
| 2 | `noise_enhancement` | 复杂噪声语音增强 | `scripts/cpu_flows/run_noise_enhancement.py` | snr_improvement、pesq_proxy、stoi_proxy |
| 3 | `cross_channel_asr` | 跨信道连续语音识别 | `scripts/cpu_flows/run_cross_channel_asr.py` | WER、CER、channel_metrics |
| 4 | `low_resource_asr` | 小语种连续语音识别 | `scripts/cpu_flows/run_low_resource_asr.py` | WER、CER |
| 5 | `code_switch_asr` | 混合语种连续语音识别 | `scripts/cpu_flows/run_code_switch_asr.py` | WER、CER |
| 6 | `speaker_verification` | 声纹识别 | `scripts/cpu_flows/run_speaker_verification.py` | eer_proxy、verification_accuracy |
| 7 | `speech_interaction` | 语音智能交互 | `scripts/cpu_flows/run_speech_interaction.py` | intent_accuracy、slot_f1 |
| 8 | `speech_translation` | 多语言语音翻译 | `scripts/cpu_flows/run_speech_translation.py` | bleu_proxy、chrf_proxy |
| 9 | `transcription` | 语音转写 | `scripts/cpu_flows/run_transcription.py` | WER、CER |
| 10 | `speech_correction` | 语音纠错 | `scripts/cpu_flows/run_speech_correction.py` | correction_precision、correction_recall、correction_f1 |

## 单流程运行方式

### 语音识别

```bash
python3 scripts/cpu_flows/run_asr.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/asr/train/checkpoint.json
../runs/cpu_flows/asr/train/train_metrics.json
../runs/cpu_flows/asr/evaluate/metrics.json
../runs/cpu_flows/asr/summary.json
```

### 复杂噪声语音增强

```bash
python3 scripts/cpu_flows/run_noise_enhancement.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/noise_enhancement/train/checkpoint.json
../runs/cpu_flows/noise_enhancement/train/train_metrics.json
../runs/cpu_flows/noise_enhancement/evaluate/metrics.json
../runs/cpu_flows/noise_enhancement/summary.json
```

### 跨信道连续语音识别

```bash
python3 scripts/cpu_flows/run_cross_channel_asr.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/cross_channel_asr/train/checkpoint.json
../runs/cpu_flows/cross_channel_asr/train/train_metrics.json
../runs/cpu_flows/cross_channel_asr/evaluate/metrics.json
../runs/cpu_flows/cross_channel_asr/summary.json
```

### 小语种连续语音识别

```bash
python3 scripts/cpu_flows/run_low_resource_asr.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/low_resource_asr/train/checkpoint.json
../runs/cpu_flows/low_resource_asr/train/train_metrics.json
../runs/cpu_flows/low_resource_asr/evaluate/metrics.json
../runs/cpu_flows/low_resource_asr/summary.json
```

### 混合语种连续语音识别

```bash
python3 scripts/cpu_flows/run_code_switch_asr.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/code_switch_asr/train/checkpoint.json
../runs/cpu_flows/code_switch_asr/train/train_metrics.json
../runs/cpu_flows/code_switch_asr/evaluate/metrics.json
../runs/cpu_flows/code_switch_asr/summary.json
```

### 声纹识别

```bash
python3 scripts/cpu_flows/run_speaker_verification.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/speaker_verification/train/checkpoint.json
../runs/cpu_flows/speaker_verification/train/train_metrics.json
../runs/cpu_flows/speaker_verification/evaluate/metrics.json
../runs/cpu_flows/speaker_verification/summary.json
```

### 语音智能交互

```bash
python3 scripts/cpu_flows/run_speech_interaction.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/speech_interaction/train/checkpoint.json
../runs/cpu_flows/speech_interaction/train/train_metrics.json
../runs/cpu_flows/speech_interaction/evaluate/metrics.json
../runs/cpu_flows/speech_interaction/summary.json
```

### 多语言语音翻译

```bash
python3 scripts/cpu_flows/run_speech_translation.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/speech_translation/train/checkpoint.json
../runs/cpu_flows/speech_translation/train/train_metrics.json
../runs/cpu_flows/speech_translation/evaluate/metrics.json
../runs/cpu_flows/speech_translation/summary.json
```

### 语音转写

```bash
python3 scripts/cpu_flows/run_transcription.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/transcription/train/checkpoint.json
../runs/cpu_flows/transcription/train/train_metrics.json
../runs/cpu_flows/transcription/evaluate/metrics.json
../runs/cpu_flows/transcription/summary.json
```

### 语音纠错

```bash
python3 scripts/cpu_flows/run_speech_correction.py --output-root ../runs/cpu_flows
```

输出：

```text
../runs/cpu_flows/speech_correction/train/checkpoint.json
../runs/cpu_flows/speech_correction/train/train_metrics.json
../runs/cpu_flows/speech_correction/evaluate/metrics.json
../runs/cpu_flows/speech_correction/summary.json
```

## 汇总运行方式

如需一次性运行 10 个独立流程，可执行：

```bash
python3 scripts/run_all_cpu_tasks.py
```

该脚本内部复用 `scripts/cpu_flows/run_cpu_flow.py`，并汇总输出：

```text
../runs/all_cpu_tasks/summary.json
```

## 已执行验证

已逐个执行 10 个独立脚本，全部生成独立 `summary.json`，并通过单元测试：

```bash
python3 -m unittest speech_train_preset/alg/speech_train_all_in_one/main/tests/test_independent_cpu_flow_scripts.py
```
