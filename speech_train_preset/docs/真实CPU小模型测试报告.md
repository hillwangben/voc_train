# 真实 CPU 小模型测试报告

## 测试目标

下载并使用真实开源 CPU 小模型进行 ASR 推理测试，测试数据集控制在 50 条以内。

## 模型

- 模型：Vosk small English `vosk-model-small-en-us-0.15`
- 模型目录：`model/real_cpu_models/vosk-model-small-en-us-0.15`
- 运行设备：CPU
- 模型大小：约 68MB 解压后目录

## 数据集

- 样本数：10
- 数据来源：本地离线 TTS 生成英文 wav，用于真实 ASR 推理链路验证
- 数据目录：`runs/real_vosk_asr/dataset`
- 数据限制：小于 50 条

## 指标

- WER：0.567857
- CER：0.252000

## 逐条结果

| utt_id | reference | prediction | WER | CER |
| --- | --- | --- | --- | --- |
| real_vosk_001 | one zero zero one nine oh two | one zero zero one i though to | 0.429 | 0.261 |
| real_vosk_002 | open the training dashboard | open the trading on board | 0.750 | 0.250 |
| real_vosk_003 | start speech recognition training | start speech recognition training | 0.000 | 0.000 |
| real_vosk_004 | check the model evaluation report | cutler model evaluation report | 0.400 | 0.207 |
| real_vosk_005 | deploy the voice correction service | the ploy my voice correction service | 0.600 | 0.161 |
| real_vosk_006 | query the speaker verification result | we're in a speaker the occasion me now | 1.600 | 0.545 |
| real_vosk_007 | run the noise enhancement test | noi and arm's length | 1.000 | 0.692 |
| real_vosk_008 | translate the speech into english | translate los each into english | 0.400 | 0.172 |
| real_vosk_009 | record the meeting transcript | that on the meeting transcript | 0.500 | 0.231 |
| real_vosk_010 | switch to channel number two | switch to channel number two | 0.000 | 0.000 |

## 结论

真实 Vosk small 模型已在 CPU 环境完成推理测试。结果不是满分模拟，说明当前报告来自真实模型识别输出。后续若替换为真实人工语音数据或更适配的语音模型，指标可进一步改善。
