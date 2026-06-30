# StepAudio2-mini MS-SWIFT 训练

## 概述

本目录包含了使用 MS-SWIFT 框架训练 StepAudio2-mini 模型的完整流程。

## 文件说明

- `convert_to_swift_format.py`: CSV 到 JSONL 格式转换脚本（同时可选预先切片为独立 wav）
- `run_prepare_data.sh`: 一键数据准备脚本（CSV → 切片 → train/val 划分 → 音频可读性校验）
- `run_train_swift.sh`: MS-SWIFT 训练脚本（默认 LoRA，允许通过 `TUNER_TYPE=full` 切换全参微调）
- `run_inference.sh`: MS-SWIFT 推理脚本（自动判别 LoRA / full 检查点，支持多卡并行）
- `run_eval.sh`: 推理结果评估脚本（threshold / precision / recall / F1 分析）
- `eval_classification.py`: `run_eval.sh` 调用的评估实现
- `data/`: 训练数据目录（`train.jsonl` / `val.jsonl` / `val.small.jsonl`）

## 使用步骤

### 0. 环境准备（关键！）

ms-swift 4.x 要求 **Python ≥ 3.9**，本机已预装好的推荐环境是 `env-3.12.11`：

```bash
# 推荐：直接激活仓库推荐环境（已预装 PyYAML / PyTorch / transformers / swift CLI）
conda activate env-3.12.11
which swift   # 应输出 /data/miniconda3/envs/env-3.12.11/bin/swift
```

如果你忘了激活环境，**`run_train_swift.sh` / `run_inference.sh` 也会自动检测并使用** `/data/miniconda3/envs/env-3.12.11/bin/swift`，无需手动切换。

> ⚠️ 切勿在 `env-3.6.8` 等 Python < 3.9 的环境下运行，会出现 `swift: command not found` 或 `ModuleNotFoundError: No module named 'yaml'` 等错误。

### 1. 数据准备（一键）

```bash
INPUT_CSV=/path/to/your/data.csv \
    bash project/stepaudio/run_prepare_data.sh
# 可选环境变量：OUTPUT_DIR / VAL_RATIO / MAX_SAMPLES / SEED / NO_SLICE
```

脚本会依次执行：
1. 调用 `convert_to_swift_format.py` 将 CSV 转为 JSONL，并根据 `bot_audio_ts[0]` 预先切片 user_audio（避免训练时 IO 开销）
2. 过滤掉音频路径不存在的样本，并随机抽样用 `soundfile` 试读一下硕认可解码
3. 按 `VAL_RATIO`（默认 0.1）随机划分为 `train.jsonl` / `val.jsonl`

输出产物默认位于 `project/stepaudio/data/`：
```
data/
  all.jsonl              # 未划分的全量样本
  train.jsonl / val.jsonl
  sliced_audios/*.wav    # 切片后的音频
```

如果不需要预先切片，可选 `--no_slice`：
```bash
python project/stepaudio/convert_to_swift_format.py \
    --input /path/to/data.csv --output project/stepaudio/data/train.jsonl --no_slice
```

### 2. 运行训练

```bash
# 默认 LoRA（推荐先跑通流程）
bash project/stepaudio/run_train_swift.sh

# 全参微调
TUNER_TYPE=full bash project/stepaudio/run_train_swift.sh
```

常用超参环境变量：`TUNER_TYPE` / `LEARNING_RATE` / `NUM_EPOCHS` / `BATCH_SIZE` / `MAX_LENGTH` /
`LORA_RANK` / `LORA_ALPHA` / `LORA_TARGET_MODULES`。

### 3. 运行推理

```bash
# 自动选中 OUTPUT_DIR 下最新的 checkpoint，自动识别是 LoRA 还是 full
bash project/stepaudio/run_inference.sh
# 或手工指定检查点：
MODEL_PATH=/path/to/output/v10-xxx/checkpoint-6 bash project/stepaudio/run_inference.sh
# 多卡并行（自动 dataset.shard 切片再合并）
MODEL_PATH=/path/to/checkpoint NPROC_PER_NODE=4 bash project/stepaudio/run_inference.sh
# 推理结果默认保存到 project/stepaudio/infer_results/result_<ckpt>_<ts>.jsonl
```

## 关键特性

1. **音频动态切片**: 音频在模型加载时根据 `_bot_audio_end_time` 自动切片（从 0 到 bot 回复时刻）
2. **TTS Label 构建**: 自动构建文本 token 和 audio token 交织的 labels
3. **自定义 Preprocessor**: 保留 messages 中的额外字段（response 和 _bot_stokens）
4. **多样本生成**: CSV 的一行自动生成多个训练样本

## 数据格式

### 输入 CSV 格式
```csv
user_audio,bot_text,bot_stokens,bot_audio_ts,speak_type,bot_info,...
```

### 输出 JSONL 格式
```jsonl
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<audio>"},
    {
      "role": "assistant",
      "content": "<speak_reply><tts_start>",
      "response": "你好",
      "_bot_stokens": [1,2,3,...]
    }
  ],
  "audios": ["/path/to/audio.wav"],
  "_bot_audio_end_time": 5.47
}
```

## 注意事项

- 确保设置 `export LOG_LEVEL=INFO` 以避免 TRACE 日志错误
- 自定义 preprocessor 会自动加载（通过 `swift/dataset/dataset/stepaudio.py`）
- 模型路径需要根据实际情况修改
