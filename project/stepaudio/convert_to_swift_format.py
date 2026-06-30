#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据格式转换脚本: CSV → MS-SWIFT JSONL
将 StepAudio CSV 格式转换为 MS-SWIFT 标准格式
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, Any
import pandas as pd
import torchaudio


# 系统提示词模板（参考 DEFAULT_SYSTEM_TEMPLATE）
DEFAULT_SYSTEM_TEMPLATE = """--- SYSTEM ---
{bot_setting}
--- INSTRUCTION ---
{instruction}
你对用户的了解是：
{user_profile}
注意：对于用户的其他信息（例如年龄、家乡、毕业学校、职业、社会关系、婚恋情况等），如果你想了解，都可以作为聊天话题的候选。如果对于用户的某些信息有疑惑，你可以问用户，而不是擅自揣测和臆想。
请结合聊天记录输出你的回复。
--- CHAT_HISTORY ---
下面是你和用户的对话历史：
{memory}

下面给出了用户最新的音频输入，你需要结合以上信息给出相应回复"""


def format_system_content(bot_info: Dict[str, Any]) -> str:
    """格式化系统提示词"""
    # 创建一个 defaultdict，缺失字段返回空字符串
    safe_bot_info = defaultdict(str, bot_info)

    # 检查是否有 env_time 字段
    if "env_time" in bot_info:
        template = """--- SYSTEM ---
{bot_setting}
--- INSTRUCTION ---
{instruction}
你对用户的了解是：
{user_profile}
注意：对于用户的其他信息（例如年龄、家乡、毕业学校、职业、社会关系、婚恋情况等），如果你想了解，都可以作为聊天话题的候选。如果对于用户的某些信息有疑惑，你可以问用户，而不是擅自揣测和臆想。
请结合聊天记录输出你的回复。
--- CHAT_HISTORY ---
下面是你和用户的对话历史：
{memory}

--- 环境信息 ---
{env_time}
下面给出了用户最新的音频输入，你需要结合以上信息给出相应回复"""
    else:
        template = DEFAULT_SYSTEM_TEMPLATE

    content = template.format_map(safe_bot_info)
    # 清理多余的空行
    return "\n".join([s.strip() for s in content.split("\n")]).strip()


def get_speak_token(speak_type: str) -> str:
    """获取 speak token"""
    mapping = {
        "reply": "<speak_reply>",
        "backchannel": "<speak_backchannel>",
    }
    return mapping.get(speak_type, "<speak_reply>")


def save_audio_slice(audio_path: str, start_sec: float, end_sec: float, output_path: str) -> str:
    """
    切片音频并保存到新文件

    Args:
        audio_path: 原始音频路径
        start_sec: 开始时间（秒）
        end_sec: 结束时间（秒）
        output_path: 输出文件路径

    Returns:
        输出文件的绝对路径
    """
    # 加载音频
    waveform, sample_rate = torchaudio.load(audio_path)

    # 转为单声道
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # 计算采样点
    start_sample = int(start_sec * sample_rate)
    end_sample = int(end_sec * sample_rate)

    # 切片
    sliced_waveform = waveform[:, start_sample:end_sample]

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, sliced_waveform, sample_rate)

    return os.path.abspath(output_path)


def convert_row_to_samples(row: pd.Series, output_audio_dir: str, row_idx: int,
                          do_slice: bool = True) -> list:
    """
    将 CSV 的一行转换为多个训练样本

    每个 bot_text/bot_stokens 对应一个样本。
    如果 do_slice=True，会根据 bot_audio_ts[0] 切片原始 user_audio 并保存到
    output_audio_dir/rowX_sampleY.wav，最终 audios 字段使用切片后的绝对路径；
    否则仅使用原始 user_audio 路径。
    """
    samples = []

    # 解析字段
    try:
        bot_text_list = eval(row["bot_text"])
        bot_stokens_list = eval(row["bot_stokens"])
        bot_audio_ts_list = eval(row["bot_audio_ts"])
        speak_type_list = eval(row["speak_type"])
        bot_info = json.loads(row["bot_info"])
    except Exception as e:
        print(f"[WARNING] 解析第 {row_idx} 行失败: {e}")
        return samples

    # 检查列表长度
    if not (len(speak_type_list) == len(bot_text_list) == len(bot_stokens_list) == len(bot_audio_ts_list)):
        print(f"[WARNING] 第 {row_idx} 行字段长度不一致，跳过")
        return samples

    if len(speak_type_list) == 0:
        return samples

    # 解析 session history
    session_history_bot_text = []
    session_history_bot_audio_ts = []
    if "session_history_bot_text" in row and pd.notna(row["session_history_bot_text"]):
        try:
            session_history_bot_text = eval(row["session_history_bot_text"])
            session_history_bot_audio_ts = eval(row["session_history_bot_audio_ts"])
        except:
            pass

    # 格式化 system prompt
    system_content = format_system_content(bot_info)

    # 为每个 bot_text 创建样本
    for idx, (bot_text, bot_stokens, bot_audio_ts, speak_type) in enumerate(
        zip(bot_text_list, bot_stokens_list, bot_audio_ts_list, speak_type_list)
    ):
        # 过滤太长或太短的样本
        if len(bot_stokens) > 750 or bot_audio_ts[0] < 2 or bot_audio_ts[0] > 60:
            continue

        if len(str(bot_info)) > 4000:
            continue

        # 构建 messages
        messages = [{"role": "system", "content": system_content}]

        # 处理音频路径（统一转换成绝对路径，避免训练时 cwd 切换后找不到文件）
        user_audio_path = row["user_audio"]
        if not os.path.isabs(user_audio_path):
            user_audio_path = os.path.abspath(user_audio_path)

        # 记录音频切片的结束时间（bot 回复开始的时刻）
        end_time = bot_audio_ts[0]

        # 可选：预先切片保存，让训练时 swift 直接读切好的音频。
        # 如果切片失败或原始文件不存在，跳过该样本，避免后续训练时报 LibsndfileError。
        sliced_audio_path = user_audio_path
        if do_slice:
            if not os.path.exists(user_audio_path):
                print(f"[WARN] 原始音频不存在，跳过 row={row_idx} sample={idx}: {user_audio_path}")
                continue
            sliced_audio_path = os.path.join(output_audio_dir, f"row{row_idx}_sample{idx}.wav")
            try:
                sliced_audio_path = save_audio_slice(
                    user_audio_path, 0.0, float(end_time), sliced_audio_path)
            except Exception as e:
                print(f"[WARN] 切片失败，跳过 row={row_idx} sample={idx}: {e}")
                continue

        # 添加 user message
        messages.append({"role": "user", "content": "<audio>"})

        # 添加 assistant message（只包含特殊 token 和 tts_start）
        # 将 response 和 _bot_stokens 作为额外字段存储在 assistant message 中
        speak_token = get_speak_token(speak_type)
        messages.append({
            "role": "assistant",
            "content": f"{speak_token}<tts_start>",
            "response": bot_text,  # 将响应文本存储在 message 中
            "_bot_stokens": bot_stokens,  # 将 audio tokens 存储在 message 中
        })

        # 构建样本
        sample = {
            "messages": messages,
            "audios": [sliced_audio_path],
            "_speak_type": speak_type,
            "_bot_audio_end_time": end_time,  # 保留元数据，便于调试；swift 本身并不读取该字段
        }

        # 更新 session history（累积）
        session_history_bot_text.append(bot_text)
        session_history_bot_audio_ts.append(bot_audio_ts)

        samples.append(sample)

    return samples


def convert_csv_to_jsonl(input_csv: str, output_jsonl: str, output_audio_dir: str = None,
                         max_samples: int = None, do_slice: bool = True):
    """
    将 StepAudio CSV 转换为 MS-SWIFT JSONL 格式

    Args:
        input_csv: 输入 CSV 文件路径
        output_jsonl: 输出 JSONL 文件路径
        output_audio_dir: 输出切片音频的目录（默认 <output_jsonl_dir>/sliced_audios）
        max_samples: 最大样本数（用于调试）
        do_slice: 是否在此阶段切片并存为独立 wav
    """
    if output_audio_dir is None:
        output_audio_dir = os.path.join(os.path.dirname(output_jsonl) or ".", "sliced_audios")
    output_audio_dir = os.path.abspath(output_audio_dir)

    print(f"读取 CSV 文件: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"CSV 共有 {len(df)} 行，切片输出目录: {output_audio_dir} (do_slice={do_slice})")

    # 转换所有行
    all_samples = []
    for idx, row in df.iterrows():
        samples = convert_row_to_samples(row, output_audio_dir, idx, do_slice=do_slice)
        all_samples.extend(samples)

        if max_samples and len(all_samples) >= max_samples:
            print(f"已达到最大样本数 {max_samples}，停止处理")
            all_samples = all_samples[:max_samples]
            break

    print(f"共生成 {len(all_samples)} 个训练样本")

    # 保存为 JSONL
    print(f"保存到 JSONL 文件: {output_jsonl}")
    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("转换完成！")

    # 打印第一个样本作为示例
    if all_samples:
        print("\n示例样本（第一个）:")
        print(json.dumps(all_samples[0], ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="将 StepAudio CSV 转换为 MS-SWIFT JSONL 格式")
    parser.add_argument("--input", type=str, required=True, help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSONL 文件路径（默认：data/train.jsonl）")
    parser.add_argument("--output_audio_dir", type=str, default=None,
                        help="输出切片音频的目录（如果需要切片）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（用于调试）")
    parser.add_argument("--no_slice", action="store_true",
                        help="不预先切片，直接引用原始 user_audio（默认则预切为独立 wav）")
    args = parser.parse_args()

    # 设置默认输出路径
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output = os.path.join(script_dir, "data", "train.jsonl")

    convert_csv_to_jsonl(
        input_csv=args.input,
        output_jsonl=args.output,
        output_audio_dir=args.output_audio_dir,
        max_samples=args.max_samples,
        do_slice=not args.no_slice,
    )


if __name__ == "__main__":
    main()
