import json
import os
from pathlib import Path

# 源文件与目标文件路径
SRC_PATH = Path("/home/chengz/LAMs/mcp_chatbot-audio/mcp_servers/tool_descriptions.json")
DST_PATH = SRC_PATH.parent / "tool_descriptions_api.json"

EXTRA_TOOLS = {
    # Audio Processor tools
    "load_audio",
    "resample_audio",
    "compute_stft",
    "compute_mfcc",
    "compute_mel_spectrogram",
    "add_reverb",
    "mix_audio",
    "apply_fade",
    "serve_local_audio",
    "stop_audio_server",
    # DSP Processor tools
    "convert_audio_format",
    "trim_audio",
    "align_audio_lengths",
    # Markdown Processor tools
    "read_file",
    "write_file",
    "modify_file",
}


def contains_api(text: str) -> bool:
    """判断字符串中是否包含 'api'（忽略大小写）。"""
    return "_api" in text.lower()


def main():
    if not SRC_PATH.exists():
        raise FileNotFoundError(f"找不到源文件: {SRC_PATH}")

    with SRC_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # 根据键名或 description / detailed_description 中是否包含 "api"，
    # 或者工具名属于 EXTRA_TOOLS 进行过滤
    filtered = {
        name: info
        for name, info in data.items()
        if (
            contains_api(name)
            or contains_api(info.get("description", ""))
            or contains_api(info.get("detailed_description", ""))
            or name in EXTRA_TOOLS
        )
    }

    # 将结果写入目标文件，保持 UTF-8 并美化缩进
    with DST_PATH.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"已成功将 {len(filtered)} 条记录写入 {DST_PATH}")


if __name__ == "__main__":
    main() 