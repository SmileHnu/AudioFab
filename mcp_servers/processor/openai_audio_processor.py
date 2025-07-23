import os
from mcp.server.fastmcp import FastMCP
from models.openai_audio_api import transcribe_audio, text_to_speech
from datetime import datetime
from typing import Dict, Any

AUDIO_OUTPUT_DIR = "/home/chengz/LAMs/mcp_chatbot-audio/output/audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

mcp = FastMCP("OpenAI Audio API Tool Server")

@mcp.tool()
def openai_transcribe(
    file_path: str,
    language: str = "zh",
    model: str = "whisper-1"
) -> Dict[str, Any]:
    """
    使用 OpenAI Whisper API 进行音频转文本（ASR）。
    支持多种语言和模型选择。
    
    参数：
        file_path: 本地音频文件路径（支持 wav/mp3/ogg/flac 等）
        language: 语言代码（如 'en', 'zh'，可选，默认 'zh'）
        model: OpenAI Whisper 模型名（可选，默认 'whisper-1'）
    返回：
        包含转录文本的字典
    """
    try:
        text = transcribe_audio(file_path, model=model, language=language)
        return {
            "success": True,
            "text": text
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@mcp.tool()
def openai_tts(
    text: str,
    voice: str = "alloy",
    model: str = "tts-1",
    output_filename: str = None
) -> Dict[str, Any]:
    """
    使用 OpenAI TTS API 进行文本转语音，支持多种声音风格和模型选择。
    生成的音频文件以 wav 格式保存在 /mnt/d/LAMs/mcp_chatbot-audio/gen_audio 目录。
    
    参数：
        text: 要合成的文本
        voice: 声音风格（如 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'，可选，默认 'alloy'）
        model: OpenAI TTS 模型名（可选，默认 'tts-1'）
        output_filename: 输出文件名（可选，不含路径和后缀，默认自动生成）
    返回：
        包含生成的音频文件路径的字典
    """
    try:
        audio_bytes = text_to_speech(text, model=model, voice=voice)
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"openai_tts_{timestamp}"
        output_path = os.path.join(AUDIO_OUTPUT_DIR, output_filename + ".wav")
        with open(output_path, "wb") as f:
            f.write(audio_bytes)
        return {
            "success": True,
            "file_path": output_path
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

if __name__ == "__main__":
    mcp.run() 