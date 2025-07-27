import requests
import base64
import os

OPENAI_API_KEY = os.getenv("LLM_API_KEY")
OPENAI_API_BASE = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
}

def transcribe_audio(file_path: str, model: str = "whisper-1", language: str = "zh") -> str:
    """
    调用 OpenAI API 进行音频转文本（ASR）。
    :param file_path: 本地音频文件路径
    :param model: OpenAI 支持的语音识别模型
    :param language: 语言代码（如 'en', 'zh'）
    :return: 转录文本
    """
    url = f"{OPENAI_API_BASE}/audio/transcriptions"
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "audio/wav")}
        data = {"model": model, "language": language}
        response = requests.post(url, headers=HEADERS, files=files, data=data)
    response.raise_for_status()
    return response.json().get("text", "")

def text_to_speech(text: str, model: str = "tts-1", voice: str = "alloy") -> bytes:
    """
    调用 OpenAI API 进行文本转语音（TTS）。
    :param text: 要合成的文本
    :param model: OpenAI 支持的 TTS 模型
    :param voice: 声音风格（如 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'）
    :return: 合成的音频二进制内容（mp3）
    """
    url = f"{OPENAI_API_BASE}/audio/speech"
    json_data = {"model": model, "input": text, "voice": voice}
    response = requests.post(url, headers={**HEADERS, "Content-Type": "application/json"}, json=json_data)
    response.raise_for_status()
    return response.content

if __name__ == "__main__":
    # 示例：音频转文本
    text = transcribe_audio("example.wav", language="zh")
    print("转录结果:", text)

    # 示例：文本转语音
    audio_bytes = text_to_speech("你好，世界！", voice="alloy")
    with open("output.mp3", "wb") as f:
        f.write(audio_bytes)
    print("TTS 音频已保存为 output.mp3") 