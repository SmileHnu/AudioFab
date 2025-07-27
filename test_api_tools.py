import os
from mcp_servers.servers.API_servers import (
    cosyvoice2tool_api,
    AudioX_api,
    Qwen2audio_api,
    clearervoice_api,
    SenseVoice_api,
    whisper_large_v3_turbo_api,
    tiger_api,
    audio_super_resolution_api,
)

# ----------------- 样例文件路径 -----------------
SAMPLE_AUDIO = "/home/chengz/LAMs/mcp_chatbot-audio/output/audio/audiogen_20250524_195721_0.wav"
SAMPLE_VOICE = "/home/chengz/LAMs/mcp_chatbot-audio/output/audio/20250525015827.wav"
SAMPLE_VIDEO = "/home/chengz/LAMs/mcp_chatbot-audio/output/video/audiox_video_20250530_211711.mp4"
OUTPUT_DIR = "/home/chengz/LAMs/mcp_chatbot-audio/api_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------- 单个工具的测试封装 -----------------

def safe_run(name: str, func, *args, **kwargs):
    """执行函数并捕获异常，打印结果或错误。"""
    print(f"\n===== 开始测试: {name} =====")
    try:
        result = func(*args, **kwargs)
        print(f"{name} 运行成功，返回: {result}")
    except Exception as e:
        print(f"{name} 运行失败，错误: {e}")


# ----------------- 各工具测试 -----------------

def test_cosyvoice():
    safe_run(
        "cosyvoice2tool_api",
        cosyvoice2tool_api,
        tts_text="你好，这是一段 CosyVoice2 测试。",
        mode="cosy",
        prompt_wav_path=SAMPLE_VOICE,
        output_path=os.path.join(OUTPUT_DIR, "cosyvoice2_output.wav"),
    )

def test_audiox():
    safe_run(
        "AudioX_api",
        AudioX_api,
        prompt="A short cinematic sound of ocean waves.",
        video_file_path=SAMPLE_VIDEO,
        seconds_total=5,
        output_audio_path=os.path.join(OUTPUT_DIR, "audiox_output.wav"),
        output_video_path=os.path.join(OUTPUT_DIR, "audiox_output.mp4"),
    )

def test_qwen2audio():
    safe_run(
        "Qwen2audio_api",
        Qwen2audio_api,
        prompt="你好，介绍一下自己。",
        audio_file_path=SAMPLE_VOICE,
        chatbot_history=None,
        action="chat",
        output_file=os.path.join(OUTPUT_DIR, "qwen_history.json"),
    )

def test_clearervoice():
    safe_run(
        "clearervoice_api",
        clearervoice_api,
        task="enhancement",
        input_path=SAMPLE_AUDIO,
        model="MossFormer2_48000Hz",
        output_audio_path=os.path.join(OUTPUT_DIR, "clearervoice_enhanced.wav"),
    )

def test_sensevoice():
    safe_run(
        "SenseVoice_api",
        SenseVoice_api,
        input_wav_path=SAMPLE_AUDIO,
        language="auto",
        output_txt_path=os.path.join(OUTPUT_DIR, "sensevoice_result.txt"),
    )

def test_whisper():
    safe_run(
        "whisper_large_v3_turbo_api",
        whisper_large_v3_turbo_api,
        audio_path=SAMPLE_VOICE,
        task="transcribe",
        output_path=os.path.join(OUTPUT_DIR, "whisper_transcript.txt"),
    )

def test_tiger():
    safe_run(
        "tiger_api",
        tiger_api,
        input_file_path=SAMPLE_AUDIO,
        task="/separate_speakers",
        output_dir=os.path.join(OUTPUT_DIR, "tiger_speakers"),
    )

def test_audio_sr():
    safe_run(
        "audio_super_resolution_api",
        audio_super_resolution_api,
        audio_file_path=SAMPLE_AUDIO,
        output_path=os.path.join(OUTPUT_DIR, "audio_sr_output.wav"),
        model_name="basic",
    )


def run_all_tests():
    """按顺序运行所有测试"""
    test_cosyvoice()
    test_audiox()
    test_qwen2audio()
    test_clearervoice()
    test_sensevoice()
    test_whisper()
    test_tiger()
    test_audio_sr()


if __name__ == "__main__":
    # run_all_tests() 
    from gradio_client import Client, handle_file
    client = Client("https://indexteam-indextts-demo.ms.show/",hf_token="36394f1b-a0cd-4895-9264-f73ad6637b4c")
    result = client.predict(
            prompt=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav'),
            text="Hello!!",
            api_name="/gen_single"
    )
    print(result)