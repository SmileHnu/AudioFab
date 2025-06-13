# 本地部署 Tool Kits 中集成的工具

开始本地部署 Tool Kits 中集成的第三方工具前，您需要确保：

- 已经部署好了 AudioFab 的其他部分
- 已将 [ToolKits_ZH.md](./ToolKits_ZH.md) 中 FunTTS MCP Servers、Music MCP Servers 和 Audio Separator MCP Servers 需要的模型下载至本地。

**FunASRTool、EmotionRecognitionTool、CosyVoice2Tool**

1. 配置虚拟环境

    ```bash
    conda create -n cosyvoice python=3.10
    conda activate cosyvoice
    pip install -r requirements/FunASRTool_EmotionRecognitionTool_CosyVoice2Tool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/FunTTS_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `FunASRTool`、`EmotionRecognitionTool`、`CosyVoice2Tool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "FunASRTool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",  # <--- 修改为 cosyvoice 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Funasr_processor.py")
    },
    "EmotionRecognitionTool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",  # <--- 修改为 cosyvoice 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Funasr_processor.py")
    },
    "CosyVoice2Tool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",  # <--- 修改为 cosyvoice 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Cosyvoice2_tool.py")
    }
    }
    ```

3. 更新模型路径

    打开 `processor/Funasr_processor.py`，更新 `SenseVoiceSmall`、`emotion2vec` 的值

    ```python
    # 将此路径修改为您下载的 SenseVoiceSmall 模型的实际存放路径
    SenseVoiceSmall = "/path/to/your/downloaded/models--FunAudioLLM--SenseVoiceSmall"
    # 将此路径修改为您下载的 emotion2vec_plus_large 模型的实际存放路径
    emotion2vec = "/path/to/your/downloaded/models--emotion2vec--emotion2vec_plus_large"
    ```

    打开 `processor/Cosyvoice2_tool.py`，更新 `cosyvoice2path` 的值

    ```python
    # 将此路径修改为您下载的 CosyVoice2-0.5B 模型的实际存放路径
    cosyvoice2path = "/path/to/your/downloaded/CosyVoice2-0.5B"
    ```

**SparkTTSTool、ClearVoiceTool、WhisperASRTool**

1. 配置虚拟环境

    ```bash
    conda create -n scw python=3.12
    conda activate scw
    pip install -r requirements/SparkTTSTool_ClearVoiceTool_WhisperASRTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/FunTTS_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `SparkTTSTool`、`ClearVoiceTool`、`WhisperASRTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "SparkTTSTool": {
        "python_path": "your/python/interpreter/path", # <--- 修改为 scw 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "sparktts_processor.py")
    },
    "ClearVoiceTool": {
        "python_path": "your/python/interpreter/path", # <--- 修改为 scw 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "ClearerVoice_tool.py")
    },
    "WhisperASRTool": {
        "python_path": "your/python/interpreter/path", # <--- 修改为 scw 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "whisper_tool.py")
    }
    }
    ```

    打开 `processor/sparktts_processor.py`，更新 `PYTHON_PATH` 的值

    ```python
    PYTHON_PATH = "your/python/interpreter/path" # <--- 修改为 scw 环境的 Python 解释器路径
    ```

3. 更新模型路径

    打开 `processor/sparktts_processor.py`，更新 `SPARKTTS_PATH` 的值

    ```python
    # 将此路径修改为您下载的 Spark-TTS-0.5B 模型的实际存放路径
    SPARKTTS_PATH = "/home/chengz/LAMs/pre_train_models/models--SparkAudio--Spark-TTS-0.5B"
    ```

    打开 `processor/whisper_tool.py`，更新 `model_path` 的值

    ```python
    # 将此路径修改为您下载的 whisper-large-v3 模型的实际存放路径
    model_path: str = "/home/chengz/LAMs/pre_train_models/models--openai--whisper-large-v3",
    ```

**VoiceCraftTool**

1. 配置虚拟环境

    ```bash
    conda create -n voicecraft python=3.9
    conda activate voicecraft
    pip install -r requirements/VoiceCraftTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/FunTTS_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `VoiceCraftTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "VoiceCraftTool": {
        "python_path": "/home/chengz/anaconda3/envs/voicecraft/bin/python", # <--- 修改为 voicecraft 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "voicecraft_processor.py")
    }
    }
    ```

    打开 `processor/voicecraft_processor.py`，更新 `PYTHON_ENV_PATH` 的值

    ```python
    PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/voicecraft/bin/python" # <--- 修改为 voicecraft 环境的 Python 解释器路径
    ```

3. 更新模型路径

    打开 `processor/voicecraft_processor.py`，更新 `PRETRAINED_MODEL_DIR` 的值

    ```python
    PRETRAINED_MODEL_DIR = Path(os.environ.get(
        "VOICECRAFT_MODEL_DIR", 
        # 将此路径修改为您下载的 VoiceCraft 模型的实际存放路径
        "/home/chengz/LAMs/pre_train_models/models--pyp1--VoiceCraft"
    ))
    ```

**Qwen2AudioTool**

1. 配置虚拟环境

    ```bash
    conda create -n Qwenaudio python=3.10
    conda activate Qwenaudio
    pip install -r requirements/Qwen2AudioTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/FunTTS_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `Qwen2AudioTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "Qwen2AudioTool": {
        "python_path": "/home/chengz/anaconda3/envs/Qwenaudio/bin/python", # <--- 修改为 Qwenaudio 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Qwen2Audio_processor.py")
    }
    }
    ```

3. 更新模型路径

    打开 `processor/Qwen2Audio_processor.py`，更新 `QWEN2_AUDIO_PATH` 的值

    ```python
    # 将此路径修改为您下载的 Qwen2-Audio-7B-Instruct 模型的实际存放路径
    QWEN2_AUDIO_PATH = "/home/chengz/LAMs/pre_train_models/models--Qwen--Qwen2-Audio-7B-Instruct"
    ```

**AudioXTool**

1. 配置虚拟环境

    ```bash
    conda create -n AudioX python=3.10
    conda activate AudioX
    pip install -r requirements/AudioXTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/music_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `AudioXTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "AudioXTool": {
        "python_path": "/home/chengz/anaconda3/envs/AudioX/bin/python", # <--- 修改为 AudioXTool 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "AudioX_processor.py")
    }
    }
    ```

3. 更新模型路径

    打开 `processor/AudioX_processor.py`，更新 `AudioX_model_path` 的值

    ```python
    # 将此路径修改为您下载的 AudioX 模型的实际存放路径
    AudioX_model_path = "/home/chengz/LAMs/pre_train_models/models--HKUSTAudio--AudioX"
    ```

**ACEStepTool**

1. 配置虚拟环境

    ```bash
    conda create -n ace_step python=3.10
    conda activate ace_step
    pip install -r requirements/ACEStepTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/music_mcp_servers.py，找到 `TOOL_ENV_CONFIG` 字典

    更新 `ACEStepTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "AudioXTool": {
        "python_path": "/home/chengz/anaconda3/envs/ace_step/bin/python", # <--- 修改为 ace_step 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "ACE_step_processor.py")
    }
    }
    ```

3. 更新模型路径

    打开 `processor/ACE_step_processor.py`，更新 `checkpoint_path`、`chinese_rap` 的值

    ```python
    # 将此路径修改为您下载的 ACE-Step-v1-3.5B 模型的实际存放路径
    checkpoint_path = '/home/chengz/LAMs/pre_train_models/models--ACE-Step--ACE-Step-v1-3.5B'
    # 将此路径修改为您下载的 ACE-Step-v1-chinese-rap-LoRA 模型的实际存放路径
    chinese_rap = "/home/chengz/LAMs/pre_train_models/models--ACE-Step--ACE-Step-v1-chinese-rap-LoRA"
    ```

**MusicGenTool、AudioGenTool**

1. 配置虚拟环境

    ```bash
    conda create -n audiocraft python=3.10
    conda activate audiocraft
    pip install -r requirements/MusicGenTool_AudioGenTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/music_mcp_servers.py，找到 `TOOL_ENV_CONFIG` 字典

    更新 `MusicGenTool`、`AudioGenTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "MusicGenTool": {
        "python_path": "/home/chengz/anaconda3/envs/audiocraft/bin/python", # <--- 修改为 audiocraft 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Audiocraft_tool_processor.py")
    },
        "AudioGenTool": {
        "python_path": "/home/chengz/anaconda3/envs/audiocraft/bin/python", # <--- 修改为 audiocraft 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Audiocraft_tool_processor.py")
    }
    }
    ```

3. 更新模型路径

    打开 `processor/Audiocraft_tool_processor.py`，更新 `musicgen_path`、`audiogen_path` 的值

    ```python
    # 将此路径修改为您下载的 musicgen-melody 模型的实际存放路径
    musicgen_path = "/home/chengz/LAMs/pre_train_models/models--facebook--musicgen-melody"
    # 将此路径修改为您下载的 audiogen-medium 模型的实际存放路径
    audiogen_path = "/home/chengz/LAMs/pre_train_models/models--facebook--audiogen-medium"
    ```

**Hallo2Tool**

1. 配置虚拟环境

    ```bash
    conda create -n hallo python=3.10
    conda activate hallo
    pip install -r requirements/Hallo2Tool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/music_mcp_servers.py，找到 `TOOL_ENV_CONFIG` 字典

    更新 `Hallo2Tool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "Hallo2Tool": {
        "python_path": "/home/chengz/anaconda3/envs/hallo/bin/python",  # <--- 修改为 hallo 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "hello2_processor.py")
    }
    }
    ```

    打开 `processor/hello2_processor.py`，更新两处 `sys.path.append()` 内参数的值，更新 `model_config` 中 `config_path` 的值，更新 `script_path` 的值

    ```python
    # 分别在第 14 行、第 290 行：根据 mcp_chatbot-audio 在您设备中的存储路径修改这两处路径
    sys.path.append("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2")

    model_config = {
        "model_path": str(HALLO2_PATH),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "weight_dtype": "fp16" if torch.cuda.is_available() else "fp32",
        # 根据 mcp_chatbot-audio 在您设备中的存储路径修改此路径
        "config_path": str(Path("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/configs/inference/long.yaml"))
    }

    # 根据 mcp_chatbot-audio 在您设备中的存储路径修改此路径
    script_path = "/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/scripts/video_sr.py"
    ```

3. 更新模型路径

    打开 `processor/hello2_processor.py`，更新 `musicgen_path`、`audiogen_path` 的值

    ```python
    # 将此路径修改为您下载的 hallo2 模型的实际存放路径
    HALLO2_PATH = Path("/home/chengz/LAMs/pre_train_models/models--fudan-generative-ai--hallo2")
    ```

**YuEETool**

1. 配置虚拟环境

    ```bash
    conda create -n yue_e python=3.12
    conda activate yue_e
    pip install -r requirements/YuEETool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/music_mcp_servers.py，找到 `TOOL_ENV_CONFIG` 字典

    更新 `YuEETool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "YuEETool": {
        "python_path": "/home/chengz/anaconda3/envs/yue_e/bin/python",  # <--- 修改为 yue_e 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "yue_e_tool.py")
    }
    }
    ```

    打开 `processor/yue_processor.py`，更新 `YUE_INFERENCE_DIR`、`PYTHON_ENV_PATH` 的值
    ```python
    # 请根据您本地环境中 mcp_chatbot-audio/models/YuE-exllamav2/src/yue 的实际存储位置，修改此路径。
    YUE_INFERENCE_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/models/YuE-exllamav2/src/yue")
    # 修改为 yue_e 环境的 Python 解释器路径
    PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/yue_e/bin/python"
    ```

3. 更新模型路径

    打开 `processor/yue_processor.py`，更新 `YUE_S1_EN_COT`、`YUE_S1_EN_ICL`、`YUE_S1_ZH_COT`、`YUE_S1_ZH_ICL`、`YUE_S2_GENERAL`、`YUE_UPSAMPLER` 的值

    ```python
    # 将此路径修改为您下载的 YuE-s1-7B-anneal-en-cot 模型的实际存放路径
    YUE_S1_EN_COT =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-cot"
    # 将此路径修改为您下载的 YuE-s1-7B-anneal-en-icl 模型的实际存放路径
    YUE_S1_EN_ICL =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-icl"
    # 将此路径修改为您下载的 YuE-s1-7B-anneal-zh-cot 模型的实际存放路径
    YUE_S1_ZH_COT =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-cot"
    # 将此路径修改为您下载的 YuE-s1-7B-anneal-zh-icl 模型的实际存放路径
    YUE_S1_ZH_ICL =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-icl"
    # 将此路径修改为您下载的 YuE-s2-1B-general 模型的实际存放路径
    YUE_S2_GENERAL = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s2-1B-general"
    # 将此路径修改为您下载的 YuE-upsampler 模型的实际存放路径
    YUE_UPSAMPLER = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-upsampler"
    ```

**DiffRhythmTool**

1. 配置虚拟环境

    ```bash
    conda create -n diffrhythm python=3.10
    conda activate diffrhythm
    pip install -r requirements/DiffRhythmTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/music_mcp_servers.py，找到 `TOOL_ENV_CONFIG` 字典

    更新 `DiffRhythmTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "DiffRhythmTool": {
        "python_path": "/home/qianshuaix/miniconda3/envs/diffrhythm/bin/python",  # <--- 修改为 diffrhythm 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "DiffRhythm_processor.py")
    }
    }
    ```

    打开 `processor/DiffRhythm_processor.py`，更新 `OUTPUT_DIR`、`PYTHON_PATH` 的值

    ```python
    # 请根据您本地环境中 mcp_chatbot-audio 的实际存储位置，修改此路径。
    OUTPUT_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/output/music")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 修改为 yue_e 环境的 Python 解释器路径
    PYTHON_PATH = "/home/qianshuaix/miniconda3/envs/diffrhythm/bin/python"
    ```

**AudioSepTool**

1. 配置虚拟环境

    ```bash
    conda create -n AudioSep python=3.10
    conda activate AudioSep
    pip install -r requirements/AudioSepTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/Audioseparator_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `AudioSepTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "AudioSepTool": {
        "python_path": "/home/chengz/anaconda3/envs/AudioSep/bin/python",  # <--- 修改为 AudioSep 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "Audiosep_processor.py")
    }
    }
    ```

    打开 `processor/Audiosep_processor.py`，更新 `SCRIPTS_DIR`、`PYTHON_ENV_PATH` 的值
    ```python
    # 请根据您本地环境中 mcp_chatbot-audio 的实际存储位置，修改此路径
    SCRIPTS_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/models/AudioSep")

    # 修改为 AudioSep 环境的 Python 解释器路径
    PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/AudioSep/bin/python"
    ```

3. 更新模型路径

    打开 `processor/Audiosep_processor.py`，更新 `AUDIOSEP_MODEL_PATH` 的值

    ```python
    # 将此路径修改为您下载的 audiosep_base_4M_steps.ckpt 的实际存放路径
    AUDIOSEP_MODEL_PATH = "/home/chengz/LAMs/pre_train_models/Audiosep_pretrain_models/audiosep_base_4M_steps.ckpt"
    ```

**AudioSeparatorTool**

1. 配置虚拟环境

    ```bash
    conda create -n voicecraft2 python=3.10
    conda activate voicecraft2
    pip install -r requirements/AudioSeparatorTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/Audioseparator_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `AudioSeparatorTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "AudioSeparatorTool": {
        "python_path": "/home/chengz/anaconda3/envs/voicecraft2/bin/python",  # <--- 修改为 voicecraft2 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "audio_separator_processor.py")
    }
    }
    ```

    打开 `processor/audio_separator_processor.py`，更新 `sys.path.append()` 内的参数

    ```python
    # 第 19 行，根据 mcp_chatbot-audio 在您设备的存储路径更新 sys.path.append() 内的参数
    sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/audio-separator')
    ```

3. 更新模型路径

    打开 `processor/audio_separator_processor.py`，更新 `MODEL_FILE_DIR` 的值

    ```python
    # 将此路径修改为包含 UVR-MDX-NET-Inst_HQ_3 模型 和 htdemucs_6s 模型所有文件的文件夹路径
    MODEL_FILE_DIR = os.path.abspath("models/audio-separator/models")
    DEFAULT_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
    DEFAULT_DEMUCS_MODEL = "htdemucs_6s.yaml"
    ```

**TIGERSpeechSeparationTool**

1. 配置虚拟环境

    ```bash
    conda create -n Tiger python=3.10
    conda activate Tiger
    pip install -r requirements/TIGERSpeechSeparationTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/Audioseparator_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `TIGERSpeechSeparationTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "TIGERSpeechSeparationTool": {
        "python_path": "/home/chengz/anaconda3/envs/Tiger/bin/python",  # <--- 修改为 Tiger 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "TIGER_speech_separation_processor.py")
    }
    }
    ```

    打开 `processor/TIGER_speech_separation_processor.py`，更新 `sys.path.append()` 内的参数， `OUTPUT_DIR` 参数

    ```python
    # 根据 mcp_chatbot-audio 在您设备的存储路径更新 sys.path.append() 内的参数及 OUTPUT_DIR
    sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/TIGER')
    OUTPUT_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/output")
    ```

3. 更新模型路径

    打开 `processor/TIGER_speech_separation_processor.py`，更新 `output_files` 中 `cache_dir` 的值

    ```python
    output_files = separate_speech(
        audio_path=audio_path,
        output_dir=str(output_dir),
        # 将此路径修改为您下载的 TIGER-speech 的实际存放路径
        cache_dir="/home/chengz/LAMs/pre_train_models/models--JusperLee--TIGER-speech"
    )
    ```

**AudioSRTool**

1. 配置虚拟环境

    ```bash
    conda create -n audiosr python=3.9
    conda activate audiosr
    pip install -r requirements/AudioSRTool_requirements.txt
    ```

2. 更新环境路径

    打开 `servers/Audioseparator_mcp_servers.py`，找到 `TOOL_ENV_CONFIG` 字典

    更新 `AudioSRTool` 对应的 `python_path` 

    ```python
    TOOL_ENV_CONFIG = {
    "AudioSRTool": {
        "python_path": "/home/qianshuaix/miniconda3/envs/audiosr/bin/python",  # <--- 修改为 audiosr 环境的 Python 解释器路径
        "script_path": str(Path(__file__).parent / "processor" / "audiosr_tool.py")
    }
    }
    ```
