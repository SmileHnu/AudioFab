# On-Premises Deployment of Integrated Tool Kits

Before you begin the on-premises deployment of the third-party tools integrated within the Tool Kits, please ensure the following:

- You have already deployed other components of AudioFab.
- You have downloaded the necessary models for FunTTS MCP Servers, Music MCP Servers, and Audio Separator MCP Servers from [ToolKits.md](./ToolKits.md) to your local environment.

**FunASRTool、EmotionRecognitionTool、CosyVoice2Tool**

1. Configure the virtual environment

    ```bash
    conda create -n cosyvoice python=3.10
    conda activate cosyvoice
    cd mcp_servers
    pip install -r requirements/FunASRTool_EmotionRecognitionTool_CosyVoice2Tool_requirements.txt
    ```

2. Update the environment path

    Open `servers/FunTTS_mcp_servers.py`, and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` for `FunASRTool`, `EmotionRecognitionTool`, and `CosyVoice2Tool`.

    ```python
    TOOL_ENV_CONFIG = {
    "FunASRTool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",  # <--- Change to the Python interpreter path of your cosyvoice environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Funasr_processor.py")
    },
    "EmotionRecognitionTool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",  # <--- Change to the Python interpreter path of your cosyvoice environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Funasr_processor.py")
    },
    "CosyVoice2Tool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",  # <--- Change to the Python interpreter path of your cosyvoice environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Cosyvoice2_tool.py")
    }
    }
    ```

3. Update the model path

    Open `processor/Funasr_processor.py`, and update the values for `SenseVoiceSmall` and `emotion2vec`.

    ```python
    # Change this path to the actual storage path of your downloaded SenseVoiceSmall model
    SenseVoiceSmall = "/path/to/your/downloaded/models--FunAudioLLM--SenseVoiceSmall"
    # Change this path to the actual storage path of your downloaded emotion2vec_plus_large model
    emotion2vec = "/path/to/your/downloaded/models--emotion2vec--emotion2vec_plus_large"
    ```
    Open `processor/Cosyvoice2_tool.py`, and update the value for `cosyvoice2path`.

    ```python
    # Change this path to the actual storage path of your downloaded CosyVoice2-0.5B model
    cosyvoice2path = "/path/to/your/downloaded/CosyVoice2-0.5B"
    ```

**SparkTTSTool、ClearVoiceTool、WhisperASRTool**

1. Configure the virtual environment

    ```bash
    conda create -n scw python=3.12
    conda activate scw
    cd mcp_servers
    pip install -r requirements/SparkTTSTool_ClearVoiceTool_WhisperASRTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/FunTTS_mcp_servers.py`, and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` for `SparkTTSTool`, `ClearVoiceTool` and`WhisperASRTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "SparkTTSTool": {
        "python_path": "your/python/interpreter/path", # <--- Change to the Python interpreter path of your scw environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "sparktts_processor.py")
    },
    "ClearVoiceTool": {
        "python_path": "your/python/interpreter/path", # <--- Change to the Python interpreter path of your scw environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "ClearerVoice_tool.py")
    },
    "WhisperASRTool": {
        "python_path": "your/python/interpreter/path", # <--- Change to the Python interpreter path of your scw environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "whisper_tool.py")
    }
    }
    ```

    Open `processor/sparktts_processor.py`, and update the values for `PYTHON_PATH`.

    ```python
    PYTHON_PATH = "your/python/interpreter/path" # <--- Change to the Python interpreter path of your scw environment
    ```

3. Update the model path

    Open `processor/sparktts_processor.py`, and update the value of `SPARKTTS_PATH`.

    ```python
    # Change this path to the actual storage path of your downloaded Spark-TTS-0.5B model
    SPARKTTS_PATH = "/home/chengz/LAMs/pre_train_models/models--SparkAudio--Spark-TTS-0.5B"
    ```

    Open `processor/whisper_tool.py,` and update the value of `model_path`.

    ```python
    # Change this path to the actual storage path of your downloaded whisper-large-v3 model
    model_path: str = "/home/chengz/LAMs/pre_train_models/models--openai--whisper-large-v3",
    ```

**VoiceCraftTool**

1. Configure the virtual environment

    ```bash
    conda create -n voicecraft python=3.9
    conda activate voicecraft
    cd mcp_servers
    pip install -r requirements/VoiceCraftTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/FunTTS_mcp_servers.py`, and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `VoiceCraftTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "VoiceCraftTool": {
        "python_path": "/home/chengz/anaconda3/envs/voicecraft/bin/python", # <--- Change this to the Python interpreter path of your voicecraft environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "voicecraft_processor.py")
    }
    }
    ```

    Open `processor/voicecraft_processor.py`, and update the value of `PYTHON_ENV_PATH`.

    ```python
    PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/voicecraft/bin/python" # <--- Change this to the Python interpreter path of your voicecraft environment
    ```

3. Update the model path

    Open `processor/voicecraft_processor.py`, and update the value of `PRETRAINED_MODEL_DIR`.

    ```python
    PRETRAINED_MODEL_DIR = Path(os.environ.get(
        "VOICECRAFT_MODEL_DIR", 
        # Change this path to the actual storage path of the VoiceCraft model you downloaded
        "/home/chengz/LAMs/pre_train_models/models--pyp1--VoiceCraft"
    ))
    ```

**Qwen2AudioTool**

1. Configure the virtual environment

    ```bash
    conda create -n Qwenaudio python=3.10
    conda activate Qwenaudio
    cd mcp_servers
    pip install -r requirements/Qwen2AudioTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/FunTTS_mcp_servers.py`, and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `Qwen2AudioTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "Qwen2AudioTool": {
        "python_path": "/home/chengz/anaconda3/envs/Qwenaudio/bin/python",  # <--- Change this to the Python interpreter path of your Qwenaudio environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Qwen2Audio_processor.py")
    }
    }
    ```

3. Update the model path

    Open `processor/Qwen2Audio_processor.py`, and update the value of `QWEN2_AUDIO_PATH`.

    ```python
    # Change this path to the actual storage path of the Qwen2-Audio-7B-Instruct model you downloaded
    QWEN2_AUDIO_PATH = "/home/chengz/LAMs/pre_train_models/models--Qwen--Qwen2-Audio-7B-Instruct"
    ```

**AudioXTool**

1. Configure the virtual environment

    ```bash
    conda create -n AudioX python=3.10
    conda activate AudioX
    cd mcp_servers
    pip install -r requirements/AudioXTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/music_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `AudioXTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "AudioXTool": {
        "python_path": "/home/chengz/anaconda3/envs/AudioX/bin/python", # <--- Change to the Python interpreter path for the AudioXTool environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "AudioX_processor.py")
    }
    }
    ```

3. Update the model path

    Open `processor/AudioX_processor.py` and update the value of `AudioX_model_path`.

    ```python
    # Change this path to the actual storage path of your downloaded AudioX model
    AudioX_model_path = "/home/chengz/LAMs/pre_train_models/models--HKUSTAudio--AudioX"
    ```

**ACEStepTool**

1. Configure the virtual environment

    ```bash
    conda create -n ace_step python=3.10
    conda activate ace_step
    cd mcp_servers
    pip install -r requirements/ACEStepTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/music_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `ACEStepTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "AudioXTool": {
        "python_path": "/home/chengz/anaconda3/envs/ace_step/bin/python", # <--- Change to the Python interpreter path for the ace_step environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "ACE_step_processor.py")
    }
    }
    ```

3. Update the model path

    Open `processor/ACE_step_processor.py` and update the values for `checkpoint_path` and `chinese_rap`.

    ```python
    # Change this path to the actual storage path of your downloaded ACE-Step-v1-3.5B model
    checkpoint_path = '/home/chengz/LAMs/pre_train_models/models--ACE-Step--ACE-Step-v1-3.5B'
    # Change this path to the actual storage path of your downloaded ACE-Step-v1-chinese-rap-LoRA model
    chinese_rap = "/home/chengz/LAMs/pre_train_models/models--ACE-Step--ACE-Step-v1-chinese-rap-LoRA"
    ```

**MusicGenTool、AudioGenTool**

1. Configure the virtual environment

    ```bash
    conda create -n audiocraft python=3.10
    conda activate audiocraft
    cd mcp_servers
    pip install -r requirements/MusicGenTool_AudioGenTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/music_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` for MusicGenTool and `AudioGenTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "MusicGenTool": {
        "python_path": "/home/chengz/anaconda3/envs/audiocraft/bin/python", # <--- Change this to the Python interpreter path of your audiocraft environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Audiocraft_tool_processor.py")
    },
        "AudioGenTool": {
        "python_path": "/home/chengz/anaconda3/envs/audiocraft/bin/python", # <--- Change this to the Python interpreter path of your audiocraft environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Audiocraft_tool_processor.py")
    }
    }
    ```

3. Update the model path

    Open `processor/Audiocraft_tool_processor.py` and update the values for `musicgen_path` and `audiogen_path`.

    ```python
    # Modify this path to the actual storage location of the musicgen-melody model you downloaded
    musicgen_path = "/home/chengz/LAMs/pre_train_models/models--facebook--musicgen-melody"
    # Modify this path to the actual storage location of the audiogen-medium model you downloaded
    audiogen_path = "/home/chengz/LAMs/pre_train_models/models--facebook--audiogen-medium"
    ```

**Hallo2Tool**

1. Configure the virtual environment

    ```bash
    conda create -n hallo python=3.10
    conda activate hallo
    cd mcp_servers
    pip install -r requirements/Hallo2Tool_requirements.txt
    ```

2. Update the environment path

    Open `servers/music_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` for `Hallo2Tool`.

    ```python
    TOOL_ENV_CONFIG = {
    "Hallo2Tool": {
        "python_path": "/home/chengz/anaconda3/envs/hallo/bin/python",  # <--- Change this to the Python interpreter path of your hallo environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "hello2_processor.py")
    }
    }
    ```

    Open `processor/hello2_processor.py`, update the parameters in the two `sys.path.append()` calls, update the value of `config_path` in model_config, and update the value of `script_path`.

    ```python
    # On line 14 and line 290 respectively: Modify these two paths according to the storage path of mcp_chatbot-audio on your device
    sys.path.append("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2")

    model_config = {
        "model_path": str(HALLO2_PATH),
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "weight_dtype": "fp16" if torch.cuda.is_available() else "fp32",
        # Modify this path according to the storage path of mcp_chatbot-audio on your device
        "config_path": str(Path("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/configs/inference/long.yaml"))
    }

    # Modify this path according to the storage path of mcp_chatbot-audio on your device
    script_path = "/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/scripts/video_sr.py"
    ```

3. Update the model path

    Open `processor/hello2_processor.py` and update the value for `HALLO2_PATH`.

    ```python
    # Modify this path to the actual storage location of the hallo2 model you downloaded
    HALLO2_PATH = Path("/home/chengz/LAMs/pre_train_models/models--fudan-generative-ai--hallo2")
    ```

**YuEETool**

1. Configure the virtual environment

    ```bash
    conda create -n yue_e python=3.12
    conda activate yue_e
    cd mcp_servers
    pip install -r requirements/YuEETool_requirements.txt
    ```

2. Update the environment path

    Open `servers/music_mcp_servers.py`, and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `YuEETool`.

    ```python
    TOOL_ENV_CONFIG = {
    "YuEETool": {
        "python_path": "/home/chengz/anaconda3/envs/yue_e/bin/python",  # <--- Change this to the Python interpreter path of the yue_e environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "yue_e_tool.py")
    }
    }
    ```

    Open `processor/yue_processor.py`, and update the values of `YUE_INFERENCE_DIR` and `PYTHON_ENV_PATH`.

    ```python
    # Please modify this path according to the actual storage location of mcp_chatbot-audio/models/YuE-exllamav2/src/yue in your local environment.
    YUE_INFERENCE_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/models/YuE-exllamav2/src/yue")
    # Change this to the Python interpreter path of the yue_e environment
    PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/yue_e/bin/python"
    ```

3. Update the model path

    Open `processor/yue_processor.py`, and update the values of `YUE_S1_EN_COT`, `YUE_S1_EN_ICL`, `YUE_S1_ZH_COT`, `YUE_S1_ZH_ICL`, `YUE_S2_GENERAL`, and `YUE_UPSAMPLER`.

    ```python
    # Change this path to the actual storage location of your downloaded YuE-s1-7B-anneal-en-cot model
    YUE_S1_EN_COT =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-cot"
    # Change this path to the actual storage location of your downloaded YuE-s1-7B-anneal-en-icl model
    YUE_S1_EN_ICL =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-icl"
    # Change this path to the actual storage location of your downloaded YuE-s1-7B-anneal-zh-cot model
    YUE_S1_ZH_COT =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-cot"
    # Change this path to the actual storage location of your downloaded YuE-s1-7B-anneal-zh-icl model
    YUE_S1_ZH_ICL =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-icl"
    # Change this path to the actual storage location of your downloaded YuE-s2-1B-general model
    YUE_S2_GENERAL = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s2-1B-general"
    # Change this path to the actual storage location of your downloaded YuE-upsampler model
    YUE_UPSAMPLER = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-upsampler"
    ```

**DiffRhythmTool**

1. Configure the virtual environment

    ```bash
    conda create -n diffrhythm python=3.10
    conda activate diffrhythm
    cd mcp_servers
    pip install -r requirements/DiffRhythmTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/music_mcp_servers.py`, and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `DiffRhythmTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "DiffRhythmTool": {
        "python_path": "/home/qianshuaix/miniconda3/envs/diffrhythm/bin/python",  # <--- Change this to the Python interpreter path of the diffrhythm environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "DiffRhythm_processor.py")
    }
    }
    ```
    Open `processor/DiffRhythm_processor.py`, and update the values of `OUTPUT_DIR` and `PYTHON_PATH`.

    ```python
    # Please modify this path according to the actual storage location of mcp_chatbot-audio in your local environment.
    OUTPUT_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/output/music")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Change this to the Python interpreter path of the diffrhythm environment
    PYTHON_PATH = "/home/qianshuaix/miniconda3/envs/diffrhythm/bin/python"
    ```

**AudioSepTool**

1. Configure the virtual environment

    ```bash
    conda create -n AudioSep python=3.10
    conda activate AudioSep
    cd mcp_servers
    pip install -r requirements/AudioSepTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/Audioseparator_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` for `AudioSepTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "AudioSepTool": {
        "python_path": "/home/chengz/anaconda3/envs/AudioSep/bin/python",  # <--- Change to the Python interpreter path of your AudioSep environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "Audiosep_processor.py")
    }
    }
    ```

    Open `processor/Audiosep_processor.py` and update the values for `SCRIPTS_DIR` and `PYTHON_ENV_PATH`.

    ```python
    # Please modify this path according to the actual storage location of mcp_chatbot-audio on your local environment
    SCRIPTS_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/models/AudioSep")

    # Change to the Python interpreter path of your AudioSep environment
    PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/AudioSep/bin/python"
    ```

3. Update the model path

    Open `processor/Audiosep_processor.py` and update the value of `AUDIOSEP_MODEL_PATH`.

    ```python
    # Change this path to the actual storage path of the audiosep_base_4M_steps.ckpt you downloaded
    AUDIOSEP_MODEL_PATH = "/home/chengz/LAMs/pre_train_models/Audiosep_pretrain_models/audiosep_base_4M_steps.ckpt"
    ```

**AudioSeparatorTool**

1. Configure the virtual environment

    ```bash
    conda create -n voicecraft2 python=3.10
    conda activate voicecraft2
    cd mcp_servers
    pip install -r requirements/AudioSeparatorTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/Audioseparator_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` for `AudioSeparatorTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "AudioSeparatorTool": {
        "python_path": "/home/chengz/anaconda3/envs/voicecraft2/bin/python",  # <--- Change to the Python interpreter path of your voicecraft2 environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "audio_separator_processor.py")
    }
    }
    ```

    Open `processor/audio_separator_processor.py` and update the parameter within `sys.path.append()`.

    ```python
    # Line 19, update the parameter in sys.path.append() according to the storage path of mcp_chatbot-audio on your device
    sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/audio-separator')
    ```

3. Update the model path

    Open `processor/audio_separator_processor.py` and update the value of `MODEL_FILE_DIR`.

    ```python
    # Change this path to the folder path containing all files for the UVR-MDX-NET-Inst_HQ_3 model and the htdemucs_6s model
    MODEL_FILE_DIR = os.path.abspath("models/audio-separator/models")
    DEFAULT_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
    DEFAULT_DEMUCS_MODEL = "htdemucs_6s.yaml"
    ```

**TIGERSpeechSeparationTool**

1. Configure the virtual environment

    ```bash
    conda create -n Tiger python=3.10
    conda activate Tiger
    cd mcp_servers
    pip install -r requirements/TIGERSpeechSeparationTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/Audioseparator_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `TIGERSpeechSeparationTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "TIGERSpeechSeparationTool": {
        "python_path": "/home/chengz/anaconda3/envs/Tiger/bin/python",  # <--- Change this to the Python interpreter path of your Tiger environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "TIGER_speech_separation_processor.py")
    }
    }
    ```

    Open `processor/TIGER_speech_separation_processor.py`, and update the parameter in `sys.path.append()` and the `OUTPUT_DIR` parameter.

    ```python
    # Update the parameter in sys.path.append() and OUTPUT_DIR based on the storage path of mcp_chatbot-audio on your device
    sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/TIGER')
    OUTPUT_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/output")
    ```

3. Update the model path

    Open `processor/TIGER_speech_separation_processor.py` and update the value of `cache_dir` in the `separate_speech` function call.

    ```python
    output_files = separate_speech(
        audio_path=audio_path,
        output_dir=str(output_dir),
        # Change this path to the actual storage location where you downloaded TIGER-speech
        cache_dir="/home/chengz/LAMs/pre_train_models/models--JusperLee--TIGER-speech"
    )
    ```

**AudioSRTool**

1. Configure the virtual environment

    ```bash
    conda create -n audiosr python=3.9
    conda activate audiosr
    cd mcp_servers
    pip install -r requirements/AudioSRTool_requirements.txt
    ```

2. Update the environment path

    Open `servers/Audioseparator_mcp_servers.py` and find the `TOOL_ENV_CONFIG` dictionary.

    Update the `python_path` corresponding to `AudioSRTool`.

    ```python
    TOOL_ENV_CONFIG = {
    "AudioSRTool": {
        "python_path": "/home/qianshuaix/miniconda3/envs/audiosr/bin/python",  # <--- Change this to the Python interpreter path of your audiosr environment
        "script_path": str(Path(__file__).parent.parent / "processor" / "audiosr_tool.py")
    }
    }
    ```
