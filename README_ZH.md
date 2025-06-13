<div align="center">

# AudioFab

<!-- Language Switch with Badges -->
<p>
  <a href="./README.md">
    <img src="https://img.shields.io/badge/Language-English-blue?style=flat-square&logo=google-translate" alt="English">
  </a>
  <a href="./README_ZH.md">
    <img src="https://img.shields.io/badge/语言-中文-blue?style=flat-square&logo=google-translate" alt="中文">
  </a>
  <a href="https://creativecommons.org/licenses/by-nc/4.0/">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" alt="许可证: CC BY-NC 4.0">
  </a>
</p>
<!-- Installation Badges with Links -->
<p>
  <a href="#🚀-安装指南">
    <img src="https://img.shields.io/badge/Install-Conda-green?style=flat-square&logo=anaconda" alt="Conda Installation">
  </a>
  <a href="#⚡-快速开始">
    <img src="https://img.shields.io/badge/Quick-Start-orange?style=flat-square&logo=lightning" alt="Quick Start">
  </a>
</p>

</div>

## 📌 简介

AudioFab 是一个专为音频领域打造的、全面且用户友好的开源智能代理框架，旨在解决音频处理工具集成复杂、依赖冲突频发以及大型语言模型在工具调用中可靠性不足等核心挑战。

通过 AudioFab，用户可以用自然语言下达指令，由 AudioFab 完成以往需要专业软件和技能才能实现的复杂音频任务。AudioFab 会智能地理解您的意图，并自动选择、调度底层各种专业的音频工具来一步步完成任务，将强大的功能聚合在统一、简单的交互之下。

<div align="center">
<img src="./assets/AudioFab.png" width="95%">
</div>

AudioFab 采用了基于模型上下文协议（MCPs, Model Context Protocols）的标准化架构，其核心是独立工具管理机制与智能化工具选择工作流。前者通过隔离的运行环境根除了工具间的依赖冲突，简化了工具集成流程；后者则通过精确的筛选与小样本（Few-shot）引导机制，有效缓解了因上下文过长导致的性能下降与工具幻觉（Tool Hallucination）问题，显著提升了系统的可靠性与可扩展性。

## ✨ 功能特性

**🧩 基于 MCPs 的独立工具管理架构**

AudioFab 引入了基于模型上下文协议（MCPs）的独立工具管理架构。该架构使每一个工具都能运行在专属的依赖环境中，从根本上杜绝了依赖冲突。一个新工具在配置完成后，仅需通过简单的注册步骤便可集成到框架中。

**🧠 抑制工具幻觉的智能选择工作流**

AudioFab 通过两阶段策略应对工具幻觉：首先，通过智能筛选精简工具列表以缩短上下文；其次，在调用前注入小样本示例（Few-Shot Exemplars）进行精确引导。此设计有效抑制了模型的错误调用，确保了代理执行的稳定与可靠。

**🎧 兼具易用性与专业性的智能音频代理**

AudioFab 为非专家提供一个易于使用，同时功能又足够专业和全面的智能音频代理。同时，其全面的功能与高可扩展性亦能满足专业人士的深度开发与研究需求。

## 🛠️ 工具集成

AudioFab 中集成了一套功能全面且强大的工具集，整个系统通过模块化的服务器形式提供服务，结构清晰，易于扩展。其核心能力涵盖了从基础到前沿的多个层面：

- **基础服务** 包括 [**Markdown Servers**](#music-mcp-servers) 提供的文本文件管理、[**DSP Servers**](#dsp-servers) 和 [**Audio Servers**](#audio-servers) 提供的专业级音频信号处理与特征提取，以及 [**Tensor Servers**](#tensor-servers) 提供的底层张量计算与GPU管理。

- **智能发现** 通过 [**Tool Query Servers**](#tool-query-servers) 提供强大的工具发现与查询功能，帮助用户在庞大的工具库中快速定位所需工具。

- **高级音视频处理** 集成了业界领先的模型，构成了 [**FunTTS MCP**](#funtts-mcp-servers)、[**Music MCP**](#music-mcp-servers) 和 [**Audio Separator MCP**](#audio-separator-mcp-servers) 三大核心服务。

- **API 支持** 在 [**API Servers**](#api-servers) 中集成了一些工具的 API。

除此之外，您还可以在 AudioFab 中集成自己的服务、工具、API，以扩充 AudioFab 的能力边界。

### **Markdown Servers**

Markdown Servers 主要提供一套基础的文件读写和管理服务，专注于处理 Markdown、TXT 和 JSON 等文本格式的文件。

| 功能名称 | 功能简介 |
|---|---|
| **read_file** | 读取指定类型的所有文件内容，支持md, txt, json文件。 |
| **write_file** | 写入（或新建）指定类型文件，支持md、txt、json。 |
| **modify_file** | 修改（覆盖）单个已存在的.md/.txt/.json文件。 |

### **DSP Servers**

DSP Servers 主要提供一系列基础的数字音频信号处理服务，涵盖了音频特征提取、格式转换以及基础编辑等功能。

| 功能名称 | 功能简介 |
|---|---|
| **compute_stft** | 计算音频信号的短时傅里叶变换（STFT） |
| **compute_mfcc** | 计算音频的梅尔频率倒谱系数（MFCC）特征 |
| **compute_mel_spectrogram** | 计算音频的梅尔频谱图并保存为数据文件 |
| **convert_audio_format** | 将音频从一种格式转换为另一种格式，并可调整参数 |
| **trim_audio** | 裁剪音频文件的指定时间区间 |
| **align_audio_lengths** | 将多个音频文件通过填充、裁剪等方式对齐到相同长度 |

### **Audio Servers**

Audio Servers 主要提供一套全面的后端音频处理服务。它涵盖了从基础的音频加载、格式处理，到复杂的数字信号处理（如特征提取、效果添加），以及便捷的网络服务功能，允许用户通过URL访问和管理音频文件。

| 功能名称 | 功能简介 |
|---|---|
| **load_audio** | 加载音频数据 |
| **resample_audio** | 重采样音频 |
| **compute_stft** | 计算短时傅里叶变换 |
| **compute_mfcc** | 计算MFCC特征 |
| **compute_mel_spectrogram** | 计算梅尔频谱图并生成可视化图像 |
| **add_reverb** | 添加混响效果 |
| **mix_audio** | 混合多个音频 |
| **apply_fade** | 应用淡入淡出效果 |
| **serve_local_audio** | 将本地音频文件转换为可访问的URL |
| **stop_audio_server** | 停止音频文件上传服务器，并释放资源 |

### **Tensor Servers**

Tensor Servers 主要提供一系列用于处理和操作 PyTorch 张量和 NumPy数组的工具，涵盖了格式转换、基本运算、数据操作以及GPU设备管理等服务。

| 功能名称 | 功能简介 |
| --- | --- |
| **get_gpu_info** | 获取 GPU 信息 |
| **set_gpu_device** | 设置当前使用的 GPU 设备 |
| **load_numpy_file** | 加载 .npy 格式的 NumPy 数组文件 |
| **load_torch_file** | 加载 .pth 格式的 PyTorch 张量文件 |
| **convert_numpy_to_tensor** | 将 NumPy 数组转换为 PyTorch 张量并保存 |
| **convert_tensor_to_numpy** | 将 PyTorch 张量转换为 NumPy 数组并保存 |
| **move_tensor_to_device** | 将张量移动到指定设备（CPU 或 CUDA） |
| **concatenate_tensors** | 沿指定维度连接多个张量 |
| **split_tensor** | 沿指定维度拆分张量 |
| **save_tensor** | 保存张量数据到 PyTorch .pth 文件 |
| **tensor_operations** | 对张量执行基本操作 |

### **Tool Query Servers**

Tool Query Servers 主要提供一个工具发现和信息查询的服务，它能帮助用户在众多可用工具中，通过列出、查询和智能搜索等方式，找到并了解如何使用最适合其任务需求的工具。

| 功能名称 | 功能简介 |
| --- | --- |
| **query_tool** | 查询任何工具的详细信息，包括其参数规格、使用示例和功能 |
| **list_available_tools** | 列出所有可用的工具及其简要描述 |
| **search_tools_by_task** | 根据自然语言任务描述，智能地搜索相关工具 |

### **FunTTS MCP Servers**

FunTTS MCP Servers 涵盖了从语音识别（Whisper, FunASR）、语音合成（CosyVoice2, SparkTTS）、声音编辑（VoiceCraft）、语音增强（ClearVoice）到情感分析和多维音频理解（Qwen2Audio, EmotionRecognition）的全链条能力。

| 工具名称 | 工具简介 | 模型下载 |
| --- | --- | --- |
| [**FunASRTool**](https://github.com/modelscope/FunASR) | 用于自动语音识别(ASR)、语音活动检测(VAD)和语言识别等任务 | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)<br>[emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| [**EmotionRecognitionTool**](https://github.com/modelscope/FunASR) | 识别语音中的情绪，支持在整个话语或逐秒的粒度上进行分析 | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)<br>[emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| [**CosyVoice2Tool**](https://github.com/FunAudioLLM/CosyVoice) | 高级文本到语音合成，支持语音克隆、跨语言合成以及带指令的情感/方言语音生成 | [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) |
| [**SparkTTSTool**](https://github.com/sparkaudio/spark-tts) | 生成语音，提供零样本语音克隆功能和可控的语音参数 | [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) |
| [**VoiceCraftTool**](https://github.com/jasonppy/VoiceCraft) | 通过替换、插入或删除单词来编辑英语语音，同时保留说话者的原始声音 | [VoiceCraft](https://huggingface.co/pyp1/VoiceCraft) |
| [**Qwen2AudioTool**](https://github.com/QwenLM/Qwen2-Audio) | 基于全面的音频理解，用于转录、音乐分析和说话人识别等任务 | [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| [**ClearVoiceTool**](https://github.com/modelscope/ClearerVoice-Studio) | 增强、分离语音或进行语音超分辨率处理 | [MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)<br>[FRCRN_SE_16K](https://huggingface.co/alibabasglab/FRCRN_SE_16K)<br>[MossFormerGAN_SE_16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)<br>[MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)<br>[MossFormer2_SR_48K](https://huggingface.co/alibabasglab/MossFormer2_SR_48K)<br>[AV_MossFormer2_TSE_16K](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K)<br>(首次调用时会自动下载所需的模型) |
| [**WhisperASRTool**](https://github.com/openai/whisper) | 对长音频进行高质量的自动语音识别(ASR)和翻译 | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |

### **Music MCP Servers**

Music MCP Servers 实现了从文本、歌词到完整歌曲（DiffRhythm, YuEETool, ACEStep）的创作，并支持通过音频驱动肖像图片生成视频（Hallo2）。

| 工具名称 | 工具简介 | 模型下载 |
| --- | --- | --- |
| [**AudioXTool**](https://github.com/ZeyueT/AudioX) | 生成音频或视频。可以从文本、音频或视频输入生成内容 | [AudioX](https://huggingface.co/HKUSTAudio/AudioX) |
| [**ACEStepTool**](https://github.com/ace-step/ACE-Step) | 生成音乐。支持文本到音乐、音乐重制(retake)、局部重绘(repaint)、编辑、扩展和音频到音频转换等多种任务 | [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)<br>[ACE-Step-v1-chinese-rap-LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA) |
| [**MusicGenTool**](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) | 根据文本描述和可选的旋律来生成音乐 | [musicgen-melody](https://huggingface.co/facebook/musicgen-melody) |
| [**AudioGenTool**](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md) | 根据文本描述生成环境音、音效等非音乐类型的音频内容 | [audiogen-medium](https://huggingface.co/facebook/audiogen-medium) |
| [**Hallo2Tool**](https://github.com/fudan-generative-vision/hallo2) | 通过一张源肖像图片和驱动音频来生成对话的动画视频。支持头部姿态、面部表情和唇形同步的权重调整 | [hallo2](https://huggingface.co/fudan-generative-ai/hallo2) |
| [**YuEETool**](https://github.com/multimodal-art-projection/YuE) | 根据流派和歌词生成带人声的完整歌曲。是YuE模型的增强版，支持多种语言和推理方法 | [YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot)<br>[YuE-s1-7B-anneal-en-icl](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl)<br>[YuE-s1-7B-anneal-zh-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot)<br>[YuE-s1-7B-anneal-zh-icl](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl)<br>[YuE-s2-1B-general](https://huggingface.co/m-a-p/YuE-s2-1B-general)<br>[YuE-upsampler](https://huggingface.co/m-a-p/YuE-upsampler) |
| [**DiffRhythmTool**](https://github.com/ASLP-lab/DiffRhythm) | 基于歌词（LRC格式）和风格提示（文本或音频）生成带人声和伴奏的完整歌曲 | [DiffRhythm-v1.2](https://huggingface.co/ASLP-lab/DiffRhythm-1_2)<br>[DiffRhythm-full](https://huggingface.co/ASLP-lab/DiffRhythm-full)<br>(首次调用时会自动下载所需的模型) |

### **Audio Separator MCP Servers**

Audio Separator MCP Servers 提供先进的音频分离技术，能将混合音轨精确分离为人声、伴奏或特定声音（AudioSep, TIGERSpeechSeparationTool），并支持音频超分辨率（AudioSRTool）来提升音质。

| 工具名称 | 工具简介 | 模型下载 |
| --- | --- | --- |
| [**AudioSepTool**](https://github.com/Audio-AGI/AudioSep) | 根据自然语言文本描述，从混合音频中分离出特定的声音事件或乐器 | [audiosep_base_4M_steps](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint) |
| **AudioSeparatorTool<br>(来自 [uvr-mdx-infer](https://github.com/seanghay/uvr-mdx-infer) 和 [Demucs](https://github.com/facebookresearch/demucs))** | 将音轨分离为多个独立的音源（如人声、伴奏、鼓、贝斯等） | [UVR-MDX-NET-Inst_HQ_3](https://huggingface.co/seanghay/uvr_models/blob/main/UVR-MDX-NET-Inst_HQ_3.onnx)<br>[htdemucs_6s](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th) |
| [**TIGERSpeechSeparationTool**](https://github.com/JusperLee/TIGER) | 从包含多人讲话的混合音频中准确地分离出每个人的语音 | [TIGER-speech](https://huggingface.co/JusperLee/TIGER-speech) |
| [**AudioSRTool**](https://github.com/haoheliu/versatile_audio_super_resolution/tree/main) | 通过超分技术增强音频质量，可将低采样率音频提升至48kHz高品质输出 | [audiosr_basic](https://huggingface.co/haoheliu/audiosr_basic)<br>[audiosr_speech](https://huggingface.co/haoheliu/audiosr_speech)<br>(首次调用时会自动下载所需的模型) |

### **API Servers**

API Servers 中集成了部分工具的 API，能实现 FunTTS MCP Servers、Music MCP Servers、Audio Separator MCP Servers 提供的部分功能。

**1. 文本转语音**

| 工具名 | 描述 |
| :--- | :--- |
| [cosyvoice2tool_api](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B) | 将文本转换为逼真语音，支持声音克隆和自然语言控制 |
| [index_tts_1.5_api](https://modelscope.cn/studios/IndexTeam/IndexTTS-Demo) | 通过克隆参考音频中的声音来生成目标文本的语音 |
| [step_audio_tts_3b_api](https://modelscope.cn/studios/Swarmeta_AI/Step-Audio-TTS-3B) | 克隆参考音频的音色以生成新的语音 |
| [sparkTTS_tool_api](https://huggingface.co/spaces/thunnai/SparkTTS) | 支持声音克隆和自定义（性别、音高、速度）的文本转语音工具 |
| [voicecraft_tts_and_edit_api](https://huggingface.co/spaces/Approximetal/VoiceCraft_gradio) | 主要用于文本转语音，也支持对生成的音频进行编辑 |

**2. 音乐与音效创作**

| 工具名 | 描述 |
| :--- | :--- |
| [diffrhythm_api](https://huggingface.co/spaces/ASLP-lab/DiffRhythm) | 从主题、歌词到最终编曲的全流程音乐生成工具 |
| [ACE_Step_api](https://huggingface.co/spaces/ACE-Step/ACE-Step) | 集成化的端到端音乐生成、编辑和扩展工具 |
| [audiocraft_jasco_api](https://huggingface.co/spaces/Tonic/audiocraft) | 基于文本、和弦、旋律和鼓点生成音乐 |
| [yue_api](https://huggingface.co/spaces/innova-ai/YuE-music-generator-demo) | 根据音乐流派、歌词或音频提示生成包含人声和伴奏的音乐 |
| [AudioX_api](https://huggingface.co/spaces/Zeyue7/AudioX) | 根据文本、视频或音频提示生成爆炸、脚步等高质量通用音效 |

**3. 音频修复与分离**

| 工具名 | 描述 |
| :--- | :--- |
| [clearervoice_api](https://huggingface.co/spaces/alibabasglab/ClearVoice) | 多功能的音频处理工具，支持语音增强、分离和超分辨率 |
| [tiger_api](https://huggingface.co/spaces/fffiloni/TIGER-audio-extraction) | 从音频或视频中分离人声、音乐和音效的音轨提取工具 |
| [audio_super_resolution_api](https://huggingface.co/spaces/Nick088/Audio-SR) | 提升音频文件分辨率以增强其质量 |

**4. 音频内容分析**

| 工具名 | 描述 |
| :--- | :--- |
| [whisper_large_v3_turbo_api](https://huggingface.co/spaces/hf-audio/whisper-large-v3-turbo) | 对本地、URL或YouTube音频进行转录或翻译 |
| [SenseVoice_api](https://www.modelscope.cn/studios/iic/SenseVoice) | 基于语音的多任务理解工具，支持识别、情感和事件检测 |
| [Qwen2audio_api](https://modelscope.cn/studios/Qwen/Qwen2-Audio-Instruct-Demo/summary/) | 支持文本和音频输入的多模态对话工具，侧重于理解音频内容 |

## 🚀 安装指南

### 1. 安装 AudioFab

1. 克隆仓库

    ```bash
    git clone https://github.com/SmileHnu/AudioFab.git
    cd AudioFab
    ```

2. 设置虚拟环境

    ```bash
    conda create -n AudioFab python=3.10
    conda activate AudioFab
    ```

3. 安装依赖

    ```bash
    pip install -r requirements.txt
    ```

4. 配置环境

    编辑 `.env` 文件：

    ```
    LLLM_API_KEY=your_llm_api_key_here
    LLM_BASE_URL=your_llm_api_base_url_here
    LLM_MODEL_NAME=your_llm_model_name_here

    OLLAMA_MODEL_NAME="your_ollama_model_name_here"
    OLLAMA_BASE_URL="your_ollama_base_url_here"

    #wsl
    MARKDOWN_FOLDER_PATH=your_markdown_folder_path_here
    RESULT_FOLDER_PATH=your_result_folder_path_here
     ```

    编辑 `mcp_servers/servers_config.json` 以匹配您的本地设置：

    - 将 `command` 替换为您的 python 解释器路径。

    - 将 `PYTHONPATH` 替换为 `mcp_servers` 在您目录中的绝对路径。

    ```json
    {
        "mcpServers": {
            "markdown_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/markdown_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "dsp_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/dsp_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "audio_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/audio_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "tensor_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/tensor_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "tool_query_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/tool_query_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "FunTTS_mcp_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/FunTTS_mcp_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "music_mcp_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/music_mcp_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "Audioseparator_mcp_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/Audioseparator_mcp_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            },
            "API_servers": {
                "command": "your/python/interpreter/path",
                "args": [
                    "mcp_servers/servers/API_servers.py"
                ],
                "env": {
                    "PYTHONPATH": "your/absolute/path/to/mcp_servers"
                }
            }
        }
    }
    ```

    可以通过运行 `scripts/check.sh` 来检查您的配置是否正确。

### 2. 外部依赖部署与配置

将在 `mcp_servers` 文件夹中完成后续的部署工作。

```bash
cd mcp_servers
```

AudioFab 在 [FunTTS MCP Servers](#funtts-mcp-servers)、[Music MCP Servers](#music-mcp-servers)、[Audio Separator MCP Servers](#audio-separator-mcp-servers) 中集成了多个第三方模型，为了确保所有功能都能正常运行，您需要[在本地环境中部署和配置](#本地部署)或[通过 API 使用](#使用-api)这些模型。

**⚠️ 重要配置说明**

由于需要本地部署的模型数量较多，导致本地部署工作繁杂，且本地运行这些模型时间会占用大量的计算资源，因此**更建议您先[通过 API 使用](#使用-api)部分模型**以快速体验 AudioFab。

#### 本地部署

开始本地部署第三方模型之前，请确保已经部署好了 AudioFab 的其他部分，并已将所有 [FunTTS MCP Servers](#funtts-mcp-servers)、[Music MCP Servers](#music-mcp-servers) 和 [Audio Separator MCP Servers](#audio-separator-mcp-servers) 中需要的模型下载至本地。

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

#### 使用 API

- 如果您**已经完成本地部署**，可以跳过这部分内容。

- 如果您**未完成本地部署**，打开 `servers_config.json`，删除以下内容以注销您未部署的三个服务即可。

  ```json
  "FunTTS_mcp_servers": {
      "command": "/home/chengz/anaconda3/bin/python",
      "args": [
          "mcp_servers/servers/FunTTS_mcp_servers.py"
      ],
      "env": {
          "PYTHONPATH": "."
      }
  },
  "music_mcp_servers": {
      "command": "/home/chengz/anaconda3/bin/python",
      "args": [
          "mcp_servers/servers/music_mcp_servers.py"
      ],
      "env": {
          "PYTHONPATH": "."
      }
  },
  "Audioseparator_mcp_servers": {
      "command": "/home/chengz/anaconda3/bin/python",
      "args": [
          "mcp_servers/servers/Audioseparator_mcp_servers.py"
      ],
      "env": {
          "PYTHONPATH": "."
      }
  },
  ```

## ⚡ 快速开始

一键运行 AudioFab

```bash
conda activate AudioFab
python scripts/start_all.py
```

## 🤝 贡献

我们欢迎任何形式的贡献，包括但不限于：

- 报告 Bug
- 提交功能请求
- 代码贡献
- 文档改进

## 🙏 参考与致谢

## 📝 许可证

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

本作品采用
[知识共享署名-非商业性使用 4.0 国际许可协议][cc-by-nc] 进行许可。

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

详细许可证条款请参阅 [`LICENSE`](./LICENSE.txt) 文件。

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## 📖引用
