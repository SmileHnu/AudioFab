# Tool Kits 集成工具详细介绍

此文档将介绍 MCP Server 中各服务内部集成的工具。

## **Markdown Servers**

Markdown Servers 主要提供一套基础的文件读写和管理服务，专注于处理 Markdown、TXT 和 JSON 等文本格式的文件。

| 功能名称 | 功能简介 |
|---|---|
| **read_file** | 读取指定类型的所有文件内容，支持md, txt, json文件。 |
| **write_file** | 写入（或新建）指定类型文件，支持md、txt、json。 |
| **modify_file** | 修改（覆盖）单个已存在的.md/.txt/.json文件。 |

## **DSP Servers**

DSP Servers 主要提供一系列基础的数字音频信号处理服务，涵盖了音频特征提取、格式转换以及基础编辑等功能。

| 功能名称 | 功能简介 |
|---|---|
| **compute_stft** | 计算音频信号的短时傅里叶变换（STFT） |
| **compute_mfcc** | 计算音频的梅尔频率倒谱系数（MFCC）特征 |
| **compute_mel_spectrogram** | 计算音频的梅尔频谱图并保存为数据文件 |
| **convert_audio_format** | 将音频从一种格式转换为另一种格式，并可调整参数 |
| **trim_audio** | 裁剪音频文件的指定时间区间 |
| **align_audio_lengths** | 将多个音频文件通过填充、裁剪等方式对齐到相同长度 |

## **Audio Servers**

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

## **Tensor Servers**

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

## **Tool Query Servers**

Tool Query Servers 主要提供一个工具发现和信息查询的服务，它能帮助用户在众多可用工具中，通过列出、查询和智能搜索等方式，找到并了解如何使用最适合其任务需求的工具。

| 功能名称 | 功能简介 |
| --- | --- |
| **query_tool** | 查询任何工具的详细信息，包括其参数规格、使用示例和功能 |
| **list_available_tools** | 列出所有可用的工具及其简要描述 |
| **search_tools_by_task** | 根据自然语言任务描述，智能地搜索相关工具 |

## **FunTTS MCP Servers**

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

## **Music MCP Servers**

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

## **Audio Separator MCP Servers**

Audio Separator MCP Servers 提供先进的音频分离技术，能将混合音轨精确分离为人声、伴奏或特定声音（AudioSep, TIGERSpeechSeparationTool），并支持音频超分辨率（AudioSRTool）来提升音质。

| 工具名称 | 工具简介 | 模型下载 |
| --- | --- | --- |
| [**AudioSepTool**](https://github.com/Audio-AGI/AudioSep) | 根据自然语言文本描述，从混合音频中分离出特定的声音事件或乐器 | [audiosep_base_4M_steps](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint) |
| **AudioSeparatorTool<br>(来自 [uvr-mdx-infer](https://github.com/seanghay/uvr-mdx-infer) 和 [Demucs](https://github.com/facebookresearch/demucs))** | 将音轨分离为多个独立的音源（如人声、伴奏、鼓、贝斯等） | [UVR-MDX-NET-Inst_HQ_3](https://huggingface.co/seanghay/uvr_models/blob/main/UVR-MDX-NET-Inst_HQ_3.onnx)<br>[htdemucs_6s](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th) |
| [**TIGERSpeechSeparationTool**](https://github.com/JusperLee/TIGER) | 从包含多人讲话的混合音频中准确地分离出每个人的语音 | [TIGER-speech](https://huggingface.co/JusperLee/TIGER-speech) |
| [**AudioSRTool**](https://github.com/haoheliu/versatile_audio_super_resolution/tree/main) | 通过超分技术增强音频质量，可将低采样率音频提升至48kHz高品质输出 | [audiosr_basic](https://huggingface.co/haoheliu/audiosr_basic)<br>[audiosr_speech](https://huggingface.co/haoheliu/audiosr_speech)<br>(首次调用时会自动下载所需的模型) |

## **API Servers**

API Servers 中集成了部分工具的 API，能实现 FunTTS MCP Servers、Music MCP Servers、Audio Separator MCP Servers 提供的部分功能。

**1. 文本转语音**

| 工具名 | 描述 |
| :--- | :--- |
| [cosyvoice2tool_api](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B) | 将文本转换为逼真语音，支持声音克隆和自然语言控制 |
| [index_tts_1.5_api](https://huggingface.co/spaces/IndexTeam/IndexTTS) | 通过克隆参考音频中的声音来生成目标文本的语音 |
| [step_audio_tts_3b_api](https://modelscope.cn/studios/Swarmeta_AI/Step-Audio-TTS-3B) | 克隆参考音频的音色以生成新的语音 |
| [sparkTTS_tool_api](https://huggingface.co/spaces/thunnai/SparkTTS) | 支持声音克隆和自定义（性别、音高、速度）的文本转语音工具 |
| [voicecraft_tts_and_edit_api](https://huggingface.co/spaces/Approximetal/VoiceCraft_gradio) | 主要用于文本转语音，也支持对生成的音频进行编辑 |

**2. 音乐与音效创作**

| 工具名 | 描述 |
| :--- | :--- |
| [diffrhythm_api](https://huggingface.co/spaces/dskill/DiffRhythm) | 从主题、歌词到最终编曲的全流程音乐生成工具 |
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
| [SenseVoice_api](https://huggingface.co/spaces/megatrump/SenseVoice) | 基于语音的多任务理解工具，支持识别、情感和事件检测 |
| [Qwen2audio_api](https://modelscope.cn/studios/Qwen/Qwen2-Audio-Instruct-Demo/summary/) | 支持文本和音频输入的多模态对话工具，侧重于理解音频内容 |
