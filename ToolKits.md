# Tool Kits: Detailed Introduction to Integrated Tools

This document provides a detailed introduction to the tools integrated within various services in the MCP Server.

## **Markdown Servers**

Markdown Servers primarily provide a basic set of file reading, writing, and management services, focusing on handling text-based file formats such as Markdown, TXT, and JSON.

| Function Name | Function Introduction |
|---|---|
| read_file | Reads the content of all files of a specified type, with support for md, txt, and json files. |
| write_file | Writes to (or creates a new) file of a specified type, with support for md, txt, and json. |
| modify_file | Modifies (overwrites) a single existing .md/.txt/.json file. |

## **DSP Servers**

DSP Servers primarily provide a series of basic digital audio signal processing services, covering functions such as audio feature extraction, format conversion, and basic editing.

| Function Name | Function Introduction |
|---|---|
| compute_stft | Computes the Short-Time Fourier Transform (STFT) of an audio signal. |
| compute_mfcc | Computes the Mel-Frequency Cepstral Coefficients (MFCC) features of an audio signal. |
| compute_mel_spectrogram | Computes the Mel spectrogram of an audio signal and saves it as a data file. |
| convert_audio_format | Converts an audio file from one format to another, with adjustable parameters. |
| trim_audio | Trims a specified time interval of an audio file. |
| align_audio_lengths | Aligns multiple audio files to the same length through methods like padding or trimming. |

## **Audio Servers**

Audio Servers provide a comprehensive set of back-end audio processing services. It covers everything from basic audio loading and format processing to complex digital signal processing (such as feature extraction and effect addition), as well as convenient network service functions that allow users to access and manage audio files via URL.

| Function Name | Function Introduction |
|---|---|
| load_audio | Loads audio data |
| resample_audio | Resamples audio |
| compute_stft | Computes the Short-Time Fourier Transform |
| compute_mfcc | Computes MFCC features |
| compute_mel_spectrogram | Computes a mel spectrogram and generates a visualization image |
| add_reverb | Adds a reverb effect |
| mix_audio | Mixes multiple audio tracks |
| apply_fade | Applies a fade-in/fade-out effect |
| serve_local_audio | Converts a local audio file into an accessible URL |
| stop_audio_server | Stops the audio file upload server and releases resources |

## **Tensor Servers**

Tensor Servers primarily provide a series of tools for processing and manipulating PyTorch tensors and NumPy arrays, covering services such as format conversion, basic operations, data manipulation, and GPU device management.

| Function Name | Function Introduction |
| --- | --- |
| get_gpu_info | Get GPU information |
| set_gpu_device | Set the current GPU device |
| load_numpy_file | Load a NumPy array file in .npy format |
| load_torch_file | Load a PyTorch tensor file in .pth format |
| convert_numpy_to_tensor | Convert a NumPy array to a PyTorch tensor and save it |
| convert_tensor_to_numpy | Convert a PyTorch tensor to a NumPy array and save it |
| move_tensor_to_device | Move a tensor to a specified device (CPU or CUDA) |
| concatenate_tensors | Concatenate multiple tensors along a specified dimension |
| split_tensor | Split a tensor along a specified dimension |
| save_tensor | Save tensor data to a PyTorch .pth file |
| tensor_operations | Perform basic operations on a tensor |

## **Tool Query Servers**

Tool Query Servers primarily provide a service for tool discovery and information query. They help users find the most suitable tools for their tasks from among many available options and learn how to use them through methods such as listing, querying, and intelligent search.

| Function Name | Function Introduction |
| --- | --- |
| query_tool | Queries the detailed information of any tool, including its parameter specifications, usage examples, and functionality. |
| list_available_tools | Lists all available tools and their brief descriptions. |
| search_tools_by_task | Intelligently searches for relevant tools based on a natural language description of a task. |

## **FunTTS MCP Servers**

FunTTS MCP Servers cover a full-chain of capabilities from speech recognition (Whisper, FunASR), speech synthesis (CosyVoice2, SparkTTS), voice editing (VoiceCraft), speech enhancement (ClearVoice), to emotion analysis and multi-dimensional audio understanding (Qwen2Audio, EmotionRecognition).

| Tool Name | Tool Introduction | Model Download |
| --- | --- | --- |
| [FunASRTool](https://github.com/modelscope/FunASR) | Used for tasks such as automatic speech recognition (ASR), voice activity detection (VAD), and language identification. | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)<br>[emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| [EmotionRecognitionTool](https://github.com/modelscope/FunASR) | Recognizes emotions in speech, supporting analysis at the whole utterance or second-by-second level. | [SenseVoiceSmall](https://huggingface.co/FunAudioLLM/SenseVoiceSmall)<br>[emotion2vec_plus_large](https://huggingface.co/emotion2vec/emotion2vec_plus_large) |
| [CosyVoice2Tool](https://github.com/FunAudioLLM/CosyVoice) | Advanced text-to-speech synthesis, supporting voice cloning, cross-lingual synthesis, and emotion/dialect speech generation with instructions. | [CosyVoice2-0.5B](https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B) |
| [SparkTTSTool](https://github.com/sparkaudio/spark-tts) | Generates speech, providing zero-shot voice cloning capabilities and controllable speech parameters. | [Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B) |
| [VoiceCraftTool](https://github.com/jasonppy/VoiceCraft) | Edits English speech by replacing, inserting, or deleting words while preserving the speaker's original voice. | [VoiceCraft](https://huggingface.co/pyp1/VoiceCraft) |
| [Qwen2AudioTool](https://github.com/QwenLM/Qwen2-Audio) | Based on comprehensive audio understanding, used for tasks such as transcription, music analysis, and speaker identification. | [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) |
| [ClearVoiceTool](https://github.com/modelscope/ClearerVoice-Studio) | Enhances, separates speech, or performs speech super-resolution processing. | [MossFormer2_SE_48K](https://huggingface.co/alibabasglab/MossFormer2_SE_48K)<br>[FRCRN_SE_16K](https://huggingface.co/alibabasglab/FRCRN_SE_16K)<br>[MossFormerGAN_SE_16K](https://huggingface.co/alibabasglab/MossFormerGAN_SE_16K)<br>[MossFormer2_SS_16K](https://huggingface.co/alibabasglab/MossFormer2_SS_16K)<br>[MossFormer2_SR_48K](https://huggingface.co/alibabasglab/MossFormer2_SR_48K)<br>[AV_MossFormer2_TSE_16K](https://huggingface.co/alibabasglab/AV_MossFormer2_TSE_16K)<br>(The required model will be downloaded automatically on the first call) |
| [WhisperASRTool](https://github.com/openai/whisper) | Performs high-quality automatic speech recognition (ASR) and translation for long audio. | [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) |

## **Music MCP Servers**

Music MCP Servers enable the creation of full songs from text and lyrics (DiffRhythm, YuEETool, ACEStep), and support audio-driven video generation from portrait images (Hallo2).

| Tool Name | Tool Introduction | Model Download |
| --- | --- | --- |
| [AudioXTool](https://github.com/ZeyueT/AudioX) | Generates audio or video. Can generate content from text, audio, or video inputs. | [AudioX](https://huggingface.co/HKUSTAudio/AudioX) |
| [ACEStepTool](https://github.com/ace-step/ACE-Step) | Generates music. Supports multiple tasks such as text-to-music, music retake, repaint, editing, extension, and audio-to-audio conversion. | [ACE-Step-v1-3.5B](https://huggingface.co/ACE-Step/ACE-Step-v1-3.5B)<br>[ACE-Step-v1-chinese-rap-LoRA](https://huggingface.co/ACE-Step/ACE-Step-v1-chinese-rap-LoRA) |
| [MusicGenTool](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md) | Generates music from text descriptions and optional melodies. | [musicgen-melody](https://huggingface.co/facebook/musicgen-melody) |
| [AudioGenTool](https://github.com/facebookresearch/audiocraft/blob/main/docs/AUDIOGEN.md) | Generates non-music audio content such as ambient sounds and sound effects from text descriptions. | [audiogen-medium](https://huggingface.co/facebook/audiogen-medium) |
| [Hallo2Tool](https://github.com/fudan-generative-vision/hallo2) | Generates an animated video of a talking head from a source portrait image and a driving audio. Supports weight adjustment for head pose, facial expressions, and lip-sync. | [hallo2](https://huggingface.co/fudan-generative-ai/hallo2) |
| [YuEETool](https://github.com/multimodal-art-projection/YuE) | Generates complete songs with vocals based on genre and lyrics. It is an enhanced version of the YuE model, supporting multiple languages and inference methods. | [YuE-s1-7B-anneal-en-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot)<br>[YuE-s1-7B-anneal-en-icl](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl)<br>[YuE-s1-7B-anneal-zh-cot](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot)<br>[YuE-s1-7B-anneal-zh-icl](https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl)<br>[YuE-s2-1B-general](https://huggingface.co/m-a-p/YuE-s2-1B-general)<br>[YuE-upsampler](https://huggingface.co/m-a-p/YuE-upsampler) |
| [DiffRhythmTool](https://github.com/ASLP-lab/DiffRhythm) | Generates complete songs with vocals and accompaniment based on lyrics (LRC format) and style prompts (text or audio). | [DiffRhythm-v1.2](https://huggingface.co/ASLP-lab/DiffRhythm-1_2)<br>[DiffRhythm-full](https://huggingface.co/ASLP-lab/DiffRhythm-full)<br>(The required model will be downloaded automatically on the first call) |

## **Audio Separator MCP Servers**

Audio Separator MCP Servers offer advanced audio separation technology, capable of precisely separating mixed audio tracks into vocals, accompaniment, or specific sounds (AudioSep, TIGERSpeechSeparationTool), and support audio super-resolution (AudioSRTool) to enhance audio quality.

| Tool Name | Tool Introduction | Model Download |
| --- | --- | --- |
| [AudioSepTool](https://github.com/Audio-AGI/AudioSep) | Separates specific sound events or musical instruments from a mixed audio based on natural language text descriptions. | [audiosep_base_4M_steps](https://huggingface.co/spaces/Audio-AGI/AudioSep/tree/main/checkpoint) |
| AudioSeparatorTool<br>(from [uvr-mdx-infer](https://github.com/seanghay/uvr-mdx-infer) and [Demucs](https://github.com/facebookresearch/demucs)) | Separates an audio track into multiple independent sources (such as vocals, accompaniment, drums, bass, etc.). | [UVR-MDX-NET-Inst_HQ_3](https://huggingface.co/seanghay/uvr_models/blob/main/UVR-MDX-NET-Inst_HQ_3.onnx)<br>[htdemucs_6s](https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th) |
| [TIGERSpeechSeparationTool](https://github.com/JusperLee/TIGER) | Accurately separates the speech of each individual from a mixed audio containing multiple speakers. | [TIGER-speech](https://huggingface.co/JusperLee/TIGER-speech) |
| [AudioSRTool](https://github.com/haoheliu/versatile_audio_super_resolution/tree/main) | Enhances audio quality through super-resolution technology, capable of upscaling low-sample-rate audio to high-quality 48kHz output. | [audiosr_basic](https://huggingface.co/haoheliu/audiosr_basic)<br>[audiosr_speech](https://huggingface.co/haoheliu/audiosr_speech)<br>(The required model will be downloaded automatically on the first call) |

## **API Servers**

The APIs of several tools are integrated into API Servers, which can realize some of the functions provided by FunTTS MCP Servers, Music MCP Servers, and Audio Separator MCP Servers.

**1. Text-to-Speech**

| Tool Name | Description |
| :--- | :--- |
| [cosyvoice2tool_api](https://www.modelscope.cn/studios/iic/CosyVoice2-0.5B) | Converts text into realistic speech, supporting voice cloning and natural language control. |
| [index_tts_1.5_api](https://huggingface.co/spaces/IndexTeam/IndexTTS) | Generates speech for a target text by cloning the voice from a reference audio. |
| [step_audio_tts_3b_api](https://modelscope.cn/studios/Swarmeta_AI/Step-Audio-TTS-3B) | Clones the timbre of a reference audio to generate new speech. |
| [sparkTTS_tool_api](https://huggingface.co/spaces/thunnai/SparkTTS) | A text-to-speech tool that supports voice cloning and customization (gender, pitch, speed). |
| [voicecraft_tts_and_edit_api](https://huggingface.co/spaces/Approximetal/VoiceCraft_gradio) | Primarily used for text-to-speech, but also supports editing the generated audio. |

**2. Music and Sound Effect Creation**

| Tool Name | Description |
| :--- | :--- |
| [diffrhythm_api](https://huggingface.co/spaces/dskill/DiffRhythm) | A full-process music generation tool, from theme and lyrics to final arrangement. |
| [ACE_Step_api](https://huggingface.co/spaces/ACE-Step/ACE-Step) | An integrated, end-to-end tool for music generation, editing, and extension. |
| [audiocraft_jasco_api](https://huggingface.co/spaces/Tonic/audiocraft) | Generates music based on text, chords, melody, and drum beats. |
| [yue_api](https://huggingface.co/spaces/innova-ai/YuE-music-generator-demo) | Generates music with vocals and accompaniment based on music genre, lyrics, or audio prompts. |
| [AudioX_api](https://huggingface.co/spaces/Zeyue7/AudioX) | Generates high-quality general sound effects like explosions and footsteps based on text, video, or audio prompts. |

**3. Audio Restoration and Separation**

| Tool Name | Description |
| :--- | :--- |
| [clearervoice_api](https://huggingface.co/spaces/alibabasglab/ClearVoice) | A multi-functional audio processing tool that supports speech enhancement, separation, and super-resolution. |
| [tiger_api](https://huggingface.co/spaces/fffiloni/TIGER-audio-extraction) | A track extraction tool for separating vocals, music, and sound effects from audio or video. |
| [audio_super_resolution_api](https://huggingface.co/spaces/Nick088/Audio-SR) | Increases the resolution of audio files to enhance their quality. |

**4. Audio Content Analysis**

| Tool Name | Description |
| :--- | :--- |
| [whisper_large_v3_turbo_api](https://huggingface.co/spaces/hf-audio/whisper-large-v3-turbo) | Transcribes or translates local, URL, or YouTube audio. |
| [SenseVoice_api](https://huggingface.co/spaces/megatrump/SenseVoice) | A speech-based multi-task understanding tool that supports recognition, emotion, and event detection. |
| [Qwen2audio_api](https://modelscope.cn/studios/Qwen/Qwen2-Audio-Instruct-Demo/summary/) | A multimodal dialogue tool that supports text and audio input, with a focus on understanding audio content. |
