import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import inspect
import tempfile
from typing import Optional, Dict, Any, List, Union, Literal

from mcp.server.fastmcp import FastMCP

# 初始化MCP服务器
mcp = FastMCP("语音识别与合成服务：集成FunASR、CosyVoice2,Qwen2audio和SparkTTS模型")

# 工具环境配置
TOOL_ENV_CONFIG = {
    "FunASRTool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Funasr_processor.py")
    },
    "EmotionRecognitionTool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Funasr_processor.py")
    },
    # "CosyVoiceTool": {
    #     "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",
    #     "script_path": str(Path(__file__).parent / "Cosyvoice2_processor.py")
    # },
    "CosyVoice2Tool": {
        "python_path": "/home/chengz/anaconda3/envs/cosyvoice/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Cosyvoice2_tool.py")
    },
    "SparkTTSTool": {
        "python_path": "/home/chengz/anaconda3/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "sparktts_processor.py")
    },
    "VoiceCraftTool": {
        "python_path": "/home/chengz/anaconda3/envs/voicecraft/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "voicecraft_processor.py")
    },
    "Qwen2AudioTool": {
        "python_path": "/home/chengz/anaconda3/envs/Qwenaudio/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Qwen2Audio_processor.py")
    },
    "ClearVoiceTool": {
        "python_path": "/home/chengz/anaconda3/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "ClearerVoice_tool.py")
    },
    "WhisperASRTool": {
        "python_path": "/home/chengz/anaconda3/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "whisper_tool.py")
    }
}

# 启动器脚本路径
LAUNCHER_SCRIPT = str(Path(__file__).parent.parent  / "mcp_tool_launcher.py")

# 创建临时目录（如果不存在）
TEMP_DIR = Path(tempfile.gettempdir()) / "mcp_temp"
TEMP_DIR.mkdir(exist_ok=True)

def execute_tool_in_env(tool_name, **kwargs):
    """在特定环境中执行工具
    
    Args:
        tool_name: 要执行的工具名称
        **kwargs: 传递给工具的参数
        
    Returns:
        工具执行结果
    """
    # 检查工具配置是否存在
    if tool_name not in TOOL_ENV_CONFIG:
        return {"success": False, "error": f"工具 '{tool_name}' 没有环境配置"}
    
    tool_config = TOOL_ENV_CONFIG[tool_name]
    
    # 创建临时参数文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_params_file = TEMP_DIR / f"{tool_name}_{timestamp}.json"
    
    # 将参数写入临时文件
    with open(temp_params_file, 'w') as f:
        json.dump(kwargs, f)
    
    try:
        # 构建命令 - 使用固定的启动器脚本
        cmd = [
            tool_config["python_path"],
            LAUNCHER_SCRIPT,
            "--tool_name", tool_name,
            "--module_path", tool_config["script_path"],
            "--params_file", str(temp_params_file)
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        

        # 清理临时文件
        try:
            os.remove(temp_params_file)
        except:
            pass  # 忽略清理错误
        
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"工具执行失败: {result.stderr}"
            }
        
        #解析结果
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {
                "success": True,
                "error": f"不影响运行结果的打印信息: {result.stdout.strip()}"#result.stdout.strip()
            }


            
    except Exception as e:
        # 清理临时文件
        try:
            os.remove(temp_params_file)
        except:
            pass  # 忽略清理错误
            
        return {"success": False, "error": str(e)}

# 直接定义工具而不导入

@mcp.tool()
def FunASRTool(
    # Input content
    audio_path: str,
    
    # Task configuration
    task: Literal["asr", "asr_itn", "lid", "vad", "punc", "timestamp", "streaming_asr"] = "asr",
    
    # Output configuration
    output_path: Optional[str] = None,
    output_format: Literal["json", "txt"] = "json",
    
    # Language options
    language: Literal["zh", "yue", "en", "ja", "ko", "auto"] = "auto",
    
    # Streaming parameters (for streaming_asr only)
    is_streaming: bool = False,
    chunk_size: Optional[List[int]] = None,      # Format: [0, 10, 5] - see FunASR docs
    encoder_chunk_look_back: Optional[int] = None,
    decoder_chunk_look_back: Optional[int] = None,
    
    # Text-related parameters
    text_file: Optional[str] = None,  # For timestamp task
    
    # Model configuration
    model_name: Optional[str] = None,
    model_revision: Optional[str] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Process audio using FunASR models for tasks like ASR, VAD, language identification, etc.
    
    Args:
        audio_path: Path to the input audio file
        
        task: The speech processing task to perform 
              - asr: Automatic Speech Recognition
              - asr_itn: ASR with Inverse Text Normalization
              - lid: Language Identification
              - vad: Voice Activity Detection
              - punc: Punctuation Restoration
              - timestamp: Timestamp Prediction
              - streaming_asr: Streaming ASR
              
        output_path: Custom path to save the output. If not provided, a default path will be used
        output_format: Format for the output (json or txt)
        
        language: Language of the audio (zh, yue, en, ja, ko, auto)
        
        is_streaming: Whether to use streaming mode
        chunk_size: Streaming configuration, e.g. [0, 10, 5]
        encoder_chunk_look_back: Number of encoder chunks to look back
        decoder_chunk_look_back: Number of decoder chunks to look back
        
        text_file: Path to text file (required for timestamp task)
        
        model_name: Custom model name to override default selection
        model_revision: Model revision to use
        device: Computing device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing processing results and metadata
    """
    return execute_tool_in_env("FunASRTool", 
                               audio_path=audio_path,
                               task=task,
                               output_path=output_path,
                               output_format=output_format,
                               language=language,
                               is_streaming=is_streaming,
                               chunk_size=chunk_size,
                               encoder_chunk_look_back=encoder_chunk_look_back,
                               decoder_chunk_look_back=decoder_chunk_look_back,
                               text_file=text_file,
                               model_name=model_name,
                               model_revision=model_revision,
                               device=device)

@mcp.tool()
def EmotionRecognitionTool(
    # Input content
    audio_path: str,
    
    # Output configuration
    output_dir: Optional[str] = None,
    
    # Processing options
    granularity: Literal["utterance", "second"] = "utterance",
    extract_embedding: bool = False,
    
    # Model configuration
    model_name: str = "/home/chengz/LAMs/pre_train_models/models--emotion2vec--emotion2vec_plus_large",
    device: str = "cuda"
) -> Dict[str, Any]:
    """Recognize emotions in speech audio using the emotion2vec model.
    
    Args:
        audio_path: Path to the input audio file
        
        output_dir: Directory to save the output. If not provided, a default path will be used
        
        granularity: Level of analysis granularity
                     - "utterance": Analyze the entire audio clip as one unit
                     - "second": Analyze emotions second by second
                     
        extract_embedding: Whether to extract and return emotion embeddings
        
        model_name: Path to the emotion recognition model
        device: Computing device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing emotion recognition results and metadata
    """
    return execute_tool_in_env("EmotionRecognitionTool",
                               audio_path=audio_path,
                               output_dir=output_dir,
                               granularity=granularity,
                               extract_embedding=extract_embedding,
                               model_name="/home/chengz/LAMs/pre_train_models/models--emotion2vec--emotion2vec_plus_large",
                               device=device)

@mcp.tool()
def CosyVoiceTool(
    # Core inputs
    text: Optional[str] = None, # Text to synthesize. Not used in Voice Conversion mode.
    source_audio_path: Optional[str] = None, # For Voice Conversion: audio to be converted.

    # Voice reference/control
    prompt_audio_path: Optional[str] = None, # Reference audio for cloning, cross-lingual, or VC target.
    prompt_text: Optional[str] = None, # Transcript of prompt_audio_path, primarily for zero-shot.
    speaker_id: Optional[str] = None, # ID of a pre-defined SFT speaker (e.g., "中文女", "中文男").
    zero_shot_speaker_id: Optional[str] = None, # An ID for a pre-cached zero-shot speaker embedding.

    # Mode selectors / high-level instructions
    cross_lingual_synthesis: bool = False, # If True, use cross-lingual mode with prompt_audio_path.
    
    # For instruct mode (requires using the instruct model)
    use_instruct_mode: bool = False, # If True, use the Instruct model for emotional, styled speech
    instruct_text: Optional[str] = None, # Custom instruction for the instruct model

    # Advanced/Optional parameters for inference methods
    speed: float = 1.0, # Speech speed.
    stream_output: bool = False, # Corresponds to 'stream' in CosyVoice methods.
    use_text_frontend: bool = True, # For text normalization. Set to False to pass raw text as in some examples.

    # Output & System Configuration
    language_tag: Optional[str] = None, # Language tag, e.g. "<|zh|>","<|en|>","<|jp|>","<|yue|>","<|ko|>"
    output_path: Optional[str] = None,
    device_hint: str = "cuda", # Hint for torch device context if needed.
    # Model loading options (effective only on first call that loads the model)
    model_fp16: bool = False,
    model_jit: bool = False,
    model_trt: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive Text-to-Speech and Voice Conversion tool using CosyVoice.

    Supports:
    - SFT (Standard Fine-Tuned) synthesis with a speaker_id.
    - Zero-Shot voice cloning from a prompt_audio_path and prompt_text.
    - Cross-Lingual synthesis using a prompt_audio_path for voice and new text.
    - Instructed Synthesis (with CosyVoice-Instruct model) for emotional, styled speech.
    - Voice Conversion (VC) from source_audio_path to the voice of prompt_audio_path.

    Args:
        text: Text to synthesize. Required for all modes except Voice Conversion.
            For cross-lingual, prepend language tag like "<|en|>" for English.
        source_audio_path: Path to the source audio file (16kHz preferred) for Voice Conversion mode.
        
        prompt_audio_path: Path to a reference audio file (16kHz preferred) for voice.
                           Used in Zero-Shot, Cross-Lingual, and as target voice for VC.
        prompt_text: Transcript of prompt_audio_path. Recommended for Zero-Shot mode.
        speaker_id: ID of a pre-defined SFT speaker (e.g., "中文女", "中文男").
        zero_shot_speaker_id: An ID for a pre-cached zero-shot speaker embedding.
        
        cross_lingual_synthesis: If True, performs cross-lingual synthesis. Requires 'text' and 'prompt_audio_path'.
        
        use_instruct_mode: If True, uses the CosyVoice-Instruct model for emotional, styled speech.
        instruct_text: Custom instruction for the instruct model (e.g. character description, style).

        speed: Controls the speed of the generated speech (default: 1.0).
        stream_output: If True, model yields audio segment by segment. Tool concatenates before saving.
        use_text_frontend: If True (default), uses CosyVoice's text normalization.
        
        language_tag: Language tag for cross-lingual synthesis ("<|zh|>","<|en|>","<|jp|>","<|yue|>","<|ko|>")
        output_path: Path to save the generated audio. Defaults to a timestamped file.
        device_hint: PyTorch device hint ("cuda", "cpu"). Model primarily uses CUDA if available.
        model_fp16/jit/trt: Advanced model loading options (effective on first initialization).

    Returns:
        Dictionary with generation results, metadata, and status.
    """
    return execute_tool_in_env("CosyVoiceTool",
                               text=text,
                               source_audio_path=source_audio_path,
                               prompt_audio_path=prompt_audio_path,
                               prompt_text=prompt_text,
                               speaker_id=speaker_id,
                               zero_shot_speaker_id=zero_shot_speaker_id,
                               cross_lingual_synthesis=cross_lingual_synthesis,
                               use_instruct_mode=use_instruct_mode,
                               instruct_text=instruct_text,
                               speed=speed,
                               stream_output=stream_output,
                               use_text_frontend=use_text_frontend,
                               language_tag=language_tag,
                               output_path=output_path,
                               device_hint=device_hint,
                               model_fp16=model_fp16,
                               model_jit=model_jit,
                               model_trt=model_trt)

@mcp.tool()
def CosyVoice2Tool(
    # Core inputs
    text: Optional[str] = None,
    source_audio_path: Optional[str] = None,
    
    # Voice reference/control
    prompt_audio_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    speaker_id: Optional[str] = None,
    zero_shot_speaker_id: Optional[str] = None,
    
    # Mode selectors / high-level instructions
    cross_lingual_synthesis: bool = False,
    use_instruct_mode: bool = False,
    instruct_text: Optional[str] = None,
    
    # Advanced/Optional parameters
    speed: float = 1.0,
    stream_output: bool = False,
    use_text_frontend: bool = True,
    
    # Output & System Configuration
    language_tag: Optional[str] = None,
    output_path: Optional[str] = None,
    device_hint: str = "cuda",
    
    # Model loading options
    model_fp16: bool = False,
    model_jit: bool = False,
    model_trt: bool = False,
    use_flow_cache: bool = False
) -> Dict[str, Any]:
    """
    Advanced Text-to-Speech synthesis using CosyVoice2 model with multiple capabilities.
    
    CosyVoice2 supports these key features:
    - Zero-shot In-context Generation: Clone any voice from a reference audio
    - Cross-lingual In-context Generation: Synthesize text in a different language with the same voice
    - Mixed-lingual In-context Generation: Support for multiple languages in the same sentence
    - Instructed Voice Generation: Role-playing and emotional voice control
    - Dialect Control: Support for various dialects (e.g., Cantonese, Shanghai, Shandong)
    - Fine-grained Control: Advanced control over pronunciation and style
    - Speaking Style Control: Emotional expression and speaking rate control
    - Voice Cloning: Convert source audio to target speaker's voice
    
    Args:
        text: Text to synthesize. Required for all modes except Voice Conversion.
            For multilingual text, can include language tags like "<|zh|>", "<|en|>", etc.
            
        source_audio_path: Path to source audio for Voice Conversion mode.
        
        prompt_audio_path: Path to reference audio for voice cloning, cross-lingual synthesis, 
                          or as target voice for conversion.
                          
        prompt_text: Transcript of prompt_audio_path. Required for Zero-shot mode.
        
        speaker_id: ID of a pre-defined speaker (if supported by the model).
        
        zero_shot_speaker_id: ID for a pre-cached zero-shot speaker embedding.
        
        cross_lingual_synthesis: If True, performs cross-lingual synthesis with the same voice.
                                Requires 'text' and 'prompt_audio_path'.
                                
        use_instruct_mode: If True, enables Instructed Voice Generation mode for emotional,
                          styled speech using the inference_instruct2 method.
                          
        instruct_text: Instructions for voice style, e.g.:
                      - Emotion: "用开心的语气说", "用伤心的语气说", "用恐惧的情感表达"
                      - Dialect: "用粤语说这句话", "用上海话说", "使用山东话说"
                      - Character: "一个忧郁的诗人，言语中总是透露出一丝哀愁和浪漫"
                      - Speaking style: "Speaking very fast", "Speaking with patience"
                      
        speed: Controls speech rate (default: 1.0).
        
        stream_output: Whether to process audio in streaming mode.
        
        use_text_frontend: Whether to use text normalization preprocessing.
        
        language_tag: Language tag to prepend to text if not already present.
                     Examples: "<|zh|>", "<|en|>", "<|jp|>", "<|yue|>", "<|ko|>"
                     
        output_path: Path to save the output audio file. Default is timestamped WAV file.
        
        device_hint: Computing device ("cuda" or "cpu").
        
        model_fp16: Use FP16 precision (requires CUDA).
        
        model_jit: Use JIT compilation for faster inference (requires CUDA).
        
        model_trt: Use TensorRT for acceleration (requires CUDA).
        
        use_flow_cache: Whether to use flow cache for faster inference.
        
    Returns:
        Dictionary containing:
        - success: Whether synthesis was successful
        - output_path: Path to the generated audio file
        - sample_rate: Sample rate of the generated audio
        - duration: Duration of the generated audio in seconds
        - text: The synthesized text
        - metadata: Additional information about the synthesis
        - error: Error message if synthesis failed
    
    Examples:
        # Zero-shot voice cloning
        result = CosyVoice2Tool(
            text="Hello world, this is a synthesized voice.",
            prompt_audio_path="reference_voice.wav",
            prompt_text="This is my reference voice sample."
        )
        
        # Cross-lingual synthesis
        result = CosyVoice2Tool(
            text="<|en|>This is spoken in English with a Chinese voice.",
            prompt_audio_path="chinese_voice.wav",
            cross_lingual_synthesis=True
        )
        
        # Instructed voice generation (emotional)
        result = CosyVoice2Tool(
            text="Life is full of wonderful surprises.",
            use_instruct_mode=True,
            instruct_text="用开心的语气说"
        )
        
        # Voice conversion
        result = CosyVoice2Tool(
            source_audio_path="source_voice.wav",
            prompt_audio_path="target_voice.wav"
        )
    """
    return execute_tool_in_env("CosyVoice2Tool", 
                               text=text,
                               source_audio_path=source_audio_path,
                               prompt_audio_path=prompt_audio_path,
                               prompt_text=prompt_text,
                               speaker_id=speaker_id,
                               zero_shot_speaker_id=zero_shot_speaker_id,
                               cross_lingual_synthesis=cross_lingual_synthesis,
                               use_instruct_mode=use_instruct_mode,
                               instruct_text=instruct_text,
                               speed=speed,
                               stream_output=stream_output,
                               use_text_frontend=use_text_frontend,
                               language_tag=language_tag,
                               output_path=output_path,
                               device_hint=device_hint,
                               model_fp16=model_fp16,
                               model_jit=model_jit,
                               model_trt=model_trt,
                               use_flow_cache=use_flow_cache)

@mcp.tool()
def SparkTTSTool(
    # Input text
    text: str,
    
    # Voice cloning options
    prompt_text: Optional[str] = None,
    prompt_speech_path: Optional[str] = None,
    
    # Voice control parameters
    gender: Optional[Literal["male", "female"]] ="female",
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = "moderate",
    
    # Output configuration
    output_path: Optional[str] = "/home/chengz/LAMs/mcp_chatbot-audio/output/audio",
    
    # Model configuration
    device: int = 0,
    
    # Verbose output
    verbose: bool = False
) -> Dict[str, Any]:
    """Generate speech using the Spark-TTS zero-shot text-to-speech system.
    
    Args:
        text: The text to convert to speech
        prompt_text: Transcript of the reference audio for voice cloning
        prompt_speech_path: Path to the reference audio file for voice cloning
        gender: Gender of the synthesized voice ("male" or "female")
        pitch: Pitch level of the voice ("very_low", "low", "moderate", "high", "very_high")
        speed: Speaking rate ("very_low", "low", "moderate", "high", "very_high")
        output_path: Custom path to save the generated audio (WAV format)
        device: CUDA device ID for inference (0, 1, etc.)
        verbose: Whether to print detailed information during processing
        
    Returns:
        Dictionary containing the path to the generated audio file and processing info
    """
    return execute_tool_in_env("SparkTTSTool",
                              text=text,
                              prompt_text=prompt_text,
                              prompt_speech_path=prompt_speech_path,
                              gender=gender,
                              pitch=pitch,
                              speed=speed,
                              output_path=output_path,
                              device=device,
                              verbose=verbose)

@mcp.tool()
def VoiceCraftTool(
    # Input audio
    audio_path: str,
    
    # Editing parameters
    edit_type: Literal["substitution", "insertion", "deletion"],
    original_transcript: Optional[str] = None,
    target_transcript: str = "",
    
    # Optional parameters
    left_margin: float = 0.08,
    right_margin: float = 0.08,
    
    # Decoding parameters
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 0.8,
    stop_repetition: int = 2,
    kvcache: bool = True,
    silence_tokens: str = "[1388,1898,131]",
    
    # Output parameters
    output_path: Optional[str] = None,
    
    # System parameters
    device: int = 0,
    seed: int = 42
) -> Dict[str, Any]:
    """Edit speech audio by substituting, inserting, or deleting words in an English audio recording.
    
    VoiceCraft allows for zero-shot speech editing in English, enabling you to naturally
    modify the content of speech recordings while preserving the speaker's voice and style.
    
    Args:
        audio_path: Path to the input audio file to edit (WAV format recommended).
        
        edit_type: Type of edit to perform:
                  - "substitution": Replace words with new ones
                  - "insertion": Add new words between existing ones
                  - "deletion": Remove words from the speech
        
        original_transcript: Transcript of the original audio.
        
        target_transcript: Desired transcript after editing. Must match the original 
                          except for the parts being edited.
        
        left_margin: Additional time margin (in seconds) before the edited segment. Margin to the left of the editing segment,,Default: 0.08.
        
        right_margin:  Additional time margin (in seconds) before the edited segment. Margin to the right of the editing segment,Default: 0.08.
        
        temperature: Controls randomness in generation (higher = more random).  Do not recommend to change
        
        top_k: Number of highest probability vocabulary tokens to keep for sampling.
              -1 means no top-k filtering.  
        
        top_p: Nucleus sampling parameter (higher = more diversity).
        
        stop_repetition: Controls repetition. When the number of consecutive repetition 
                        of a token is bigger than this, stop it. -1 for speech editing. -1 means do not adjust prob of silence tokens. 
        
        kvcache: Whether to use key-value caching for faster inference.Set to 0 to use less VRAM, but with slower inference
        
        silence_tokens: List of token IDs that represent silence, in string format.
                       Default is "[1388,1898,131]".
        
        output_path: Custom path to save the edited audio file (WAV format).
        
        device: CUDA device ID (just need ID number eg. 0, 1, 2, etc.) to use for inference.
        
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing paths to the original and edited audio files, transcripts,
        and information about the editing process.
    """
    return execute_tool_in_env("VoiceCraftTool",
                               audio_path=audio_path,
                               edit_type=edit_type,
                               original_transcript=original_transcript,
                               target_transcript=target_transcript,
                               left_margin=left_margin,
                               right_margin=right_margin,
                               temperature=temperature,
                               top_k=top_k,
                               top_p=top_p,
                               stop_repetition=stop_repetition,
                               kvcache=kvcache,
                               silence_tokens=silence_tokens,
                               output_path=output_path,
                               device=device,
                               seed=seed)

@mcp.tool()
def Qwen2AudioTool(
    # Task selection (expanded with AIR-Bench capabilities)
    task: Literal[
        # Basic tasks
        "transcribe", "chat", "evaluate", 
        # AIR-Bench speech tasks
        "speech_grounding", "language_identification", "speaker_gender", 
        "emotion_recognition", "speaker_age", "speech_entity", 
        "intent_classification", "speaker_verification", "synthesized_voice_detection",
        # AIR-Bench audio tasks
        "audio_grounding", "vocal_classification", "acoustic_scene", 
        "sound_qa", "music_instruments", "music_genre", 
        "music_note_pitch", "music_note_velocity", "music_qa", "music_emotion"
    ] = "chat",
    
    # Input content
    audio_path: str = None,  # Required for all tasks
    text: Optional[str] = None,  # Optional text input for context
    reference_audio_path: Optional[str] = None,  # For comparison tasks
    
    # Task-specific parameters
    prompt: str = "",  # Task-specific instructions
    evaluation_criteria: str = "",  # For evaluation tasks
    evaluation_prompt_name: str = "",  # Predefined evaluation prompts
    target_language: Optional[str] = None,  # For translation tasks
    
    # Generation parameters
    temperature: float = 0.7,  # Controls randomness (0.0-1.0)
    top_p: float = 0.9,  # Nucleus sampling parameter (0.0-1.0)
    max_new_tokens: int = 2048,  # Maximum response length
    do_sample: bool = True,  # Whether to use sampling
    
    # Output options
    output_path: Optional[str] = None,  # Custom output path
    output_format: Literal["json", "txt"] = "json",  # Output format
    
    # Compute options
    device: str = "cuda"  # Computing device
) -> Dict[str, Any]:
    """Process audio with Qwen2-Audio model for comprehensive audio understanding tasks.
    
    This tool provides a wide range of audio processing capabilities based on the Qwen2-Audio
    large speech language model. It supports both basic audio tasks and advanced AIR-Bench
    capabilities for detailed audio analysis and understanding.
    
    Args:
        task: Task to perform 
            Basic tasks:
            - "transcribe": Convert speech to text with high accuracy
            - "chat": General audio processing with natural language interaction
            - "evaluate": Comprehensive audio quality assessment
            
            AIR-Bench speech tasks:
            - "speech_grounding": Locate and identify specific elements in speech
            - "language_identification": Detect spoken language with high accuracy
            - "speaker_gender": Identify speaker gender from voice characteristics
            - "emotion_recognition": Analyze emotional content in speech
            - "speaker_age": Estimate speaker age range
            - "speech_entity": Extract named entities and key information
            - "intent_classification": Determine speaker's intent and purpose
            - "speaker_verification": Compare voice samples for identity verification
            - "synthesized_voice_detection": Detect AI-generated or synthetic speech
            
            AIR-Bench audio tasks:
            - "audio_grounding": Identify and locate specific sounds in audio
            - "vocal_classification": Classify different types of vocal sounds
            - "acoustic_scene": Analyze and classify acoustic environments
            - "sound_qa": Answer questions about audio content
            - "music_instruments": Identify musical instruments in audio
            - "music_genre": Classify music genres and styles
            - "music_note_pitch": Analyze musical note frequencies
            - "music_note_velocity": Analyze musical note dynamics
            - "music_qa": Answer questions about musical content
            - "music_emotion": Detect emotional content in music
            
        audio_path: Path to input audio file (required for all tasks)
        text: Additional text input for context or specific instructions
        reference_audio_path: Path to reference audio for comparison tasks
        
        prompt: Task-specific instructions or guidance
        evaluation_criteria: Custom criteria for audio evaluation
        evaluation_prompt_name: Name of predefined evaluation prompt
        target_language: Target language for translation tasks
        
        temperature: Controls response randomness (0.0-1.0)
        top_p: Nucleus sampling parameter (0.0-1.0)
        max_new_tokens: Maximum length of generated response
        do_sample: Whether to use sampling in generation
        
        output_path: Custom path to save output
        output_format: Format for saving output (json or txt)
        device: Computing device ("cuda" or "cpu")
        
    Returns:
        Dictionary containing:
        - success: bool - Whether the task was successful
        - response: str - The model's response or analysis
        - processing_time: float - Time taken for processing
        - task: str - The performed task
        - output_path: str - Path to saved output (if any)
        - error: str - Error message (if any)
        
    Examples:
        # Basic transcription
        result = Qwen2AudioTool(
            task="transcribe",
            audio_path="speech.wav"
        )
        
        # Emotion analysis with custom prompt
        result = Qwen2AudioTool(
            task="emotion_recognition",
            audio_path="speech.wav",
            prompt="Analyze the emotional content in detail"
        )
        
        # Music genre classification
        result = Qwen2AudioTool(
            task="music_genre",
            audio_path="music.wav",
            temperature=0.3  # More focused response
        )
    """
    return execute_tool_in_env("Qwen2AudioTool",
                               task=task,
                               audio_path=audio_path,
                               text=text,
                               reference_audio_path=reference_audio_path,
                               prompt=prompt,
                               evaluation_criteria=evaluation_criteria,
                               evaluation_prompt_name=evaluation_prompt_name,
                               target_language=target_language,
                               temperature=temperature,
                               top_p=top_p,
                               max_new_tokens=max_new_tokens,
                               do_sample=do_sample,
                               output_path=output_path,
                               output_format=output_format,
                               device=device)

@mcp.tool()
def ClearVoiceTool(
    # Task selection
    task: Literal["speech_enhancement", "speech_separation", "speech_super_resolution", "target_speaker_extraction"],
    
    # Model selection - based on task
    model_name: Literal[
        # Speech Enhancement models
        "MossFormer2_SE_48K", "MossFormerGAN_SE_16K", "FRCRN_SE_16K",
        # Speech Separation models
        "MossFormer2_SS_16K",
        # Speech Super-Resolution models
        "MossFormer2_SR_48K",
        # Target Speaker Extraction models
        "AV_MossFormer2_TSE_16K"
    ],
    
    # Input/output paths
    input_path: str,
    output_path: Optional[str] = None,
    
    # Processing options
    online_write: bool = True,
    
    # Batch processing
    batch_process: bool = False,
    input_directory: Optional[str] = None,
    
    # Compute options
    device: str = "cuda"
) -> Dict[str, Any]:
    """Process audio using ClearerVoice models for speech enhancement, separation, and more.
    
    ClearerVoice provides unified models for various speech processing tasks:
    
    1. Speech Enhancement (SE): Remove noise and improve speech quality
       - MossFormer2_SE_48K: High-quality 48kHz model
       - MossFormerGAN_SE_16K: 16kHz model with GAN-based enhancement
       - FRCRN_SE_16K: Lightweight 16kHz enhancement model
       
    2. Speech Separation (SS): Separate multiple speakers in an audio
       - MossFormer2_SS_16K: 16kHz model for separating speakers
       
    3. Speech Super-Resolution (SR): Improve audio quality and resolution
       - MossFormer2_SR_48K: 48kHz model for speech super-resolution
       
    4. Audio-Visual Target Speaker Extraction (TSE): Extract specific speaker from audio/video
       - AV_MossFormer2_TSE_16K: 16kHz model for target speaker extraction
    
    Args:
        task: The audio processing task to perform
        
        model_name: Specific model to use for the selected task
        
        input_path: Path to the input audio file (for speech tasks) or video file (for TSE).
                    Required unless batch_process=True.
                    
        output_path: Directory to save the processed output.
                     If not provided, a default path will be used.
                     
        online_write: Whether to automatically save the processed audio.
        
        batch_process: Process multiple files in a directory.
                      When True, input_directory must be provided.
                      
        input_directory: Directory containing files to process in batch mode.
                         Required when batch_process=True.
                         
        device: Computing device to use ('cuda' or 'cpu').
        
    Returns:
        Dictionary containing processing results and metadata
        
    Examples:
        # Speech enhancement
        result = ClearVoiceTool(
            task="speech_enhancement",
            model_name="MossFormer2_SE_48K",
            input_path="noisy_speech.wav",
            output_path="enhanced_output"
        )
        
        # Batch processing of files
        result = ClearVoiceTool(
            task="speech_enhancement",
            model_name="MossFormer2_SE_48K",
            batch_process=True,
            input_directory="noisy_files",
            output_path="enhanced_files"
        )
    """
    if batch_process:
        if not input_directory:
            return {
                "success": False,
                "error": "input_directory must be provided when batch_process=True"
            }
        
        return execute_tool_in_env("ClearVoiceTool",
                                   method="process_directory",
                                   input_dir=input_directory,
                                   output_dir=output_path or "clearvoice_output",
                                   task=task,
                                   model_name=model_name)
    else:
        if not input_path:
            return {
                "success": False,
                "error": "input_path must be provided when batch_process=False"
            }
            
        return execute_tool_in_env("ClearVoiceTool",
                                   method="ClearVoice_tool",
                                   task=task,
                                   model_name=model_name,
                                   input_path=input_path,
                                   output_path=output_path,
                                   online_write=online_write,
                                   device=device)

@mcp.tool()
def WhisperASRTool(
    # Input content
    audio_path: str,
    
    # Task configuration
    task: Literal["transcribe", "translate"] = "transcribe",
    
    # Language options
    language: Optional[str] = "auto",  # e.g., "english", "chinese", "auto" for auto-detection
    
    # Generation parameters
    max_new_tokens: int = 440,
    num_beams: int = 1,
    temperature: Union[float, tuple] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: float = 1.35,
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_prev_tokens: bool = False,
    
    # Timestamp options
    return_timestamps: Union[bool, str] = False,  # True for sentence-level, "word" for word-level
    
    # Batch processing
    batch_size: int = 1,
    
    # Model configuration
    model_path: str = "/home/chengz/LAMs/pre_train_models/models--openai--whisper-large-v3",
    torch_dtype: str = "float16",  # "float16" or "float32"
    low_cpu_mem_usage: bool = True,
    use_safetensors: bool = True,
    
    # Output configuration
    output_path: Optional[str] = None,
    output_format: Literal["json", "txt"] = "json",
    
    # Compute options
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
) -> Dict[str, Any]:
    """
    Automatic Speech Recognition using OpenAI Whisper large-v3 model.
    
    This tool provides high-quality speech recognition with support for multiple languages,
    translation, and timestamp generation using the Whisper large-v3 model with sequential
    long-form algorithm for processing audio files longer than 30 seconds.
    
    Features:
    - High-quality automatic speech recognition (ASR)
    - Speech translation to English
    - Multi-language support (99 languages)
    - Timestamp generation (sentence or word level)
    - Batch processing for multiple files
    - Sequential long-form algorithm for long audio files
    - Local model support with customizable paths
    
    Args:
        audio_path: Path to the input audio file (supports various formats: wav, mp3, m4a, etc.)
        
        task: Task to perform
              - "transcribe": Convert speech to text in the same language as the audio
              - "translate": Translate speech to English text
              
        language: Source language of the audio (optional, auto-detected if not specified)
                 Examples: "english", "chinese", "spanish", "french", "japanese", etc.
                 Use None or "auto" for automatic language detection
                 
        max_new_tokens: Maximum number of tokens to generate (default: 128)
        num_beams: Number of beams for beam search (1 for greedy decoding, default: 1)
        temperature: Temperature for sampling, supports fallback strategy
                    Can be a single float or tuple like (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        compression_ratio_threshold: Threshold for zlib compression ratio in token space (default: 1.35)
        logprob_threshold: Log probability threshold for token acceptance (default: -1.0)
        no_speech_threshold: Threshold for no-speech detection (default: 0.6)
        condition_on_prev_tokens: Whether to condition generation on previous tokens (default: False)
        
        return_timestamps: Timestamp generation mode
                          - False: No timestamps
                          - True: Sentence-level timestamps  
                          - "word": Word-level timestamps
                          
        batch_size: Batch size for processing multiple files (default: 1)
        
        model_path: Path to the local Whisper model directory
                   Default: "/home/chengz/LAMs/pre_train_models/models--openai--whisper-large-v3"
        torch_dtype: PyTorch data type ("float16" for GPU, "float32" for CPU, default: "float16")
        low_cpu_mem_usage: Whether to use low CPU memory usage during model loading (default: True)
        use_safetensors: Whether to use safetensors format for faster loading (default: True)
        
        output_path: Custom path to save the transcription output
                    If not provided, saves to "whisper_output/whisper_result_{timestamp}.{format}"
        output_format: Output format ("json" for structured data, "txt" for plain text, default: "json")
        
        device: Computing device ("auto" for automatic selection, "cuda", "cpu", or specific device like "cuda:0")
        
    Returns:
        Dictionary containing:
        - success: bool - Whether transcription was successful
        - transcription: str - The transcribed text
        - chunks: list - Timestamp information (if return_timestamps is enabled)
        - audio_path: str - Path to the input audio file
        - task: str - The performed task ("transcribe" or "translate")
        - language: str - Detected or specified language
        - model_path: str - Path to the used model
        - device: str - Device used for processing
        - output_path: str - Path to the saved output file
        - parameters: dict - Processing parameters used
        - timestamp: str - Processing timestamp
        - error: str - Error message (if failed)
        
    """
    return execute_tool_in_env("WhisperASRTool",
                               audio_path=audio_path,
                               task=task,
                               language=language,
                               max_new_tokens=max_new_tokens,
                               num_beams=num_beams,
                               temperature=temperature,
                               compression_ratio_threshold=compression_ratio_threshold,
                               logprob_threshold=logprob_threshold,
                               no_speech_threshold=no_speech_threshold,
                               condition_on_prev_tokens=condition_on_prev_tokens,
                               return_timestamps=return_timestamps,
                               batch_size=batch_size,
                               model_path=model_path,
                               torch_dtype=torch_dtype,
                               low_cpu_mem_usage=low_cpu_mem_usage,
                               use_safetensors=use_safetensors,
                               output_path=output_path,
                               output_format=output_format,
                               device=device)

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run()
