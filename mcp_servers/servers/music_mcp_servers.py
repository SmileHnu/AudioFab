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
mcp = FastMCP("音樂和視頻生成服務：集成多種AI音樂/視頝模型")

# 工具环境配置
TOOL_ENV_CONFIG = {
    "AudioXTool": {
        "python_path": "/home/chengz/anaconda3/envs/AudioX/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "AudioX_processor.py")
    },
    "ACEStepTool": {
        "python_path": "/home/chengz/anaconda3/envs/ace_step/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "ACE_step_processor.py")
    },
    "MusicGenTool": {
        "python_path": "/home/chengz/anaconda3/envs/audiocraft/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Audiocraft_tool_processor.py")
    },
    "AudioGenTool": {
        "python_path": "/home/chengz/anaconda3/envs/audiocraft/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "Audiocraft_tool_processor.py")
    },
    "Hallo2Tool": {
        "python_path": "/home/chengz/anaconda3/envs/hallo/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor" / "hello2_processor.py")
    },
    "Hallo2VideoEnhancementTool": {
        "python_path": "/home/chengz/anaconda3/envs/hallo/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor"  / "hello2_processor.py")
    },
    # "YuEMusicGenerationTool": {
    #     "python_path": "/home/chengz/anaconda3/envs/yue/bin/python",
    #     "script_path": str(Path(__file__).parent.parent / "yue_processor.py")
    # },
    "YuEETool": {
        "python_path": "/home/chengz/anaconda3/envs/yue_e/bin/python",
        "script_path": str(Path(__file__).parent.parent / "processor"  / "yue_e_tool.py")
    },
    "DiffRhythmTool": {
        "python_path": "/home/qianshuaix/miniconda3/envs/diffrhythm/bin/python",
        "script_path": sstr(Path(__file__).parent.parent / "processor" / "DiffRhythm_processor.py")
    }
}

# 启动器脚本路径
LAUNCHER_SCRIPT = str(Path(__file__).parent.parent / "mcp_tool_launcher.py")

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
        
        # 解析结果
        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": f"一些可能无影响结果的信息: {result.stdout}"
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
def AudioXTool(
    # Input modalities
    text_prompt: Optional[str] = None,
    audio_path: Optional[str] = None,
    video_path: Optional[str] = None,
    
    # Output paths
    output_audio_path: Optional[str] = None,
    output_video_path: Optional[str] = None,
    
    # Generation parameters
    seconds_start: int = 0,
    seconds_total: int = 10,
    steps: int = 250, 
    cfg_scale: float = 7.0,
    
    # Sampler parameters
    sigma_min: float = 0.3,
    sigma_max: float = 500.0,
    sampler_type: str = "dpmpp-3m-sde",
    
    # Additional parameters
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    
    # Model configuration
    device_selection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate audio or video using the AudioX multimodal model.
    Content can be generated from text, audio, and/or video inputs.
    Outputs audio (.wav) and optionally video (.mp4) if a valid video_path was used for conditioning.

    Args:
        text_prompt: Text description for the content to generate.
        audio_path: Path to an audio file for audio prompt. If None or invalid, a default silent audio prompt will be used.
        video_path: Path to a video file for video prompt. If None or invalid, a default blank video prompt will be used.
        
        output_audio_path: Custom path to save generated audio (WAV). Default: output/audio/audiox_audio_[timestamp].wav
        output_video_path: Custom path to save generated video (MP4). Default: output/video/audiox_video_[timestamp].mp4. Video is only generated if input video_path was valid and used.
        
        seconds_start: Start time (seconds) for conditioning and generation window.
        seconds_total: Total duration (seconds) of generated content.
        steps: Number of inference steps.
        cfg_scale: Classifier-free guidance scale.
        
        sigma_min: Minimum sigma for sampler.
        sigma_max: Maximum sigma for sampler.
        sampler_type: Sampler type (e.g., "dpmpp-3m-sde").
        
        negative_prompt: Text describing undesired content.
        seed: Random seed. If None, a random seed is used.
        
        device_selection: Preferred device ("cuda" or "cpu"). Defaults to CUDA if available.
        
    Returns:
        Dictionary with output paths, success status, and parameters.
    """
    return execute_tool_in_env("AudioXTool", 
                              text_prompt=text_prompt,
                              audio_path=audio_path,
                              video_path=video_path,
                              output_audio_path=output_audio_path,
                              output_video_path=output_video_path,
                              seconds_start=seconds_start,
                              seconds_total=seconds_total,
                              steps=steps,
                              cfg_scale=cfg_scale,
                              sigma_min=sigma_min,
                              sigma_max=sigma_max,
                              sampler_type=sampler_type,
                              negative_prompt=negative_prompt,
                              seed=seed,
                              device_selection=device_selection)

@mcp.tool()
def ACEStepTool(
    # Generation type
    task: Literal["text2music", "retake", "repaint", "edit", "extend", "audio2audio"] = "text2music",
    
    # Input content
    prompt: Optional[str] = None,
    lyrics: Optional[str] = None,
    
    # Reference audio
    audio_prompt: Optional[str] = None,
    ref_audio_strength: float = 0.5,
    
    # Output configuration
    output_path: Optional[str] = None,
    format: str = "wav",
    audio_duration: float = 30.0,
    
    # Generation parameters
    infer_step: int = 60,
    guidance_scale: float = 15.0,
    scheduler_type: Literal["euler", "heun", "pingpong"] = "euler",
    cfg_type: Literal["apg", "cfg", "cfg_star"] = "apg",
    omega_scale: float = 10.0,
    seed: Optional[int] = None,
    
    # Advanced generation parameters
    guidance_interval: float = 0.5,
    guidance_interval_decay: float = 0.0,
    min_guidance_scale: float = 3.0,
    use_erg_tag: bool = True,
    use_erg_lyric: bool = True,
    use_erg_diffusion: bool = True,
    oss_steps: Optional[str] = None,
    
    # Double condition guidance
    guidance_scale_text: float = 0.0,
    guidance_scale_lyric: float = 0.0,
    
    # Retake parameters
    retake_variance: float = 0.5,
    
    # Repaint/extend parameters
    repaint_start: int = 0,
    repaint_end: int = 0,
    
    # Source audio for edit/repaint/extend
    src_audio_path: Optional[str] = None,
    
    # Edit parameters
    edit_target_prompt: Optional[str] = None,
    edit_target_lyrics: Optional[str] = None,
    edit_n_min: float = 0.0,
    edit_n_max: float = 1.0,
    edit_n_avg: int = 1,
    
    # LoRA adaptation
    lora_name_or_path: str = "none",
    
    # Model configuration
    device_id: int = 0,
    bf16: bool = True,
    torch_compile: bool = False,
    cpu_offload: bool = False,
    overlapped_decode: bool = False,
    debug: bool = False
) -> Dict[str, Any]:
    """Generate music using the ACE-Step foundation model.
    
    ACE-Step supports various music generation tasks: text-to-music generation, variations (retake),
    partial regeneration (repaint), editing, extending existing music, and audio-to-audio generation.
    
    Args:
        task: The generation task type (text2music, retake, repaint, edit, extend, audio2audio)
        
        prompt: Text description for the music to generate
        lyrics: Lyrics for the generated music with structure tags like [verse], [chorus]
        
        audio_prompt: Path to a reference audio file for audio2audio generation
        ref_audio_strength: Strength of reference audio influence (0-1, higher = more influence)
        
        output_path: Custom path to save the generated audio
        format: Output audio format (wav)
        audio_duration: Duration of generated audio in seconds
        
        infer_step: Number of inference steps (higher = better quality but slower)
        guidance_scale: Classifier-free guidance scale (higher = more adherence to prompt)
        scheduler_type: Type of diffusion scheduler (euler, heun, pingpong)
        cfg_type: Type of classifier-free guidance method
        omega_scale: Scale for omega parameter
        seed: Random seed for reproducible generation
        
        guidance_interval: Interval for applying guidance during generation
        guidance_interval_decay: Decay rate for guidance scale during generation
        min_guidance_scale: Minimum value for guidance scale after decay
        use_erg_tag: Whether to use enhanced tag generation
        use_erg_lyric: Whether to use enhanced lyric generation
        use_erg_diffusion: Whether to use enhanced diffusion process
        oss_steps: Comma-separated list of one-step sampling steps
        
        guidance_scale_text: Guidance scale for text in double condition mode
        guidance_scale_lyric: Guidance scale for lyrics in double condition mode
        
        retake_variance: Variance for retake generation (0-1, higher = more different)
        
        repaint_start: Start time (seconds) for repainting
        repaint_end: End time (seconds) for repainting
        
        src_audio_path: Path to source audio for edit/repaint/extend tasks
        
        edit_target_prompt: Target prompt for edit task
        edit_target_lyrics: Target lyrics for edit task
        edit_n_min: Minimum normalized time step for edit diffusion
        edit_n_max: Maximum normalized time step for edit diffusion
        edit_n_avg: Number of prediction averages for edit diffusion
        
        lora_name_or_path: Path or name of a LoRA adaptation to use
        
        device_id: GPU device ID to use
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch.compile for optimization
        cpu_offload: Whether to offload model to CPU when not in use
        overlapped_decode: Whether to use overlapped decoding for long audio
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing the path to the generated audio file and generation parameters
    """
    return execute_tool_in_env("ACEStepTool", 
                              task=task,
                              prompt=prompt,
                              lyrics=lyrics,
                              audio_prompt=audio_prompt,
                              ref_audio_strength=ref_audio_strength,
                              output_path=output_path,
                              format=format,
                              audio_duration=audio_duration,
                              infer_step=infer_step,
                              guidance_scale=guidance_scale,
                              scheduler_type=scheduler_type,
                              cfg_type=cfg_type,
                              omega_scale=omega_scale,
                              seed=seed,
                              guidance_interval=guidance_interval,
                              guidance_interval_decay=guidance_interval_decay,
                              min_guidance_scale=min_guidance_scale,
                              use_erg_tag=use_erg_tag,
                              use_erg_lyric=use_erg_lyric,
                              use_erg_diffusion=use_erg_diffusion,
                              oss_steps=oss_steps,
                              guidance_scale_text=guidance_scale_text,
                              guidance_scale_lyric=guidance_scale_lyric,
                              retake_variance=retake_variance,
                              repaint_start=repaint_start,
                              repaint_end=repaint_end,
                              src_audio_path=src_audio_path,
                              edit_target_prompt=edit_target_prompt,
                              edit_target_lyrics=edit_target_lyrics,
                              edit_n_min=edit_n_min,
                              edit_n_max=edit_n_max,
                              edit_n_avg=edit_n_avg,
                              lora_name_or_path=lora_name_or_path,
                              device_id=device_id,
                              bf16=bf16,
                              torch_compile=torch_compile,
                              cpu_offload=cpu_offload,
                              overlapped_decode=overlapped_decode,
                              debug=debug)

@mcp.tool()
def MusicGenTool(
    # Input content
    prompt: Union[str, List[str]],
    
    # Melody conditioning
    melody_path: Optional[str] = None,
    melody_sample_rate: int = 44100,
    
    # Output configuration
    output_path: Optional[str] = None,
    format: str = "wav",
    
    # Generation parameters
    duration: float = 10.0,
    guidance_scale: float = 3.0,
    seed: Optional[int] = None,
    
    # Audio normalization
    apply_loudness_normalization: bool = True,
    
    # Model configuration
    model_path: str = "/home/chengz/LAMs/pre_train_models/models--facebook--musicgen-melody",
    device: str = "cuda",
    dtype: str = "float16",
    
    # Batch processing
    batch_size: int = 1
) -> Dict[str, Any]:
    """Generate music using Facebook's MusicGen model based on text prompts and optional melody.
    
    Args:
        prompt: Text description for the music to generate. Can be a single string or a list of strings for batch generation.
        
        melody_path: Optional path to a melody audio file to use as reference for the generation
        melody_sample_rate: Sample rate of the melody file if provided
        
        output_path: Custom path to save the generated audio. If not provided, a default path will be used.
        format: Output audio format (wav)
        
        duration: Duration of generated audio in seconds (maximum 30 seconds)
        guidance_scale: Classifier-free guidance scale (higher = more adherence to prompt)
        seed: Random seed for reproducible generation
        
        apply_loudness_normalization: Whether to apply loudness normalization to the generated audio
        
        model_path: Path to the MusicGen model
        device: Computing device to use ('cuda' or 'cpu')
        dtype: Precision to use for model inference ('float16' or 'float32')
        
        batch_size: Number of audio samples to generate in parallel
        
    Returns:
        Dictionary containing the path(s) to the generated audio file(s) and generation parameters
    """
    return execute_tool_in_env("MusicGenTool",
                              prompt=prompt,
                              melody_path=melody_path,
                              melody_sample_rate=melody_sample_rate,
                              output_path=output_path,
                              format=format,
                              duration=duration,
                              guidance_scale=guidance_scale,
                              seed=seed,
                              apply_loudness_normalization=apply_loudness_normalization,
                              model_path=model_path,
                              device=device,
                              dtype=dtype,
                              batch_size=batch_size)

@mcp.tool()
def AudioGenTool(
    # Input content
    prompt: Union[str, List[str]],
    
    # Output configuration
    output_path: Optional[str] = None,
    format: str = "wav",
    
    # Generation parameters
    duration: float = 5.0,
    guidance_scale: float = 3.0,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    seed: Optional[int] = None,
    
    # Audio normalization
    apply_loudness_normalization: bool = True,
    
    # Model configuration
    model_path: str = "/home/chengz/LAMs/pre_train_models/models--facebook--audiogen-medium",
    device: str = "cuda",
    dtype: str = "float16",
    
    # Extended generation parameters
    extend_stride: float = 2.0,
    
    # Batch processing
    batch_size: int = 1
) -> Dict[str, Any]:
    """Generate audio (environmental sounds, effects) using Facebook's AudioGen model based on text prompts.
    
    Args:
        prompt: Text description for the audio to generate. Can be a single string or a list of strings for batch generation.
        
        output_path: Custom path to save the generated audio. If not provided, a default path will be used.
        format: Output audio format (wav)
        
        duration: Duration of generated audio in seconds (maximum 30 seconds)
        guidance_scale: Classifier-free guidance scale (higher = more adherence to prompt)
        temperature: Temperature for sampling (higher = more random)
        top_k: Top-k sampling parameter
        top_p: Top-p sampling parameter (0.0 means disabled)
        seed: Random seed for reproducible generation
        
        apply_loudness_normalization: Whether to apply loudness normalization to the generated audio
        
        model_path: Path to the AudioGen model
        device: Computing device to use ('cuda' or 'cpu')
        dtype: Precision to use for model inference ('float16' or 'float32')
        
        extend_stride: Stride for extended generation (for audio > 10s)
        
        batch_size: Number of audio samples to generate in parallel
        
    Returns:
        Dictionary containing the path(s) to the generated audio file(s) and generation parameters
    """
    return execute_tool_in_env("AudioGenTool",
                              prompt=prompt,
                              output_path=output_path,
                              format=format,
                              duration=duration,
                              guidance_scale=guidance_scale,
                              temperature=temperature,
                              top_k=top_k,
                              top_p=top_p,
                              seed=seed,
                              apply_loudness_normalization=apply_loudness_normalization,
                              model_path=model_path,
                              device=device,
                              dtype=dtype,
                              extend_stride=extend_stride,
                              batch_size=batch_size)

@mcp.tool()
def Hallo2Tool(
    # Input paths
    source_image: str,
    driving_audio: str,
    
    # Animation control parameters
    pose_weight: Optional[float] = 0.3, 
    face_weight: Optional[float] = 0.7,
    lip_weight: Optional[float] = 1.0,
    face_expand_ratio: Optional[float] = 1.5,
    
    # Output configuration
    output_path: Optional[str] = None,
    
    # Model configuration
    config_path: Optional[str] ="/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/configs/inference/long.yaml",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Generate audio-driven portrait animation using the Hallo2 model.
    
    Args:
        source_image: Path to the source portrait image. The portrait should face forward.
        driving_audio: Path to the driving audio file. Supported formats: WAV.
        
        pose_weight: Weight for head pose motion. Higher values increase head movement.
        face_weight: Weight for facial expression. Higher values enhance facial expressions.
        lip_weight: Weight for lip sync accuracy. Higher values improve lip synchronization.
        face_expand_ratio: Expand ratio for the detected face region. Higher values include more context.
        
        output_path: Custom path to save the generated video file (MP4 format).
        
        config_path: Path to custom config file. If not provided, default config will be used.
        device: Computing device to use for inference ('cuda' or 'cpu').
    
    Returns:
        Dictionary containing the path to the generated video file and processing info.
    """
    return execute_tool_in_env("Hallo2Tool",
                              source_image=source_image,
                              driving_audio=driving_audio,
                              pose_weight=pose_weight,
                              face_weight=face_weight,
                              lip_weight=lip_weight,
                              face_expand_ratio=face_expand_ratio,
                              output_path=output_path,
                              config_path=config_path,
                              device=device)

@mcp.tool()
def Hallo2VideoEnhancementTool(
    # Input paths
    input_video: str,
    
    # Enhancement parameters
    fidelity_weight: Optional[float] = 0.5,
    upscale: Optional[int] = 2,
    
    # Output configuration
    output_path: Optional[str] = None,
    
    # Enhancement options
    bg_upsampler: Optional[str] = "realesrgan",
    face_upsample: Optional[bool] = True,
    detection_model: Optional[str] = "retinaface_resnet50",
    bg_tile: Optional[int] = 400,
    only_center_face: Optional[bool] = False
) -> Dict[str, Any]:
    """Enhance video quality with high-resolution processing using Hallo2's video enhancement module.
    
    Args:
        input_video: Path to the input video file to enhance.
        
        fidelity_weight: Balance between quality and fidelity (0-1). Lower values prioritize quality, higher values preserve fidelity.
        upscale: Upscaling factor for the image (2, 3, or 4).
        
        output_path: Custom path to save the enhanced video. If not provided, a default path will be used.
        
        bg_upsampler: Background upsampler to use. Set to "None" to disable background upsampling.
        face_upsample: Whether to apply additional upsampling to the face regions.
        detection_model: Face detection model to use.
        bg_tile: Tile size for background upsampling. Smaller values use less memory but may reduce quality.
        only_center_face: Whether to only enhance the center face in the video.
    
    Returns:
        Dictionary containing the path to the enhanced video file and processing info.
    """
    return execute_tool_in_env("Hallo2VideoEnhancementTool",
                              input_video=input_video,
                              fidelity_weight=fidelity_weight,
                              upscale=upscale,
                              output_path=output_path,
                              bg_upsampler=bg_upsampler,
                              face_upsample=face_upsample,
                              detection_model=detection_model,
                              bg_tile=bg_tile,
                              only_center_face=only_center_face)

@mcp.tool()
def YuEMusicGenerationTool(
    # Required inputs
    genre: str,
    lyrics: str,
    
    # Optional configuration
    language: Literal["english", "chinese"] = "english",
    reasoning_method: Literal["cot", "icl"] = "cot",
    max_new_tokens: int = 3000,
    repetition_penalty: float = 1.1,
    run_n_segments: int = 1,
    output_file: Optional[str] = None,
    
    # Audio prompt options
    use_audio_prompt: bool = False,
    audio_prompt_path: Optional[str] = None,
    prompt_start_time: float = 0.0,
    prompt_end_time: float = 30.0,
    
    # Dual tracks prompt options
    use_dual_tracks_prompt: bool = False,
    vocal_track_prompt_path: Optional[str] = None,
    instrumental_track_prompt_path: Optional[str] = None,
    
    # Additional settings
    keep_intermediate: bool = False,
    disable_offload_model: bool = False,
    cuda_idx: int = 0,
    seed: int = 42,
    rescale: bool = True
) -> Dict[str, Any]:
    """Generate music with lyrics using the YuE AI music generation model.
    
    Args:
        genre: Musical genre and style description (e.g., "pop, electronic, energetic")
        lyrics: Lyrics for the song to be generated
        language: Language for the lyrics (english or chinese)
        reasoning_method: Reasoning method for generation (cot = chain-of-thought, icl = in-context learning)
        max_new_tokens: Maximum new tokens to generate
        repetition_penalty: Penalty for repetition (1.0-2.0, higher values reduce repetition)
        run_n_segments: Number of segments to process
        output_file: Path to save the generated audio file (optional)
        use_audio_prompt: Whether to use an audio file as a prompt
        audio_prompt_path: Path to the audio prompt file
        prompt_start_time: Start time in seconds for audio prompt extraction
        prompt_end_time: End time in seconds for audio prompt extraction
        use_dual_tracks_prompt: Whether to use dual tracks as prompt
        vocal_track_prompt_path: Path to vocal track prompt file
        instrumental_track_prompt_path: Path to instrumental track prompt file
        keep_intermediate: Whether to keep intermediate files
        disable_offload_model: Whether to disable model offloading
        cuda_idx: CUDA device index
        seed: Random seed for reproducibility
        rescale: Whether to rescale output to avoid clipping
        
    Returns:
        Dictionary containing success status, output file path, and parameters used
    """
    return execute_tool_in_env("YuEMusicGenerationTool",
                              genre=genre,
                              lyrics=lyrics,
                              language=language,
                              reasoning_method=reasoning_method,
                              max_new_tokens=max_new_tokens,
                              repetition_penalty=repetition_penalty,
                              run_n_segments=run_n_segments,
                              output_file=output_file,
                              use_audio_prompt=use_audio_prompt,
                              audio_prompt_path=audio_prompt_path,
                              prompt_start_time=prompt_start_time,
                              prompt_end_time=prompt_end_time,
                              use_dual_tracks_prompt=use_dual_tracks_prompt,
                              vocal_track_prompt_path=vocal_track_prompt_path,
                              instrumental_track_prompt_path=instrumental_track_prompt_path,
                              keep_intermediate=keep_intermediate,
                              disable_offload_model=disable_offload_model,
                              cuda_idx=cuda_idx,
                              seed=seed,
                              rescale=rescale)

@mcp.tool()
def YuEETool(
    # Required inputs
    genre: str,
    lyrics: str,
    
    # Optional configuration
    language: Literal["english", "chinese"] = "english",
    reasoning_method: Literal["cot", "icl"] = "cot",
    max_new_tokens: int = 3000,
    repetition_penalty: float = 1.1,
    run_n_segments: int = 2,
    output_file: Optional[str] = None,
    
    # Stage model configurations
    stage1_use_exl2: bool = True,
    stage2_use_exl2: bool = True,
    stage2_batch_size: int = 4,
    stage1_cache_size: int = 16384,
    stage2_cache_size: int = 32768,
    stage1_cache_mode: str = "FP16",
    stage2_cache_mode: str = "FP16",
    stage1_no_guidance: bool = False,
    
    # Audio prompt options
    use_audio_prompt: bool = False,
    audio_prompt_path: Optional[str] = None,
    prompt_start_time: float = 0.0,
    prompt_end_time: float = 30.0,
    
    # Dual tracks prompt options
    use_dual_tracks_prompt: bool = False,
    vocal_track_prompt_path: Optional[str] = None,
    instrumental_track_prompt_path: Optional[str] = None,
    
    # Additional settings
    keep_intermediate: bool = False,
    disable_offload_model: bool = False,
    cuda_idx: int = 0,
    seed: Optional[int] = None,
    rescale: bool = True
) -> Dict[str, Any]:
    """Generate music with lyrics using the YuE-E extended AI music generation model.
    
    YuE-E is an enhanced version of YuE with support for various language models
    and reasoning methods for high-quality music generation with lyrics. The model 
    automatically selects the appropriate pre-trained model based on language
    and reasoning method settings.
    
    Args:
        genre: Musical genre and style description (e.g., "pop, electronic, energetic")
        lyrics: Lyrics for the song to be generated
        language: Language for the lyrics (english or chinese)
        reasoning_method: Reasoning method for generation (cot = chain-of-thought, icl = in-context learning)
        max_new_tokens: Maximum new tokens to generate
        repetition_penalty: Penalty for repetition (1.0-2.0, higher values reduce repetition)
        run_n_segments: Number of segments to process during generation
        output_file: Path to save the generated audio file (optional)
        
        stage1_use_exl2: Use exllamav2 to load and run stage 1 model
        stage2_use_exl2: Use exllamav2 to load and run stage 2 model
        stage2_batch_size: Non-exl2 batch size used in Stage 2 inference
        stage1_cache_size: Cache size used in Stage 1 inference
        stage2_cache_size: Exl2 cache size used in Stage 2 inference
        stage1_cache_mode: Cache mode for Stage 1 (FP16, Q8, Q6, Q4)
        stage2_cache_mode: Cache mode for Stage 2 (FP16, Q8, Q6, Q4)
        stage1_no_guidance: Disable classifier-free guidance for stage 1
        
        use_audio_prompt: Whether to use an audio file as a prompt
        audio_prompt_path: Path to the audio prompt file
        prompt_start_time: Start time in seconds for audio prompt extraction
        prompt_end_time: End time in seconds for audio prompt extraction
        
        use_dual_tracks_prompt: Whether to use dual tracks as prompt
        vocal_track_prompt_path: Path to vocal track prompt file
        instrumental_track_prompt_path: Path to instrumental track prompt file
        
        keep_intermediate: Whether to keep intermediate files
        disable_offload_model: Whether to disable model offloading
        cuda_idx: CUDA device index
        seed: Random seed for reproducibility (None for random)
        rescale: Whether to rescale output to avoid clipping
        
    Returns:
        Dictionary containing success status, output file path, and parameters used
    """
    return execute_tool_in_env("YuEETool", 
                              genre=genre,
                              lyrics=lyrics,
                              language=language,
                              reasoning_method=reasoning_method,
                              max_new_tokens=max_new_tokens,
                              repetition_penalty=repetition_penalty,
                              run_n_segments=run_n_segments,
                              output_file=output_file,
                              stage1_use_exl2=stage1_use_exl2,
                              stage2_use_exl2=stage2_use_exl2,
                              stage2_batch_size=stage2_batch_size,
                              stage1_cache_size=stage1_cache_size,
                              stage2_cache_size=stage2_cache_size,
                              stage1_cache_mode=stage1_cache_mode,
                              stage2_cache_mode=stage2_cache_mode,
                              stage1_no_guidance=stage1_no_guidance,
                              use_audio_prompt=use_audio_prompt,
                              audio_prompt_path=audio_prompt_path,
                              prompt_start_time=prompt_start_time,
                              prompt_end_time=prompt_end_time,
                              use_dual_tracks_prompt=use_dual_tracks_prompt,
                              vocal_track_prompt_path=vocal_track_prompt_path,
                              instrumental_track_prompt_path=instrumental_track_prompt_path,
                              keep_intermediate=keep_intermediate,
                              disable_offload_model=disable_offload_model,
                              cuda_idx=cuda_idx,
                              seed=seed,
                              rescale=rescale)

@mcp.tool()
def DiffRhythmTool(
    # Model selection
    model_version: Literal["v1.2", "full"] = "v1.2",
    
    # Input content - lyrics
    lrc_path: Optional[str] = None,
    lrc_text: Optional[str] = None,
    
    # Style input (one of these must be provided)
    ref_prompt: Optional[str] = None,
    ref_audio_path: Optional[str] = None,
    
    # Generation parameters
    chunked: bool = False,
    batch_infer_num: int = 1,
    
    # Edit mode parameters
    edit: bool = False,
    ref_song: Optional[str] = None,
    edit_segments: Optional[str] = None,  # Format: [[start1,end1],...] with -1 for audio start/end
    
    # Output configuration
    output_path: Optional[str] = None,
    
    # Verbose output
    verbose: bool = False
) -> Dict[str, Any]:
    """Generate music using DiffRhythm model with lyrics, style prompts, or references.
    
    DiffRhythm is the first open-sourced diffusion-based music generation model capable of creating 
    full-length songs with both vocals and accompaniment. It supports two main models:
    
    - DiffRhythm-v1.2: Latest model with best quality, generates up to 1m35s songs
    - DiffRhythm-full: Full-length model, generates up to 4m45s songs
    
    Args:
        model_version: Model version to use ("v1.2" for latest quality or "full" for long generation)
        
        lrc_path: Path to lyrics file (.lrc format with timestamps)
        lrc_text: Direct lyrics text input in .lrc format (e.g., "[00:00.00]Hello world\n[00:05.00]This is a test")
        
        ref_prompt: Text prompt describing the musical style (e.g., "pop energetic electronic")
        ref_audio_path: Path to reference audio file for style conditioning
        
        chunked: Use chunked decoding (recommended for 8GB VRAM, reduces memory usage)
        batch_infer_num: Number of songs to generate in parallel
        
        edit: Enable edit mode for modifying existing songs
        ref_song: Path to reference song for editing (required when edit=True)
        edit_segments: Time segments to edit in format "[[start1,end1],...]" with -1 for audio start/end
        
        output_path: Custom output path. If not provided, saves to /home/chengz/LAMs/mcp_chatbot-audio/output/music/
        
        verbose: Print detailed generation information
        
    Returns:
        Dictionary containing generation results, output path, and metadata
        
    Examples:
        # Generate with text prompt
        result = DiffRhythmTool(
            ref_prompt="happy energetic pop song",
            lrc_text="[00:00.00]Hello world\n[00:05.00]This is a test song"
        )
        
        # Generate full-length song with audio reference
        result = DiffRhythmTool(
            model_version="full",
            ref_audio_path="/path/to/reference.wav",
            lrc_path="/path/to/lyrics.lrc"
        )
        
        # Edit existing song
        result = DiffRhythmTool(
            edit=True,
            ref_song="/path/to/song.wav",
            edit_segments="[[0,30]]",
            ref_prompt="jazz style"
        )
    """
    return execute_tool_in_env("DiffRhythmTool",
                              model_version=model_version,
                              lrc_path=lrc_path,
                              lrc_text=lrc_text,
                              ref_prompt=ref_prompt,
                              ref_audio_path=ref_audio_path,
                              chunked=chunked,
                              batch_infer_num=batch_infer_num,
                              edit=edit,
                              ref_song=ref_song,
                              edit_segments=edit_segments,
                              output_path=output_path,
                              verbose=verbose)

if __name__ == "__main__":
    # 启动MCP服务器
    mcp.run()
