import os
import sys
from pathlib import Path
import torch
import torchaudio
import json
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import numpy as np

# Add ACE-Step to path
ACE_STEP_PATH = Path("models/ACE_Step") # 确保这个路径正确
sys.path.append(str(ACE_STEP_PATH.resolve())) # 使用resolve()获取绝对路径

# 修改导入方式，直接尝试从本地目录导入
try:
    # 首先尝试直接从本地目录导入
    sys.path.append(str(Path("models").resolve()))
    from ACE_Step.infer import ACEStepPipeline
except ImportError:
    try:
        # 如果失败，尝试使用官方API导入格式
        from acestep.pipeline_ace_step import ACEStepPipeline
    except ImportError:
        print("无法导入ACEStepPipeline，请确认ACE-Step已正确安装或路径配置正确")
        print(f"当前Python路径: {sys.path}")
        raise

from mcp.server.fastmcp import FastMCP

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
checkpoint_path='/home/chengz/LAMs/pre_train_models/models--ACE-Step--ACE-Step-v1-3.5B'
chinese_rap = "/home/chengz/LAMs/pre_train_models/models--ACE-Step--ACE-Step-v1-chinese-rap-LoRA"
# Initialize MCP server
mcp = FastMCP("ACE-Step: A Music Generation Foundation Model")

# Global variable for model instance
ace_step_model = None

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def initialize_model(
    device_id=0, 
    dtype="bfloat16", 
    torch_compile=False, 
    cpu_offload=False,
    overlapped_decode=False
):
    """Initialize the ACE-Step model if it hasn't been loaded yet."""
    global ace_step_model
    
    if ace_step_model is None:
        print(f"Initializing ACE-Step model (device ID: {device_id}, precision: {dtype})...")
        
        # Set visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        
        # Initialize model
        ace_step_model = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            device_id=0,  # Since we've already set CUDA_VISIBLE_DEVICES
            dtype=dtype,
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode
        )
        
        # Load checkpoint
        ace_step_model.load_checkpoint(checkpoint_dir=checkpoint_path)
        print("ACE-Step model initialized successfully")
    
    return ace_step_model

# @mcp.tool()
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
    lora_name_or_path: str = '',
    
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
        
        checkpoint_path: Path to model checkpoint directory
        device_id: GPU device ID to use
        bf16: Whether to use bfloat16 precision
        torch_compile: Whether to use torch.compile for optimization
        cpu_offload: Whether to offload model to CPU when not in use
        overlapped_decode: Whether to use overlapped decoding for long audio
        debug: Whether to print debug information
        
    Returns:
        Dictionary containing the path to the generated audio file and generation parameters
    """
    try:
        # Validate inputs based on task
        if task == "text2music" and not prompt:
            return {
                "success": False,
                "error": "Text prompt is required for text2music task"
            }
        
        if task == "audio2audio" and not audio_prompt:
            return {
                "success": False,
                "error": "Audio prompt is required for audio2audio task"
            }
        
        if task in ["edit", "repaint", "extend"] and not src_audio_path:
            return {
                "success": False,
                "error": f"Source audio path is required for {task} task"
            }
        
        if task == "edit" and not (edit_target_prompt or edit_target_lyrics):
            return {
                "success": False,
                "error": "Target prompt or lyrics are required for edit task"
            }
        
        # Handle audio_prompt parameter for audio2audio task
        audio2audio_enable = False
        ref_audio_input = None
        
        if audio_prompt and os.path.exists(audio_prompt):
            audio2audio_enable = True
            ref_audio_input = audio_prompt
        
        # Handle seed parameter
        actual_seed = seed if seed is not None else np.random.randint(0, 2147483647)
        manual_seeds = [actual_seed] if seed is not None else None
        retake_seeds = [actual_seed + 1] if seed is not None else None  # Different seed for retake
        
        # Initialize the model
        model = initialize_model(
            device_id=device_id,
            dtype="bfloat16" if bf16 else "float32",
            torch_compile=torch_compile,
            cpu_offload=cpu_offload,
            overlapped_decode=overlapped_decode
        )
        
        # Determine output path
        if output_path is None:
            timestamp = get_timestamp()
            output_path = str(AUDIO_DIR / f"acestep_{task}_{timestamp}.{format}")
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Generate the music
        result = model(
            format=format,
            audio_duration=audio_duration,
            prompt=prompt or "",
            lyrics=lyrics or "",
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=oss_steps,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_audio_input=ref_audio_input,
            lora_name_or_path=lora_name_or_path,
            retake_seeds=retake_seeds,
            retake_variance=retake_variance,
            task=task,
            repaint_start=repaint_start,
            repaint_end=repaint_end,
            src_audio_path=src_audio_path,
            edit_target_prompt=edit_target_prompt,
            edit_target_lyrics=edit_target_lyrics,
            edit_n_min=edit_n_min,
            edit_n_max=edit_n_max,
            edit_n_avg=edit_n_avg,
            save_path=output_path,
            batch_size=1,
            debug=debug
        )
        
        # Extract actual paths and parameters from result
        audio_path = result[0] if len(result) > 0 else None
        params_json = result[-1] if len(result) > 1 else {}
        
        # Remove any duplicate parameter information
        if isinstance(params_json, dict) and "input_params_json" in params_json:
            del params_json["input_params_json"]
            
        # Get audio duration
        try:
            if audio_path and os.path.exists(audio_path):
                audio_info = torchaudio.info(audio_path)
                duration = audio_info.num_frames / audio_info.sample_rate
            else:
                duration = None
        except Exception as e:
            duration = None
            print(f"Could not get audio duration: {str(e)}")
        
        return {
            "success": True,
            "audio_path": audio_path,
            "generation_parameters": {
                "task": task,
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_duration": audio_duration if duration is None else duration,
                "lora_name_or_path": lora_name_or_path,
                "seed": actual_seed,
                "scheduler_type": scheduler_type,
                "guidance_scale": guidance_scale,
                "infer_step": infer_step,
                "cfg_type": cfg_type
            },
            "all_parameters": params_json
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Music generation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

# if __name__ == "__main__":
#     # Run the MCP server
#     mcp.run()
