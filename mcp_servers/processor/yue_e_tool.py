import os
import sys
import json
import subprocess
from pathlib import Path
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Literal

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Create temp directory
TEMP_DIR = Path(tempfile.gettempdir()) / "mcp_temp"
TEMP_DIR.mkdir(exist_ok=True)

# Path constants for pre-trained models
YUE_S1_EN_COT = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-cot"
YUE_S1_EN_ICL = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-icl"
YUE_S1_ZH_COT = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-cot"
YUE_S1_ZH_ICL = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-icl"
YUE_S2_GENERAL = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s2-1B-general"
YUE_UPSAMPLER = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-upsampler"

# YuE  推理脚本目录，优先使用环境变量 YUE_INFERENCE_DIR，否则默认相对项目根目录 <project_root>/models/YuE-exllamav2/src/yue
# <project_root> 通过当前文件的祖先目录推断得到（向上两级）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
YUE_INFERENCE_DIR = Path(os.getenv("YUE_INFERENCE_DIR", PROJECT_ROOT / "models" / "YuE-exllamav2" / "src" / "yue"))
YUE_INFER_SCRIPT = str(YUE_INFERENCE_DIR / "infer.py")

# Python environment path
PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/yue_e/bin/python"

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_temp_text_file(content, prefix="yue_"):
    """Create a temporary text file with the given content."""
    fd, path = tempfile.mkstemp(suffix=".txt", prefix=prefix, dir=TEMP_DIR)
    with os.fdopen(fd, 'w') as f:
        f.write(content)
    return path

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
    
    Args:
        genre: Musical genre and style description (e.g., "pop, electronic, energetic")
        lyrics: Lyrics for the song to be generated
        language: Language for the lyrics (english or chinese)
        reasoning_method: Reasoning method for generation (cot = chain-of-thought, icl = in-context learning)
        max_new_tokens: Maximum new tokens to generate
        repetition_penalty: Penalty for repetition (1.0-2.0, higher values reduce repetition)
        run_n_segments: Number of segments to process during generation
        output_file: Path to save the generated audio file (optional) 注意：是个文件夹
        
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
    try:
        # Select the appropriate model based on language and reasoning method
        if language == "english":
            if reasoning_method == "cot":
                stage1_model = YUE_S1_EN_COT
            else:  # icl
                stage1_model = YUE_S1_EN_ICL
        else:  # chinese
            if reasoning_method == "cot":
                stage1_model = YUE_S1_ZH_COT
            else:  # icl
                stage1_model = YUE_S1_ZH_ICL
        
        stage2_model = YUE_S2_GENERAL
        
        # Create temporary files for genre and lyrics
        genre_file = create_temp_text_file(genre, prefix="yue_genre_")
        lyrics_file = create_temp_text_file(lyrics, prefix="yue_lyrics_")
        
        # Set output path
        if output_file is None:
            timestamp = get_timestamp()
            output_file = str(AUDIO_DIR / f"yue_e_{timestamp}")
        else:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Build command arguments
        cmd = [
            PYTHON_ENV_PATH,
            YUE_INFER_SCRIPT,
            "--stage1_model", stage1_model,
            "--stage2_model", stage2_model,
            "--genre_txt", genre_file,
            "--lyrics_txt", lyrics_file,
            "--output_dir", output_file,
            "--max_new_tokens", str(max_new_tokens),
            "--repetition_penalty", str(repetition_penalty),
            "--run_n_segments", str(run_n_segments),
            "--stage1_cache_size", str(stage1_cache_size),
            "--stage2_cache_size", str(stage2_cache_size),
            "--stage1_cache_mode", stage1_cache_mode,
            "--stage2_cache_mode", stage2_cache_mode,
            "--stage2_batch_size", str(stage2_batch_size),
            "--cuda_idx", str(cuda_idx)
        ]
        
        # Add optional flags
        if stage1_use_exl2:
            cmd.append("--stage1_use_exl2")
        if stage2_use_exl2:
            cmd.append("--stage2_use_exl2")
        if stage1_no_guidance:
            cmd.append("--stage1_no_guidance")
        if keep_intermediate:
            cmd.append("--keep_intermediate")
        if disable_offload_model:
            cmd.append("--disable_offload_model")
        if rescale:
            cmd.append("--rescale")
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        
        # Audio prompt options
        if use_audio_prompt:
            cmd.append("--use_audio_prompt")
            if audio_prompt_path:
                cmd.extend(["--audio_prompt_path", audio_prompt_path])
            cmd.extend(["--prompt_start_time", str(prompt_start_time)])
            cmd.extend(["--prompt_end_time", str(prompt_end_time)])
        
        # Dual tracks prompt options
        if use_dual_tracks_prompt:
            cmd.append("--use_dual_tracks_prompt")
            if vocal_track_prompt_path:
                cmd.extend(["--vocal_track_prompt_path", vocal_track_prompt_path])
            if instrumental_track_prompt_path:
                cmd.extend(["--instrumental_track_prompt_path", instrumental_track_prompt_path])
        
        start_time = datetime.now()
        
        # Execute the YuE script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Clean up temporary files
        try:
            os.remove(genre_file)
            os.remove(lyrics_file)
        except:
            pass  # Ignore cleanup errors
        
        # 判断是否有 import/module 错误
        error_keywords = ["ModuleNotFoundError", "Traceback"]
        has_import_error = any(kw in result.stderr for kw in error_keywords)
        success = (result.returncode == 0) and (not has_import_error)

        # Build paths for mixed, instrumental, and vocal track files
        mixed_output_path = os.path.join(output_file, "mixed.wav") if success else None
        itrack_output_path = os.path.join(output_file, "itrack.wav") if success else None
        vtrack_output_path = os.path.join(output_file, "vtrack.wav") if success else None

        return {
            "success": success,
            "output_path": output_file,
            "mixed_output_path": mixed_output_path,
            "itrack_output_path": itrack_output_path,
            "vtrack_output_path": vtrack_output_path,
            "processing_time": processing_time,
            "stdout": result.stdout,
            "parameters": {
                "genre": genre,
                "lyrics_length": len(lyrics),
                "language": language,
                "reasoning_method": reasoning_method,
                "stage1_model": stage1_model,
                "stage2_model": stage2_model,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "run_n_segments": run_n_segments
            },
            "cmd_result_stderr": {
                "stderr": result.stderr
            }
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
