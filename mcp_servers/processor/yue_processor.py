import os
import sys
import uuid
import json
import torch
import torchaudio
import numpy as np
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Literal


# Path constants for pre-trained models
YUE_S1_EN_COT =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-cot"
YUE_S1_EN_ICL =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-en-icl"
YUE_S1_ZH_COT =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-cot"
YUE_S1_ZH_ICL =  "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s1-7B-anneal-zh-icl"
YUE_S2_GENERAL = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-s2-1B-general"
YUE_UPSAMPLER = "/home/chengz/LAMs/pre_train_models/models--m-a-p--YuE-upsampler"

# Add YuE paths to system path
YUE_BASE_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/models/YuE")
YUE_INFERENCE_DIR = YUE_BASE_DIR / "inference"

# Python environment path for YuE
PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/yue/bin/python"

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
YUE_TEMP_DIR = OUTPUT_DIR / "yue_temp"
for dir_path in [AUDIO_DIR, YUE_TEMP_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize MCP server


def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def YuEMusicGenerationTool(
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
    try:
        timestamp = get_timestamp()
        unique_id = uuid.uuid4().hex[:8]
        
        # Create temporary files for genre and lyrics
        genre_file = YUE_TEMP_DIR / f"genre_{timestamp}_{unique_id}.txt"
        lyrics_file = YUE_TEMP_DIR / f"lyrics_{timestamp}_{unique_id}.txt"
        
        # Write genre and lyrics to files
        with open(genre_file, 'w', encoding='utf-8') as f:
            f.write(genre)
        
        with open(lyrics_file, 'w', encoding='utf-8') as f:
            f.write(lyrics)
        
        # Set output file if not provided
        if output_file is None:
            output_dir = str(AUDIO_DIR)
            output_name = f"yue_generated_{timestamp}_{unique_id}.mp3"
        else:
            output_dir = os.path.dirname(os.path.abspath(output_file))
            output_name = os.path.basename(output_file)
            # Make sure the directory exists
            os.makedirs(output_dir, exist_ok=True)
        
        # Determine which Stage 1 model to use based on language and reasoning method
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
        
        # Build the command to run infer.py
        cmd = [
            PYTHON_ENV_PATH,
            str(YUE_INFERENCE_DIR / "infer.py"),
            "--cuda_idx", str(cuda_idx),
            "--stage1_model", stage1_model,
            "--stage2_model", YUE_S2_GENERAL,
            "--genre_txt", str(genre_file),
            "--lyrics_txt", str(lyrics_file),
            "--run_n_segments", str(run_n_segments),
            "--stage2_batch_size", "4",  # Default value
            "--output_dir", output_dir,
            "--max_new_tokens", str(max_new_tokens),
            "--repetition_penalty", str(repetition_penalty),
            "--seed", str(seed)
        ]
        
        # Add optional flags
        if rescale:
            cmd.append("--rescale")
        
        if disable_offload_model:
            cmd.append("--disable_offload_model")
        
        # Add audio prompt options if enabled
        if use_audio_prompt and audio_prompt_path:
            if not os.path.exists(audio_prompt_path):
                return {
                    "success": False,
                    "error": f"Audio prompt file not found: {audio_prompt_path}"
                }
            cmd.extend([
                "--use_audio_prompt",
                "--audio_prompt_path", audio_prompt_path,
                "--prompt_start_time", str(prompt_start_time),
                "--prompt_end_time", str(prompt_end_time)
            ])
        
        # Add dual tracks prompt options if enabled
        if use_dual_tracks_prompt and vocal_track_prompt_path and instrumental_track_prompt_path:
            if not os.path.exists(vocal_track_prompt_path):
                return {
                    "success": False,
                    "error": f"Vocal track prompt file not found: {vocal_track_prompt_path}"
                }
            if not os.path.exists(instrumental_track_prompt_path):
                return {
                    "success": False,
                    "error": f"Instrumental track prompt file not found: {instrumental_track_prompt_path}"
                }
            cmd.extend([
                "--use_dual_tracks_prompt",
                "--vocal_track_prompt_path", vocal_track_prompt_path,
                "--instrumental_track_prompt_path", instrumental_track_prompt_path
            ])
        
        print(f"Starting YuE music generation (genre: {genre[:30]}{'...' if len(genre) > 30 else ''}, lyrics length: {len(lyrics)} chars)...")
        start_time = datetime.now()
        
        # Execute the command
        import subprocess
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"YuE processing completed in {processing_time:.2f} seconds")
        
        # Clean up temporary files if not keeping intermediate files
        if not keep_intermediate:
            try:
                os.remove(genre_file)
                os.remove(lyrics_file)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary files: {e}")
        
        # Find the output file
        final_output_file = None
        for file in os.listdir(output_dir):
            if file.endswith('.mp3') and 'final_mix' in file:
                final_output_file = os.path.join(output_dir, file)
                break
        
        # If no final mix found, try to find any mix file
        if not final_output_file:
            for file in os.listdir(output_dir):
                if file.endswith('.mp3') and 'mix' in file:
                    final_output_file = os.path.join(output_dir, file)
                    break
        
        # If still no file found, try to find any mp3 file
        if not final_output_file:
            for file in os.listdir(output_dir):
                if file.endswith('.mp3'):
                    final_output_file = os.path.join(output_dir, file)
                    break
        
        # Process the result
        if final_output_file and os.path.exists(final_output_file):
            # Rename the file to the expected output name if needed
            if os.path.basename(final_output_file) != output_name:
                new_path = os.path.join(output_dir, output_name)
                os.rename(final_output_file, new_path)
                final_output_file = new_path
            
            return {
                "success": True,
                "output_file": final_output_file,
                "parameters": {
                    "genre": genre[:50] + "..." if len(genre) > 50 else genre,
                    "lyrics_length": len(lyrics),
                    "language": language,
                    "reasoning_method": reasoning_method,
                    "used_audio_prompt": use_audio_prompt,
                    "used_dual_tracks": use_dual_tracks_prompt,
                    "processing_time": processing_time
                }
            }
        else:
            error_message = f"Failed to generate audio. "
            if stderr:
                error_message += f"Error: {stderr}"
            else:
                error_message += "No output file was generated."
            
            return {
                "success": False,
                "error": error_message
            }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

