import os
import sys
import subprocess
import json
from pathlib import Path
import random
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import tempfile

# Create output directories

OUTPUT_DIR = Path("/home/chengz/LAMs/mcp_chatbot-audio/output/music")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from mcp.server.fastmcp import FastMCP
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Initialize MCP server
mcp = FastMCP("DiffRhythm: Diffusion-Based High-Quality Full-Length Music Generation System")

# Global constants
PYTHON_PATH = "/home/qianshuaix/miniconda3/envs/diffrhythm/bin/python"
INFER_SCRIPT = "models/DiffRhythm/infer/infer.py"

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

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
    """
    try:
        # Validate input parameters
        if ref_prompt is None and ref_audio_path is None:
            return {
                "success": False,
                "error": "Either ref_prompt or ref_audio_path must be provided"
            }
        
        if ref_prompt is not None and ref_audio_path is not None:
            return {
                "success": False,
                "error": "Only one of ref_prompt or ref_audio_path should be provided"
            }
            
        if edit and (ref_song is None or edit_segments is None):
            return {
                "success": False,
                "error": "Reference song and edit segments must be provided when edit mode is enabled"
            }
        
        # Set model repo based on version
        if model_version == "v1.2":
            repo_id = "ASLP-lab/DiffRhythm-1_2"
            audio_length = 95
        elif model_version == "full":
            repo_id = "ASLP-lab/DiffRhythm-full"
            audio_length = 285
        else:
            return {
                "success": False,
                "error": f"Invalid model version: {model_version}. Must be 'v1.2' or 'full'"
            }
        
        # Set output path
        if output_path is None:
            timestamp = get_timestamp()
            filename = f"diffrhythm_{model_version}_{timestamp}.wav"
            output_path = str(OUTPUT_DIR / filename)
        else:
            output_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(output_dir, exist_ok=True)
        
        output_dir = os.path.dirname(output_path)
        
        # Create a temporary lyrics file if lrc_text was provided
        temp_lrc_path = None
        if lrc_path is None and lrc_text is not None:
            temp_lrc_path = str(OUTPUT_DIR / f"temp_lyrics_{get_timestamp()}.lrc")
            with open(temp_lrc_path, "w", encoding="utf-8") as f:
                f.write(lrc_text)
            lrc_path = temp_lrc_path
        
        # Build command
        cmd = [
            PYTHON_PATH,
            INFER_SCRIPT,
            "--output-dir", output_dir,
            "--audio-length", str(audio_length),
            "--batch-infer-num", str(batch_infer_num),
            "--repo-id", repo_id
        ]
        
        # Add lyrics path if provided
        if lrc_path:
            cmd.extend(["--lrc-path", lrc_path])
        
        # Add style reference
        if ref_prompt:
            cmd.extend(["--ref-prompt", ref_prompt])
        elif ref_audio_path:
            cmd.extend(["--ref-audio-path", ref_audio_path])
        
        # Add edit mode parameters if enabled
        if edit:
            cmd.append("--edit")
            cmd.extend(["--ref-song", ref_song])
            cmd.extend(["--edit-segments", edit_segments])
        
        # Add chunked decoding if enabled
        if chunked:
            cmd.append("--chunked")
        
        # Run the command
        if verbose:
            print(f"Running DiffRhythm with command: {' '.join(cmd)}")
            print(f"Model: {model_version} ({repo_id})")
            print(f"Audio length: {audio_length}s")
        
        start_time_inference = datetime.now()
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd="/home/chengz/LAMs/mcp_chatbot-audio/models/DiffRhythm"  # Set working directory to DiffRhythm root
        )
        
        stdout, stderr = process.communicate()
        
        inference_time = (datetime.now() - start_time_inference).total_seconds()
        
        if process.returncode != 0:
            return {
                "success": False,
                "error": f"DiffRhythm execution failed: {stderr}",
                "stdout": stdout,
                "stderr": stderr,
                "command": " ".join(cmd)
            }
        
        # Check if output file exists
        default_output_path = os.path.join(output_dir, "output.wav")
        if os.path.exists(default_output_path):
            # Rename to desired output path if necessary
            if default_output_path != output_path:
                os.rename(default_output_path, output_path)
                final_output_path = output_path
            else:
                final_output_path = default_output_path
        else:
            return {
                "success": False,
                "error": f"Output file was not generated at {default_output_path}",
                "stdout": stdout,
                "stderr": stderr,
                "command": " ".join(cmd)
            }
        
        # Clean up temporary files
        if temp_lrc_path and os.path.exists(temp_lrc_path):
            os.remove(temp_lrc_path)
        
        # Get file size and duration info
        file_size = os.path.getsize(final_output_path) if os.path.exists(final_output_path) else 0
        
        # Return result
        return {
            "success": True,
            "output_path": final_output_path,
            "parameters": {
                "model_version": model_version,
                "repo_id": repo_id,
                "audio_length": audio_length,
                "chunked": chunked,
                "edit": edit,
                "batch_infer_num": batch_infer_num,
                "inference_time": inference_time,
                "file_size_mb": round(file_size / (1024 * 1024), 2)
            },
            "stdout": stdout if verbose else None,
            "model_info": {
                "name": "DiffRhythm",
                "version": model_version,
                "max_duration": f"{audio_length}s",
                "description": "Diffusion-based end-to-end full-length song generation"
            }
        }
    
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        if verbose:
            print(f"Error in DiffRhythmTool: {e}")
            print(error_traceback)
        return {
            "success": False,
            "error": str(e),
            "traceback": error_traceback
        }

if __name__ == "__main__":
    print("开始测试 DiffRhythm 工具...")
    print("=" * 50)
    
    # 测试用例1：基本文本提示词生成 (v1.2)
    print("\n测试用例1：DiffRhythm-v1.2 基本文本提示词生成")
    print("-" * 40)
    result1 = DiffRhythmTool(
        model_version="v1.2",
        lrc_text="[00:00.00]This is a test song\n[00:05.00]Generated by DiffRhythm\n[00:10.00]AI music creation",
        ref_prompt="happy energetic pop song with electronic elements",
        chunked=True,
        verbose=True
    )
    print(json.dumps(result1, indent=2, ensure_ascii=False))
    
    # 测试用例2：使用参考音频生成 (v1.2)
    print("\n测试用例2：DiffRhythm-v1.2 参考音频生成")
    print("-" * 40)
    test_audio = "/home/chengz/LAMs/mcp_chatbot-audio/user_media/audio/audio_dfb01034-13d5-447f-b458-327d12ddd56a.mp3"
    if os.path.exists(test_audio):
        result2 = DiffRhythmTool(
            model_version="v1.2",
            lrc_text="[00:00.00]Reference audio style\n[00:05.00]AI generated vocals\n[00:10.00]Perfect synchronization",
            ref_audio_path=test_audio,
            chunked=True,
            verbose=True
        )
        print(json.dumps(result2, indent=2, ensure_ascii=False))
    else:
        print(f"测试音频文件不存在: {test_audio}")
    
    # 测试用例3：长音频生成 (full)
    print("\n测试用例3：DiffRhythm-full 长音频生成")
    print("-" * 40)
    result3 = DiffRhythmTool(
        model_version="full",
        lrc_text="[00:00.00]This is a full-length song\n[00:30.00]With multiple verses\n[01:00.00]And choruses\n[01:30.00]Generated by AI\n[02:00.00]Up to four minutes long",
        ref_prompt="epic orchestral ballad with piano and strings",
        chunked=True,
        verbose=True
    )
    print(json.dumps(result3, indent=2, ensure_ascii=False))
    
    # 测试用例4：纯音乐生成（无歌词）
    print("\n测试用例4：DiffRhythm-v1.2 纯音乐生成")
    print("-" * 40)
    result4 = DiffRhythmTool(
        model_version="v1.2",
        ref_prompt="instrumental jazz piano solo with smooth saxophone",
        chunked=True,
        verbose=True
    )
    print(json.dumps(result4, indent=2, ensure_ascii=False))
    
    # 测试用例5：批量生成
    print("\n测试用例5：DiffRhythm-v1.2 批量生成")
    print("-" * 40)
    result5 = DiffRhythmTool(
        model_version="v1.2",
        lrc_text="[00:00.00]Batch generation test\n[00:05.00]Multiple versions",
        ref_prompt="rock song with electric guitar",
        batch_infer_num=2,
        chunked=True,
        verbose=True
    )
    print(json.dumps(result5, indent=2, ensure_ascii=False))
    
    print("\n所有测试完成！") 
