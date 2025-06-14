import os
import sys
from pathlib import Path
import torch
import numpy as np
import librosa
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
from scipy.io.wavfile import write
import subprocess
import json

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Scripts directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# 如果设置了环境变量 AUDIOSEP_SCRIPTS_DIR，则优先使用；否则默认 <project_root>/models/AudioSep
SCRIPTS_DIR = Path(os.getenv("AUDIOSEP_SCRIPTS_DIR", PROJECT_ROOT / "models" / "AudioSep"))
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIOSEP_SCRIPT_PATH = SCRIPTS_DIR / "run_audiosep.py"

# Add AudioSep path to system path
sys.path.append(os.path.abspath("models/AudioSep"))

# Set environment variable to avoid PyDantic compatibility issues
os.environ["LIGHTNING_PYDANTIC_V2"] = "False"



# Global constants
AUDIOSEP_MODEL_PATH = "/home/chengz/LAMs/pre_train_models/Audiosep_pretrain_models/audiosep_base_4M_steps.ckpt"
AUDIOSEP_CONFIG_PATH = os.path.abspath("models/AudioSep/config/audiosep_base.yaml")
PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/AudioSep/bin/python"

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")




def AudioSepTool(
    # Input content
    audio_file: str,
    text: str,
    
    # Output configuration
    output_file: Optional[str] = None,
    
    # Processing options
    use_chunk: bool = False,
    
    # Computing device
    device: str = "cuda"
) -> Dict[str, Any]:
    """Separate specific sounds from an audio file using textual descriptions.
    
    Args:
        audio_file: Path to the input audio file to process
        text: Textual description of the sound to separate (e.g., "piano playing", "dog barking")
        output_file: Path to save the separated audio file. If not provided, a default path will be used
        use_chunk: Whether to use chunked processing for memory efficiency with longer audio files
        device: Computing device to use (cuda or cpu)
        
    Returns:
        Dictionary containing the separation results and metadata
    """
    try:
        # Validate input parameters
        if not os.path.exists(audio_file):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_file}"
            }
        
        if not text or text.strip() == "":
            return {
                "success": False,
                "error": "Text query cannot be empty"
            }
        
        # Set output path
        if output_file is None:
            timestamp = get_timestamp()
            output_file = str(AUDIO_DIR / f"audiosep_{timestamp}.wav")
        else:
            # Make sure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        start_time = datetime.now()
        
        # Build command arguments
        cmd = [
            PYTHON_ENV_PATH,
            str(AUDIOSEP_SCRIPT_PATH),
            "--audio_file", audio_file,
            "--text", text,
            "--output_file", output_file,
            "--device", device,
            "--model_path", AUDIOSEP_MODEL_PATH,
            "--config_path", AUDIOSEP_CONFIG_PATH
        ]
        
        if use_chunk:
            cmd.append("--use_chunk")
        
        # Execute the AudioSep script
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Check the subprocess result
            if result.returncode != 0:
                return {
                    "success": False,
                    "error": f"AudioSep processing failed: {result.stderr}"
                }
            
            # Try to parse JSON output from the subprocess
            try:
                last_line = result.stdout.strip().split('\n')[-1]
                subprocess_result = json.loads(last_line)
                if not subprocess_result.get("success", False):
                    return {
                        "success": False,
                        "error": subprocess_result.get("error", "Unknown error in AudioSep processing")
                    }
            except json.JSONDecodeError:
                # If there's no JSON output, check if the output file exists
                if not os.path.exists(output_file):
                    return {
                        "success": False,
                        "error": "AudioSep processing failed to create output file"
                    }
            
            # Return result
            return {
                "success": True,
                "output_path": output_file,
                "processing_time": processing_time,
                "parameters": {
                    "text_query": text,
                    "use_chunk": use_chunk,
                    "device": device
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to execute AudioSep subprocess: {str(e)}"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


