import os
import sys
from pathlib import Path
import subprocess
import json
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# SparkTTS paths
SPARKTTS_PATH = "/home/chengz/LAMs/pre_train_models/models--SparkAudio--Spark-TTS-0.5B"
INFERENCE_SCRIPT = "models/SparkTTS/cli/inference.py"
PYTHON_PATH = "/home/chengz/anaconda3/bin/python"

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def SparkTTSTool(
    # Input text
    text: str,
    
    # Voice cloning options
    prompt_text: Optional[str] = None,
    prompt_speech_path: Optional[str] = None,
    
    # Voice control parameters
    gender: Optional[Literal["male", "female"]] = None,
    pitch: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
    speed: Optional[Literal["very_low", "low", "moderate", "high", "very_high"]] = None,
    
    # Output configuration
    output_path: Optional[str] = None,
    
    # Model configuration
    device: int = 0,
    
    # Verbose output
    verbose: bool = False
) -> Dict[str, Any]:
    """Generate speech using the Spark-TTS zero-shot text-to-speech system.
    
    Args:
        text: The text to convert to speech
        prompt_text: Transcript of the reference audio for voice cloning (needed with prompt_speech_path)
        prompt_speech_path: Path to the reference audio file for voice cloning
        gender: Gender of the synthesized voice ("male" or "female")
        pitch: Pitch level of the voice ("very_low", "low", "moderate", "high", "very_high")
        speed: Speaking rate ("very_low", "low", "moderate", "high", "very_high")
        output_path: Custom path to save the generated audio  (WAV format)
        device: CUDA device ID for inference (0, 1, etc.)
        verbose: Whether to print detailed information during processing
        
    Returns:
        Dictionary containing the path to the generated audio file and processing info
    """
    try:
        # Determine output path
        if output_path is None:
            timestamp = get_timestamp()
            output_path = str(AUDIO_DIR / f"sparktts_{timestamp}.wav")
        else:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Build command
        cmd = [
            PYTHON_PATH,
            INFERENCE_SCRIPT,
            "--model_dir", SPARKTTS_PATH,
            "--save_dir", os.path.dirname(output_path),
            "--device", str(device),
            "--text", text
        ]
        
        # Add optional arguments
        if prompt_text:
            cmd.extend(["--prompt_text", prompt_text])
        if prompt_speech_path:
            cmd.extend(["--prompt_speech_path", prompt_speech_path])
        if gender:
            cmd.extend(["--gender", gender])
        if pitch:
            cmd.extend(["--pitch", pitch])
        if speed:
            cmd.extend(["--speed", speed])
        
        # Run the command
        if verbose:
            print(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return {
                "success": False,
                "error": f"SparkTTS execution failed: {stderr}",
                "stdout": stdout,
                "stderr": stderr
            }
        
        # Determine the mode used
        if gender is not None:
            mode = "controllable_tts"
        elif prompt_speech_path is not None:
            mode = "voice_cloning"
        else:
            mode = "default"
        
        return {
            "success": True,
            "audio_path": output_path,
            "mode": mode,
            "processing_info": {
                "text": text,
                "device_id": device,
                "stdout": stdout
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Speech generation failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

if __name__ == "__main__":
    # Test the tool
    result = SparkTTSTool(
        text="Hello, this is a test of the SparkTTS system.",
        verbose=True,
        device=0,
        gender="female",
        pitch="very_low",
        speed="very_low"
    )
    print(json.dumps(result, indent=2))
