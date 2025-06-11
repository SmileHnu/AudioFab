import os
import sys
from pathlib import Path
import subprocess
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import json
import librosa
import tempfile

# Set HuggingFace mirror endpoint
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Python interpreter path for the ClearerVoice environment
PYTHON_ENV_PATH = "/home/qianshuaix/miniconda3/envs/envTest/bin/python"

# MCP tool launcher script
MCP_LAUNCHER_SCRIPT = str(Path(__file__).parent / "mcp_tool_launcher.py")

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_audio_info(audio_path):
    """Get audio file information (sample rate, channels) to help with model selection."""
    try:
        # Load audio file using librosa directly
        y, sr = librosa.load(audio_path, sr=None)
        channels = 2 if len(y.shape) > 1 and y.shape[0] > 1 else 1
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Return audio information
        return {
            "success": True,
            "sample_rate": sr,
            "channels": channels,
            "duration": duration,
            "bit_depth": "16-bit" if y.dtype == 'float32' else "Unknown"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error getting audio info: {str(e)}"
        }

def ClearerVoiceTool(
    # Input content
    input_path: str,
    
    # Task selection
    task: Literal["speech_enhancement", "speech_separation", "speech_super_resolution", "target_speaker_extraction"] = "speech_enhancement",
    
    # Model selection
    model_name: Literal[
        "MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K",  # Speech Enhancement
        "MossFormer2_SS_16K",  # Speech Separation
        "MossFormer2_SR_48K",  # Speech Super Resolution
        "AV_MossFormer2_TSE_16K"  # Audio-Visual Target Speaker Extraction
    ] = "MossFormer2_SE_48K",
    
    # Output configuration
    output_path: Optional[str] = None,
    online_write: bool = True,
    
    # Auto-select model based on audio properties
    auto_select_model: bool = True
) -> Dict[str, Any]:
    """Process audio using ClearerVoice-Studio for speech enhancement, separation, super-resolution, and target speaker extraction.
    
    ClearerVoice-Studio is a unified inference platform for speech audio processing tasks. It supports multiple state-of-the-art
    pretrained models and provides a simple interface for processing audio files.
    
    Supported audio formats: wav, aac, ac3, aiff, flac, m4a, mp3, ogg, opus, wma, webm, etc.
    Supports both mono and stereo channels with 16-bit or 32-bit precision.
    
    Args:
        input_path: Path to the input audio file or video file (for target speaker extraction)
        
        task: The audio processing task to perform:
              - speech_enhancement: Remove noise and improve speech quality
              - speech_separation: Separate multiple speakers in an audio
              - speech_super_resolution: Improve audio quality and resolution
              - target_speaker_extraction: Extract target speaker from audio/video
              
        model_name: The specific model to use for the task:
                   For speech_enhancement:
                   - MossFormer2_SE_48K: For 48kHz high-quality audio
                   - FRCRN_SE_16K: For 16kHz audio, faster processing
                   - MossFormerGAN_SE_16K: For 16kHz audio, higher quality
                   
                   For speech_separation:
                   - MossFormer2_SS_16K: For separating multiple speakers
                   
                   For speech_super_resolution:
                   - MossFormer2_SR_48K: For upsampling to 48kHz
                   
                   For target_speaker_extraction:
                   - AV_MossFormer2_TSE_16K: For extracting speakers from audiovisual input
        
        output_path: Directory to save the processed output. If not provided, a default path will be used
        
        online_write: Whether to automatically save the processed audio
        
        auto_select_model: If True, automatically select the appropriate model based on the audio properties
        
    Returns:
        Dictionary containing the processing results and metadata
    """
    try:
        # Validate input parameters
        if not os.path.exists(input_path):
            return {
                "success": False,
                "error": f"Input file not found: {input_path}"
            }
        
        # Set output path
        if output_path is None:
            timestamp = get_timestamp()
            output_path = str(AUDIO_DIR / f"clearervoice_{timestamp}")
            os.makedirs(output_path, exist_ok=True)
        else:
            # Make sure the directory exists
            os.makedirs(output_path, exist_ok=True)
        
        # Get audio info for auto model selection if needed
        audio_info = None
        if auto_select_model:
            audio_info = get_audio_info(input_path)
            if not audio_info.get("success", False):
                return {
                    "success": False,
                    "error": f"Failed to analyze audio: {audio_info.get('error', 'Unknown error')}"
                }
            
            # Auto-select the most appropriate model based on audio properties
            sample_rate = audio_info.get("sample_rate", 16000)
            
            if task == "speech_enhancement":
                if sample_rate >= 44100:
                    model_name = "MossFormer2_SE_48K"  # High quality for high sample rate
                else:
                    model_name = "MossFormerGAN_SE_16K"  # Better quality for standard sample rate
            
            elif task == "speech_super_resolution":
                model_name = "MossFormer2_SR_48K"
            
            elif task == "speech_separation":
                model_name = "MossFormer2_SS_16K"
            
            elif task == "target_speaker_extraction":
                model_name = "AV_MossFormer2_TSE_16K"
        
        # Map the selected model to model_names array for ClearVoice
        model_names = [model_name]
        
        # Create a temporary file for parameters
        temp_params_file = Path(tempfile.gettempdir()) / f"clearervoice_params_{get_timestamp()}.json"
        params = {
            "task": task,
            "model_names": model_names,
            "input_path": input_path,
            "output_path": output_path,
            "online_write": online_write
        }
        
        with open(temp_params_file, 'w') as f:
            json.dump(params, f)
        
        # Build command to run the MCP launcher
        cmd = [
            PYTHON_ENV_PATH,
            MCP_LAUNCHER_SCRIPT,
            "--tool_name", "ClearVoice",
            "--module_path", str(Path(__file__).parent / "ClearerVoice_tool.py"),
            "--params_file", str(temp_params_file)
        ]
        
        # Execute the command
        start_time = datetime.now()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up the temporary file
        try:
            os.remove(temp_params_file)
        except:
            pass
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Check for execution errors
        if result.returncode != 0:
            return {
                "success": False,
                "error": f"ClearerVoice processing failed: {result.stderr}"
            }
        
        # Parse the output
        try:
            cv_result = json.loads(result.stdout.strip())
            if not cv_result.get("success", False):
                return {
                    "success": False,
                    "error": cv_result.get("error", "Unknown error in ClearerVoice processing")
                }
            
            # Add additional information to the result
            cv_result["audio_info"] = audio_info
            cv_result["input_path"] = input_path
            cv_result["processing_time"] = processing_time
            
            return cv_result
            
        except json.JSONDecodeError:
            # If there's no valid JSON output, check if files were created
            output_files = os.listdir(output_path)
            if not output_files:
                return {
                    "success": False,
                    "error": "ClearerVoice processing did not produce any output files"
                }
            
            # Return basic success information
            return {
                "success": True,
                "output_path": output_path,
                "output_files": output_files,
                "task": task,
                "model": model_name,
                "processing_time": processing_time,
                "message": "Processing completed but detailed information unavailable"
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def SpeechEnhancementTool(
    # Input content
    input_path: str,
    
    # Model selection
    model_name: Literal["MossFormer2_SE_48K", "FRCRN_SE_16K", "MossFormerGAN_SE_16K"] = "MossFormer2_SE_48K",
    
    # Output configuration
    output_path: Optional[str] = None,
    online_write: bool = True,
    
    # Auto-select model based on audio properties
    auto_select_model: bool = True
) -> Dict[str, Any]:
    """Enhance speech audio by removing noise and improving clarity.
    
    This tool uses state-of-the-art speech enhancement models to clean up noisy audio recordings.
    It can effectively remove background noise, room reverberation, and other distortions.
    
    Supported audio formats: wav, aac, ac3, aiff, flac, m4a, mp3, ogg, opus, wma, webm, etc.
    Supports both mono and stereo channels with 16-bit or 32-bit precision.
    
    Args:
        input_path: Path to the input audio file
        
        model_name: The specific model to use:
                   - MossFormer2_SE_48K: Best for 48kHz high-quality audio
                   - FRCRN_SE_16K: Faster processing for 16kHz audio
                   - MossFormerGAN_SE_16K: Higher quality for 16kHz audio
        
        output_path: Directory to save the enhanced audio. If not provided, a default path will be used
        
        online_write: Whether to automatically save the processed audio
        
        auto_select_model: If True, automatically select the appropriate model based on audio sample rate
        
    Returns:
        Dictionary containing the enhancement results and output file information
    """
    return ClearerVoiceTool(
        input_path=input_path,
        task="speech_enhancement",
        model_name=model_name,
        output_path=output_path,
        online_write=online_write,
        auto_select_model=auto_select_model
    )

def SpeechSeparationTool(
    # Input content
    input_path: str,
    
    # Model selection - only one option currently available
    model_name: Literal["MossFormer2_SS_16K"] = "MossFormer2_SS_16K",
    
    # Output configuration
    output_path: Optional[str] = None,
    online_write: bool = True
) -> Dict[str, Any]:
    """Separate multiple speakers from a mixed audio recording.
    
    This tool separates speech from different speakers in a mixed audio recording.
    It can identify and isolate up to two different speakers.
    
    Supported audio formats: wav, aac, ac3, aiff, flac, m4a, mp3, ogg, opus, wma, webm, etc.
    Supports both mono and stereo channels with 16-bit or 32-bit precision.
    
    Args:
        input_path: Path to the input audio file containing multiple speakers
        
        model_name: The specific model to use:
                   - MossFormer2_SS_16K: Model for separating multiple speakers
        
        output_path: Directory to save the separated audio files. If not provided, a default path will be used
        
        online_write: Whether to automatically save the processed audio
        
    Returns:
        Dictionary containing the separation results and paths to individual speaker files
        The output will include separate WAV files for each identified speaker
    """
    return ClearerVoiceTool(
        input_path=input_path,
        task="speech_separation",
        model_name=model_name,
        output_path=output_path,
        online_write=online_write,
        auto_select_model=False
    )

def SpeechSuperResolutionTool(
    # Input content
    input_path: str,
    
    # Model selection - only one option currently available
    model_name: Literal["MossFormer2_SR_48K"] = "MossFormer2_SR_48K",
    
    # Output configuration
    output_path: Optional[str] = None,
    online_write: bool = True
) -> Dict[str, Any]:
    """Improve audio quality by upsampling to higher resolution.
    
    This tool enhances low-quality audio recordings by upsampling them to higher resolution.
    It can improve clarity, reduce artifacts, and enhance overall audio quality.
    
    Supported audio formats: wav, aac, ac3, aiff, flac, m4a, mp3, ogg, opus, wma, webm, etc.
    Supports both mono and stereo channels with 16-bit or 32-bit precision.
    
    Args:
        input_path: Path to the input audio file to enhance
        
        model_name: The specific model to use:
                   - MossFormer2_SR_48K: Model for upsampling to 48kHz
        
        output_path: Directory to save the enhanced audio. If not provided, a default path will be used
        
        online_write: Whether to automatically save the processed audio
        
    Returns:
        Dictionary containing the super-resolution results and output file information
    """
    return ClearerVoiceTool(
        input_path=input_path,
        task="speech_super_resolution",
        model_name=model_name,
        output_path=output_path,
        online_write=online_write,
        auto_select_model=False
    )

def TargetSpeakerExtractionTool(
    # Input content
    input_path: str,
    
    # Model selection - only one option currently available
    model_name: Literal["AV_MossFormer2_TSE_16K"] = "AV_MossFormer2_TSE_16K",
    
    # Output configuration
    output_path: Optional[str] = None,
    online_write: bool = True
) -> Dict[str, Any]:
    """Extract the target speaker's voice from an audio/video recording.
    
    This tool extracts the voice of a specific speaker from an audio or video recording
    that contains multiple speakers or background noise. It uses audio-visual cues
    to identify and isolate the target speaker.
    
    Supported formats: wav, mp4, avi and other video formats with audio tracks.
    
    Args:
        input_path: Path to the input audio or video file
        
        model_name: The specific model to use:
                   - AV_MossFormer2_TSE_16K: Audio-visual model for target speaker extraction
        
        output_path: Directory to save the extracted audio. If not provided, a default path will be used
        
        online_write: Whether to automatically save the processed audio
        
    Returns:
        Dictionary containing the extraction results and output file information
        The output will include the extracted target speaker audio and possibly background audio
    """
    return ClearerVoiceTool(
        input_path=input_path,
        task="target_speaker_extraction",
        model_name=model_name,
        output_path=output_path,
        online_write=online_write,
        auto_select_model=False
    ) 