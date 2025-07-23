import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import time
import json
from pathlib import Path
import tempfile
from typing import Optional, Dict, Any, List, Union, Literal

def ClearVoice_tool(
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
    
    # Compute options
    device: str = "cuda"
) -> Dict[str, Any]:
    """Process audio using ClearerVoice models for speech enhancement, separation, and more.
    
    ClearerVoice is a unified platform for various speech processing tasks:
    
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
        task: The audio processing task to perform:
              - speech_enhancement: Remove noise and improve speech quality
              - speech_separation: Separate multiple speakers in an audio
              - speech_super_resolution: Improve audio quality and resolution
              - target_speaker_extraction: Extract target speaker from audio/video
              
        model_name: Specific model to use for the selected task
        
        input_path: Path to the input audio file (for speech tasks) or video file (for TSE)
                   Supported formats: .wav, .mp4, .avi, .aac, .ac3, .aiff, .flac, 
                                      .m4a, .mp3, .ogg, .opus, .wma, .webm
                                      
        output_path: Directory to save the processed output
                    If not provided, a default temporary directory will be used
                    
        online_write: Whether to automatically save the processed audio
        
        device: Computing device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing:
        - success: Whether processing was successful
        - output_path: Path to the processed audio file(s)
        - task: The task that was performed
        - model: The model that was used
        - processing_time: Time taken to process the audio
        - error: Error message if processing failed
        
    Examples:
        # Speech enhancement with 48kHz model
        result = ClearVoice_tool(
            task="speech_enhancement",
            model_name="MossFormer2_SE_48K",
            input_path="noisy_speech.wav",
            output_path="enhanced_output"
        )
        
        # Speaker separation
        result = ClearVoice_tool(
            task="speech_separation",
            model_name="MossFormer2_SS_16K",
            input_path="mixed_voices.wav",
            output_path="separated_output"
        )
        
        # Target speaker extraction from video
        result = ClearVoice_tool(
            task="target_speaker_extraction",
            model_name="AV_MossFormer2_TSE_16K",
            input_path="video_with_multiple_speakers.mp4",
            output_path="extracted_speaker"
        )
    """
    try:
        # Create default output path if not provided
        if output_path is None:
            output_path = str(Path(tempfile.gettempdir()) / f"clearvoice_output_{int(time.time())}")
        
        # Ensure the output directory exists
        os.makedirs(output_path, exist_ok=True)
        
        # Import ClearVoice
        from clearvoice import ClearVoice
        
        # Validate model and task compatibility
        task_model_map = {
            "speech_enhancement": ["MossFormer2_SE_48K", "MossFormerGAN_SE_16K", "FRCRN_SE_16K"],
            "speech_separation": ["MossFormer2_SS_16K"],
            "speech_super_resolution": ["MossFormer2_SR_48K"],
            "target_speaker_extraction": ["AV_MossFormer2_TSE_16K"]
        }
        
        if model_name not in task_model_map.get(task, []):
            return {
                "success": False,
                "error": f"Model '{model_name}' is not compatible with task '{task}'. Compatible models: {task_model_map.get(task, [])}"
            }
        
        # Initialize ClearVoice with the specified model
        start_time = time.time()
        cv = ClearVoice(
            task=task,
            model_names=[model_name]
        )
        
        # Process the audio
        output = cv(
            input_path=input_path,
            online_write=online_write,
            output_path=output_path
        )
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create response
        response = {
            "success": True,
            "output_path": output_path,
            "task": task,
            "model": model_name,
            "processing_time": processing_time
        }
        
        # For speech separation, multiple files are generated
        if task == "speech_separation":
            # List output files
            output_files = [f for f in os.listdir(output_path) if f.startswith(f"output_{model_name}_spk")]
            response["output_files"] = output_files
        
        return response
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Batch processing function
def process_directory(
    input_dir: str, 
    output_dir: str, 
    task: Literal["speech_enhancement", "speech_separation", "speech_super_resolution", "target_speaker_extraction"],
    model_name: Optional[str] = None
) -> Dict[str, Any]:
    """Process all audio files in a directory using ClearVoice.
    
    Args:
        input_dir: Directory containing audio files to process
        output_dir: Directory to save processed files
        task: The task to perform
        model_name: Model to use for processing (if None, will use default for task)
        
    Returns:
        Dictionary containing processing results
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Select default model based on task if not specified
    if model_name is None:
        model_name = {
            "speech_enhancement": "MossFormer2_SE_48K",
            "speech_separation": "MossFormer2_SS_16K",
            "speech_super_resolution": "MossFormer2_SR_48K",
            "target_speaker_extraction": "AV_MossFormer2_TSE_16K"
        }.get(task)
    
    # Get all audio/video files
    supported_formats = ('.wav', '.mp4', '.avi', '.aac', '.ac3', '.aiff', 
                       '.flac', '.m4a', '.mp3', '.ogg', '.opus', '.wma', '.webm')
    media_files = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    
    if not media_files:
        return {
            "success": False,
            "error": f"No supported media files found in {input_dir}"
        }
    
    results = []
    
    # Process each file
    for media_file in media_files:
        input_path = os.path.join(input_dir, media_file)
        file_output_dir = os.path.join(output_dir, Path(media_file).stem)
        
        # Process the file
        result = ClearVoice_tool(
            task=task,
            model_name=model_name,
            input_path=input_path,
            output_path=file_output_dir
        )
        
        results.append({
            "file": media_file,
            "success": result.get("success", False),
            "output_path": result.get("output_path", None),
            "error": result.get("error", None) if not result.get("success", False) else None
        })
    
    return {
        "success": True,
        "processed_files": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "results": results,
        "output_path": output_dir
    }

# Method dispatcher for launcher script
def ClearVoiceTool(method="ClearVoice_tool", **kwargs):
    """Dispatcher function for the ClearVoiceTool.
    
    This function is called by the MCP launcher script and routes the call
    to the appropriate method (either ClearVoice_tool or process_directory).
    
    Args:
        method: The method to call ("ClearVoice_tool" or "process_directory")
        **kwargs: Arguments to pass to the method
        
    Returns:
        Dictionary containing the results from the called method
    """
    if method == "ClearVoice_tool":
        return ClearVoice_tool(**kwargs)
    elif method == "process_directory":
        return process_directory(**kwargs)
    else:
        return {
            "success": False,
            "error": f"Unknown method: {method}"
        }

