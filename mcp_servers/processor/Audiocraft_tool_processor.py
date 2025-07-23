import os
import sys
from pathlib import Path
import torch
import torchaudio
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime

import numpy as np
import scipy.io.wavfile

# Add patch for older PyTorch versions that don't have get_default_device
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.get_default_device = get_default_device

# Direct imports from audiocraft
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)



def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Global variables for models
musicgen_model = None
audiogen_model = None
jasco_model = None

musicgen_path = "/home/chengz/LAMs/pre_train_models/models--facebook--musicgen-melody"
audiogen_path = "/home/chengz/LAMs/pre_train_models/models--facebook--audiogen-medium"

def initialize_musicgen(
    model_path: str = musicgen_path,
    device: str = "cuda",
    dtype: str = "float16",
):
    """Initialize the MusicGen model if it hasn't been loaded yet."""
    global musicgen_model
    
    if musicgen_model is None:

        musicgen_model  = MusicGen.get_pretrained(model_path, device=device)
        
        
    
    return musicgen_model

def initialize_audiogen(
    model_path: str = audiogen_path,
    device: str = "cuda",
    dtype: str = "float16",
):
    """Initialize the AudioGen model if it hasn't been loaded yet."""
    global audiogen_model
    
    if audiogen_model is None:

        audiogen_model = AudioGen.get_pretrained(model_path, device=device)
    
    
    return audiogen_model



# @mcp.tool()
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
    model_path: str = musicgen_path,
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
    try:
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Make prompt a list if it's a single string
        if isinstance(prompt, str):
            descriptions = [prompt]
        else:
            descriptions = prompt[:batch_size]  # Limit to batch_size
            
        # Initialize model
        model = initialize_musicgen(model_path, device, dtype)
        
        # Set generation parameters
        model.set_generation_params(
            duration=duration,  # Cap at 30 seconds
            cfg_coef=guidance_scale,
            use_sampling=True,
            temperature=1.0
        )
        
        # Ensure output directory exists
        if output_path is None:
            timestamp = get_timestamp()
            output_paths = [str(AUDIO_DIR / f"musicgen_{timestamp}_{i}.{format}") for i in range(len(descriptions))]
        else:
            if len(descriptions) == 1:
                output_paths = [output_path]
            else:
                # If multiple prompts but single output path, create multiple paths
                base, ext = os.path.splitext(output_path)
                if not ext:
                    ext = f".{format}"
                output_paths = [f"{base}_{i}{ext}" for i in range(len(descriptions))]
        
        # Generate audio
        if melody_path is not None:
            # Load melody
            melody_wav, sr = torchaudio.load(melody_path)
            # Generate with melody conditioning
            wav = model.generate_with_chroma(
                descriptions=descriptions,
                melody_wavs=melody_wav,
                melody_sample_rate=melody_sample_rate,
                progress=True
            )
        else:
            # Generate with text only
            wav = model.generate(
                descriptions=descriptions,
                progress=True
            )
        
        # Save outputs
        sample_rate = model.sample_rate
        for i, one_wav in enumerate(wav):
            # Will save under output_path, with optional loudness normalization
            audio_write(
                output_paths[i].split('.')[0],  # Remove extension as audio_write adds it
                one_wav.cpu(),
                sample_rate,
                strategy="loudness" if apply_loudness_normalization else "peak",
                loudness_compressor=apply_loudness_normalization
            )
        
        # Return result
        return {
            "success": True,
            "output_paths": output_paths if len(output_paths) > 1 else output_paths[0]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# @mcp.tool()
def AudioGenTool(
    # Input content
    prompt: Union[str, List[str]],
    
    # Output configuration
    output_path: Optional[str] = None,
    format: str = "wav",
    
    # Generation parameters
    duration: float = 10.0,
    guidance_scale: float = 3.0,
    temperature: float = 1.0,
    top_k: int = 250,
    top_p: float = 0.0,
    seed: Optional[int] = None,
    
    # Audio normalization
    apply_loudness_normalization: bool = True,
    
    # Model configuration
    model_path: str = audiogen_path,
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
    try:

        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Make prompt a list if it's a single string
        if isinstance(prompt, str):
            descriptions = [prompt]
        else:
            descriptions = prompt[:batch_size]  # Limit to batch_size
            
        # Initialize model
        model = initialize_audiogen(model_path, device, dtype)
        
        # Set generation parameters
        model.set_generation_params(
            use_sampling=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            duration=duration,
            cfg_coef=guidance_scale,
            extend_stride=extend_stride
        )
        
        # Ensure output directory exists
        if output_path is None:
            timestamp = get_timestamp()
            output_paths = [str(AUDIO_DIR / f"audiogen_{timestamp}_{i}.{format}") for i in range(len(descriptions))]
        else:
            if len(descriptions) == 1:
                output_paths = [output_path]
            else:
                # If multiple prompts but single output path, create multiple paths
                base, ext = os.path.splitext(output_path)
                if not ext:
                    ext = f".{format}"
                output_paths = [f"{base}_{i}{ext}" for i in range(len(descriptions))]
        
        # Generate audio
        wav = model.generate(descriptions, progress=True)
        
        # Save outputs
        sample_rate = model.sample_rate
        for i, one_wav in enumerate(wav):
            # Will save under output_path, with optional loudness normalization
            audio_write(
                output_paths[i].split('.')[0],  # Remove extension as audio_write adds it
                one_wav.cpu(),
                sample_rate,
                strategy="loudness" if apply_loudness_normalization else "peak",
                loudness_compressor=apply_loudness_normalization
            )
        
        # Return result
        return {
            "success": True,
            "output_paths": output_paths if len(output_paths) > 1 else output_paths[0]
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


