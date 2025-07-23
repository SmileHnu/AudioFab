import os
import sys
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from datetime import datetime

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
SEPARATED_DIR = AUDIO_DIR #/ "separated"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
SEPARATED_DIR.mkdir(parents=True, exist_ok=True)

# Add audio-separator path to system path

# sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/audio-separator')
sys.path.append(os.path.abspath("models/audio-separator"))
from audio_separator.separator import Separator

# Global constants
MODEL_FILE_DIR = os.path.abspath("models/audio-separator/models")
DEFAULT_UVR_MODEL = "UVR-MDX-NET-Inst_HQ_3.onnx"
DEFAULT_DEMUCS_MODEL = "htdemucs_6s.yaml"

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Global variables for model
separator = None

def initialize_separator(
    model_file_dir: str = MODEL_FILE_DIR,
    output_dir: str = None,
    output_format: str = "WAV"
):
    """Initialize the audio separator if it hasn't been loaded yet."""
    global separator
    
    if separator is None:
        print(f"Initializing Audio Separator with model dir: {model_file_dir}")
        
        try:
            # Initialize the Separator class
            separator = Separator(
                model_file_dir=model_file_dir,
                output_dir=output_dir,
                output_format=output_format,
                use_soundfile=True,
                log_level=logging.INFO
            )
            
            print("Audio Separator initialized successfully")
        except Exception as e:
            print(f"Error initializing Audio Separator: {e}")
            raise
    
    return separator

def get_output_names_by_model(model_name):
    """Generate appropriate output names based on the model."""
    timestamp = get_timestamp()
    
    if model_name == DEFAULT_UVR_MODEL:
        # For UVR model, create vocal and instrumental stem names
        return {
            "Vocals": f"vocals_{timestamp}_output",
            "Instrumental": f"instrumental_{timestamp}_output"
        }
    elif model_name == DEFAULT_DEMUCS_MODEL:
        # For htdemucs model, create names for all 6 stems
        return {
            "Vocals": f"vocals_{timestamp}_output",
            "Drums": f"drums_{timestamp}_output",
            "Bass": f"bass_{timestamp}_output",
            "Other": f"other_{timestamp}_output",
            "Guitar": f"guitar_{timestamp}_output",
            "Piano": f"piano_{timestamp}_output"
        }
    
    # Default case: return empty dict to use original naming
    return {}

def AudioSeparatorTool(
    # Input content
    audio_path: str,
    
    # Model selection
    model_name: str = DEFAULT_UVR_MODEL,
    
    # Output options
    output_dir: Optional[str] = None,
    output_format: str = "WAV"
) -> Dict[str, Any]:
    """Separate audio into different stems (vocals, instrumental, drums, bass, etc.).
    
    Args:
        audio_path: Path to the input audio file
        model_name: Model to use for separation (UVR-MDX-NET-Inst_HQ_3.onnx or htdemucs_6s.yaml)
        output_dir: Directory to save separated stems
        output_format: Format of output audio files (WAV, MP3, FLAC, etc.)
    
    Returns:
        Dictionary containing the separation results and paths to output files
    """
    try:
        # Validate input parameters
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}"
            }
        
        # Set output directory
        if output_dir is None:
            timestamp = get_timestamp()
            audio_filename = os.path.splitext(os.path.basename(audio_path))[0]
            output_dir_path = SEPARATED_DIR / f"{audio_filename}_{timestamp}"
            output_dir = str(output_dir_path)
        
        # Make sure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        sep = initialize_separator(
            output_dir=output_dir,
            output_format=output_format
        )
        
        # Load the selected model
        sep.load_model(model_filename=model_name)
        
        # Create output names based on the model
        output_names = get_output_names_by_model(model_name)
        
        print(f"Starting audio separation using model: {model_name}")
        print(f"Input file: {audio_path}")
        print(f"Output directory: {output_dir}")
        
        start_time = datetime.now()
        
        # Perform separation
        output_files = sep.separate(audio_path, output_names)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        print(f"Separation completed in {processing_time:.2f} seconds")
        print(f"Generated {len(output_files)} stems: {', '.join(os.path.basename(f) for f in output_files)}")
        
        # Return result
        return {
            "success": True,
            "model_used": model_name,
            "output_dir": output_dir,
            "output_files": output_files,
            "processing_time": processing_time
        }
    
    except Exception as e:
        import traceback
        print(f"Error in AudioSeparatorTool: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

