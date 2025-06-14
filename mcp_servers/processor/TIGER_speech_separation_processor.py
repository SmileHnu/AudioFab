import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

# Add the models directory to the path
sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/TIGER')
from inference_speech import separate_speech

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# @mcp.tool()
def TIGERSpeechSeparationTool(
    # Input audio
    audio_path: str,
    
    # Output configuration
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Separate speech from audio mixtures using TIGER model.
    
    Args:
        audio_path: Path to the input audio file containing mixed speech to separate.
        output_dir: Directory to save the separated speech files. Each speaker 
                    will be saved as a separate WAV file.
    
    Returns:
        Dictionary containing paths to all separated audio files
    """
    try:
        # Check if required input is provided
        if not audio_path:
            return {
                "success": False,
                "error": "Missing required parameter: audio_path"
            }
        
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}"
            }
        
        # Determine output directory
        if output_dir is None:
            output_dir = AUDIO_DIR / f"separated_{get_timestamp()}"
        else:
            output_dir = Path(output_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Start time for processing
        start_time = datetime.now()
        
        # Call the speech separation function
        output_files = separate_speech(
            audio_path=audio_path,
            output_dir=str(output_dir),
            cache_dir="/home/chengz/LAMs/pre_train_models/models--JusperLee--TIGER-speech"
        )
        
        # End time for processing
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        if not output_files:
            return {
                "success": False,
                "error": "No audio files were generated during separation"
            }
        
        return {
            "success": True,
            "message": f"Successfully separated speech using TIGER model",
            "output_files": output_files,
            "num_speakers": len(output_files),
            "processing_info": {
                "processing_time": f"{processing_time:.2f} seconds"
            }
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Error during speech separation: {str(e)}"
        }

# if __name__ == "__main__":
#     # Run the MCP server
#     mcp.run()
