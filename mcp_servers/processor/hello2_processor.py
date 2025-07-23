import os
import sys
from pathlib import Path
import torch
import numpy as np
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
from omegaconf import OmegaConf
import shutil

# Add the models directory to the path
HALLO2_PATH = Path("/home/chengz/LAMs/pre_train_models/models--fudan-generative-ai--hallo2")
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
HALLO2_SRC_DIR = PROJECT_ROOT_DIR / "models" / "hallo2"
sys.path.append(str(HALLO2_PATH))
sys.path.append(str(HALLO2_SRC_DIR))
# sys.path.append("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2")

from mcp.server.fastmcp import FastMCP

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
VIDEO_DIR = OUTPUT_DIR / "video"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# Initialize MCP server
mcp = FastMCP("Hallo2: Audio-driven Portrait Animation")

# Global variables for model configuration
model_config = {
    "model_path": str(HALLO2_PATH),
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "weight_dtype": "fp16" if torch.cuda.is_available() else "fp32",
    # "config_path": str(Path("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/configs/inference/long.yaml"))
    "config_path": str(HALLO2_SRC_DIR / "configs" / "inference" / "long.yaml")
}

# Global variable for model instance
hallo2_model = None

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def initialize_model():
    """Initialize the Hallo2 model components if they haven't been loaded yet."""
    global hallo2_model
    
    if hallo2_model is not None:
        return hallo2_model
        
    try:
        # Import here to avoid early import errors
        import sys
        # sys.path.append('/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2')
        
        # We'll create a stub for the model initialization here
        # In a real implementation, you would load all the model components
        hallo2_model = {
            "initialized": True,
            "config": OmegaConf.load(model_config["config_path"])
        }
        

        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Hallo2 model: {str(e)}")
    
    return hallo2_model

# @mcp.tool()
def Hallo2Tool(
    # Input paths
    source_image: str,
    driving_audio: str,
    
    # Animation control parameters
    pose_weight: Optional[float] = 0.3, 
    face_weight: Optional[float] = 0.7,
    lip_weight: Optional[float] = 1.0,
    face_expand_ratio: Optional[float] = 1.5,
    
    # Output configuration
    output_path: Optional[str] = None,
    
    # Model configuration
    config_path: Optional[str] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Generate audio-driven portrait animation using the Hallo2 model.
    
    Args:
        source_image: Path to the source portrait image. The portrait should face forward.
        driving_audio: Path to the driving audio file. Supported formats: WAV.
        
        pose_weight: Weight for head pose motion. Higher values increase head movement.
        face_weight: Weight for facial expression. Higher values enhance facial expressions.
        lip_weight: Weight for lip sync accuracy. Higher values improve lip synchronization.
        face_expand_ratio: Expand ratio for the detected face region. Higher values include more context.
        
        output_path: Custom path to save the generated video file (MP4 format).
        
        config_path: Path to custom config file. If not provided, default config will be used.
        device: Computing device to use for inference ('cuda' or 'cpu').
    
    Returns:
        Dictionary containing the path to the generated video file and processing info.
    """
    try:
        # Check if required inputs are provided
        if not source_image or not os.path.exists(source_image):
            return {
                "success": False,
                "error": f"Source image not found: {source_image}"
            }
        
        if not driving_audio or not os.path.exists(driving_audio):
            return {
                "success": False,
                "error": f"Driving audio not found: {driving_audio}"
            }
        
        # Create output path and directory
        timestamp = get_timestamp()
        if output_path is None:
            output_dir = VIDEO_DIR / f"hallo2_{timestamp}"
            output_file = output_dir / "animation.mp4"
            output_path = str(output_file)
        else:
            output_dir = Path(os.path.dirname(output_path))
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use default or custom config
        if config_path is None:
            config_path = model_config["config_path"]
        elif not os.path.exists(config_path):
            return {
                "success": False,
                "error": f"Config file not found: {config_path}"
            }
        
        # Load config
        config = OmegaConf.load(config_path)
        
        # Set required parameters
        config.source_image = source_image
        config.driving_audio = driving_audio
        config.save_path = str(output_dir)
        
        # Set optional parameters if specified
        if pose_weight is not None:
            config.pose_weight = pose_weight
        if face_weight is not None:
            config.face_weight = face_weight
        if lip_weight is not None:
            config.lip_weight = lip_weight
        if face_expand_ratio is not None:
            config.face_expand_ratio = face_expand_ratio
        
        # Save the modified config
        tmp_config_path = output_dir / "config.yaml"
        OmegaConf.save(config, tmp_config_path)
        
        print(f"Starting Hallo2 animation process...")
        print(f"Source image: {source_image}")
        print(f"Driving audio: {driving_audio}")
        print(f"Output directory: {output_dir}")
        
        # Measure processing time
        start_time = datetime.now()
        
        # Import the inference module
        from scripts.inference_long import inference_process, merge_videos
        
        # Create a simple class to hold arguments
        class SimpleArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Create args object with the config path
        args = SimpleArgs(config=str(tmp_config_path))
        
        # Run the inference process
        seg_video_path = inference_process(args)
        
        # Merge the segmented videos
        final_video_path = os.path.join(Path(seg_video_path).parent, "merge_video.mp4")
        if not os.path.exists(final_video_path):
            # Try to merge videos manually
            merge_videos(seg_video_path, final_video_path)
            
            if not os.path.exists(final_video_path):
                return {
                    "success": False,
                    "error": "Failed to generate or merge animation videos"
                }
        
        # If custom output path is specified, copy the result
        if output_path != str(final_video_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy(final_video_path, output_path)
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "video_path": output_path,
            "processing_info": {
                "source_image": source_image,
                "driving_audio": driving_audio,
                "animation_parameters": {
                    "pose_weight": pose_weight,
                    "face_weight": face_weight,
                    "lip_weight": lip_weight,
                    "face_expand_ratio": face_expand_ratio
                },
                "processing_time": f"{processing_time:.2f} seconds",
                "device": device or str(model_config["device"])
            }
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "success": False,
            "error": f"Error during portrait animation: {str(e)}",
            "traceback": tb
        }

# @mcp.tool()
def Hallo2VideoEnhancementTool(
    # Input paths
    input_video: str,
    
    # Enhancement parameters
    fidelity_weight: Optional[float] = 0.5,
    upscale: Optional[int] = 2,
    
    # Output configuration
    output_path: Optional[str] = None,
    
    # Enhancement options
    bg_upsampler: Optional[str] = "realesrgan",
    face_upsample: Optional[bool] = True,
    detection_model: Optional[str] = "retinaface_resnet50",
    bg_tile: Optional[int] = 400,
    only_center_face: Optional[bool] = False
) -> Dict[str, Any]:
    """Enhance video quality with high-resolution processing using Hallo2's video enhancement module.
    
    Args:
        input_video: Path to the input video file to enhance.
        
        fidelity_weight: Balance between quality and fidelity (0-1). Lower values prioritize quality, higher values preserve fidelity.
        upscale: Upscaling factor for the image (2, 3, or 4).
        
        output_path: Custom path to save the enhanced video. If not provided, a default path will be used.
        
        bg_upsampler: Background upsampler to use. Set to "None" to disable background upsampling.
        face_upsample: Whether to apply additional upsampling to the face regions.
        detection_model: Face detection model to use.
        bg_tile: Tile size for background upsampling. Smaller values use less memory but may reduce quality.
        only_center_face: Whether to only enhance the center face in the video.
    
    Returns:
        Dictionary containing the path to the enhanced video file and processing info.
    """
    try:
        # Check if required inputs are provided
        if not input_video or not os.path.exists(input_video):
            return {
                "success": False,
                "error": f"Input video not found: {input_video}"
            }
        
        # Create output path
        if output_path is None:
            timestamp = get_timestamp()
            video_name = Path(input_video).stem
            output_dir = VIDEO_DIR / f"hallo2_enhanced_{video_name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Add model paths to sys.path
        sys.path.append(str(HALLO2_PATH))
        sys.path.append("/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2")
        
        print(f"Starting Hallo2 video enhancement process...")
        print(f"Input video: {input_video}")
        print(f"Output directory: {output_dir}")
        print(f"Enhancement parameters: fidelity={fidelity_weight}, upscale={upscale}")
        
        # Measure processing time
        start_time = datetime.now()
        
        # Create a simple class to hold arguments
        class SimpleArgs:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)
        
        # Create args object with all parameters
        args = SimpleArgs(
            input_path=input_video,
            output_path=str(output_dir),
            fidelity_weight=fidelity_weight,
            upscale=upscale,
            has_aligned=False,
            only_center_face=only_center_face,
            draw_box=False,
            detection_model=detection_model,
            bg_upsampler=bg_upsampler if bg_upsampler.lower() != "none" else "None",
            face_upsample=face_upsample,
            bg_tile=bg_tile,
            suffix=None
        )
        
        # Run video enhancement directly
        import importlib
        import torch
        
        # Import the standardized subprocess wrapper
        import sys
        sys.path.append(str(Path(__file__).parent.parent))
        from utils.subprocess_wrapper import SubprocessWrapper, MCPSubprocessTool
        
        script_path = "/home/chengz/LAMs/mcp_chatbot-audio/models/hallo2/scripts/video_sr.py"
        cmd = [
            sys.executable, 
            script_path,
            "-i", input_video,
            "-o", str(output_dir),
            "-w", str(fidelity_weight),
            "-s", str(upscale),
            "--detection_model", detection_model,
            "--bg_upsampler", bg_upsampler if bg_upsampler.lower() != "none" else "None",
            "--bg_tile", str(bg_tile)
        ]
        
        if face_upsample:
            cmd.append("--face_upsample")
        if only_center_face:
            cmd.append("--only_center_face")
            
        print(f"Running command: {' '.join(cmd)}")
        
        # Use standardized subprocess execution
        result = SubprocessWrapper.run_command(
            cmd=cmd,
            output_dir=str(output_dir),
            expected_files=["*.mp4"],
            check_output_files=True
        )
        
        if not result.success:
            # Use standardized error handling
            return MCPSubprocessTool.format_error_result(
                result, 
                "Hallo2VideoEnhancementTool",
                "Video enhancement failed"
            )
            
        # Use the first enhanced video found
        enhanced_videos = result.output_files
        if not enhanced_videos:
            return {
                "success": False,
                "error": "No enhanced video was found in the output directory after processing",
                "details": {
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "output_dir": str(output_dir)
                }
            }
        
        enhanced_video_path = enhanced_videos[0]
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        return {
            "success": True,
            "video_path": enhanced_video_path,
            "processing_info": {
                "input_video": input_video,
                "enhancement_parameters": {
                    "fidelity_weight": fidelity_weight,
                    "upscale": upscale,
                    "bg_upsampler": bg_upsampler,
                    "face_upsample": face_upsample
                },
                "processing_time": f"{processing_time:.2f} seconds"
            }
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "success": False,
            "error": f"Error during video enhancement: {str(e)}",
            "traceback": tb
        }

# if __name__ == "__main__":
#     # Run the MCP server
#     mcp.run()
