#!/usr/bin/env python3
"""
AudioX Processor: Multi-modal audio generation using AudioX model
Supports text-to-audio, video-to-audio, text-to-music, video-to-music generation
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from datetime import datetime
import tempfile
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import AudioX dependencies
try:
    import torch
    import torchaudio
    from einops import rearrange
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from stable_audio_tools.data.utils import read_video, merge_video_audio, load_and_process_audio
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

# Constants
AudioX_model_path = "/home/chengz/LAMs/pre_train_models/models--HKUSTAudio--AudioX"
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_BASE_DIR = PROJECT_ROOT_DIR / "output"
AUDIO_DIR = OUTPUT_BASE_DIR / "audio"
VIDEO_DIR = OUTPUT_BASE_DIR / "video"

# Create output directories
for dir_path in [AUDIO_DIR, VIDEO_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class AudioXProcessor:
    """AudioX processor for multi-modal audio generation"""
    
    def __init__(self, device_selection=None):
        """Initialize AudioX processor
        
        Args:
            device_selection: Preferred device ("cuda", "cpu", or None for auto)
        """
        self.device = self._setup_device(device_selection)
        self.model = None
        self.model_config = None
        self.sample_rate = None
        self.sample_size = None
        self.target_fps = None
        
    def _setup_device(self, device_selection):
        """Setup computing device"""
        if device_selection and device_selection.lower() != "auto":
            if device_selection.lower() == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            elif device_selection.lower() == "cpu":
                return torch.device("cpu")
            else:
                # Try to use the specific device
                try:
                    return torch.device(device_selection)
                except:
                    pass
        
        # Auto device selection
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Load AudioX model"""
        try:
            print(f"Loading AudioX model from: {AudioX_model_path}")
            print(f"Using device: {self.device}")
            
            # Load pretrained model from local path
            if os.path.exists(AudioX_model_path):
                # Load config
                config_path = os.path.join(AudioX_model_path, "config.json")
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                
                with open(config_path, 'r') as f:
                    self.model_config = json.load(f)
                
                print(f"Loaded config: {self.model_config.get('model_type', 'unknown')}")
                
                # Create model from config
                from stable_audio_tools.models.factory import create_model_from_config
                self.model = create_model_from_config(self.model_config)
                
                # Load model weights
                model_ckpt_path = os.path.join(AudioX_model_path, "model.ckpt")
                if not os.path.exists(model_ckpt_path):
                    # Try safetensors format
                    model_ckpt_path = os.path.join(AudioX_model_path, "model.safetensors")
                    if not os.path.exists(model_ckpt_path):
                        raise FileNotFoundError(f"Model checkpoint not found: {model_ckpt_path}")
                
                print(f"Loading weights from: {model_ckpt_path}")
                from stable_audio_tools.models.utils import load_ckpt_state_dict
                state_dict = load_ckpt_state_dict(model_ckpt_path)
                self.model.load_state_dict(state_dict)
                
            else:
                # Fall back to HuggingFace
                print("Local model not found, loading from HuggingFace...")
                from stable_audio_tools.models.pretrained import get_pretrained_model
                self.model, self.model_config = get_pretrained_model("HKUSTAudio/AudioX")
            
            # Move to device
            self.model = self.model.to(self.device).eval().requires_grad_(False)
            
            # Extract model configuration
            self.sample_rate = self.model_config["sample_rate"]
            self.sample_size = self.model_config["sample_size"]
            self.target_fps = self.model_config.get("video_fps", 5)
            
            print(f"Model loaded successfully!")
            print(f"Sample rate: {self.sample_rate}, Sample size: {self.sample_size}")
            print(f"Target FPS: {self.target_fps}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            return False
    
    def load_and_process_audio(self, audio_path, seconds_start, seconds_total):
        """Load and process audio input"""
        if audio_path is None or not os.path.exists(audio_path):
            # Return default silent audio
            return torch.zeros((2, int(self.sample_rate * seconds_total)))
        
        try:
            audio_tensor, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio_tensor = resampler(audio_tensor)
            
            # Extract time segment
            start_index = int(self.sample_rate * seconds_start)
            target_length = int(self.sample_rate * seconds_total)
            end_index = start_index + target_length
            
            if start_index < audio_tensor.shape[1]:
                audio_tensor = audio_tensor[:, start_index:end_index]
            else:
                audio_tensor = torch.zeros((audio_tensor.shape[0], target_length))
            
            # Pad if too short
            if audio_tensor.shape[1] < target_length:
                pad_length = target_length - audio_tensor.shape[1]
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad_length))
            
            # Ensure stereo
            if audio_tensor.shape[0] == 1:
                audio_tensor = audio_tensor.repeat(2, 1)
            elif audio_tensor.shape[0] > 2:
                audio_tensor = audio_tensor[:2, :]
            
            return audio_tensor
            
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return torch.zeros((2, int(self.sample_rate * seconds_total)))
    
    def read_video_input(self, video_path, seconds_start, seconds_total):
        """Read and process video input"""
        if video_path is None or not os.path.exists(video_path):
            # Return default blank video tensor
            # Create a blank video tensor with expected dimensions
            # Typical shape: [frames, channels, height, width]
            num_frames = int(self.target_fps * seconds_total)
            return torch.zeros((num_frames, 3, 224, 224))  # Assuming 224x224 RGB frames
        
        try:
            video_tensor = read_video(video_path, seek_time=seconds_start, 
                                    duration=seconds_total, target_fps=self.target_fps)
            return video_tensor
        except Exception as e:
            print(f"Error reading video {video_path}: {e}")
            # Return blank video tensor
            num_frames = int(self.target_fps * seconds_total)
            return torch.zeros((num_frames, 3, 224, 224))
    
    def generate_audio(self, 
                      text_prompt=None,
                      audio_path=None, 
                      video_path=None,
                      output_audio_path=None,
                      output_video_path=None,
                      seconds_start=0,
                      seconds_total=10,
                      steps=250,
                      cfg_scale=7.0,
                      sigma_min=0.3,
                      sigma_max=500.0,
                      sampler_type="dpmpp-3m-sde",
                      negative_prompt=None,
                      seed=None):
        """Generate audio using AudioX model
        
        Args:
            text_prompt: Text description for generation
            audio_path: Path to audio prompt file
            video_path: Path to video prompt file  
            output_audio_path: Custom output audio path
            output_video_path: Custom output video path
            seconds_start: Start time for conditioning
            seconds_total: Total duration
            steps: Number of inference steps
            cfg_scale: Classifier-free guidance scale
            sigma_min: Minimum sigma for sampler
            sigma_max: Maximum sigma for sampler  
            sampler_type: Sampler type
            negative_prompt: Negative prompt
            seed: Random seed
            
        Returns:
            Dictionary with results
        """
        try:
            if not self.model:
                if not self.load_model():
                    return {"success": False, "error": "Failed to load model"}
            
            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)
            
            # Generate timestamp for output files
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Set default output paths
            if output_audio_path is None:
                output_audio_path = str(AUDIO_DIR / f"audiox_audio_{timestamp}.wav")
            if output_video_path is None:
                output_video_path = str(VIDEO_DIR / f"audiox_video_{timestamp}.mp4")
            
            # Process inputs
            print("Processing inputs...")
            video_tensor = self.read_video_input(video_path, seconds_start, seconds_total)
            audio_tensor = self.load_and_process_audio(audio_path, seconds_start, seconds_total)
            
            # Move tensors to device
            video_tensor = video_tensor.to(self.device)
            audio_tensor = audio_tensor.to(self.device)
            
            print(f"Video tensor shape: {video_tensor.shape}")
            print(f"Audio tensor shape: {audio_tensor.shape}")
            
            # Prepare conditioning
            conditioning = [{
                "video_prompt": [video_tensor.unsqueeze(0)],        
                "text_prompt": text_prompt or "",
                "audio_prompt": audio_tensor.unsqueeze(0),
                "seconds_start": seconds_start,
                "seconds_total": seconds_total
            }]
            
            # Prepare negative conditioning if provided
            negative_conditioning = None
            if negative_prompt:
                negative_conditioning = [{
                    "video_prompt": [video_tensor.unsqueeze(0)],        
                    "text_prompt": negative_prompt,
                    "audio_prompt": audio_tensor.unsqueeze(0),
                    "seconds_start": seconds_start,
                    "seconds_total": seconds_total
                }]
            
            print("Starting generation...")
            
            # Generate audio
            output = generate_diffusion_cond(
                self.model,
                steps=steps,
                cfg_scale=cfg_scale,
                conditioning=conditioning,
                negative_conditioning=negative_conditioning,
                sample_size=self.sample_size,
                sigma_min=sigma_min,
                sigma_max=sigma_max,
                sampler_type=sampler_type,
                device=self.device
            )
            
            print("Generation completed!")
            
            # Rearrange audio batch to a single sequence
            output = rearrange(output, "b d n -> d (b n)")
            
            # Peak normalize, clip, convert to int16
            output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            
            # Save audio
            torchaudio.save(output_audio_path, output, self.sample_rate)
            print(f"Audio saved to: {output_audio_path}")
            
            result = {
                "success": True,
                "output_audio_path": output_audio_path,
                "parameters": {
                    "text_prompt": text_prompt,
                    "audio_path": audio_path,
                    "video_path": video_path,
                    "seconds_start": seconds_start,
                    "seconds_total": seconds_total,
                    "steps": steps,
                    "cfg_scale": cfg_scale,
                    "sigma_min": sigma_min,
                    "sigma_max": sigma_max,
                    "sampler_type": sampler_type,
                    "negative_prompt": negative_prompt,
                    "seed": seed
                }
            }
            
            # Merge video and audio if video input was provided and valid
            if video_path and os.path.exists(video_path):
                try:
                    merge_video_audio(video_path, output_audio_path, output_video_path, 
                                    seconds_start, seconds_total)
                    result["output_video_path"] = output_video_path
                    print(f"Video saved to: {output_video_path}")
                except Exception as e:
                    print(f"Warning: Could not merge video and audio: {e}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error during generation: {e}"
            print(error_msg)
            traceback.print_exc()
            return {"success": False, "error": error_msg}

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="AudioX Multi-modal Audio Generation")
    parser.add_argument("--params_file", required=True, help="JSON file with parameters")
    
    args = parser.parse_args()
    
    # Load parameters
    try:
        with open(args.params_file, 'r') as f:
            params = json.load(f)
    except Exception as e:
        result = {"success": False, "error": f"Failed to load parameters: {e}"}
        print(json.dumps(result))
        return
    
    # Initialize processor
    processor = AudioXProcessor(device_selection=params.get("device_selection"))
    
    # Extract parameters
    generate_params = {
        "text_prompt": params.get("text_prompt"),
        "audio_path": params.get("audio_path"),
        "video_path": params.get("video_path"),
        "output_audio_path": params.get("output_audio_path"),
        "output_video_path": params.get("output_video_path"),
        "seconds_start": params.get("seconds_start", 0),
        "seconds_total": params.get("seconds_total", 10),
        "steps": params.get("steps", 250),
        "cfg_scale": params.get("cfg_scale", 7.0),
        "sigma_min": params.get("sigma_min", 0.3),
        "sigma_max": params.get("sigma_max", 500.0),
        "sampler_type": params.get("sampler_type", "dpmpp-3m-sde"),
        "negative_prompt": params.get("negative_prompt"),
        "seed": params.get("seed")
    }
    
    # Generate audio
    result = processor.generate_audio(**generate_params)
    
    # Output result as JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
