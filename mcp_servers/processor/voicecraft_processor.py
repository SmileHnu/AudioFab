import os
import sys
import time
import torch
import torchaudio
import numpy as np
import random
import pickle
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Literal

# 使用相对路径定位模型目录
current_dir = Path(__file__).parent
project_root = current_dir.parent
VOICECRAFT_BASE_DIR = project_root / "models" / "VoiceCraft"

# 可从环境变量获取预训练模型路径，默认值作为备选
PRETRAINED_MODEL_DIR = Path(os.environ.get(
    "VOICECRAFT_MODEL_DIR", 
    "/home/chengz/LAMs/pre_train_models/models--pyp1--VoiceCraft"
))

# Python环境路径
PYTHON_ENV_PATH = "/home/chengz/anaconda3/envs/voicecraft/bin/python"

# 脚本路径
INFERENCE_SCRIPT_PATH = VOICECRAFT_BASE_DIR / "infers_voicevraft.py"

# Create output directories
OUTPUT_DIR = Path("output")
VOICECRAFT_OUTPUT_DIR = OUTPUT_DIR
VOICECRAFT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Global variables to hold models (loaded only when needed)
voicecraft_model = None
text_tokenizer = None
audio_tokenizer = None
whisperx_model = None
phn2num = None

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def seed_everything(seed=42):
    """Set seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def initialize_model(device=None):
    """Initialize the VoiceCraft model, tokenizers, and phoneme mapping."""
    global voicecraft_model, text_tokenizer, audio_tokenizer, phn2num
    
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    
    # Load model weights
    try:
        # Model paths
        model_file = PRETRAINED_MODEL_DIR / "giga830M.pth"
        encodec_file = PRETRAINED_MODEL_DIR / "encodec_4cb2048_giga.th"
        
        # Check if files exist
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not encodec_file.exists():
            raise FileNotFoundError(f"Encodec model file not found: {encodec_file}")
        
        # Load VoiceCraft model
        ckpt = torch.load(model_file, map_location='cpu')
        model_args = ckpt['config']
        phn2num = ckpt['phn2num']
        
        voicecraft_model = voicecraft.VoiceCraft(model_args)
        voicecraft_model.load_state_dict(ckpt['model'])
        voicecraft_model.to(device)
        voicecraft_model.eval()
        
        # Initialize tokenizers
        text_tokenizer = TextTokenizer(backend="espeak")
        audio_tokenizer = AudioTokenizer(signature=str(encodec_file), device=device)
        
   
    except Exception as e:
        print(f"Error initializing VoiceCraft model: {e}")
        raise
    
    return voicecraft_model, model_args, text_tokenizer, audio_tokenizer, phn2num



def get_transcribe_state(segments):
    """Extract transcript and word boundaries from WhisperX segments."""
    words_info = [word_info for segment in segments for word_info in segment["words"]]
    return {
        "transcript": " ".join([segment["text"].strip() for segment in segments]),
        "word_bounds": [
            {"word": word["word"], "start": word["start"], "end": word["end"]}
            for word in words_info
        ],
    }

def get_mask_interval_from_word_bounds(word_bounds, word_span_ind, edit_type):
    """Calculate time boundaries for editing based on word indices."""
    tmp = word_span_ind.split(",")
    s, e = int(tmp[0]), int(tmp[-1])
    start = None
    
    for j, item in enumerate(word_bounds):
        if j == s:
            if edit_type == "insertion":
                start = float(item["end"])
            else:
                start = float(item["start"])
        if j == e:
            if edit_type == "insertion":
                end = float(item["start"])
            else:
                end = float(item["end"])
            assert start is not None
            break
    
    return (start, end)

@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_path, target_text, mask_interval, device, decode_config):
    """
    Perform inference for speech editing on a single audio sample.
    
    Args:
        model: The VoiceCraft model
        model_args: Model configuration
        phn2num: Phoneme to number mapping
        text_tokenizer: Text tokenizer
        audio_tokenizer: Audio tokenizer
        audio_path: Path to the input audio file
        target_text: Target transcript
        mask_interval: Time intervals to edit
        device: Device for inference
        decode_config: Decoding configuration
        
    Returns:
        Tuple of original and edited audio
    """
    # Phonemize text
    text_tokens = [phn2num[phn] for phn in
            tokenize_text(
                text_tokenizer, text=target_text.strip()
            ) if phn in phn2num
        ]
    text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
    text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])
    
    # Encode audio using tokenize_audio function
    encoded_frames = tokenize_audio(audio_tokenizer, audio_path)
    original_audio = encoded_frames[0][0].transpose(2, 1)  # [1,T,K]
    assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == model_args.n_codebooks, original_audio.shape
    
    # Perform inference
    start_time = time.time()
    encoded_frames = model.inference(
        text_tokens.to(device),
        text_tokens_lens.to(device),
        original_audio[...,:model_args.n_codebooks].to(device),  # [1,T,8]
        mask_interval=mask_interval.unsqueeze(0).to(device),
        top_k=decode_config['top_k'],
        top_p=decode_config['top_p'],
        temperature=decode_config['temperature'],
        stop_repetition=decode_config['stop_repetition'],
        kvcache=decode_config['kvcache'],
        silence_tokens=decode_config['silence_tokens'],
    )  # output is [1,K,T]
    
    if isinstance(encoded_frames, tuple):
        encoded_frames = encoded_frames[0]
    
    # Decode audio
    original_sample = audio_tokenizer.decode(
        [(original_audio.transpose(2, 1), None)]  # [1,T,8] -> [1,8,T]
    )
    generated_sample = audio_tokenizer.decode(
        [(encoded_frames, None)]
    )
    
    print(f"Inference completed in {time.time() - start_time:.2f} seconds")
    return original_sample, generated_sample

def VoiceCraftTool(
    # Input audio
    audio_path: str,
    
    # Editing parameters
    edit_type: Literal["substitution", "insertion", "deletion"],
    original_transcript: Optional[str] = None,
    target_transcript: str = "",
    
    # Optional parameters
    left_margin: float = 0.08,
    right_margin: float = 0.08,
    
    # Decoding parameters
    temperature: float = 1.0,
    top_k: int = -1,
    top_p: float = 0.8,
    stop_repetition: int = 2,
    kvcache: bool = True,
    silence_tokens: str = "[1388,1898,131]",
    
    # Output parameters
    output_path: Optional[str] = None,
    
    # System parameters
    device: int = 0,
    seed: int = 42
) -> Dict[str, Any]:
    """Edit speech audio by substituting, inserting, or deleting words in an English audio recording."""
    try:
        # Set up seed for reproducibility
        seed_everything(seed)
        
        # Create timestamp for output naming
        timestamp = get_timestamp()
        if output_path is None:
            output_dir = VOICECRAFT_OUTPUT_DIR 
            os.makedirs(output_dir, exist_ok=True)
            output_path = output_dir / f"edited_audio_{timestamp}.wav"
        else:
            output_dir = Path(output_path).parent
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if audio file exists
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}"
            }
        
        # Check if original transcript is provided
        if original_transcript is None or len(original_transcript.strip()) == 0:
            return {
                "success": False,
                "error": "Original transcript must be provided"
            }
        
        # Prepare the command
        cmd = [
            PYTHON_ENV_PATH,
            str(INFERENCE_SCRIPT_PATH),
            "--audio_path", audio_path,
            "--edit_type", edit_type,
            "--original_transcript", original_transcript,
            "--target_transcript", target_transcript,
            "--output_path", str(output_path),
            "--model_path", str(PRETRAINED_MODEL_DIR / "giga830M.pth"),
            "--encodec_path", str(PRETRAINED_MODEL_DIR / "encodec_4cb2048_giga.th"),
            "--left_margin", str(left_margin),
            "--right_margin", str(right_margin),
            "--temperature", str(temperature),
            "--top_k", str(top_k),
            "--top_p", str(top_p),
            "--stop_repetition", str(stop_repetition),
            "--kvcache", str(1 if kvcache else 0),
            "--silence_tokens", silence_tokens,
            "--device", f"cuda:{device}" if torch.cuda.is_available() else "cpu",
            "--seed", str(seed)
        ]
        
        # Execute the script
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        # Check the result
        if result.returncode != 0:
            # Try to extract JSON error from output
            try:
                for line in reversed(result.stdout.strip().split('\n')):
                    if line.startswith('{') and line.endswith('}'):
                        error_json = json.loads(line)
                        if not error_json.get("success", True):
                            return error_json
            except:
                pass
                
            return {
                "success": False,
                "error": f"VoiceCraft processing failed with return code {result.returncode}",
                "stderr": result.stderr
            }
        
        # Try to parse JSON output from the subprocess
        try:
            # First try to find the last JSON object in the output
            json_lines = [line for line in result.stdout.strip().split('\n') 
                         if line.strip().startswith('{') and line.strip().endswith('}')]
            if json_lines:
                result_json = json.loads(json_lines[-1])
                if "success" in result_json:
                    return result_json
            
            # If no JSON found in stdout, check if the output files exist
            if os.path.exists(output_path):
                metadata_path = os.path.join(os.path.dirname(output_path), "metadata.json")
                original_path = os.path.join(os.path.dirname(output_path), "original_audio.wav")
                
                # Try to read metadata if it exists
                metadata = {}
                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    except:
                        pass
                
                return {
                    "success": True,
                    "original_audio_path": original_path,
                    "edited_audio_path": str(output_path),
                    "original_transcript": original_transcript,
                    "target_transcript": target_transcript,
                    "edit_type": edit_type,
                    "metadata_path": metadata_path if os.path.exists(metadata_path) else None,
                    "metadata": metadata
                }
            
            # If we get here, something went wrong
            return {
                "success": False,
                "error": "Could not parse VoiceCraft output",
                "stderr": result.stderr
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Error parsing VoiceCraft output: {str(e)}",
                "stderr": result.stderr
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Speech editing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }

# If called directly, can be used to test functionality
if __name__ == "__main__":
    # Support command line arguments to run the tool directly
    import argparse
    
    parser = argparse.ArgumentParser(description="VoiceCraft Speech Editing Tool")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file")
    parser.add_argument("--edit_type", required=True, choices=["substitution", "insertion", "deletion"], 
                        help="Type of edit to perform")
    parser.add_argument("--original_transcript", help="Original transcript of the audio")
    parser.add_argument("--target_transcript", required=True, help="Target transcript after editing")
    parser.add_argument("--output_path", help="Path to save the edited audio")
    
    args = parser.parse_args()
    
    result = VoiceCraftTool(
        audio_path=args.audio_path,
        edit_type=args.edit_type,
        original_transcript=args.original_transcript,
        target_transcript=args.target_transcript,
        output_path=args.output_path
    )
    
    print(json.dumps(result, indent=2))
