import os
import sys
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any, List
from datetime import datetime

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Add CosyVoice path to system path
# Ensure this path is correct for your environment
COSYVOICE_BASE_PATH = os.path.abspath("models/CosyVoice") 
if COSYVOICE_BASE_PATH not in sys.path:
    sys.path.append(COSYVOICE_BASE_PATH)

MATCHA_TTS_PATH = os.path.join(COSYVOICE_BASE_PATH, 'third_party/Matcha-TTS')
if MATCHA_TTS_PATH not in sys.path:
    sys.path.append(MATCHA_TTS_PATH)

from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
cosyvoice2path = "/home/chengz/LAMs/pre_train_models/CosyVoice2-0.5B"


class CosyVoice2Manager:
    """Tool for text-to-speech synthesis using CosyVoice2 model."""
    
    def __init__(self):
        self.model = None
        self.model_dir = cosyvoice2path
        self.loaded = False
        
    def _ensure_model_loaded(self, model_fp16=False, model_jit=False, model_trt=False, use_flow_cache=False):
        """Ensure the model is loaded with the specified parameters."""
        if not self.loaded or (
            self.loaded_config.get('model_fp16') != model_fp16 or
            self.loaded_config.get('model_jit') != model_jit or
            self.loaded_config.get('model_trt') != model_trt or
            self.loaded_config.get('use_flow_cache') != use_flow_cache
        ):
            # Load model with specified parameters
            print(f"Loading CosyVoice2 model from {self.model_dir}...")
            self.model = CosyVoice2(
                model_dir=self.model_dir,
                load_jit=model_jit,
                load_trt=model_trt,
                fp16=model_fp16,
                use_flow_cache=use_flow_cache
            )
            self.loaded = True
            self.loaded_config = {
                'model_fp16': model_fp16,
                'model_jit': model_jit,
                'model_trt': model_trt,
                'use_flow_cache': use_flow_cache
            }
            print("CosyVoice2 model loaded successfully.")
    
    def synthesize(
        self,
        # Core inputs
        text: Optional[str] = None,
        source_audio_path: Optional[str] = None,
        
        # Voice reference/control
        prompt_audio_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        speaker_id: Optional[str] = None,
        zero_shot_speaker_id: Optional[str] = None,
        
        # Mode selectors
        cross_lingual_synthesis: bool = False,
        use_instruct_mode: bool = False,
        instruct_text: Optional[str] = None,
        
        # Advanced parameters
        speed: float = 1.0,
        stream_output: bool = False,
        use_text_frontend: bool = True,
        
        # Output & System Configuration
        language_tag: Optional[str] = None,
        output_path: Optional[str] = None,
        device_hint: str = "cuda",
        
        # Model loading options
        model_fp16: bool = False,
        model_jit: bool = False,
        model_trt: bool = False,
        use_flow_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Synthesize speech using the CosyVoice2 model with various capabilities.
        
        Returns:
            Dictionary containing generation results, metadata, and status.
        """
        # Set device
        device = torch.device(device_hint if torch.cuda.is_available() else "cpu")
        if device.type == "cpu" and (model_fp16 or model_jit or model_trt):
            print("Warning: FP16/JIT/TRT optimizations require CUDA. Falling back to CPU mode.")
            model_fp16, model_jit, model_trt = False, False, False
        
        # Ensure model is loaded with correct parameters
        self._ensure_model_loaded(model_fp16, model_jit, model_trt, use_flow_cache)
        
        # Prepare output path
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"cosyvoice2_{timestamp}.wav"
            output_path = str(AUDIO_DIR / output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Prepend language tag to text if provided
        if language_tag and text and not text.startswith(language_tag):
            text = f"{language_tag}{text}"
        
        try:
            # Load audio files if provided
            prompt_speech_16k = None
            if prompt_audio_path:
                prompt_speech_16k = load_wav(prompt_audio_path, 16000)
                
            source_speech_16k = None
            if source_audio_path:
                source_speech_16k = load_wav(source_audio_path, 16000)
            
            # Determine synthesis mode and call appropriate method
            audio_chunks = []
            
            # Voice Conversion mode
            if source_audio_path and prompt_audio_path:
                print("Performing Voice Conversion...")
                for output in self.model.inference_vc(
                    source_speech_16k=source_speech_16k,
                    prompt_speech_16k=prompt_speech_16k,
                    stream=stream_output,
                    speed=speed
                ):
                    audio_chunks.append(output['tts_speech'][0].cpu().numpy())
            
            # Instructed Voice Generation mode
            elif use_instruct_mode and text and instruct_text:
                if not prompt_audio_path:
                    return {
                        "success": False,
                        "error": "prompt_audio_path is required for instructed voice generation"
                    }
                print(f"Performing Instructed Voice Generation with instruction: {instruct_text}")
                for output in self.model.inference_instruct2(
                    tts_text=text,
                    instruct_text=instruct_text,
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id=zero_shot_speaker_id or '',
                    stream=stream_output,
                    speed=speed,
                    text_frontend=use_text_frontend
                ):
                    audio_chunks.append(output['tts_speech'][0].cpu().numpy())
            
            # Cross-lingual synthesis mode
            elif cross_lingual_synthesis and text and prompt_audio_path:
                print("Performing Cross-lingual Synthesis...")
                for output in self.model.inference_cross_lingual(
                    tts_text=text,
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id=zero_shot_speaker_id or '',
                    stream=stream_output,
                    speed=speed,
                    text_frontend=use_text_frontend
                ):
                    audio_chunks.append(output['tts_speech'][0].cpu().numpy())
            
            # Zero-shot synthesis mode
            elif text and prompt_audio_path and prompt_text:
                print("Performing Zero-shot In-context Generation...")
                for output in self.model.inference_zero_shot(
                    tts_text=text,
                    prompt_text=prompt_text,
                    prompt_speech_16k=prompt_speech_16k,
                    zero_shot_spk_id=zero_shot_speaker_id or '',
                    stream=stream_output,
                    speed=speed,
                    text_frontend=use_text_frontend
                ):
                    audio_chunks.append(output['tts_speech'][0].cpu().numpy())
            
            else:
                return {
                    "success": False,
                    "error": "Invalid combination of parameters. Please check documentation for supported modes."
                }
            
            # Concatenate audio chunks and save to file
            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)
                sf.write(output_path, audio_data, self.model.sample_rate)
                
                return {
                    "success": True,
                    "output_path": output_path,
                    "sample_rate": self.model.sample_rate,
                    "duration": len(audio_data) / self.model.sample_rate,
                    "text": text,
                    "metadata": {
                        "speed": speed,
                    }
                }
            else:
                return {
                    "success": False,
                    "error": "No audio generated."
                }
                
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                "success": False,
                "error": str(e),
                "error_details": error_details
            }

# Create an instance of the tool
cosyvoice2_manager = CosyVoice2Manager()

# Function to be called by the MCP tool launcher
def CosyVoice2Tool(
    # Core inputs
    text: Optional[str] = None,
    source_audio_path: Optional[str] = None,
    
    # Voice reference/control
    prompt_audio_path: Optional[str] = None,
    prompt_text: Optional[str] = None,
    speaker_id: Optional[str] = None,
    zero_shot_speaker_id: Optional[str] = None,
    
    # Mode selectors / high-level instructions
    cross_lingual_synthesis: bool = False,
    use_instruct_mode: bool = False,
    instruct_text: Optional[str] = None,
    
    # Advanced/Optional parameters
    speed: float = 1.0,
    stream_output: bool = False,
    use_text_frontend: bool = True,
    
    # Output & System Configuration
    language_tag: Optional[str] = None,
    output_path: Optional[str] = None,
    device_hint: str = "cuda",
    
    # Model loading options
    model_fp16: bool = False,
    model_jit: bool = False,
    model_trt: bool = False,
    use_flow_cache: bool = False
):
    """Process function to be called by the tool launcher."""
    return cosyvoice2_manager.synthesize(
        text=text,
        source_audio_path=source_audio_path,
        prompt_audio_path=prompt_audio_path,
        prompt_text=prompt_text,
        speaker_id=speaker_id,
        zero_shot_speaker_id=zero_shot_speaker_id,
        cross_lingual_synthesis=cross_lingual_synthesis,
        use_instruct_mode=use_instruct_mode,
        instruct_text=instruct_text,
        speed=speed,
        stream_output=stream_output,
        use_text_frontend=use_text_frontend,
        language_tag=language_tag,
        output_path=output_path,
        device_hint=device_hint,
        model_fp16=model_fp16,
        model_jit=model_jit,
        model_trt=model_trt,
        use_flow_cache=use_flow_cache
    )