import os
import sys
import logging
from pathlib import Path
import torch
import numpy as np
import soundfile as sf
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import json
import time
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
TEXT_DIR = OUTPUT_DIR / "text"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)

# Model paths
QWEN2_AUDIO_PATH = "/home/chengz/LAMs/pre_train_models/models--Qwen--Qwen2-Audio-7B-Instruct"

# Global variables for models
qwen2_audio_model = None

def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def initialize_qwen2_audio_model(
    model_path: str = QWEN2_AUDIO_PATH,
    device: str = "cuda"
):
    """Initialize the Qwen2-Audio model if it hasn't been loaded yet."""
    global qwen2_audio_model
    if qwen2_audio_model is None:
        print(f"Initializing Qwen2-Audio model (model: {model_path}, device: {device})...")
        try:
            print(f"Loading Qwen2-Audio model from {model_path}")
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = Qwen2AudioForConditionalGeneration.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            qwen2_audio_model = {
                "processor": processor,
                "model": model
            }
            print("Qwen2-Audio model loaded successfully")
        except Exception as e:
            print(f"Error initializing Qwen2-Audio model: {e}")
            import traceback
            traceback.print_exc()
            raise
    return qwen2_audio_model

def process_audio_input(audio_path: str, processor) -> np.ndarray:
    """Process audio file into the format expected by the model."""
    try:
        # Load audio using librosa with explicit sampling rate
        sampling_rate = processor.feature_extractor.sampling_rate
        audio, sr = librosa.load(audio_path, sr=sampling_rate)
        return audio
    except Exception as e:
        print(f"Error processing audio file {audio_path}: {e}")
        raise

def Qwen2AudioTool(
    # Task selection (expanded with AIR-Bench capabilities)
    task: Literal[
        # Basic tasks
        "transcribe", "chat", "evaluate", 
        # AIR-Bench speech tasks
        "speech_grounding", "language_identification", "speaker_gender", 
        "emotion_recognition", "speaker_age", "speech_entity", 
        "intent_classification", "speaker_verification", "synthesized_voice_detection",
        # AIR-Bench audio tasks
        "audio_grounding", "vocal_classification", "acoustic_scene", 
        "sound_qa", "music_instruments", "music_genre", 
        "music_note_pitch", "music_note_velocity", "music_qa", "music_emotion"
    ] = "chat",
    
    # Input content
    audio_path: str = None,
    text: Optional[str] = None,
    reference_audio_path: Optional[str] = None,
    
    # Task-specific parameters
    prompt: str = "",
    evaluation_criteria: str = "",
    evaluation_prompt_name: str = "",
    target_language: Optional[str] = None,
    
    # Generation parameters
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 512,
    do_sample: bool = True,
    
    # Output options
    output_path: Optional[str] = None,
    output_format: Literal["json", "txt"] = "json",
    
    # Compute options
    device: str = "cuda"
) -> Dict[str, Any]:
    """Process audio with Qwen2-Audio model for various audio understanding tasks.
    
    Args:
        task: Task to perform 
            # Basic tasks
            - "transcribe": Convert speech to text
            - "chat": General audio processing with optional text prompt
            - "evaluate": Evaluate audio quality
            
            # speech tasks
            - "speech_grounding": Identify specific elements mentioned in speech
            - "language_identification": Identify the language being spoken
            - "speaker_gender": Identify speaker gender
            - "emotion_recognition": Analyze emotions in speech
            - "speaker_age": Estimate speaker age
            - "speech_entity": Extract entities from speech
            - "intent_classification": Classify speaker intent
            - "speaker_verification": Verify if two recordings are from same speaker
            - "synthesized_voice_detection": Detect if voice is synthesized
            
            # audio tasks
            - "audio_grounding": Identify specific sounds in audio
            - "vocal_classification": Classify vocal sounds
            - "acoustic_scene": Classify acoustic scenes/environments
            - "sound_qa": Question answering about sounds
            - "music_instruments": Identify musical instruments
            - "music_genre": Identify music genre
            - "music_note_pitch": Analyze musical note pitch
            - "music_note_velocity": Analyze musical note velocity
            - "music_qa": Question answering about music
            - "music_emotion": Detect emotion in music
            
        audio_path: Path to input audio file (required for all tasks)
        text: Text input for additional context or specific instructions
        reference_audio_path: Path to reference audio for comparison
        
        prompt: Text prompt to guide the model's response or specify task details
        evaluation_criteria: Custom criteria for audio evaluation
        evaluation_prompt_name: Name of predefined evaluation prompt to use
        target_language: Target language for translation
        
        temperature: Controls randomness in generation (higher = more random)
        top_p: Nucleus sampling parameter
        max_new_tokens: Maximum number of tokens to generate in the response
        do_sample: Whether to use sampling in generation
        
        output_path: Custom path to save output
        output_format: Format for saving output (json or txt)
        device: Computing device for inference ("cuda" or "cpu")
        
    Returns:
        Dictionary containing the results of the requested task
    """
    try:
        # Validate audio path
        if audio_path is None:
            return {
                "success": False,
                "error": f"audio_path is required for {task} task"
            }
            
        if not os.path.exists(audio_path):
            return {
                "success": False,
                "error": f"Audio file not found: {audio_path}"
            }
        
        # Initialize model
        model_components = initialize_qwen2_audio_model(device=device)
        processor = model_components["processor"]
        model = model_components["model"]
        
        # Set default output path if not provided
        if output_path is None:
            timestamp = get_timestamp()
            if output_format == "json":
                output_path = str(TEXT_DIR / f"qwen2audio_{task}_{timestamp}.json")
            else:
                output_path = str(TEXT_DIR / f"qwen2audio_{task}_{timestamp}.txt")
        
        # Process based on task type
        start_time = time.time()
        
        # Prepare conversation format
        conversation = [
            {'role': 'system', 'content': 'You are a helpful assistant.'}
        ]
        
        # Add user message with audio and text
        user_content = []
        if audio_path:
            audio = process_audio_input(audio_path, processor)
            user_content.append({"type": "audio", "audio": audio})
        
        if prompt:
            user_content.append({"type": "text", "text": prompt})
        elif task == "transcribe":
            user_content.append({"type": "text", "text": f"Fist,please transcribe the audio accurately. then, {prompt}"})
        else:
            user_content.append({"type": "text", "text": f"What's in this audio? and {prompt}"})
            
        conversation.append({"role": "user", "content": user_content})
        
        # Process conversation
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(ele["audio"])
        
        # Prepare model inputs and move to correct device
        inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
        # Move all input tensors to the specified device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            generate_ids = model.generate(
                **inputs,
                max_length=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample
            )
            # Get the length of input_ids from the dictionary
            input_ids_length = inputs["input_ids"].size(1)
            generate_ids = generate_ids[:, input_ids_length:]
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Prepare result
        result = {
            "success": True,
            "response": response,
            "processing_time": time.time() - start_time,
            "task": task
        }
        
        # Save output to file
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_format == "json":
                json.dump(result, f, ensure_ascii=False, indent=2)
            else:
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key != "success" and key != "processing_time":
                            f.write(f"{key}: {value}\n")
                else:
                    f.write(str(result))
        
        result["output_path"] = output_path
        return result
        
    except Exception as e:
        import traceback
        print(f"Error in Qwen2AudioTool: {e}")
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e)
        }

# Example usage if run directly
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio with Qwen2-Audio model")
    parser.add_argument("--task", default="chat", help="Task to perform")
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--prompt", default="", help="Prompt to guide the model")
    parser.add_argument("--reference", default=None, help="Path to reference audio (for verification tasks)")
    
    args = parser.parse_args()
    
    result = Qwen2AudioTool(
        task=args.task,
        audio_path=args.audio,
        prompt=args.prompt,
        reference_audio_path=args.reference
    )
    
    print(json.dumps(result, indent=2, ensure_ascii=False)) 