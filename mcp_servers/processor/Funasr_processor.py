import os
import sys
from pathlib import Path
import torch
import soundfile as sf
import numpy as np
from typing import Optional, Dict, Any, List, Union, Literal
from datetime import datetime
import json

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
TEXT_DIR = OUTPUT_DIR / "text"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TEXT_DIR.mkdir(parents=True, exist_ok=True)



def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

# Global variables for models
funasr_model = None
emotion_model = None
SenseVoiceSmall = "/home/chengz/LAMs/pre_train_models/models--FunAudioLLM--SenseVoiceSmall"
emotion2vec = "/home/chengz/LAMs/pre_train_models/models--emotion2vec--emotion2vec_plus_large"

def initialize_funasr_model(
    model_name: str = SenseVoiceSmall,
    device: str = "cuda",
    model_revision: Optional[str] = None
):
    """Initialize the FunASR model if it hasn't been loaded yet."""
    global funasr_model
    
    if funasr_model is None:
        print(f"Initializing FunASR model (model: {model_name}, device: {device})...")
        
        try:
            from funasr import AutoModel
            model = AutoModel(model=model_name, model_revision=model_revision, device=device, disable_update=True)
            funasr_model = model
            print("FunASR model initialized successfully")
        except Exception as e:
            print(f"Error initializing FunASR model: {e}")
            raise
    
    return funasr_model

def initialize_emotion_model(
    model_name: str = emotion2vec,
    device: str = "cuda"
):
    """Initialize the Emotion Recognition model if it hasn't been loaded yet."""
    global emotion_model
    
    if emotion_model is None:
        print(f"Initializing Emotion Recognition model (model: {model_name}, device: {device})...")
        
        try:
            from funasr import AutoModel
            model = AutoModel(model=model_name, device=device, disable_update=True)
            emotion_model = model
            print("Emotion Recognition model initialized successfully")
        except Exception as e:
            print(f"Error initializing Emotion Recognition model: {e}")
            raise
    
    return emotion_model

# @mcp.tool()
def FunASRTool(
    # Input content
    audio_path: str,
    
    # Task configuration
    task: Literal["asr", "asr_itn", "lid", "vad", "punc", "timestamp", "streaming_asr"] = "asr",
    
    # Output configuration
    output_path: Optional[str] = None,
    output_format: Literal["json", "txt"] = "json",
    
    # Language options
    language: Literal["zh", "yue", "en", "ja", "ko", "auto"] = "auto",
    
    # Streaming parameters (for streaming_asr only)
    is_streaming: bool = False,
    chunk_size: Optional[List[int]] = None,      # Format: [0, 10, 5] - see FunASR docs
    encoder_chunk_look_back: Optional[int] = None,
    decoder_chunk_look_back: Optional[int] = None,
    
    # Text-related parameters
    text_file: Optional[str] = None,  # For timestamp task
    
    # Model configuration
    model_name: Optional[str] = None,
    model_revision: Optional[str] = None,
    device: str = "cuda"
) -> Dict[str, Any]:
    """Process audio using FunASR models for tasks like ASR, VAD, language identification, etc.
    
    Args:
        audio_path: Path to the input audio file
        
        task: The speech processing task to perform 
              - asr: Automatic Speech Recognition
              - asr_itn: ASR with Inverse Text Normalization
              - lid: Language Identification
              - vad: Voice Activity Detection
              - punc: Punctuation Restoration
              - timestamp: Timestamp Prediction
              - streaming_asr: Streaming ASR
              
        output_path: Custom path to save the output. If not provided, a default path will be used
        output_format: Format for the output (json or txt)
        
        language: Language of the audio (zh, yue, en, ja, ko, auto)
        
        is_streaming: Whether to use streaming mode
        chunk_size: Streaming configuration, e.g. [0, 10, 5]
        encoder_chunk_look_back: Number of encoder chunks to look back
        decoder_chunk_look_back: Number of decoder chunks to look back
        
        text_file: Path to text file (required for timestamp task)
        
        model_name: Custom model name to override default selection
        model_revision: Model revision to use
        device: Computing device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing processing results and metadata
    """
    try:
        # Always use SenseVoiceSmall model
        selected_model = SenseVoiceSmall
        
        # Initialize model
        model = initialize_funasr_model(model_name=selected_model, device=device, model_revision=model_revision)
        
        # Ensure output directory exists
        if output_path is None:
            timestamp = get_timestamp()
            if output_format == "json":
                output_path = str(TEXT_DIR / f"funasr_{task}_{timestamp}.json")
            else:
                output_path = str(TEXT_DIR / f"funasr_{task}_{timestamp}.txt")
        
        # Process based on task type
        if task == "timestamp" and text_file is not None:
            # For timestamp prediction which needs both audio and text
            result = model.generate(input=(audio_path, text_file), data_type=("sound", "text"))
        elif task == "streaming_asr" and is_streaming:
            # Handle streaming ASR mode
            if chunk_size is None:
                chunk_size = [0, 10, 5]  # Default 10*60=600ms chunk, 5*60=300ms lookahead
            
            if encoder_chunk_look_back is None:
                encoder_chunk_look_back = 4  # Default: number of encoder chunks to lookback
                
            if decoder_chunk_look_back is None:
                decoder_chunk_look_back = 1  # Default: number of encoder chunks to lookback for decoder
            
            # Read audio
            speech, sample_rate = sf.read(audio_path)
            chunk_stride = chunk_size[1] * 960  # 600ms
            
            # Process in streaming mode
            cache = {}
            total_chunk_num = int(len(speech - 1) / chunk_stride + 1)
            streaming_results = []
            
            for i in range(total_chunk_num):
                speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
                is_final = i == total_chunk_num - 1
                res = model.generate(
                    input=speech_chunk, 
                    cache=cache, 
                    is_final=is_final, 
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back, 
                    decoder_chunk_look_back=decoder_chunk_look_back
                )
                streaming_results.append(res)
            
            result = streaming_results
        else:
            # Standard non-streaming processing for all other tasks
            result = model.generate(input=audio_path)
        
        # Save output
        with open(output_path, 'w', encoding='utf-8') as f:
            if output_format == "json":
                json.dump({"result": result}, f, ensure_ascii=False, indent=2)
            else:
                if isinstance(result, list):
                    for item in result:
                        f.write(str(item) + '\n')
                else:
                    f.write(str(result))
        
        # Return result
        return {
            "success": True,
            "result": result,
            "output_path": output_path
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def EmotionRecognitionTool(
    # Input content
    audio_path: str,
    
    # Output configuration
    output_dir: Optional[str] = None,
    
    # Processing options
    granularity: Literal["utterance", "second"] = "utterance",
    extract_embedding: bool = False,
    
    # Model configuration
    model_name: str = "emotion2vec/emotion2vec_plus_large",
    device: str = "cuda"
) -> Dict[str, Any]:
    """Analyze emotions in speech using speech emotion recognition models.
    
    Args:
        audio_path: Path to the input audio file
        
        output_dir: Directory to save results and embeddings. If not provided, a default path will be used
        
        granularity: Level of analysis - 'utterance' for entire audio or 'second' for second-by-second analysis
        extract_embedding: Whether to extract and save emotion embeddings
        
        model_name: Name of the emotion recognition model to use
        device: Computing device to use ('cuda' or 'cpu')
        
    Returns:
        Dictionary containing detected emotions, confidence scores, and metadata
    """
    try:
        # Initialize model
        model = initialize_emotion_model(model_name=model_name, device=device)
        
        # Ensure output directory exists
        if output_dir is None:
            timestamp = get_timestamp()
            output_dir = str(OUTPUT_DIR / f"emotion_{timestamp}")
            
            # Create directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
        
        # Process audio
        result = model.generate(
            audio_path, 
            output_dir=output_dir, 
            granularity=granularity, 
            extract_embedding=extract_embedding
        )
        
        # Return result
        return {
            "success": True,
            "result": result,
            "output_dir": output_dir
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


