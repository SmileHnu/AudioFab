import os
import sys
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Union, Literal

def WhisperASRTool(
    # Input content
    audio_path: str,
    
    # Task configuration
    task: Literal["transcribe", "translate"] = "transcribe",
    
    # Language options
    language: Optional[str] = None,  # e.g., "english", "chinese", "auto" for auto-detection
    
    # Generation parameters
    max_new_tokens: int = 440,
    num_beams: int = 1,
    temperature: Union[float, tuple] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold: float = 1.35,
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_prev_tokens: bool = False,
    
    # Timestamp options
    return_timestamps: Union[bool, str] = False,  # True for sentence-level, "word" for word-level
    
    # Batch processing
    batch_size: int = 1,
    
    # Model configuration
    model_path: str = "/home/chengz/LAMs/pre_train_models/models--openai--whisper-large-v3",
    torch_dtype: str = "float16",  # "float16" or "float32"
    low_cpu_mem_usage: bool = True,
    use_safetensors: bool = True,
    
    # Output configuration
    output_path: Optional[str] = None,
    output_format: Literal["json", "txt"] = "json",
    
    # Compute options
    device: str = "auto"  # "auto", "cuda", "cpu", or specific device like "cuda:0"
) -> Dict[str, Any]:
    """
    Automatic Speech Recognition using OpenAI Whisper large-v3 model.
    
    This tool provides high-quality speech recognition with support for multiple languages,
    translation, and timestamp generation using the Whisper large-v3 model.
    
    Args:
        audio_path: Path to the input audio file
        
        task: Task to perform
              - "transcribe": Convert speech to text in the same language
              - "translate": Translate speech to English text
              
        language: Source language of the audio (optional, auto-detected if not specified)
                 Examples: "english", "chinese", "spanish", "french", etc.
                 
        max_new_tokens: Maximum number of tokens to generate
        num_beams: Number of beams for beam search (1 for greedy decoding)
        temperature: Temperature for sampling. Can be a single float or tuple for fallback
        compression_ratio_threshold: Threshold for compression ratio in token space
        logprob_threshold: Log probability threshold for token acceptance
        no_speech_threshold: Threshold for no-speech detection
        condition_on_prev_tokens: Whether to condition on previous tokens
        
        return_timestamps: Timestamp generation mode
                          - False: No timestamps
                          - True: Sentence-level timestamps  
                          - "word": Word-level timestamps
                          
        batch_size: Batch size for processing multiple files
        
        model_path: Path to the local Whisper model
        torch_dtype: PyTorch data type ("float16" or "float32")
        low_cpu_mem_usage: Whether to use low CPU memory usage
        use_safetensors: Whether to use safetensors format
        
        output_path: Custom path to save the output
        output_format: Output format ("json" or "txt")
        
        device: Computing device ("auto", "cuda", "cpu", or specific device,(eg, "cuda:1"))
        
    Returns:
        Dictionary containing transcription results and metadata
    """
    try:
        # Import required libraries
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        from datasets import load_dataset, Audio
        
        # Device configuration
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Torch dtype configuration  
        if torch_dtype == "float16":
            torch_dtype_obj = torch.float16 if torch.cuda.is_available() else torch.float32
        else:
            torch_dtype_obj = torch.float32
            
        # Load model and processor
        print(f"Loading Whisper model from {model_path}...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_path,
            torch_dtype=torch_dtype_obj,
            low_cpu_mem_usage=low_cpu_mem_usage,
            use_safetensors=use_safetensors
        )
        model.to(device)
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Create pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype_obj,
            device=device,
        )
        
        # Prepare generation kwargs
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "condition_on_prev_tokens": condition_on_prev_tokens,
            "compression_ratio_threshold": compression_ratio_threshold,
            "temperature": temperature,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold,
            "return_timestamps": return_timestamps,
        }
        
        # Add language and task if specified
        if language:
            generate_kwargs["language"] = language
        if task:
            generate_kwargs["task"] = task
            
        # Process audio
        print(f"Processing audio file: {audio_path}")
        
        # Check if input is a single file or list of files
        if isinstance(audio_path, str):
            # Single file processing
            if not os.path.exists(audio_path):
                return {
                    "success": False,
                    "error": f"Audio file not found: {audio_path}"
                }
                
            result = pipe(audio_path, generate_kwargs=generate_kwargs, batch_size=batch_size)
        else:
            # Multiple files processing
            result = pipe(audio_path, generate_kwargs=generate_kwargs, batch_size=batch_size)
        
        # Prepare output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_dir = Path("whisper_output")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / f"whisper_result_{timestamp}.{output_format}"
        else:
            output_path = Path(output_path)
            if output_path.is_dir():
                output_path = output_path / f"whisper_result_{timestamp}.{output_format}"
        
        # Prepare result data
        result_data = {
            "success": True,
            "timestamp": timestamp,
            "task": task,
            "language": language,
            "transcription": result.get("text", ""),
            "chunks": result.get("chunks", []) if return_timestamps else [],
            "output_path": str(output_path),
        }
        
        # Save output file
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                if output_format == "json":
                    json.dump(result_data, f, ensure_ascii=False, indent=2)
                else:  # txt format
                    f.write(result_data["transcription"])
            
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save output file: {e}")
            result_data["save_warning"] = str(e)
        
        return result_data
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Whisper ASR processing failed: {str(e)}",
            "audio_path": audio_path,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Whisper ASR Tool")
    parser.add_argument("--params_file", required=True, help="Path to parameters JSON file")
    
    args = parser.parse_args()
    
    try:
        # Load parameters from file
        with open(args.params_file, 'r') as f:
            params = json.load(f)
        
        # Execute the tool
        result = WhisperASRTool(**params)
        
        # Output result as JSON
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": f"Failed to execute WhisperASRTool: {str(e)}"
        }
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()
