import os
import sys
import time
import torch
import torchaudio
import numpy as np
import random
import json
import argparse
from pathlib import Path

# Add VoiceCraft to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from data.tokenizer import AudioTokenizer, TextTokenizer, tokenize_text, tokenize_audio
from models import voicecraft
from edit_utils import get_span

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def inference_one_sample(model, model_args, phn2num, text_tokenizer, audio_tokenizer, audio_path, target_text, mask_interval, device, decode_config):
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

def main():
    parser = argparse.ArgumentParser(description="VoiceCraft Speech Editing")
    parser.add_argument("--audio_path", required=True, help="Path to input audio file")
    parser.add_argument("--edit_type", required=True, choices=["substitution", "insertion", "deletion"], help="Type of edit")
    parser.add_argument("--original_transcript", required=True, help="Original transcript")
    parser.add_argument("--target_transcript", required=True, help="Target transcript")
    parser.add_argument("--output_path", required=True, help="Output path for edited audio")
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--encodec_path", required=True, help="Path to encodec model")
    parser.add_argument("--left_margin", type=float, default=0.08, help="Left margin")
    parser.add_argument("--right_margin", type=float, default=0.08, help="Right margin")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k")
    parser.add_argument("--top_p", type=float, default=0.8, help="Top-p")
    parser.add_argument("--stop_repetition", type=int, default=2, help="Stop repetition")
    parser.add_argument("--kvcache", type=int, default=1, help="Use KV cache")
    parser.add_argument("--silence_tokens", type=str, default="[1388,1898,131]", help="Silence tokens")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    seed_everything(args.seed)
    
    device = torch.device(args.device)
    
    try:
        # Load model weights
        print(f"Loading model from {args.model_path}")
        ckpt = torch.load(args.model_path, map_location='cpu')
        model_args = ckpt['config']
        phn2num = ckpt['phn2num']
        
        model = voicecraft.VoiceCraft(model_args)
        model.load_state_dict(ckpt['model'])
        model.to(device)
        model.eval()
        
        # Initialize tokenizers
        text_tokenizer = TextTokenizer(backend="espeak")
        audio_tokenizer = AudioTokenizer(signature=args.encodec_path, device=device)
        
        # Get audio info
        audio_info = torchaudio.info(args.audio_path)
        audio_dur = audio_info.num_frames / audio_info.sample_rate
        
        # Determine edit spans
        orig_span, new_span = get_span(args.original_transcript, args.target_transcript, args.edit_type)
        
        # Handle edge cases
        if orig_span[0] > orig_span[1]:
            raise RuntimeError(f"Invalid span for {args.audio_path}")
        
        # Format spans for mask interval calculation
        if orig_span[0] == orig_span[1]:
            orig_span_save = [orig_span[0]]
        else:
            orig_span_save = orig_span
        
        orig_span_save = ",".join([str(item) for item in orig_span_save])
        
        # Create a simple word bounds list based on token indices
        words = args.original_transcript.split()
        word_bounds = []
        
        # Approximate word timings based on audio duration
        avg_word_duration = audio_dur / len(words)
        current_time = 0
        
        for word in words:
            word_end = current_time + avg_word_duration
            word_bounds.append({"word": word, "start": current_time, "end": word_end})
            current_time = word_end
        
        # Get time boundaries for editing
        if args.edit_type == "insertion":
            # For insertion, we need to find the point between words
            word_idx = int(orig_span_save.split(",")[0])
            if word_idx >= len(word_bounds):
                # Insert at the end
                start = end = word_bounds[-1]["end"]
            elif word_idx <= 0:
                # Insert at the beginning
                start = end = word_bounds[0]["start"]
            else:
                # Insert between words
                start = end = word_bounds[word_idx-1]["end"]
        else:
            # For substitution and deletion
            indices = [int(idx) for idx in orig_span_save.split(",")]
            start_idx = min(indices)
            end_idx = min(max(indices), len(word_bounds) - 1)
            start = word_bounds[start_idx]["start"]
            end = word_bounds[end_idx]["end"]
        
        # Apply margins
        codec_sr = 50  # Sample rate of the codec (50 Hz)
        morphed_span = (
            max(start - args.left_margin, 1 / codec_sr),
            min(end + args.right_margin, audio_dur)
        )  # in seconds
        
        # Convert to codec frames
        mask_interval = [
            [round(morphed_span[0] * codec_sr), round(morphed_span[1] * codec_sr)]
        ]
        mask_interval = torch.LongTensor(mask_interval)  # [M,2]
        
        # Set up decoding configuration
        decode_config = {
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "stop_repetition": args.stop_repetition,
            "kvcache": 1 if args.kvcache else 0,
            "codec_audio_sr": 16000,  # Sample rate of audio that codec expects
            "codec_sr": 50,  # Sample rate of codec codes
            "silence_tokens": eval(args.silence_tokens) if isinstance(args.silence_tokens, str) else args.silence_tokens,
        }
        
        # Perform speech editing
        print(f"Performing {args.edit_type} edit...")
        original_audio, edited_audio = inference_one_sample(
            model,
            model_args,
            phn2num,
            text_tokenizer,
            audio_tokenizer,
            args.audio_path,
            args.target_transcript,
            mask_interval,
            device,
            decode_config
        )
        
        # Save original audio resynth
        original_output_path = str(Path(args.output_path).parent / "original_audio.wav")
        original_audio = original_audio[0].cpu()
        torchaudio.save(original_output_path, original_audio, decode_config["codec_audio_sr"])
        
        # Save edited audio
        edited_audio = edited_audio[0].cpu()
        torchaudio.save(args.output_path, edited_audio, decode_config["codec_audio_sr"])
        
        # Save metadata
        metadata = {
            "edit_type": args.edit_type,
            "original_transcript": args.original_transcript,
            "target_transcript": args.target_transcript,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "left_margin": args.left_margin,
            "right_margin": args.right_margin,
            "edit_span_seconds": f"{morphed_span[0]:.2f} - {morphed_span[1]:.2f}",
        }
        
        with open(str(Path(args.output_path).parent / "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        result = {
            "success": True,
            "original_audio_path": original_output_path,
            "edited_audio_path": str(args.output_path),
            "original_transcript": args.original_transcript,
            "target_transcript": args.target_transcript,
            "edit_type": args.edit_type,
            "edit_span_seconds": f"{morphed_span[0]:.2f} - {morphed_span[1]:.2f}",
            "metadata_path": str(Path(args.output_path).parent / "metadata.json"),
        }
        
        print(json.dumps(result, indent=2))
        return 0
        
    except Exception as e:
        import traceback
        error_result = {
            "success": False,
            "error": f"Speech editing failed: {str(e)}",
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result, indent=2))
        return 1

if __name__ == "__main__":
    sys.exit(main())
