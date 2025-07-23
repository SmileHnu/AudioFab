import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio
import torchaudio.transforms as T # Added for resampling

def separate_speech(audio_path, output_dir, cache_dir=None):
    """
    Separate speech sources using Look2Hear TIGER model.
    
    Args:
        audio_path (str): Path to audio file (mixture)
        output_dir (str): Directory to save separated audio files
        cache_dir (str, optional): Directory to cache downloaded model
        
    Returns:
        list: List of paths to the separated audio files
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(f"Using device: {device}")

    # Load model
    # print("Loading TIGER model...")
    # Ensure cache directory exists if specified
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    Tiger_speech = "/home/chengz/LAMs/pre_train_models/models--JusperLee--TIGER-speech"
    model = look2hear.models.TIGER.from_pretrained(Tiger_speech, cache_dir=cache_dir)
    model.to(device)
    model.eval()

    # --- Audio Loading and Preprocessing ---
    target_sr = 16000
    # print(f"Loading audio from: {audio_path}")
    try:
        waveform, original_sr = torchaudio.load(audio_path)
    except Exception as e:
        # print(f"Error loading audio file {audio_path}: {e}")
        return []

    # print(f"Original sample rate: {original_sr} Hz, Target sample rate: {target_sr} Hz")

    # Resample if necessary
    if original_sr != target_sr:
        # print(f"Resampling audio from {original_sr} Hz to {target_sr} Hz...")
        resampler = T.Resample(orig_freq=original_sr, new_freq=target_sr)
        waveform = resampler(waveform)
        # print("Resampling complete.")
        
    # Move waveform to the target device
    audio = waveform.to(device)

    # Prepare the input tensor for the model
    if audio.dim() == 1:
        audio = audio.unsqueeze(0) # Add channel dimension -> [1, T]
    audio_input = audio.unsqueeze(0).to(device)
    # print(f"Audio tensor prepared with shape: {audio_input.shape}")

    # --- Speech Separation ---
    os.makedirs(output_dir, exist_ok=True)
    # print(f"Output directory: {output_dir}")
    # print("Performing separation...")

    with torch.no_grad():
        ests_speech = model(audio_input)  # Expected output shape: [B, num_spk, T]

    # Process the estimated sources
    ests_speech = ests_speech.squeeze(0)
    num_speakers = ests_speech.shape[0]
    # print(f"Separation complete. Detected {num_speakers} potential speakers.")

    # --- Save Separated Audio ---
    output_files = []
    for i in range(num_speakers):
        output_filename = os.path.join(output_dir, f"spk{i+1}.wav")
        speaker_track = ests_speech[i].cpu()
        
        # Ensure the tensor has the correct shape [channels, time]
        if speaker_track.dim() == 1:
            speaker_track = speaker_track.unsqueeze(0)  # Add channel dimension [1, T]
        
        # print(f"Saving speaker {i+1} to {output_filename}")
        try:
            torchaudio.save(
                output_filename,
                speaker_track,
                target_sr
            )
            output_files.append(output_filename)
        except Exception as e:
            # print(f"Error saving file {output_filename}: {e}")
            pass

    return output_files

if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Separate speech sources using Look2Hear TIGER model.")
    parser.add_argument("--audio_path", default="test/mix.wav", help="Path to audio file (mixture).")
    parser.add_argument("--output_dir", default="separated_audio", help="Directory to save separated audio files.")
    parser.add_argument("--model_cache_dir", default="cache", help="Directory to cache downloaded model.")

    args = parser.parse_args()
    
    # Run speech separation
    output_files = separate_speech(
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        cache_dir=args.model_cache_dir
    )
    
    # # Print results
    # if output_files:
    #     print("\nSuccessfully separated audio files:")
    #     for file in output_files:
    #         print(f"- {file}")
    # else:
    #     print("\nNo audio files were generated.")
