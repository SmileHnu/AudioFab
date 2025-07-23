import yaml
import os
import look2hear.models
import argparse
import torch
import torchaudio
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# audio path
parser = argparse.ArgumentParser()
parser.add_argument("--audio_path", default="test/test_mixture_466.wav", help="Path to audio file")

device = torch.device("cuda:7") if torch.cuda.is_available() else torch.device("cpu")
print(device)

# # Load model
model = look2hear.models.TIGERDNR.from_pretrained("JusperLee/TIGER-DnR", cache_dir="cache")
model.to(device)
model.eval()

audio = torchaudio.load(parser.parse_args().audio_path)[0].to(device)

with torch.no_grad():
    all_target_dialog, all_target_effect, all_target_music = model(audio[None])

torchaudio.save("test/test_target_dialog_466.wav", all_target_dialog.cpu(), 44100)
torchaudio.save("test/test_target_effect_466.wav", all_target_effect.cpu(), 44100)
torchaudio.save("test/test_target_music_466.wav", all_target_music.cpu(), 44100)