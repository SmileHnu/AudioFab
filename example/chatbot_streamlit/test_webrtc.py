import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import numpy as np
import soundfile as sf
import os
import datetime

AUDIO_OUTPUT_DIR = "./gen_audio"
os.makedirs(AUDIO_OUTPUT_DIR, exist_ok=True)

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.frames = []
        self.samplerate = 48000
    def recv(self, frame):
        pcm = frame.to_ndarray()
        self.frames.append(pcm)
        self.samplerate = frame.sample_rate
        return frame

st.title("WebRTC Audio Test")
ctx = webrtc_streamer(
    key="audio_recorder_webrtc",
    mode="SENDRECV",
    audio_processor_factory=AudioProcessor,
)
if ctx and ctx.state.playing and ctx.audio_processor:
    st.info("Recording... Stop to save.")
if ctx and ctx.state == ctx.STATE_STOPPED and hasattr(ctx, "audio_processor") and ctx.audio_processor:
    proc = ctx.audio_processor
    frames = proc.frames
    if frames:
        audio_np = np.concatenate(frames, axis=1).T
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_audio_{timestamp}.wav"
        file_path = os.path.join(AUDIO_OUTPUT_DIR, filename)
        sf.write(file_path, audio_np, proc.samplerate)
        st.success(f"Recording saved as {filename}")
        st.audio(file_path)