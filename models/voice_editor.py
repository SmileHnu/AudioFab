import torch
import torchaudio
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from dsp.dsp_utils import DSPProcessor

class VoiceEditor:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize VoiceCraft voice editor
        
        Args:
            model_path (str, optional): Path to pretrained model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.dsp = DSPProcessor()
        
        if model_path:
            self.load_model(model_path)
            
    def load_model(self, model_path: str) -> None:
        """
        Load VoiceCraft model and processor
        
        Args:
            model_path (str): Path to pretrained model
        """
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
    def edit_voice(self, 
                  audio_path: str,
                  target_text: str,
                  start_time: float,
                  end_time: float,
                  output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Edit voice segment with target text
        
        Args:
            audio_path (str): Path to input audio file
            target_text (str): Target text for the edited segment
            start_time (float): Start time in seconds
            end_time (float): End time in seconds
            output_path (str, optional): Path to save output audio
            
        Returns:
            Tuple[np.ndarray, int]: Edited audio data and sample rate
        """
        # Load and preprocess audio
        audio, sr = self.dsp.load_audio(audio_path)
        
        # Convert time to samples
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        
        # Extract segment to edit
        segment = audio[start_sample:end_sample]
        
        # Prepare inputs
        inputs = self.processor(
            audio=segment,
            text=target_text,
            sampling_rate=sr,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate edited audio
        with torch.no_grad():
            output = self.model.generate(**inputs)
            
        # Convert output to audio
        edited_segment = self.processor.batch_decode(output, skip_special_tokens=True)[0]
        edited_segment = np.array(edited_segment)
        
        # Replace original segment with edited one
        edited_audio = audio.copy()
        edited_audio[start_sample:end_sample] = edited_segment
        
        # Save if output path provided
        if output_path:
            self.dsp.save_audio(edited_audio, output_path, sr)
            
        return edited_audio, sr
        
    def batch_edit(self,
                  audio_path: str,
                  edits: List[Dict[str, Any]],
                  output_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
        """
        Apply multiple edits to audio
        
        Args:
            audio_path (str): Path to input audio file
            edits (List[Dict]): List of edit specifications
            output_path (str, optional): Path to save output audio
            
        Returns:
            Tuple[np.ndarray, int]: Edited audio data and sample rate
        """
        audio, sr = self.dsp.load_audio(audio_path)
        
        for edit in edits:
            audio, sr = self.edit_voice(
                audio_path=audio_path,
                target_text=edit["text"],
                start_time=edit["start_time"],
                end_time=edit["end_time"]
            )
            
        if output_path:
            self.dsp.save_audio(audio, output_path, sr)
            
        return audio, sr 