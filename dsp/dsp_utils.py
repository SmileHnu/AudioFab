import librosa
import numpy as np
import soundfile as sf
from scipy import signal
import torch
import torchaudio
import matplotlib.pyplot as plt
from typing import List, Tuple, Union, Optional, Callable
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DSPProcessor")

class DSPProcessor:
    def __init__(self, default_sr: int = 16000, fallback_mode: str = "auto"):
        """
        Initialize DSP processor with default sample rate
        
        Args:
            default_sr (int): Default sample rate for audio processing
            fallback_mode (str): Fallback mode for audio processing
                - "auto": Try librosa first, fallback to torchaudio, then soundfile
                - "librosa": Only use librosa
                - "torchaudio": Only use torchaudio
                - "soundfile": Only use soundfile
        """
        self.sample_rate = default_sr
        self.fallback_mode = fallback_mode
        
    def _with_fallback(self, primary_func: Callable, fallback_funcs: List[Callable], *args, **kwargs):
        """
        Generic fallback mechanism
        
        Args:
            primary_func (Callable): Primary function to try
            fallback_funcs (List[Callable]): List of fallback functions
            *args, **kwargs: Arguments to pass to the functions
            
        Returns:
            Result of the first successful function call
        """
        if self.fallback_mode == "auto":
            # Try primary function first
            try:
                return primary_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Primary function failed: {str(e)}")
                
                # Try fallback functions
                for i, func in enumerate(fallback_funcs):
                    try:
                        logger.info(f"Trying fallback function {i+1}")
                        return func(*args, **kwargs)
                    except Exception as e:
                        logger.warning(f"Fallback function {i+1} failed: {str(e)}")
                
                # If all fallbacks fail, raise the original exception
                raise
        else:
            # Use the specified mode
            if self.fallback_mode == "librosa":
                return primary_func(*args, **kwargs)
            elif self.fallback_mode == "torchaudio" and len(fallback_funcs) >= 1:
                return fallback_funcs[0](*args, **kwargs)
            elif self.fallback_mode == "soundfile" and len(fallback_funcs) >= 2:
                return fallback_funcs[1](*args, **kwargs)
            else:
                return primary_func(*args, **kwargs)
        
    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Load audio file and optionally resample
        
        Args:
            file_path (str): Path to audio file
            target_sr (int, optional): Target sample rate
            
        Returns:
            Tuple[np.ndarray, int]: Audio data and sample rate
        """
        # Librosa implementation
        def _load_with_librosa():
            return librosa.load(file_path, sr=target_sr or self.sample_rate)
        
        # Torchaudio implementation
        def _load_with_torchaudio():
            waveform, sr = torchaudio.load(file_path)
            waveform = waveform.numpy().mean(axis=0)  # Convert to mono
            if target_sr and sr != target_sr:
                waveform = torchaudio.functional.resample(
                    torch.from_numpy(waveform), sr, target_sr
                ).numpy()
                sr = target_sr
            return waveform, sr
        
        # Soundfile implementation
        def _load_with_soundfile():
            waveform, sr = sf.read(file_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)  # Convert to mono
            if target_sr and sr != target_sr:
                # Use scipy for resampling
                waveform = signal.resample_poly(waveform, target_sr, sr)
                sr = target_sr
            return waveform, sr
            
        return self._with_fallback(
            _load_with_librosa,
            [_load_with_torchaudio, _load_with_soundfile]
        )
        
    def resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio (np.ndarray): Input audio data
            orig_sr (int): Original sample rate
            target_sr (int): Target sample rate
            
        Returns:
            np.ndarray: Resampled audio data
        """
        # Librosa implementation
        def _resample_with_librosa():
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        
        # Torchaudio implementation
        def _resample_with_torchaudio():
            audio_tensor = torch.from_numpy(audio)
            resampled = torchaudio.functional.resample(audio_tensor, orig_sr, target_sr)
            return resampled.numpy()
        
        # Scipy implementation
        def _resample_with_scipy():
            return signal.resample_poly(audio, target_sr, orig_sr)
            
        return self._with_fallback(
            _resample_with_librosa,
            [_resample_with_torchaudio, _resample_with_scipy]
        )
        
    def stft(self, audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """
        Compute Short-time Fourier transform
        
        Args:
            audio (np.ndarray): Input audio data
            n_fft (int): FFT window size
            hop_length (int): Number of samples between successive frames
            
        Returns:
            np.ndarray: STFT matrix
        """
        # Librosa implementation
        def _stft_with_librosa():
            return librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        
        # Torchaudio implementation
        def _stft_with_torchaudio():
            audio_tensor = torch.from_numpy(audio)
            stft_matrix = torch.stft(
                audio_tensor, 
                n_fft=n_fft, 
                hop_length=hop_length,
                return_complex=True
            )
            return stft_matrix.numpy()
        
        # Scipy implementation
        def _stft_with_scipy():
            # Convert to complex format compatible with librosa
            _, _, Zxx = signal.stft(
                audio, 
                fs=self.sample_rate, 
                nperseg=n_fft, 
                noverlap=n_fft-hop_length
            )
            return Zxx
            
        return self._with_fallback(
            _stft_with_librosa,
            [_stft_with_torchaudio, _stft_with_scipy]
        )
        
    def istft(self, stft_matrix: np.ndarray, hop_length: int = 512) -> np.ndarray:
        """
        Inverse Short-time Fourier transform
        
        Args:
            stft_matrix (np.ndarray): STFT matrix
            hop_length (int): Number of samples between successive frames
            
        Returns:
            np.ndarray: Reconstructed audio signal
        """
        # Librosa implementation
        def _istft_with_librosa():
            return librosa.istft(stft_matrix, hop_length=hop_length)
        
        # Torchaudio implementation
        def _istft_with_torchaudio():
            stft_tensor = torch.from_numpy(stft_matrix)
            if not torch.is_complex(stft_tensor):
                stft_tensor = torch.complex(stft_tensor.real, stft_tensor.imag)
            reconstructed = torch.istft(
                stft_tensor, 
                n_fft=2*(stft_matrix.shape[0]-1),  # Derive n_fft from matrix shape
                hop_length=hop_length
            )
            return reconstructed.numpy()
        
        # Scipy implementation
        def _istft_with_scipy():
            # Assuming stft_matrix is in the format returned by scipy.signal.stft
            n_fft = 2*(stft_matrix.shape[0]-1)  # Derive n_fft from matrix shape
            _, reconstructed = signal.istft(
                stft_matrix, 
                fs=self.sample_rate, 
                nperseg=n_fft, 
                noverlap=n_fft-hop_length
            )
            return reconstructed
            
        return self._with_fallback(
            _istft_with_librosa,
            [_istft_with_torchaudio, _istft_with_scipy]
        )
        
    def mfcc(self, audio: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio (np.ndarray): Input audio data
            n_mfcc (int): Number of MFCC coefficients
            
        Returns:
            np.ndarray: MFCC features
        """
        # Librosa implementation
        def _mfcc_with_librosa():
            return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        
        # Torchaudio implementation
        def _mfcc_with_torchaudio():
            waveform = torch.from_numpy(audio).unsqueeze(0)  # Add batch dimension
            mfcc_transform = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=n_mfcc
            )
            mfcc_features = mfcc_transform(waveform)
            return mfcc_features.squeeze(0).numpy()  # Remove batch dimension
            
        return self._with_fallback(
            _mfcc_with_librosa,
            [_mfcc_with_torchaudio]
        )
        
    def mel_spectrogram(self, audio: np.ndarray, n_mels: int = 128) -> np.ndarray:
        """
        Generate mel spectrogram
        
        Args:
            audio (np.ndarray): Input audio data
            n_mels (int): Number of mel bands
            
        Returns:
            np.ndarray: Mel spectrogram
        """
        # Librosa implementation
        def _melspec_with_librosa():
            return librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=n_mels)
        
        # Torchaudio implementation
        def _melspec_with_torchaudio():
            waveform = torch.from_numpy(audio).unsqueeze(0)  # Add batch dimension
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=n_mels
            )
            mel_spec = mel_transform(waveform)
            return mel_spec.squeeze(0).numpy()  # Remove batch dimension
            
        return self._with_fallback(
            _melspec_with_librosa,
            [_melspec_with_torchaudio]
        )
        
    def plot_mel_spectrogram(self, mel_spec: np.ndarray, title: str = "Mel Spectrogram", sr: int = None, save_path: str = None, dpi: int = 300) -> plt.Figure:
        """
        Plot mel spectrogram
        
        Args:
            mel_spec (np.ndarray): Mel spectrogram data
            title (str): Plot title
            sr (int, optional): Sample rate for axis scaling
            save_path (str, optional): Path to save the figure
            dpi (int): DPI for saved figure
            
        Returns:
            Figure object
        """
        plt.figure(figsize=(10, 4))
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(
            mel_spec_db,
            y_axis="mel", 
            x_axis="time",
            sr=sr
        )
        plt.title(title)
        plt.colorbar(format="%+2.0f dB")
        plt.tight_layout()
        
        # 如果提供了保存路径，则保存图像
        if save_path is not None:
            plt.savefig(save_path, dpi=dpi)
            
        fig = plt.gcf()
        if save_path is not None:
            plt.close()
        return fig
        
    def get_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch using librosa's pitch tracking
        
        Args:
            audio (np.ndarray): Input audio data
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Pitches and magnitudes
        """
        # Librosa implementation
        def _pitch_with_librosa():
            return librosa.piptrack(y=audio, sr=self.sample_rate)
        
        # Simple frequency domain alternative 
        def _pitch_with_freq_domain():
            D = np.abs(librosa.stft(audio))
            pitches = np.zeros_like(D)
            magnitudes = D
            for i in range(D.shape[1]):
                index = np.argmax(D[:, i])
                pitches[index, i] = librosa.fft_frequencies(sr=self.sample_rate)[index]
            return pitches, magnitudes
            
        return self._with_fallback(
            _pitch_with_librosa,
            [_pitch_with_freq_domain]
        )
        
    def add_reverb(self, audio: np.ndarray, room_scale: float = 0.8) -> np.ndarray:
        """
        Add reverb effect using convolution
        
        Args:
            audio (np.ndarray): Input audio data
            room_scale (float): Room scale factor (0-1)
            
        Returns:
            np.ndarray: Audio with reverb effect
        """
        # Generate impulse response
        impulse_response = np.exp(-np.arange(room_scale * self.sample_rate) / (room_scale * self.sample_rate))
        impulse_response = impulse_response / np.sum(impulse_response)
        
        # Apply convolution
        return signal.convolve(audio, impulse_response, mode='full')
        
    def mix_audio(self, audio_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Mix multiple audio tracks
        
        Args:
            audio_list (List[np.ndarray]): List of audio data arrays
            weights (List[float], optional): Mixing weights
            
        Returns:
            np.ndarray: Mixed audio data
        """
        if weights is None:
            weights = [1.0] * len(audio_list)
        return np.average(audio_list, weights=weights, axis=0)
        
    def get_audio_duration(self, audio: np.ndarray) -> float:
        """
        Get audio duration in seconds
        
        Args:
            audio (np.ndarray): Input audio data
            
        Returns:
            float: Duration in seconds
        """
        return len(audio) / self.sample_rate
        
    def save_audio(self, audio: np.ndarray, file_path: str, sr: Optional[int] = None) -> None:
        """
        Save audio to file
        
        Args:
            audio (np.ndarray): Audio data to save
            file_path (str): Output file path
            sr (int, optional): Sample rate
        """
        # Soundfile implementation
        def _save_with_soundfile():
            sf.write(file_path, audio, sr or self.sample_rate)
            
        # Scipy implementation
        def _save_with_scipy():
            from scipy.io import wavfile
            wavfile.write(file_path, sr or self.sample_rate, audio.astype(np.float32))
            
        # Torchaudio implementation
        def _save_with_torchaudio():
            tensor = torch.from_numpy(audio)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)  # Add channel dimension
            torchaudio.save(file_path, tensor, sr or self.sample_rate)
            
        return self._with_fallback(
            _save_with_soundfile,
            [_save_with_torchaudio, _save_with_scipy]
        )
        
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range
        
        Args:
            audio (np.ndarray): Input audio data
            
        Returns:
            np.ndarray: Normalized audio data
        """
        return librosa.util.normalize(audio)
        
    def apply_fade(self, audio: np.ndarray, fade_duration: float) -> np.ndarray:
        """
        Apply fade in/out effect
        
        Args:
            audio (np.ndarray): Input audio data
            fade_duration (float): Fade duration in seconds
            
        Returns:
            np.ndarray: Audio with fade effect
        """
        fade_length = int(fade_duration * self.sample_rate)
        fade_in = np.linspace(0, 1, fade_length)
        fade_out = np.linspace(1, 0, fade_length)
        
        audio_copy = audio.copy()
        audio_copy[:fade_length] *= fade_in
        audio_copy[-fade_length:] *= fade_out
        return audio_copy
        
    def set_fallback_mode(self, mode: str) -> None:
        """
        Set the fallback mode for audio processing
        
        Args:
            mode (str): Fallback mode
                - "auto": Try librosa first, fallback to torchaudio, then soundfile
                - "librosa": Only use librosa
                - "torchaudio": Only use torchaudio
                - "soundfile": Only use soundfile
        """
        valid_modes = ["auto", "librosa", "torchaudio", "soundfile"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid fallback mode: {mode}. Valid modes: {valid_modes}")
        self.fallback_mode = mode
        logger.info(f"Fallback mode set to: {mode}") 