import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from abc import ABC, abstractmethod
import torch

# Create output directories
OUTPUT_DIR = Path("output")
AUDIO_DIR = OUTPUT_DIR / "audio"
AUDIO_RESULTS_DIR = AUDIO_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

class LSLMBase(ABC):
    """Base class for Large Speech Language Models (LSLMs).
    
    This class provides common functionality for audio processing models
    that can handle tasks like speech recognition, audio understanding,
    audio-to-text chat, and speech generation/conversation.
    """
    
    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        log_level: int = logging.INFO,
        supports_audio_output: bool = False,
        supports_evaluation: bool = False
    ):
        """Initialize the LSLM base class.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model directory/file
            device: Device to run the model on ("cuda" or "cpu")
            log_level: Logging level
            supports_audio_output: Whether the model can generate audio output
            supports_evaluation: Whether the model supports audio evaluation
        """
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.log_level = log_level
        self.supports_audio_output = supports_audio_output
        self.supports_evaluation = supports_evaluation
        self.model = None
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(model_name)
        
        # Check if model exists
        if not os.path.exists(model_path):
            self.logger.warning(f"Model path does not exist: {model_path}")
    
    @abstractmethod
    def load_model(self):
        """Load the model. To be implemented by subclasses."""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio to text.
        
        Args:
            audio_path: Path to audio file
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing transcription results
        """
        pass
    
    @abstractmethod
    def chat(self, audio_path: str, prompt: str = "", output_type: str = "text", **kwargs) -> Dict[str, Any]:
        """Chat with the model using audio and optional text prompt.
        
        Args:
            audio_path: Path to audio file
            prompt: Optional text prompt to guide the model
            output_type: Type of output to generate ("text", "audio", or "both")
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing chat results, including text response and optionally audio output
        """
        pass
    
    @abstractmethod
    def generate_audio(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate audio from text.
        
        Args:
            text: Text to convert to speech
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing audio generation results
        """
        pass
    
    @abstractmethod
    def evaluate_audio(self, 
                      audio_path: str, 
                      evaluation_criteria: str = "",
                      reference_audio_path: Optional[str] = None,
                      **kwargs) -> Dict[str, Any]:
        """Evaluate audio quality based on specified criteria.
        
        Args:
            audio_path: Path to audio file to evaluate
            evaluation_criteria: Specific criteria or aspects to evaluate
            reference_audio_path: Optional path to reference audio for comparison
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary containing evaluation results
        """
        pass
    
    def format_chat_message(self, prompt: str, system_prompt: str = "") -> str:
        """Format a chat message with optional system prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt/instructions
            
        Returns:
            Formatted message
        """
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}" if prompt else system_prompt
        return prompt
    
    def validate_audio_path(self, audio_path: str) -> bool:
        """Validate that audio file exists and is readable.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            True if file exists and is readable, False otherwise
        """
        if not os.path.exists(audio_path):
            self.logger.error(f"Audio file not found: {audio_path}")
            return False
        
        try:
            with open(audio_path, 'rb') as f:
                # Just check if we can open the file
                pass
            return True
        except Exception as e:
            self.logger.error(f"Error accessing audio file {audio_path}: {e}")
            return False
        
    def get_output_path(self, prefix: str = "", suffix: str = "", ext: str = ".wav") -> str:
        """Generate a unique output path for results.
        
        Args:
            prefix: Prefix for filename
            suffix: Suffix for filename
            ext: File extension
            
        Returns:
            Output file path
        """
        import time
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}" if prefix else timestamp
        if suffix:
            filename = f"{filename}_{suffix}"
        filename = f"{filename}{ext}"
        
        return str(AUDIO_RESULTS_DIR / filename)
    
    def load_evaluation_prompt(self, prompt_name: str) -> str:
        """Load a predefined evaluation prompt.
        
        Args:
            prompt_name: Name of the prompt to load
            
        Returns:
            Evaluation prompt text
        """
        # Predefined evaluation prompts for different scenarios
        evaluation_prompts = {
            "general_quality": """
                Evaluate the audio quality by considering the following aspects:
                1. Sound clarity and intelligibility
                2. Background noise level
                3. Naturalness and coherence
                4. Overall audio quality
                
                Provide a rating from 1-10 for each aspect and explain your reasoning.
            """,
            "speech_quality": """
                Evaluate the speech quality by considering:
                1. Pronunciation clarity
                2. Intonation and rhythm
                3. Speech rate
                4. Voice naturalness
                5. Emotional expressiveness
                
                Rate each aspect from 1-10 and provide a detailed assessment.
            """,
            "audio_comparison": """
                Compare the input audio with the reference audio based on:
                1. Similarity in content
                2. Similarity in style/tone
                3. Quality differences
                4. Notable improvements or degradations
                
                Provide a detailed comparison for each aspect.
            """,
            "music_quality": """
                Evaluate the musical audio quality considering:
                1. Instrumental clarity
                2. Balance between instruments/voices
                3. Dynamic range
                4. Stereo imaging/soundstage
                5. Overall production quality
                
                Rate each aspect from 1-10 and provide a detailed assessment.
            """
        }
        
        return evaluation_prompts.get(prompt_name, "Please evaluate the audio quality in detail.")
