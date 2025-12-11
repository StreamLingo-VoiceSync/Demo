"""Configuration for the TTS system."""

from __future__ import annotations
import pathlib

# Logging
from common.logger import setup_tts_logger
log = setup_tts_logger()

# Output directories
TTS_OUTPUT_DIR = pathlib.Path("./tts_output")
for d in ["audio", "logs", "models"]:
    (TTS_OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)

# Language configuration
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi", 
    "es": "Spanish",
    "fr": "French"
}

# Model configuration
# Voice cloning enabled models with neural vocoders
MODEL_CONFIG = {
    "en": {
        "model_name": "tts_models/multilingual/multi-dataset/your_tts",
        "vocoder_name": "vocoder_models/universal/libri-tts/fullband-melgan",
        "sample_rate": 22050,
        "supports_voice_cloning": True,
        "voice_cloning_method": "your_tts"
    },
    "hi": {
        "model_name": "tts_models/multilingual/multi-dataset/your_tts",
        "vocoder_name": "vocoder_models/universal/libri-tts/fullband-melgan",
        "sample_rate": 22050,
        "supports_voice_cloning": True,
        "voice_cloning_method": "your_tts"
    },
    "es": {
        "model_name": "tts_models/es/mai/tacotron2-DDC",
        "vocoder_name": "vocoder_models/es/mai/hifigan_v2",
        "sample_rate": 22050,
        "supports_voice_cloning": False,
        "voice_cloning_method": "none"
    },
    "fr": {
        "model_name": "tts_models/fr/mai/tacotron2-DDC",
        "vocoder_name": "vocoder_models/fr/mai/hifigan_v2",
        "sample_rate": 22050,
        "supports_voice_cloning": False,
        "voice_cloning_method": "none"
    }
}

# Voice cloning methods
VOICE_CLONING_METHODS = {
    "your_tts": {
        "description": "Zero-shot voice cloning with YourTTS",
        "capabilities": ["multi_language", "voice_cloning", "prosody_transfer"],
        "supported_languages": ["en", "hi", "es", "fr"]
    },
    "vits": {
        "description": "VITS model with voice cloning",
        "capabilities": ["single_language", "voice_cloning"],
        "supported_languages": ["en"]
    },
    "stargan": {
        "description": "StarGANv2-VC for voice conversion",
        "capabilities": ["voice_conversion", "multi_speaker"],
        "supported_languages": ["en"]
    }
}

# Neural vocoders
NEURAL_VOCODERS = {
    "hifigan": {
        "description": "HiFi-GAN neural vocoder",
        "quality": "high",
        "speed": "fast"
    },
    "melgan": {
        "description": "MelGAN neural vocoder",
        "quality": "medium",
        "speed": "very_fast"
    },
    "waveglow": {
        "description": "WaveGlow neural vocoder",
        "quality": "very_high",
        "speed": "slow"
    }
}

# Default settings
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_VC_METHOD = "your_tts"
DEFAULT_VOCODER = "hifigan"