"""Utility functions for the TTS system."""

from __future__ import annotations
import numpy as np
from typing import List


def bytes_to_array(audio_bytes: bytes) -> np.ndarray:
    """
    Convert audio bytes to numpy array
    
    Args:
        audio_bytes: Audio data as bytes
        
    Returns:
        Audio data as numpy array
    """
    return np.frombuffer(audio_bytes, dtype=np.int16)


def array_to_bytes(audio_array: np.ndarray) -> bytes:
    """
    Convert numpy array to audio bytes
    
    Args:
        audio_array: Audio data as numpy array
        
    Returns:
        Audio data as bytes
    """
    return audio_array.astype(np.int16).tobytes()


def calculate_audio_duration(audio_bytes: bytes, sample_rate: int) -> float:
    """
    Calculate audio duration in seconds
    
    Args:
        audio_bytes: Audio data as bytes
        sample_rate: Audio sample rate
        
    Returns:
        Duration in seconds
    """
    # 16-bit audio = 2 bytes per sample
    num_samples = len(audio_bytes) // 2
    return num_samples / sample_rate