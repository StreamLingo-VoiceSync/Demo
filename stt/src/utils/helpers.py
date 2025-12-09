"""Utility functions for the STT system."""

from __future__ import annotations
import numpy as np

# Import shared utilities
from common.utils import gen_id, safe_float_conversion, compute_rms, write_wav_int16

def resample_if_needed(audio: np.ndarray, src_sr: int, tgt_sr: int = 16000) -> np.ndarray:
    """Resample audio if needed"""
    audio = safe_float_conversion(audio)
    if audio.size == 0 or src_sr == tgt_sr:
        return audio
    
    # Try librosa first
    try:
        import librosa
        return librosa.resample(audio, orig_sr=src_sr, target_sr=tgt_sr, res_type='kaiser_fast').astype(np.float32)
    except:
        pass
    
    # Fallback linear interpolation
    ratio = float(tgt_sr) / float(src_sr)
    new_len = max(1, int(len(audio) * ratio))
    indices = np.linspace(0, len(audio) - 1, new_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)