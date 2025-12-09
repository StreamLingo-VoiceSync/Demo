"""Shared utilities extracted from STT and MT systems"""

import numpy as np
import json
import time
import uuid
from pathlib import Path
from typing import Any, List, Tuple


def gen_id(prefix: str = "id") -> str:
    """Generate unique ID"""
    return f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"


def safe_float_conversion(x) -> np.ndarray:
    """Convert input to float32 numpy array safely"""
    if x is None:
        return np.zeros(0, dtype=np.float32)
    arr = np.asarray(x)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float64) / float(info.max)
    else:
        arr = arr.astype(np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


def compute_rms(x: np.ndarray) -> float:
    """Compute RMS energy"""
    a = safe_float_conversion(x)
    if a.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(a * a)))


def write_wav_int16(path: Path, data: np.ndarray, sr: int = 16000):
    """Write audio as 16-bit WAV file"""
    data = safe_float_conversion(data)
    if data.size == 0:
        data = np.zeros(int(sr * 0.1), dtype=np.float32)
    peak = np.max(np.abs(data))
    if peak > 0.05:
        data = data / peak * 0.92
    int16 = (data * 32767.0).astype(np.int16)
    try:
        import soundfile as sf
        sf.write(str(path), int16, sr, subtype="PCM_16")
    except Exception as e:
        try:
            from scipy.io import wavfile
            wavfile.write(str(path), sr, int16)
        except Exception as e2:
            print(f"WAV write failed: {e}; {e2}")