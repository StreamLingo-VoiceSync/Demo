"""
REAL-TIME STT SYSTEM - LATENCY OPTIMIZED (<2s)

================================================================

Multi-language support: EN, HI, ES, FR
MT/TTS-ready output with word-level timestamps, prosody, and embeddings

"""

from __future__ import annotations
import math
import numpy as np

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Callable

# ============================================================================ 
# CONFIGURATION - OPTIMIZED FOR <2s LATENCY
# ============================================================================

# Audio Settings
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.03  # 30ms chunks
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)

# CRITICAL: Aggressive silence padding for <2s latency (reduced from 400ms)
SILENCE_PADDING = {
    "en": 0.25,  # 250ms - CRITICAL for latency
    "hi": 0.25,  # 250ms - CRITICAL
    "es": 0.25,  # 250ms - CRITICAL
    "fr": 0.25   # 250ms - CRITICAL
}

MIN_SEGMENT_DURATION = 0.5
MAX_SEGMENT_DURATION = 10.0  # Faster processing

# Voice thresholds (optimized for <1% WER)
RMS_THRESHOLD = {
    "en": 0.003,
    "hi": 0.0025,
    "es": 0.003,
    "fr": 0.003
}

CONFIDENCE_THRESHOLD = {
    "en": 0.65,
    "hi": 0.60,
    "es": 0.65,
    "fr": 0.65
}

VAD_DECISION_THRESHOLD = 0.10
VAD_AGGRESSIVENESS = {"en": 0, "hi": 0, "es": 0, "fr": 0}
NOISE_REDUCTION_STRENGTH = 0.35  # Reduced for speed (was 0.50)
MAX_AMPLIFICATION_GAIN = 5.0

# Balanced beam settings for speed vs accuracy
BEAM_SETTINGS = {
    "en": {"beam_size": 5, "best_of": 5},
    "hi": {"beam_size": 5, "best_of": 5},
    "es": {"beam_size": 5, "best_of": 5},
    "fr": {"beam_size": 5, "best_of": 5}
}

# Output directories
OUTDIR = Path("./stt_outputs").resolve()
AUDIO_DIR = OUTDIR / "audio_segments"
TRANSCRIPTS_DIR = OUTDIR / "transcripts"
PROSODY_DIR = OUTDIR / "prosody_features"
EMBEDDINGS_DIR = OUTDIR / "speaker_embeddings"
METRICS_DIR = OUTDIR / "metrics"

for d in (AUDIO_DIR, TRANSCRIPTS_DIR, PROSODY_DIR, EMBEDDINGS_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MODELS_CONFIG = {
    "en": {"primary": "medium"},
    "hi": {"primary": "medium"},
    "es": {"primary": "medium"},
    "fr": {"primary": "medium"}
}