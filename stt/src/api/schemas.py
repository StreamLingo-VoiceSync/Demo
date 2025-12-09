"""Data schemas for the STT system."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List
import time
import numpy as np

@dataclass
class Segment:
    id: str
    call_id: str
    speaker_id: str
    audio: np.ndarray
    sr: int
    duration: float
    language: str
    vad_conf: float
    ts: float
    start_time: float = 0.0
    end_time: float = 0.0

@dataclass
class ASRResult:
    segment_id: str
    call_id: str
    speaker_id: str
    text: str
    words: List[Dict[str, Any]]  # Word-level timestamps for MT/TTS
    confidence: float
    language: str
    processing_time: float
    model_used: str
    timestamp: str
    start_time: float = 0.0
    end_time: float = 0.0

@dataclass
class CallParticipant:
    client_id: str
    speaker_id: str
    language: str
    websocket: Any = None
    connected_at: float = field(default_factory=time.time)

@dataclass
class Call:
    call_id: str
    participants: Dict[str, CallParticipant] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    active: bool = True