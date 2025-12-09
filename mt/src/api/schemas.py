"""Data schemas for the MT system."""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import uuid
import time
import numpy as np

@dataclass
class ProcessedToken:
    """Processed token with timing"""
    text: str
    start_ms: float
    end_ms: float
    confidence: float
    call_id: str
    speaker_id: str
    source_language: str
    segment_id: str = ""
    processing_path: str = "path_1"
    source_words: List[str] = field(default_factory=list)

@dataclass
class TranslationResult:
    """Final translation with TTS readiness"""
    session_id: str
    call_id: str
    speaker_id: str
    segment_id: str
    source_language: str
    target_language: str
    source_text: str
    translated_text: str
    tts_text: str
    processing_path: str
    
    source_words: List[str] = field(default_factory=list)
    target_words: List[str] = field(default_factory=list)
    word_alignment: Dict[int, List[int]] = field(default_factory=dict)
    target_word_timestamps: List[List[float]] = field(default_factory=list)
    
    confidence: float = 0.0
    bleu_score: float = 0.0
    grammar_valid: bool = True
    
    ssml: str = ""
    pause_hints: List[Dict[str, Any]] = field(default_factory=list)
    prosody_hints: Dict[str, Any] = field(default_factory=dict)
    speaker_embedding: List[float] = field(default_factory=list)
    character_duration_map: Dict[str, float] = field(default_factory=dict)

    processing_time_ms: float = 0.0
    cache_hit: bool = False
    
    # Dual-lane specific
    cross_lane_consistency_score: float = 0.95
    path_errors: List[str] = field(default_factory=list)