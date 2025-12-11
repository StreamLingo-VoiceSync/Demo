"""Data schemas for the TTS system."""

from __future__ import annotations
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import time


class VoiceSignature(BaseModel):
    """Voice signature with embedding for voice cloning"""
    embedding: List[float]
    duration_sec: float
    rms_energy: float
    peak_amplitude: float
    zero_crossing_rate: float
    mean_pitch_hz: float
    pitch_std_dev: float
    mean_mfcc: float
    std_mfcc: float
    speech_rate: float
    silence_ratio: float


class SynthesisRequest(BaseModel):
    """Request model for TTS synthesis with voice cloning support"""
    session_id: str = Field(..., description="Unique session identifier")
    call_id: str = Field(..., description="Call identifier")
    speaker_id: str = Field(..., description="Speaker identifier")
    processing_path: str = Field(..., description="Processing path identifier")
    source_language: str = Field(..., description="Source language code")
    target_language: str = Field(..., description="Target language code")
    source_text: str = Field(..., description="Original source text")
    translated_text: str = Field(..., description="Translated text")
    tts_text: str = Field(..., description="Text to synthesize", max_length=5000)
    ssml: str = Field("", description="SSML formatted text for enhanced speech synthesis")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Translation confidence score")
    target_word_timestamps: List[List[float]] = Field([], description="Word-level timestamps for prosody")
    pause_hints: List[Dict[str, Any]] = Field([], description="Pause hints for natural speech")
    prosody_hints: Dict[str, Any] = Field({}, description="Prosody hints for voice characteristics")
    speaker_embedding: List[float] = Field([], description="256-dimensional speaker embedding for voice cloning")
    character_duration_map: Dict[str, float] = Field({}, description="Character duration mapping")
    processing_time_ms: float = Field(0.0, description="Processing time in milliseconds")
    cache_hit: bool = Field(False, description="Whether result was from cache")
    voice_signature: Optional[VoiceSignature] = Field(None, description="Detailed voice signature with acoustic features")
    
    @validator('tts_text')
    def validate_text_length(cls, v):
        if len(v) > 5000:
            raise ValueError('Text too long')
        return v
    
    @validator('source_language', 'target_language')
    def validate_language_code(cls, v):
        supported_languages = ['en', 'hi', 'es', 'fr']
        if v not in supported_languages:
            raise ValueError(f'Unsupported language: {v}')
        return v


class SynthesisResponse(BaseModel):
    """Response model for TTS synthesis"""
    session_id: str = Field(..., description="Unique session identifier")
    audio_bytes: str = Field(..., description="Base64 encoded generated audio bytes")
    sample_rate: int = Field(..., description="Audio sample rate")
    duration_ms: float = Field(..., description="Audio duration in milliseconds")
    synthesis_time_ms: float = Field(..., description="Synthesis time in milliseconds")
    language: str = Field(..., description="Language code")
    speaker_id: str = Field(..., description="Speaker identifier")
    
    class Config:
        schema_extra = {
            "example": {
                "session_id": "sess_123456",
                "audio_bytes": "base64_encoded_audio_data...",
                "sample_rate": 22050,
                "duration_ms": 2500.5,
                "synthesis_time_ms": 150.2,
                "language": "en",
                "speaker_id": "speaker_1"
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: float = Field(default_factory=time.time)


class StatsResponse(BaseModel):
    """Statistics response"""
    status: str
    stats: Dict[str, Any]


class ModelsResponse(BaseModel):
    """Models information response"""
    models: Dict[str, Dict[str, Any]]