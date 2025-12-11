"""API routes for the TTS system."""

from __future__ import annotations
import json
import time
import numpy as np
from typing import Dict, Any
import re
from functools import wraps

from fastapi import APIRouter, HTTPException, Request, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import ValidationError
import asyncio
import hashlib

# Import shared utilities
from common.logger import setup_tts_logger

# Import schemas
from .schemas import SynthesisRequest, SynthesisResponse

# Import services
from ..services.engine import TTSEngine
from ..services.processing import AudioProcessor, OutputManager
from ..core.exceptions import TTSError, LanguageNotSupportedError, ModelLoadError, SynthesisError

log = setup_tts_logger()
router = APIRouter()

# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Rate limiting
REQUEST_LIMITS = {}  # In-memory store for rate limiting (would use Redis in production)
RATE_LIMIT = 100  # requests per minute
RATE_LIMIT_WINDOW = 60  # seconds

# Initialize TTS engine and processors
tts_engine = TTSEngine(cache_size=1000)
audio_processor = AudioProcessor()
output_manager = OutputManager()

def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API key"""
    # In production, this would check against a database or environment variable
    # For demo purposes, we'll allow requests without API key but log the attempt
    if api_key_header:
        log.debug("API key provided")
    return api_key_header

def rate_limit(dependency: str = ""):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Fix: Properly handle both Request and other object types
            if isinstance(request, Request) and hasattr(request, 'client') and request.client:
                client_ip = request.client.host
            elif isinstance(request, Request):
                # Extract client IP from headers or use default
                client_ip = request.headers.get("x-forwarded-for", "127.0.0.1").split(",")[0].strip()
            else:
                # For non-Request objects, use a default identifier
                client_ip = "127.0.0.1"
            
            current_time = time.time()
            
            # Clean up old entries
            expired_keys = [key for key, value in REQUEST_LIMITS.items() 
                           if current_time - value["timestamp"] > RATE_LIMIT_WINDOW]
            for key in expired_keys:
                del REQUEST_LIMITS[key]
            
            # Check rate limit
            key = f"{client_ip}:{dependency}"
            if key in REQUEST_LIMITS:
                if REQUEST_LIMITS[key]["count"] >= RATE_LIMIT:
                    log.warning(f"Rate limit exceeded for {client_ip}")
                    raise HTTPException(status_code=429, detail="Rate limit exceeded")
                REQUEST_LIMITS[key]["count"] += 1
                REQUEST_LIMITS[key]["timestamp"] = current_time
            else:
                REQUEST_LIMITS[key] = {"count": 1, "timestamp": current_time}
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

def validate_input(func):
    """Input validation decorator"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Extract request data
        request_data = kwargs.get('request')
        if hasattr(request_data, 'tts_text'):
            text = request_data.tts_text
            # Check for potentially harmful content
            if text and len(text) > 5000:  # Limit text length
                raise HTTPException(status_code=400, detail="Text too long")
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script.*?>.*?</script>',  # Script tags
                r'javascript:',  # JavaScript URLs
                r'on\w+\s*=',  # Event handlers
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    log.warning(f"Suspicious content detected: {text[:100]}...")
                    raise HTTPException(status_code=400, detail="Invalid content")
        
        return await func(*args, **kwargs)
    return wrapper

@router.post("/synthesize", response_model=SynthesisResponse)
@rate_limit("synthesize")
@validate_input
async def synthesize(request: SynthesisRequest, api_key: str = Security(get_api_key)):
    """
    Synthesize speech from text with voice cloning
    
    Args:
        request: SynthesisRequest with text, language, and voice embedding
        
    Returns:
        SynthesisResponse with audio bytes and metadata
    """
    try:
        log.info(f"Synthesis request for {request.speaker_id} in {request.target_language}")
        
        # Extract voice signature if available
        voice_embedding = None
        reference_wav = None
        speaker_wav = None
        
        # Priority: voice_signature > speaker_embedding > reference paths
        if request.voice_signature:
            voice_embedding = request.voice_signature.embedding
        elif request.speaker_embedding and len(request.speaker_embedding) > 0:
            voice_embedding = request.speaker_embedding
        elif hasattr(request, 'reference_audio_path'):
            reference_wav = request.reference_audio_path
        elif hasattr(request, 'speaker_audio_path'):
            speaker_wav = request.speaker_audio_path
        
        # Synthesize audio with voice cloning capabilities
        audio_bytes, sample_rate, duration_ms, synthesis_time, cache_hit = tts_engine.synthesize(
            text=request.tts_text,
            lang=request.target_language,
            embedding=voice_embedding,
            reference_wav=reference_wav,
            speaker_wav=speaker_wav
        )
        
        # Apply prosody modifications if provided
        if request.prosody_hints:
            audio_bytes = audio_processor.apply_prosody(
                audio_bytes, request.prosody_hints, sample_rate
            )
        
        # Apply pause hints if provided
        if request.pause_hints:
            audio_bytes = audio_processor.apply_pause_hints(
                audio_bytes, request.pause_hints, sample_rate
            )
        
        # Save audio and metadata if needed (optional)
        # This could be controlled by a flag in the request
        if hasattr(request, 'save_output') and request.save_output:
            metadata = {
                "session_id": request.session_id,
                "speaker_id": request.speaker_id,
                "language": request.target_language,
                "text": request.tts_text,
                "duration_ms": duration_ms,
                "synthesis_time_ms": synthesis_time,
                "sample_rate": sample_rate,
                "voice_cloning_used": bool(voice_embedding or reference_wav or speaker_wav),
                "cache_hit": cache_hit
            }
            output_manager.save_audio(
                request.session_id, audio_bytes, sample_rate, request.target_language
            )
            output_manager.save_metadata(request.session_id, metadata)
        
        # Encode audio bytes as base64 for JSON serialization
        import base64
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        response = SynthesisResponse(
            session_id=request.session_id,
            audio_bytes=audio_base64,  # Return base64 encoded audio
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            synthesis_time_ms=synthesis_time,
            language=request.target_language,
            speaker_id=request.speaker_id
        )
        
        log.info(f"Synthesis completed for {request.speaker_id} in {request.target_language} | Duration: {duration_ms:.0f}ms | Time: {synthesis_time:.0f}ms | Cache: {cache_hit} | Voice Cloning: {bool(voice_embedding)}")
        
        return response
        
    except ValidationError as e:
        log.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    except LanguageNotSupportedError as e:
        log.error(f"Language not supported: {e}")
        raise HTTPException(status_code=400, detail=f"Language not supported: {e}")
    except ModelLoadError as e:
        log.error(f"Model load error: {e}")
        raise HTTPException(status_code=500, detail=f"Model unavailable: {e}")
    except SynthesisError as e:
        log.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@router.post("/clone_voice")
@rate_limit("clone_voice")
async def clone_voice(reference_audio: str, target_text: str, language: str = "en", api_key: str = Security(get_api_key)):
    """
    Clone voice from reference audio and synthesize target text
    
    Args:
        reference_audio: Path to reference audio file
        target_text: Text to synthesize with cloned voice
        language: Target language
        
    Returns:
        SynthesisResponse with cloned voice audio
    """
    try:
        log.info(f"Voice cloning request for language {language}")
        
        # Validate inputs
        if len(target_text) > 5000:
            raise HTTPException(status_code=400, detail="Text too long")
        
        if language not in ["en", "hi", "es", "fr"]:
            raise HTTPException(status_code=400, detail="Unsupported language")
        
        # Synthesize with reference audio
        audio_bytes, sample_rate, duration_ms, synthesis_time, cache_hit = tts_engine.synthesize(
            text=target_text,
            lang=language,
            reference_wav=reference_audio
        )
        
        response = SynthesisResponse(
            session_id=f"vc_{int(time.time())}",
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            duration_ms=duration_ms,
            synthesis_time_ms=synthesis_time,
            language=language,
            speaker_id="cloned_voice"
        )
        
        log.info(f"Voice cloning completed | Duration: {duration_ms:.0f}ms | Time: {synthesis_time:.0f}ms | Cache: {cache_hit}")
        
        return response
        
    except Exception as e:
        log.error(f"Voice cloning error: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {e}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "tts"}

@router.get("/stats")
async def get_stats(api_key: str = Security(get_api_key)):
    """Get TTS engine statistics"""
    stats = tts_engine.get_stats()
    return {"status": "ok", "stats": stats}

@router.post("/clear_cache")
async def clear_cache(api_key: str = Security(get_api_key)):
    """Clear synthesis cache"""
    tts_engine.clear_cache()
    return {"status": "ok", "message": "Cache cleared"}

@router.get("/models")
async def list_models(api_key: str = Security(get_api_key)):
    """List available models"""
    models = {}
    for lang, model in tts_engine.models.items():
        models[lang] = {
            "language": lang,
            "loaded": model is not None,
            "vocoder_available": tts_engine.vocoders.get(lang) is not None
        }
    return {"models": models}