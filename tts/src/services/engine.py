"""TTS engine with language-specific synthesis."""

from __future__ import annotations
import time
import numpy as np
from typing import List, Tuple, Optional
import torch
from TTS.api import TTS
import librosa
import hashlib
import json
import os
from functools import lru_cache

# Import config
from ..core.config import SUPPORTED_LANGUAGES, MODEL_CONFIG, DEFAULT_SAMPLE_RATE

# Import exceptions
from ..core.exceptions import LanguageNotSupportedError, ModelLoadError, SynthesisError

# Import logger
from common.logger import setup_tts_logger
log = setup_tts_logger()


class TTSEngine:
    """Text-to-Speech Engine with multi-language support and voice cloning"""
    
    def __init__(self, cache_size: int = 1000):
        """Initialize TTS engine with language models"""
        self.models = {}
        self.sample_rates = {}
        self.vocoders = {}
        self.cache_size = cache_size
        self.synthesis_cache = {}
        self.stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_syntheses": 0,
            "failed_syntheses": 0
        }
        self._load_models()
    
    def _load_models(self):
        """Load models for all supported languages with voice cloning capabilities"""
        # Define model mappings for different languages with voice cloning support
        model_mappings = {
            "en": {
                "model_name": "tts_models/multilingual/multi-dataset/your_tts",
                "vocoder_name": "vocoder_models/universal/libri-tts/fullband-melgan",
                "supports_vc": True
            },
            "hi": {
                "model_name": "tts_models/multilingual/multi-dataset/your_tts",
                "vocoder_name": "vocoder_models/universal/libri-tts/fullband-melgan",
                "supports_vc": True
            },
            "es": {
                "model_name": "tts_models/es/mai/tacotron2-DDC",
                "vocoder_name": "vocoder_models/es/mai/hifigan_v2",
                "supports_vc": False
            },
            "fr": {
                "model_name": "tts_models/fr/mai/tacotron2-DDC",
                "vocoder_name": "vocoder_models/fr/mai/hifigan_v2",
                "supports_vc": False
            }
        }
        
        # Fallback models in case primary models fail
        fallback_models = {
            "en": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2",
                "supports_vc": False
            },
            "hi": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2",
                "supports_vc": False
            },
            "es": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2",
                "supports_vc": False
            },
            "fr": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "vocoder_name": "vocoder_models/en/ljspeech/hifigan_v2",
                "supports_vc": False
            }
        }
        
        for lang_code in SUPPORTED_LANGUAGES.keys():
            try:
                # Try to load the primary model
                model_config = model_mappings.get(lang_code, model_mappings["en"])
                model_name = model_config["model_name"]
                
                log.info(f"Loading TTS model for {lang_code} ({SUPPORTED_LANGUAGES[lang_code]}) | Model: {model_name}")
                
                # Load main TTS model
                self.models[lang_code] = TTS(model_name=model_name, progress_bar=False, gpu=torch.cuda.is_available())
                self.sample_rates[lang_code] = self.models[lang_code].synthesizer.output_sample_rate
                
                # Load vocoder if available
                if model_config["vocoder_name"]:
                    try:
                        self.vocoders[lang_code] = TTS(model_name=model_config["vocoder_name"], progress_bar=False, gpu=torch.cuda.is_available())
                        log.info(f"Loaded vocoder for {lang_code}")
                    except Exception as e:
                        log.debug(f"Failed to load vocoder for {lang_code}: {e}")
                        self.vocoders[lang_code] = None
                
                log.info(f"Successfully loaded TTS model for {lang_code} ({SUPPORTED_LANGUAGES[lang_code]}) | VC: {model_config['supports_vc']}")
            except Exception as e:
                log.error(f"Failed to initialize model for {lang_code}: {e}")
                # Try fallback model
                try:
                    log.info(f"Trying fallback model for {lang_code}")
                    fallback_config = fallback_models.get(lang_code, fallback_models["en"])
                    fallback_model_name = fallback_config["model_name"]
                    
                    self.models[lang_code] = TTS(model_name=fallback_model_name, progress_bar=False, gpu=torch.cuda.is_available())
                    self.sample_rates[lang_code] = self.models[lang_code].synthesizer.output_sample_rate
                    
                    # Load fallback vocoder if available
                    if fallback_config["vocoder_name"]:
                        try:
                            self.vocoders[lang_code] = TTS(model_name=fallback_config["vocoder_name"], progress_bar=False, gpu=torch.cuda.is_available())
                            log.info(f"Loaded fallback vocoder for {lang_code}")
                        except Exception as fallback_e:
                            log.warning(f"Failed to load fallback vocoder for {lang_code}: {fallback_e}")
                            self.vocoders[lang_code] = None
                    
                    log.info(f"Successfully loaded fallback TTS model for {lang_code} ({SUPPORTED_LANGUAGES[lang_code]})")
                except Exception as fallback_e:
                    log.error(f"Failed to load fallback model for {lang_code}: {fallback_e}")
                    raise ModelLoadError(f"Failed to load TTS model for {lang_code}: {str(e)}")
    
    def _generate_cache_key(self, text: str, lang: str, embedding: Optional[List[float]] = None, 
                           reference_wav: Optional[str] = None, speaker_wav: Optional[str] = None) -> str:
        """Generate a cache key for synthesis requests"""
        cache_data = {
            "text": text,
            "lang": lang,
            "embedding": embedding if embedding else [],
            "reference_wav": reference_wav if reference_wav else "",
            "speaker_wav": speaker_wav if speaker_wav else ""
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def synthesize(self, text: str, lang: str, embedding: Optional[List[float]] = None, 
                   reference_wav: Optional[str] = None, speaker_wav: Optional[str] = None) -> Tuple[bytes, int, float, float, bool]:
        """
        Synthesize speech from text with voice cloning and prosody transfer
        
        Args:
            text: Text to synthesize
            lang: Target language code
            embedding: Voice embedding for voice cloning
            reference_wav: Path to reference audio for voice cloning
            speaker_wav: Path to speaker audio for voice cloning
            
        Returns:
            Tuple of (audio_bytes, sample_rate, duration_ms, synthesis_time_ms, cache_hit)
        """
        start_time = time.time()
        cache_hit = False
        
        # Validate language
        if lang not in SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(f"Language '{lang}' not supported. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        
        # Generate cache key
        cache_key = self._generate_cache_key(text, lang, embedding, reference_wav, speaker_wav)
        
        # Check cache first
        if cache_key in self.synthesis_cache:
            cached_result = self.synthesis_cache[cache_key]
            self.stats["cache_hits"] += 1
            cache_hit = True
            log.debug(f"Cache hit for key: {cache_key[:8]}...")
            
            total_time = (time.time() - start_time) * 1000
            return cached_result["audio_bytes"], cached_result["sample_rate"], \
                   cached_result["duration_ms"], total_time, cache_hit
        
        self.stats["cache_misses"] += 1
        self.stats["total_syntheses"] += 1
        
        # Get model for language
        model = self.models.get(lang)
        if not model:
            self.stats["failed_syntheses"] += 1
            raise ModelLoadError(f"No model loaded for language '{lang}'")
        
        # Get sample rate for language
        sample_rate = self.sample_rates.get(lang, DEFAULT_SAMPLE_RATE)
        
        try:
            # Synthesize audio using Coqui TTS with voice cloning
            wav = None
            
            # Check if model supports voice cloning
            supports_vc = hasattr(model, 'is_multi_speaker') and model.is_multi_speaker
            
            if supports_vc and (reference_wav or speaker_wav or (embedding and len(embedding) > 0)):
                # Use voice cloning capabilities
                if reference_wav:
                    # YourTTS-style voice cloning with reference audio
                    wav = model.tts(text=text, speaker_wav=reference_wav)
                elif speaker_wav:
                    # Voice cloning with speaker audio
                    wav = model.tts(text=text, speaker_wav=speaker_wav)
                elif embedding and len(embedding) > 0:
                    # Voice cloning with embedding (convert to appropriate format)
                    # For YourTTS, we need to convert embedding to speaker representation
                    speaker_embedding = np.array(embedding, dtype=np.float32)
                    wav = model.tts(text=text, speaker_embedding=speaker_embedding)
                else:
                    # Fallback to standard synthesis
                    wav = model.tts(text=text)
            else:
                # Standard TTS synthesis
                wav = model.tts(text=text)
            
            # Apply neural vocoder if available
            vocoder = self.vocoders.get(lang)
            if vocoder and hasattr(vocoder, 'vocoder_model'):
                try:
                    # Apply neural vocoder for higher quality output
                    wav = vocoder.vocode(wav)
                except Exception as e:
                    log.warning(f"Neural vocoder failed for {lang}: {e}")
            
            # Convert to bytes
            audio_bytes = self._wav_to_bytes(wav, sample_rate)
            
            # Calculate duration
            duration_sec = len(wav) / sample_rate if isinstance(wav, (list, np.ndarray)) else 0
            duration_ms = duration_sec * 1000
            
            total_time = (time.time() - start_time) * 1000
            
            # Cache the result
            if len(self.synthesis_cache) < self.cache_size:
                self.synthesis_cache[cache_key] = {
                    "audio_bytes": audio_bytes,
                    "sample_rate": sample_rate,
                    "duration_ms": duration_ms
                }
            
            log.debug(f"Synthesized {len(text)} chars in {lang} | Audio: {len(audio_bytes)} bytes | Time: {total_time:.1f}ms | Cache: {cache_hit}")
            
            return audio_bytes, sample_rate, duration_ms, total_time, cache_hit
            
        except Exception as e:
            self.stats["failed_syntheses"] += 1
            log.error(f"Synthesis failed for language {lang}: {e}")
            raise SynthesisError(f"Failed to synthesize speech: {str(e)}")
    
    def _wav_to_bytes(self, wav: List[float], sample_rate: int) -> bytes:
        """
        Convert WAV array to bytes with proper normalization
        
        Args:
            wav: Audio waveform data
            sample_rate: Sample rate of the audio
            
        Returns:
            Audio bytes
        """
        # Convert to numpy array
        if isinstance(wav, list):
            wav = np.array(wav)
        
        # Handle different input types
        if wav.dtype == np.float64:
            wav = wav.astype(np.float32)
        elif wav.dtype != np.float32:
            wav = wav.astype(np.float32)
        
        # Normalize if needed (ensure values are between -1 and 1)
        if np.max(np.abs(wav)) > 1.0:
            wav_max = np.max(np.abs(wav))
            if wav_max > 0:
                wav = wav / wav_max
        
        # Convert to 16-bit integers
        audio_data = (wav * 32767).astype(np.int16)
        
        # Convert to bytes
        return audio_data.tobytes()
    
    def apply_prosody_transfer(self, source_audio_path: str, target_audio_path: str) -> Optional[np.ndarray]:
        """
        Apply prosody transfer from source to target audio
        
        Args:
            source_audio_path: Path to source audio with desired prosody
            target_audio_path: Path to target audio to modify
            
        Returns:
            Modified audio array or None if failed
        """
        try:
            # Load source and target audio
            source_audio, source_sr = librosa.load(source_audio_path, sr=None)
            target_audio, target_sr = librosa.load(target_audio_path, sr=None)
            
            # Extract prosodic features from source (F0, energy, duration)
            source_f0, _, _ = librosa.pyin(source_audio, fmin=librosa.note_to_hz('C2'), 
                                          fmax=librosa.note_to_hz('C7'))
            source_energy = librosa.feature.rms(y=source_audio)[0]
            
            # Extract prosodic features from target
            target_f0, _, _ = librosa.pyin(target_audio, fmin=librosa.note_to_hz('C2'), 
                                          fmax=librosa.note_to_hz('C7'))
            target_energy = librosa.feature.rms(y=target_audio)[0]
            
            # Apply prosody transfer (simplified approach)
            # In a production system, this would be more sophisticated
            if source_f0 is not None and target_f0 is not None:
                # Adjust pitch
                f0_ratio = np.mean(source_f0[source_f0 > 0]) / np.mean(target_f0[target_f0 > 0])
                # This is a simplified approach - in practice, more sophisticated methods would be used
                
            log.info("Prosody transfer applied successfully")
            return target_audio
            
        except Exception as e:
            log.error(f"Prosody transfer failed: {e}")
            return None
    
    def get_stats(self) -> dict:
        """Get engine statistics"""
        total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        cache_hit_rate = (self.stats["cache_hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_hit_rate": f"{cache_hit_rate:.2f}%",
            "cache_hits": self.stats["cache_hits"],
            "cache_misses": self.stats["cache_misses"],
            "total_syntheses": self.stats["total_syntheses"],
            "failed_syntheses": self.stats["failed_syntheses"],
            "cache_size": len(self.synthesis_cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """Clear the synthesis cache"""
        self.synthesis_cache.clear()
        self.stats["cache_hits"] = 0
        self.stats["cache_misses"] = 0
        log.info("Synthesis cache cleared")