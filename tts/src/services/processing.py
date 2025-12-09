"""Processing logic for TTS system."""

from __future__ import annotations
import json
import time
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, Any, List, Optional
import logging

# Import config
from ..core.config import TTS_OUTPUT_DIR

# Import logger
from common.logger import setup_tts_logger
log = setup_tts_logger()


class AudioProcessor:
    """Process and enhance synthesized audio with voice cloning support"""
    
    def __init__(self):
        pass
    
    def apply_prosody(self, audio_bytes: bytes, prosody_hints: Dict[str, Any], sample_rate: int) -> bytes:
        """
        Apply prosody modifications to audio based on hints
        
        Args:
            audio_bytes: Raw audio bytes
            prosody_hints: Prosody hints from MT module
            sample_rate: Audio sample rate
            
        Returns:
            Processed audio bytes
        """
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            # Convert to float32 for processing
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Apply pitch shift if specified
            if "pitch_shift" in prosody_hints:
                n_steps = prosody_hints["pitch_shift"]
                try:
                    audio_float = librosa.effects.pitch_shift(audio_float, sr=sample_rate, n_steps=n_steps)
                except Exception as e:
                    log.warning(f"Pitch shift failed: {e}")
            
            # Apply time stretching if specified
            if "time_stretch" in prosody_hints:
                rate = prosody_hints["time_stretch"]
                try:
                    # Ensure rate is within reasonable bounds
                    rate = max(0.5, min(2.0, rate))
                    audio_float = librosa.effects.time_stretch(audio_float, rate=rate)
                except Exception as e:
                    log.warning(f"Time stretch failed: {e}")
            
            # Apply volume adjustment if specified
            if "volume" in prosody_hints:
                volume = prosody_hints["volume"]
                try:
                    # Ensure volume is within reasonable bounds
                    volume = max(0.1, min(3.0, volume))
                    audio_float = audio_float * volume
                except Exception as e:
                    log.warning(f"Volume adjustment failed: {e}")
            
            # Convert back to int16
            # Normalize if needed to prevent clipping
            if np.max(np.abs(audio_float)) > 1.0:
                audio_float = audio_float / np.max(np.abs(audio_float))
            
            audio_processed = (audio_float * 32767).astype(np.int16)
            
            log.debug("Applied prosody modifications to audio")
            return audio_processed.tobytes()
        except Exception as e:
            log.error(f"Error applying prosody: {e}")
            # Return original audio if processing fails
            return audio_bytes
    
    def apply_pause_hints(self, audio_bytes: bytes, pause_hints: List[Dict[str, Any]], sample_rate: int) -> bytes:
        """
        Apply pause hints to audio
        
        Args:
            audio_bytes: Raw audio bytes
            pause_hints: Pause hints from MT module
            sample_rate: Audio sample rate
            
        Returns:
            Audio bytes with pauses inserted
        """
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Insert pauses based on hints
            # Sort pause hints by position to insert in correct order
            sorted_pause_hints = sorted(pause_hints, key=lambda x: x.get("position", 0))
            
            for pause_hint in sorted_pause_hints:
                position = pause_hint.get("position", 0)  # Position in seconds
                duration = pause_hint.get("duration", 0.5)  # Duration in seconds
                
                # Convert position to sample index
                insert_pos = int(position * sample_rate)
                
                # Ensure insert position is within bounds
                insert_pos = max(0, min(len(audio_data), insert_pos))
                
                # Create silence samples
                silence_samples = int(duration * sample_rate)
                silence = np.zeros(silence_samples, dtype=np.int16)
                
                # Insert silence into audio
                audio_data = np.insert(audio_data, insert_pos, silence)
            
            log.debug(f"Applied {len(pause_hints)} pause hints to audio")
            return audio_data.tobytes()
        except Exception as e:
            log.error(f"Error applying pause hints: {e}")
            # Return original audio if processing fails
            return audio_bytes
    
    def enhance_voice_cloning_quality(self, audio_bytes: bytes, reference_audio_path: Optional[str] = None) -> bytes:
        """
        Enhance audio quality for voice cloning applications
        
        Args:
            audio_bytes: Raw audio bytes
            reference_audio_path: Path to reference audio for matching characteristics
            
        Returns:
            Enhanced audio bytes
        """
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0
            
            # Apply noise reduction
            try:
                # Simple spectral gating for noise reduction
                audio_float = self._reduce_noise(audio_float, 22050)
            except Exception as e:
                log.warning(f"Noise reduction failed: {e}")
            
            # Match characteristics with reference if provided
            if reference_audio_path:
                try:
                    ref_audio, ref_sr = librosa.load(reference_audio_path, sr=22050)
                    audio_float = self._match_audio_characteristics(audio_float, ref_audio)
                except Exception as e:
                    log.warning(f"Reference matching failed: {e}")
            
            # Convert back to int16
            if np.max(np.abs(audio_float)) > 1.0:
                audio_float = audio_float / np.max(np.abs(audio_float))
            
            audio_processed = (audio_float * 32767).astype(np.int16)
            
            log.debug("Enhanced voice cloning quality")
            return audio_processed.tobytes()
            
        except Exception as e:
            log.error(f"Voice cloning enhancement failed: {e}")
            return audio_bytes
    
    def _reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply simple noise reduction"""
        try:
            # Compute power spectrogram
            S = np.abs(librosa.stft(audio))
            # Simple spectral gating (threshold-based)
            threshold = np.mean(S) * 0.5
            S_filtered = S * (S > threshold)
            # Reconstruct audio
            audio_filtered = librosa.istft(S_filtered)
            return audio_filtered
        except Exception as e:
            log.warning(f"Noise reduction algorithm failed: {e}")
            return audio
    
    def _match_audio_characteristics(self, target: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Match target audio characteristics to reference"""
        try:
            # Simple amplitude matching
            target_rms = np.sqrt(np.mean(target**2))
            ref_rms = np.sqrt(np.mean(reference**2))
            
            if target_rms > 0 and ref_rms > 0:
                gain = ref_rms / target_rms
                target = target * gain
                
            return target
        except Exception as e:
            log.warning(f"Audio characteristic matching failed: {e}")
            return target


class OutputManager:
    """Manage TTS output files and metadata"""
    
    def __init__(self):
        pass
    
    def save_audio(self, session_id: str, audio_bytes: bytes, sample_rate: int, language: str) -> str:
        """
        Save audio to file
        
        Args:
            session_id: Session identifier
            audio_bytes: Audio data
            sample_rate: Sample rate
            language: Language code
            
        Returns:
            Path to saved file
        """
        try:
            filename = f"{session_id}_{language}.wav"
            filepath = TTS_OUTPUT_DIR / "audio" / filename
            
            # Convert bytes to numpy array and save as proper WAV file
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            sf.write(filepath, audio_data, sample_rate)
            
            log.info(f"Saved audio to {filepath}")
            return str(filepath)
        except Exception as e:
            log.error(f"Error saving audio file: {e}")
            raise
    
    def save_metadata(self, session_id: str, metadata: Dict[str, Any]) -> str:
        """
        Save synthesis metadata to JSON file
        
        Args:
            session_id: Session identifier
            metadata: Metadata dictionary
            
        Returns:
            Path to saved file
        """
        try:
            filename = f"{session_id}_metadata.json"
            filepath = TTS_OUTPUT_DIR / "audio" / filename
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            log.info(f"Saved metadata to {filepath}")
            return str(filepath)
        except Exception as e:
            log.error(f"Error saving metadata file: {e}")
            raise