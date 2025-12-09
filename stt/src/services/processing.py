"""Core business logic for STT processing."""

from __future__ import annotations
import threading
import time
import numpy as np
from collections import deque
import asyncio

# Import shared utilities
from common.logger import setup_stt_logger

# Import config
from ..core.config import (
    SAMPLE_RATE, 
    CHUNK_SIZE, 
    SILENCE_PADDING, 
    RMS_THRESHOLD, 
    VAD_AGGRESSIVENESS,
    NOISE_REDUCTION_STRENGTH,
    MAX_AMPLIFICATION_GAIN
)

# Import schemas
from ..api.schemas import Segment

# Import helpers
from ..utils.helpers import safe_float_conversion, compute_rms

# Import engine
from .engine import TextDeduplicator

# Import exceptions
from ..core.exceptions import VADError, AudioProcessingError

log = setup_stt_logger()

# ============================================================================ 
# PROSODY & EMBEDDINGS EXTRACTION
# ============================================================================

def extract_prosody_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, float]:
    """Extract prosody features for TTS synthesis"""
    a = safe_float_conversion(audio)
    duration = len(a) / sr
    rms = compute_rms(a)
    peak = float(np.max(np.abs(a))) if len(a) > 0 else 0.0
    zcr = float(np.sum(np.abs(np.diff(np.sign(a))))) / len(a) if len(a) > 0 else 0.0
    
    features = {
        "duration_sec": round(duration, 4),
        "rms_energy": round(rms, 6),
        "peak_amplitude": round(peak, 6),
        "zero_crossing_rate": round(zcr, 6),
    }
    
    # Pitch extraction
    if len(a) > 512:
        try:
            import librosa
            pitch = librosa.yin(a, fmin=50, fmax=400, sr=sr)
            pitch_valid = pitch[~np.isnan(pitch)]
            if len(pitch_valid) > 0:
                features["mean_pitch_hz"] = round(float(np.mean(pitch_valid)), 2)
                features["pitch_std_dev"] = round(float(np.std(pitch_valid)), 2)
            else:
                features["mean_pitch_hz"] = 0.0
                features["pitch_std_dev"] = 0.0
        except:
            features["mean_pitch_hz"] = 0.0
            features["pitch_std_dev"] = 0.0
    else:
        features["mean_pitch_hz"] = 0.0
        features["pitch_std_dev"] = 0.0
    
    # MFCC features
    if len(a) > 512:
        try:
            import librosa
            mfcc = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=13)
            features["mean_mfcc"] = round(float(np.mean(mfcc)), 2)
            features["std_mfcc"] = round(float(np.std(mfcc)), 2)
        except:
            features["mean_mfcc"] = 0.0
            features["std_mfcc"] = 0.0
    else:
        features["mean_mfcc"] = 0.0
        features["std_mfcc"] = 0.0
    
    features["speech_rate"] = round(len(a) / (duration + 1e-6), 2)
    threshold = rms * 0.1
    silent_frames = np.sum(np.abs(a) < threshold)
    features["silence_ratio"] = round(float(silent_frames / len(a)) if len(a) > 0 else 0.0, 4)
    
    return features

class SpeakerEmbeddingExtractor:
    """Extract speaker embeddings for voice cloning/TTS"""
    
    def __init__(self):
        self.cache = {}
    
    def extract(self, audio: np.ndarray, sr: int = SAMPLE_RATE, speaker_id: str = None) -> List[float]:
        if speaker_id and speaker_id in self.cache:
            return self.cache[speaker_id]
        
        a = safe_float_conversion(audio)
        embedding = np.zeros(256, dtype=np.float32)
        
        # MFCC-based embedding
        if len(a) > 512:
            try:
                import librosa
                mfcc = librosa.feature.mfcc(y=a, sr=sr, n_mfcc=13)
                embedding[:13] = np.mean(mfcc, axis=1).astype(np.float32)
                embedding[13:26] = np.std(mfcc, axis=1).astype(np.float32)
            except:
                pass
        
        # Prosody features
        prosody = extract_prosody_features(a, sr)
        prosody_vals = [
            prosody.get("duration_sec", 0) / 10.0,
            prosody.get("rms_energy", 0) * 10,
            prosody.get("peak_amplitude", 0),
            prosody.get("zero_crossing_rate", 0),
            prosody.get("mean_pitch_hz", 0) / 400.0,
            prosody.get("pitch_std_dev", 0) / 50.0,
            prosody.get("mean_mfcc", 0),
            prosody.get("std_mfcc", 0),
            prosody.get("speech_rate", 0) / 10000.0,
            prosody.get("silence_ratio", 0)
        ]
        embedding[26:36] = np.array(prosody_vals, dtype=np.float32)
        
        # Mel spectrogram features
        if len(a) > 512:
            try:
                import librosa
                S = librosa.feature.melspectrogram(y=a, sr=sr, n_mels=13)
                embedding[36:49] = np.mean(S, axis=1).astype(np.float32)
                embedding[49:62] = np.std(S, axis=1).astype(np.float32)
            except:
                pass
        
        # Frame-level RMS
        frame_len = int(sr * 0.02)
        for i in range(0, min(45, len(a) // frame_len)):
            frame = a[i*frame_len:(i+1)*frame_len]
            if len(frame) > 0:
                embedding[75 + i] = compute_rms(frame)
        
        # Random noise for diversity
        embedding[120:256] = np.random.randn(136).astype(np.float32) * 0.001
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if speaker_id:
            self.cache[speaker_id] = embedding.tolist()
        
        return embedding.tolist()

# ============================================================================ 
# VOICE ACTIVITY DETECTION
# ============================================================================

class VAD:
    def __init__(self, language: str = "en"):
        self.language = language
        self.rms_threshold = RMS_THRESHOLD.get(language, 0.003)
        self.aggressiveness = VAD_AGGRESSIVENESS.get(language, 0)
        self.vad = None
        self.history = deque(maxlen=100)
        
        try:
            import webrtcvad
            HAS_WEBRTCVAD = True
        except:
            HAS_WEBRTCVAD = False
            
        if HAS_WEBRTCVAD:
            try:
                self.vad = webrtcvad.Vad(self.aggressiveness)
                log.info(f"WebRTC VAD initialized for {language.upper()} (mode={self.aggressiveness})")
            except Exception as e:
                log.warning(f"WebRTC VAD init failed: {e}")
    
    def is_speech(self, audio: np.ndarray) -> Tuple[bool, float]:
        a = safe_float_conversion(audio)
        if a.size == 0:
            return False, 0.0
        
        rms = compute_rms(a)
        self.history.append(rms)
        rms_score = min(0.99, max(0.0, rms / (self.rms_threshold + 1e-12)))
        is_speech_rms = rms_score > 0.10  # VAD_DECISION_THRESHOLD
        
        webrtc_decision = False
        if self.vad and len(a) == CHUNK_SIZE:
            try:
                int16 = (a * 32767.0).astype(np.int16)
                webrtc_decision = self.vad.is_speech(int16.tobytes(), SAMPLE_RATE)
            except:
                pass
        
        decision = is_speech_rms or webrtc_decision
        return decision, rms_score

# ============================================================================ 
# AUDIO ENHANCEMENT
# ============================================================================

class AudioEnhancer:
    def __init__(self, target_rms: float = 0.1):
        self.target_rms = target_rms
    
    def enhance(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, Dict]:
        a = safe_float_conversion(audio)
        if a.size < 16:
            return a, {}
        
        stats = {"processing_steps": []}
        
        # DC removal
        a = a - np.mean(a)
        stats["processing_steps"].append("dc_removal")
        
        # Noise reduction (reduced strength for speed)
        try:
            import noisereduce
            HAS_NOISEREDUCE = True
        except:
            HAS_NOISEREDUCE = False
            
        if HAS_NOISEREDUCE and len(a) >= sr // 2:
            try:
                a = noisereduce.reduce_noise(y=a, sr=sr, stationary=True, prop_decrease=NOISE_REDUCTION_STRENGTH)
                stats["processing_steps"].append(f"noise_reduction_{int(NOISE_REDUCTION_STRENGTH*100)}pct")
            except:
                pass
        
        # Amplification
        current_rms = compute_rms(a) + 1e-12
        if current_rms < self.target_rms * 0.4:
            gain = min(MAX_AMPLIFICATION_GAIN, self.target_rms / current_rms)
            a = a * gain
            stats["processing_steps"].append(f"amplification_{gain:.1f}x")
        
        # Soft limiting
        peak = np.max(np.abs(a))
        if peak > 0.92:
            a = a * (0.92 / peak)
            stats["processing_steps"].append("soft_limiting")
        
        stats["enhanced_rms"] = compute_rms(a)
        return a.astype(np.float32), stats

# ============================================================================ 
# SEGMENTER - CRITICAL FOR <2s LATENCY
# ============================================================================

class Segmenter:
    """Segments audio into processable chunks with aggressive finalization"""
    
    def __init__(self):
        self.states = {}
    
    def process_chunk(
        self,
        chunk: np.ndarray,
        vad_result: Tuple[bool, float],
        call_id: str,
        speaker_id: str,
        language: str
    ) -> Optional[Segment]:
        """Process chunk and return finalized segment if ready"""
        chunk = safe_float_conversion(chunk)
        key = f"{call_id}_{speaker_id}_{language}"
        
        state = self.states.setdefault(key, {
            "current": None,
            "silence_count": 0,
            "speech_count": 0,
            "audio_buffer": [],
            "start_time": time.time()
        })
        
        is_speech, vad_conf = vad_result
        silence_padding_duration = SILENCE_PADDING.get(language, 0.25)
        silence_frames_needed = int(silence_padding_duration / 0.03)  # CHUNK_DURATION
        
        if is_speech and vad_conf > 0.10:  # VAD_DECISION_THRESHOLD
            state["silence_count"] = 0
            state["speech_count"] += 1
            state["audio_buffer"].append(chunk)
        else:
            state["silence_count"] += 1
            if state["speech_count"] > 0:
                state["audio_buffer"].append(chunk)
        
        # Finalize if silence threshold reached
        if state["silence_count"] >= silence_frames_needed and state["speech_count"] > 0:
            if state["audio_buffer"]:
                audio = np.concatenate(state["audio_buffer"])
                duration = len(audio) / SAMPLE_RATE
                
                # Import config constants
                from ..core.config import MIN_SEGMENT_DURATION, MAX_SEGMENT_DURATION
                
                if MIN_SEGMENT_DURATION <= duration <= MAX_SEGMENT_DURATION:
                    from common.utils import gen_id
                    
                    segment = Segment(
                        id=gen_id("seg"),
                        call_id=call_id,
                        speaker_id=speaker_id,
                        audio=audio,
                        sr=SAMPLE_RATE,
                        duration=duration,
                        language=language,
                        vad_conf=vad_conf,
                        ts=time.time(),
                        start_time=state["start_time"],
                        end_time=time.time()
                    )
                    
                    # Reset state
                    state["current"] = None
                    state["silence_count"] = 0
                    state["speech_count"] = 0
                    state["audio_buffer"] = []
                    state["start_time"] = time.time()
                    
                    return segment
            
            # Reset state
            state["current"] = None
            state["silence_count"] = 0
            state["speech_count"] = 0
            state["audio_buffer"] = []
            state["start_time"] = time.time()
        
        # Finalize if max duration reached
        if state["speech_count"] > 0:
            duration = (len(state["audio_buffer"]) * CHUNK_SIZE) / SAMPLE_RATE
            
            # Import config constant
            from ..core.config import MAX_SEGMENT_DURATION
            
            if duration >= MAX_SEGMENT_DURATION:
                if state["audio_buffer"]:
                    audio = np.concatenate(state["audio_buffer"])
                    
                    from common.utils import gen_id
                    
                    segment = Segment(
                        id=gen_id("seg"),
                        call_id=call_id,
                        speaker_id=speaker_id,
                        audio=audio,
                        sr=SAMPLE_RATE,
                        duration=len(audio) / SAMPLE_RATE,
                        language=language,
                        vad_conf=vad_conf,
                        ts=time.time(),
                        start_time=state["start_time"],
                        end_time=time.time()
                    )
                    
                    # Reset state
                    state["current"] = None
                    state["silence_count"] = 0
                    state["speech_count"] = 0
                    state["audio_buffer"] = []
                    state["start_time"] = time.time()
                    
                    return segment
        
        return None
    
    def force_finalize_all(self) -> List[Segment]:
        """Force finalize all pending segments (on disconnect)"""
        segments = []
        for key, state in self.states.items():
            if state["speech_count"] > 0 and state["audio_buffer"]:
                audio = np.concatenate(state["audio_buffer"])
                duration = len(audio) / SAMPLE_RATE
                
                # Import config constant
                from ..core.config import MIN_SEGMENT_DURATION
                
                if duration >= MIN_SEGMENT_DURATION:
                    call_id = key.split("_")[0]
                    speaker_id = "_".join(key.split("_")[1:-1])
                    language = key.split("_")[-1]
                    
                    from common.utils import gen_id
                    
                    segment = Segment(
                        id=gen_id("seg"),
                        call_id=call_id,
                        speaker_id=speaker_id,
                        audio=audio,
                        sr=SAMPLE_RATE,
                        duration=duration,
                        language=language,
                        vad_conf=0.9,
                        ts=time.time(),
                        start_time=state["start_time"],
                        end_time=time.time()
                    )
                    segments.append(segment)
        
        self.states.clear()
        return segments