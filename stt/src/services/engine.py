"""STT engine with model loading and initialization."""

from __future__ import annotations
import threading
import queue
import time
import math
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import Callable

# Import shared utilities
from common.logger import setup_stt_logger

# Import config
from ..core.config import (
    MODELS_CONFIG, 
    BEAM_SETTINGS, 
    SAMPLE_RATE
)

# Import schemas
from ..api.schemas import Segment, ASRResult

# Import helpers
from ..utils.helpers import safe_float_conversion

# Import exceptions
from ..core.exceptions import ModelLoadError

log = setup_stt_logger()

class SharedASRManager:
    """Manages ASR models and processing queue with parallel workers"""
    
    def __init__(self, num_workers: int = 4):
        self.models = {}
        self.model_locks = {}
        self.worker_queue = queue.Queue(maxsize=500)
        self.model_stats = defaultdict(lambda: {
            "transcriptions": 0,
            "errors": 0,
            "rejected_repetitive": 0
        })
        self.workers_running = False
        self.worker_threads = []
        self.deduplicator = TextDeduplicator()
        self.recent_texts = defaultdict(lambda: deque(maxlen=8))
        self.recent_texts_lock = threading.Lock()
        self._init_models()
        self._start_workers(num_workers)
    
    def _init_models(self):
        try:
            from faster_whisper import WhisperModel
            HAS_FASTER_WHISPER = True
        except:
            HAS_FASTER_WHISPER = False
            
        if not HAS_FASTER_WHISPER:
            log.error("Faster-Whisper not available!")
            return
        
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        except:
            device = "cpu"
            compute_type = "int8"
        
        for language in ["en", "hi", "es", "fr"]:
            config = MODELS_CONFIG.get(language, {})
            model_size = config.get("primary", "medium")
            try:
                log.info(f"Loading {language.upper()} model: {model_size}")
                model = WhisperModel(
                    model_size,
                    device=device,
                    compute_type=compute_type,
                    num_workers=2
                )
                self.models[language] = model
                self.model_locks[language] = threading.Lock()
                log.info(f"{language.upper()} model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load {language} model: {e}")
    
    def _start_workers(self, num_workers: int):
        self.workers_running = True
        for i in range(num_workers):
            thread = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        log.info(f"Started {num_workers} ASR worker threads")
    
    def _worker_loop(self, worker_id: int):
        log.info(f"Worker {worker_id} started")
        while self.workers_running:
            try:
                item = self.worker_queue.get(timeout=0.5)
                segment, callback = item
                result = self._transcribe_internal(segment)
                if callback:
                    callback(result)
                self.worker_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                log.exception(f"Worker {worker_id} error: {e}")
    
    def submit_segment(self, segment: Segment, callback: Callable):
        try:
            self.worker_queue.put((segment, callback), timeout=1.0)
        except queue.Full:
            log.warning(f"ASR queue full - dropping segment {segment.id}")
    
    def _transcribe_internal(self, segment: Segment) -> ASRResult:
        """Transcribe segment with word-level timestamps for MT/TTS"""
        t0 = time.time()
        model = self.models.get(segment.language)
        
        if model is None:
            return ASRResult(
                segment_id=segment.id,
                call_id=segment.call_id,
                speaker_id=segment.speaker_id,
                text="",
                words=[],
                confidence=0.0,
                language=segment.language,
                processing_time=time.time() - t0,
                model_used="none",
                timestamp=datetime.now().isoformat(),
                start_time=segment.start_time,
                end_time=segment.end_time
            )
        
        lock = self.model_locks.get(segment.language)
        
        try:
            with lock:
                beam_settings = BEAM_SETTINGS.get(segment.language, {"beam_size": 5, "best_of": 5})
                
                # Transcribe with word timestamps for MT/TTS
                segments_iter, info = model.transcribe(
                    segment.audio,
                    language=segment.language,
                    beam_size=beam_settings["beam_size"],
                    best_of=beam_settings["best_of"],
                    temperature=0.0,
                    condition_on_previous_text=True,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        threshold=0.5,
                        min_speech_duration_ms=250
                    ),
                    word_timestamps=True,  # CRITICAL for MT/TTS alignment
                    initial_prompt=None,
                    no_speech_threshold=0.6
                )
                
                segment_list = list(segments_iter)
                
                # Extract text and word-level info
                texts = []
                all_words = []
                
                for seg in segment_list:
                    if seg.text.strip():
                        texts.append(seg.text.strip())
                    
                    # Extract word timestamps for MT/TTS
                    if hasattr(seg, 'words') and seg.words:
                        for word in seg.words:
                            word_info = {
                                "word": word.word.strip(),
                                "start": round(word.start, 3),
                                "end": round(word.end, 3),
                                "confidence": round(word.probability, 4) if hasattr(word, 'probability') else 0.95
                            }
                            all_words.append(word_info)
                
                text = " ".join(texts)
                
                # Aggressive deduplication
                text = self.deduplicator.clean_whisper_artifacts(text)
                text = self.deduplicator.deduplicate(text)
                
                # Check for repetitiveness
                if self.deduplicator.is_repetitive(text):
                    log.warning(f"REJECTED REPETITIVE: {segment.speaker_id} - {text[:60]}...")
                    self.model_stats[segment.language]["rejected_repetitive"] += 1
                    text = ""
                    all_words = []
                
                # Cross-segment duplicate check
                if text:
                    with self.recent_texts_lock:
                        recent = self.recent_texts[segment.speaker_id]
                        text_lower = text.lower().strip()
                        if text_lower in [r.lower().strip() for r in recent]:
                            log.warning(f"REJECTED DUPLICATE: {segment.speaker_id} - {text[:60]}...")
                            text = ""
                            all_words = []
                        else:
                            recent.append(text)
                
                # Calculate confidence
                if segment_list and text:
                    confidences = []
                    for seg in segment_list:
                        if hasattr(seg, 'avg_logprob') and seg.avg_logprob is not None:
                            logprob = float(seg.avg_logprob)
                            prob = max(0.0, min(1.0, math.exp(max(-10, logprob))))
                            confidences.append(prob)
                    confidence = float(np.mean(confidences)) if confidences else 0.85
                else:
                    confidence = 0.0
                
                processing_time = time.time() - t0
                self.model_stats[segment.language]["transcriptions"] += 1
                
                return ASRResult(
                    segment_id=segment.id,
                    call_id=segment.call_id,
                    speaker_id=segment.speaker_id,
                    text=text,
                    words=all_words,  # Word-level timestamps for MT/TTS
                    confidence=confidence,
                    language=segment.language,
                    processing_time=processing_time,
                    model_used=f"faster_whisper_{MODELS_CONFIG[segment.language]['primary']}",
                    timestamp=datetime.now().isoformat(),
                    start_time=segment.start_time,
                    end_time=segment.end_time
                )
        
        except Exception as e:
            log.exception(f"ASR transcription error: {e}")
            self.model_stats[segment.language]["errors"] += 1
            return ASRResult(
                segment_id=segment.id,
                call_id=segment.call_id,
                speaker_id=segment.speaker_id,
                text="",
                words=[],
                confidence=0.0,
                language=segment.language,
                processing_time=time.time() - t0,
                model_used="error",
                timestamp=datetime.now().isoformat(),
                start_time=segment.start_time,
                end_time=segment.end_time
            )
    
    def stop(self):
        self.workers_running = False
        for thread in self.worker_threads:
            thread.join(timeout=2.0)
        log.info("ASR workers stopped")

# Text Deduplicator (moved from main)
class TextDeduplicator:
    """Advanced deduplication to eliminate repetitions"""
    
    @staticmethod
    def remove_consecutive_duplicates(text: str) -> str:
        """Remove consecutive duplicate words"""
        words = text.split()
        if len(words) < 2:
            return text
        result = [words[0]]
        for i in range(1, len(words)):
            if words[i].lower() != words[i-1].lower():
                result.append(words[i])
        return ' '.join(result)
    
    @staticmethod
    def remove_phrase_duplicates(text: str) -> str:
        """Remove duplicate phrases (2-5 words)"""
        words = text.split()
        if len(words) < 4:
            return text
        for phrase_len in range(5, 1, -1):
            i = 0
            result = []
            while i < len(words):
                if i + phrase_len * 2 <= len(words):
                    phrase1 = ' '.join(words[i:i+phrase_len]).lower()
                    phrase2 = ' '.join(words[i+phrase_len:i+phrase_len*2]).lower()
                    if phrase1 == phrase2:
                        result.extend(words[i:i+phrase_len])
                        i += phrase_len * 2
                        continue
                result.append(words[i])
                i += 1
            words = result
        return ' '.join(words)
    
    @staticmethod
    def deduplicate(text: str, threshold: float = 0.85) -> str:
        """Main deduplication pipeline"""
        if not text or len(text) < 10:
            return text
        
        # Step 1: Remove consecutive duplicates
        text = TextDeduplicator.remove_consecutive_duplicates(text)
        
        # Step 2: Remove phrase duplicates
        text = TextDeduplicator.remove_phrase_duplicates(text)
        
        # Step 3: Sentence-level deduplication
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        unique_sentences = []
        seen = set()
        for sent in sentences:
            sent_lower = sent.lower()
            if sent_lower not in seen:
                unique_sentences.append(sent)
                seen.add(sent_lower)
        
        if len(unique_sentences) == 0:
            return ""
        
        result = '. '.join(unique_sentences)
        if result and not result.endswith('.'):
            result += '.'
        return result
    
    @staticmethod
    def is_repetitive(text: str, threshold: float = 0.35) -> bool:
        """Check if text is overly repetitive"""
        if not text or len(text) < 20:
            return False
        words = text.lower().split()
        if len(words) < 3:
            return False
        unique_count = len(set(words))
        total_count = len(words)
        return (unique_count / total_count) < threshold
    
    @staticmethod
    def clean_whisper_artifacts(text: str) -> str:
        """Clean Whisper-specific artifacts"""
        text = ' '.join(text.split())
        while '  ' in text:
            text = text.replace('  ', ' ')
        while '..' in text:
            text = text.replace('..', '.')
        while ',,' in text:
            text = text.replace(',,', ',')
        text = text.replace(',.', '.')
        text = text.replace(', .', '.')
        return text.strip()