"""Core business logic for MT processing."""

from __future__ import annotations
import re
import json
import time
import uuid
import numpy as np
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
from threading import RLock

# Import NLTK
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import sent_tokenize, word_tokenize
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

# Import config
from ..core.config import (
    LANGUAGE_CODES,
    LANGUAGE_NAMES,
    IDIOM_DATABASE,
    HOMOPHONES_DB,
    SYNONYMS_DB,
    GRAMMAR_RULES,
    PUNCTUATION_RULES
)

# Import schemas
from ..api.schemas import ProcessedToken, TranslationResult

# Import helpers
from ..utils.helpers import ASRTokenProcessor

class PathRouter:
    """Route translations to appropriate paths"""
    
    @staticmethod
    def determine_path(speaker_id: str, source_lang: str, target_lang: str) -> str:
        """Determine which path (A->B or B->A) to use"""
        path = "path_1" if speaker_id.lower() in ["speaker_a", "a", "client_a"] else "path_2"
        
        # Import logger
        from common.logger import setup_mt_logger
        log = setup_mt_logger()
        log.debug(f"PathRouter: {speaker_id} {source_lang}->{target_lang} -> {path}")
        return path

class DualLaneContextManager:
    """Manage separate context for both speakers"""
    
    def __init__(self, context_window_size: int = 5):
        self.context_window_size = context_window_size
        self.speaker_contexts = {
            "speaker_A": deque(maxlen=context_window_size),
            "speaker_B": deque(maxlen=context_window_size),
            "call_context": {}
        }
        self.lock = RLock()
    
    def add_context(self, speaker_id: str, text: str, translation: str):
        """Add to speaker-specific context"""
        with self.lock:
            speaker_key = "speaker_A" if speaker_id.lower() in ["speaker_a", "a"] else "speaker_B"
            self.speaker_contexts[speaker_key].append({
                "original": text,
                "translation": translation,
                "timestamp": time.time()
            })
            
            # Import logger
            from common.logger import setup_mt_logger
            log = setup_mt_logger()
            log.debug(f"Context added for {speaker_key}")
    
    def get_context(self, speaker_id: str) -> List[Dict[str, Any]]:
        """Get context for specific speaker"""
        with self.lock:
            speaker_key = "speaker_A" if speaker_id.lower() in ["speaker_a", "a"] else "speaker_B"
            return list(self.speaker_contexts[speaker_key])
    
    def clear_context(self, call_id: str):
        """Clear context for new call"""
        with self.lock:
            for key in self.speaker_contexts:
                if key != "call_context":
                    self.speaker_contexts[key].clear()
            
            # Import logger
            from common.logger import setup_mt_logger
            log = setup_mt_logger()
            log.info(f"Context cleared for call {call_id}")

class GrammarAnalyzer:
    """Analyze and improve grammar"""
    
    def __init__(self):
        self.rules = GRAMMAR_RULES
    
    def analyze_grammar(self, text: str, language: str) -> Tuple[bool, str]:
        """Analyze grammar correctness"""
        sentences = sent_tokenize(text) if _HAS_NLTK else text.split(".")
        
        issues = []
        for sent in sentences:
            if len(sent.strip()) > 0:
                if language == "eng_Latn":
                    if not sent[0].isupper():
                        issues.append(f"Capitalization: {sent}")
        
        is_valid = len(issues) == 0
        
        # Import logger
        from common.logger import setup_mt_logger
        log = setup_mt_logger()
        log.debug(f"Grammar analysis for {language}: valid={is_valid}, issues={len(issues)}")
        return is_valid, "; ".join(issues)

class HomophoneResolver:
    """Resolve homophones based on context"""
    
    def __init__(self):
        self.homophones = HOMOPHONES_DB
    
    def resolve_homophones(self, text: str, language: str, context: str = "") -> str:
        """Resolve homophones using context"""
        resolved_text = text
        lang_homophones = self.homophones.get(language, {})
        
        for word, alternatives in lang_homophones.items():
            if word.lower() in text.lower():
                for alt in alternatives:
                    if alt.lower() in context.lower():
                        pattern = re.compile(re.escape(word), re.IGNORECASE)
                        resolved_text = pattern.sub(alt, resolved_text)
                        
                        # Import logger
                        from common.logger import setup_mt_logger
                        log = setup_mt_logger()
                        log.debug(f"Homophone resolved: {word} -> {alt}")
        
        return resolved_text

class SynonymSelector:
    """Select context-appropriate synonyms"""
    
    def __init__(self):
        self.synonyms = SYNONYMS_DB
    
    def select_synonym(self, word: str, language: str, context: str = "") -> str:
        """Select appropriate synonym based on context"""
        lang_synonyms = self.synonyms.get(language, {})
        word_lower = word.lower()
        
        if word_lower in lang_synonyms:
            selected = lang_synonyms[word_lower][0]
            
            # Import logger
            from common.logger import setup_mt_logger
            log = setup_mt_logger()
            log.debug(f"Synonym selected: {word} -> {selected}")
            return selected
        
        return word

class IdiomHandler:
    """Handle idioms and phrases"""
    
    def __init__(self):
        self.idioms = IDIOM_DATABASE
    
    def detect_and_preserve_idioms(self, text: str, language: str, speaker_id: str = "") -> Tuple[str, List[str]]:
        """Detect idioms and prepare text for translation"""
        
        detected_idioms = []
        processed_text = text
        
        lang_idioms = self.idioms.get(language, {})
        
        for idiom, (meaning, idiom_type, context) in lang_idioms.items():
            if idiom.lower() in processed_text.lower():
                detected_idioms.append(f"{idiom}:{meaning}")
                pattern = re.compile(re.escape(idiom), re.IGNORECASE)
                processed_text = pattern.sub(meaning, processed_text)
                
                # Import logger
                from common.logger import setup_mt_logger
                log = setup_mt_logger()
                log.debug(f"[{speaker_id}] Idiom detected: '{idiom}' -> '{meaning}'")
        
        return processed_text, detected_idioms

class PunctuationRestorer:
    """Restore punctuation and truecasing"""
    
    def __init__(self, target_language: str):
        self.target_language = target_language
        self.rules = PUNCTUATION_RULES.get(target_language, {})
    
    def restore_punctuation(self, text: str, tokens: Optional[List[ProcessedToken]] = None) -> str:
        """Restore proper punctuation"""
        
        try:
            if _HAS_NLTK:
                sentences = sent_tokenize(text)
            else:
                sentences = text.split(".")
        except Exception:
            sentences = text.split(".")
        
        result = []
        for i, sent in enumerate(sentences):
            sent = sent.strip()
            if not sent:
                continue
            
            sent = sent[0].upper() + sent[1:] if len(sent) > 0 else ""
            
            if i < len(sentences) - 1:
                sent = sent.rstrip(".?!।") + self.rules.get("sentence_end", ".")
            else:
                if not sent.endswith((".", "?", "!", "।")):
                    sent = sent + self.rules.get("sentence_end", ".")
            
            result.append(sent)
        
        return " ".join(result)

class AlignmentExtractor:
    """Extract cross-attention alignments"""
    
    def extract_alignments(self,
                         source_tokens: List[str],
                         target_tokens: List[str]) -> Dict[int, List[int]]:
        """Extract alignments from source to target"""
        
        alignments = {}
        num_src = len(source_tokens)
        num_tgt = len(target_tokens)
        
        for src_idx in range(num_src):
            tgt_idx = int((src_idx / max(num_src, 1)) * num_tgt)
            tgt_idx = min(tgt_idx, num_tgt - 1)
            alignments[src_idx] = [tgt_idx]
        
        return alignments

class ConfidenceFusion:
    """Fuse MT and ASR confidence"""
    
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha
    
    def fuse_confidences(self,
                       mt_confidence: float,
                       asr_confidences: List[float],
                       alignments: Dict[int, List[int]]) -> float:
        """Fuse MT and ASR confidences"""
        
        if not asr_confidences:
            return mt_confidence
        
        asr_conf = np.mean(asr_confidences)
        fused = self.alpha * mt_confidence + (1 - self.alpha) * asr_conf
        
        return float(np.clip(fused, 0.0, 1.0))

class TTSPreparator:
    """Prepare TTS-ready output with SSML and prosody"""
    
    def __init__(self):
        self.punctuation_restorer = {}
    
    def prepare_for_tts(self,
                       translation_result: TranslationResult,
                       source_tokens: List[ProcessedToken],
                       target_tokens: List[str],
                       alignments: Dict[int, List[int]],
                       speaker_embedding: Optional[List[float]] = None) -> TranslationResult:
        """Prepare translation result for TTS with voice cloning support"""
        
        call_id = translation_result.call_id
        speaker_id = translation_result.speaker_id
        source_lang = translation_result.source_language
        target_lang = translation_result.target_language
        translated_text = translation_result.translated_text
        processing_path = translation_result.processing_path
        
        log.debug(f"[{call_id}] Preparing TTS data for {speaker_id} in {target_lang}")
        
        # Generate target word timestamps for TTS prosody
        target_word_timestamps = self._generate_timestamps(source_tokens, target_tokens, alignments)
        
        # Generate pause hints based on source timing
        pause_hints = self._generate_pause_hints(source_tokens, alignments)
        
        # Generate SSML markup
        ssml = self._generate_ssml(translated_text, pause_hints, target_lang)
        
        # Restore punctuation for better TTS
        if target_lang not in self.punctuation_restorer:
            self.punctuation_restorer[target_lang] = PunctuationRestorer(target_lang)
        
        restorer = self.punctuation_restorer[target_lang]
        tts_text = restorer.restore_punctuation(translated_text, source_tokens)
        
        # Generate prosody hints
        prosody_hints = {
            "tone_pattern": "neutral",
            "speech_rate": "normal",
            "speaker_context": True,
        }
        
        # Prepare the final result with voice cloning support
        result = TranslationResult(
            session_id=translation_result.session_id,
            call_id=call_id,
            speaker_id=speaker_id,
            segment_id=translation_result.segment_id,
            source_language=source_lang,
            target_language=target_lang,
            source_text=translation_result.source_text,
            translated_text=translated_text,
            tts_text=tts_text,
            processing_path=processing_path,
            source_words=[t.text for t in source_tokens],
            target_words=target_tokens,
            word_alignment=alignments,
            target_word_timestamps=target_word_timestamps,
            confidence=translation_result.confidence,
            ssml=ssml,
            pause_hints=pause_hints,
            prosody_hints=prosody_hints,
            processing_time_ms=translation_result.processing_time_ms,
            cache_hit=translation_result.cache_hit,
            # Add speaker embedding for voice cloning
            speaker_embedding=speaker_embedding if speaker_embedding else []
        )
        return result

    def _generate_timestamps(self,
                            source_tokens: List[ProcessedToken],
                            target_tokens: List[str],
                            alignments: Dict[int, List[int]]) -> List[List[float]]:
        """Generate timestamps for target words - FIX: Better edge case handling"""
        
        target_timestamps = []
        
        for tgt_idx in range(len(target_tokens)):
            aligned_sources = []
            
            # FIX #5: Find all source tokens aligned to this target token
            for src_idx, tgt_indices in alignments.items():
                if tgt_idx in tgt_indices and src_idx < len(source_tokens):
                    aligned_sources.append(source_tokens[src_idx])
            
            # FIX #5: Better edge case handling
            if aligned_sources:
                # Use actual aligned source times
                start_ms = min(t.start_ms for t in aligned_sources)
                end_ms = max(t.end_ms for t in aligned_sources)
            elif source_tokens:
                # If no alignment found, use average of all source tokens
                start_ms = np.mean([t.start_ms for t in source_tokens])
                end_ms = start_ms + 100.0
            else:
                # Last resort fallback
                start_ms = 0.0
                end_ms = 100.0
            
            target_timestamps.append([start_ms, end_ms])
        
        return target_timestamps
    
    def _generate_pause_hints(self,
                             source_tokens: List[ProcessedToken],
                             alignments: Dict[int, List[int]]) -> List[Dict[str, Any]]:
        """Generate pause hints based on gaps - FIX: Handle negative gaps"""
        
        pause_hints = []
        
        if len(source_tokens) < 2:
            return pause_hints
        
        for i in range(len(source_tokens) - 1):
            # FIX #4: Calculate gap properly
            gap_ms = source_tokens[i + 1].start_ms - source_tokens[i].end_ms
            
            # FIX #4: Skip if gap is negative (tokens overlap) or too small
            if gap_ms > 250:
                pause_hints.append({
                    "after_token": source_tokens[i].text,
                    "pause_ms": int(gap_ms),
                })
        
        return pause_hints
    
    def _generate_ssml(self, text: str, pause_hints: List[Dict[str, Any]], target_language: str) -> str:
        """Generate SSML markup - FIX: Better word tokenization with punctuation handling"""
        
        lang_code = self._get_lang_code(target_language)
        ssml = f'<speak lang="{lang_code}">'
        
        # FIX #3: Better word tokenization that preserves punctuation structure
        words = self._tokenize_words_for_ssml(text)
        
        for i, word in enumerate(words):
            ssml += word
            
            # FIX #3: Match pause hints against clean word (without punctuation)
            clean_word = self._clean_word(word)
            
            for hint in pause_hints:
                if hint.get("after_token", "").lower() == clean_word.lower():
                    pause_ms = hint.get("pause_ms", 300)
                    ssml += f'<break time="{pause_ms}ms"/>'
            
            if i < len(words) - 1:
                ssml += " "
        
        ssml += "</speak>"
        return ssml
    
    def _tokenize_words_for_ssml(self, text: str) -> List[str]:
        """
        FIX #3: Proper word tokenization that handles punctuation
        Preserves sentence structure while splitting words
        """
        words = []
        current_word = ""
        
        for char in text:
            if char in " \t\n":
                if current_word:
                    words.append(current_word)
                    current_word = ""
            else:
                current_word += char
        
        if current_word:
            words.append(current_word)
        
        return words if words else [text]
    
    def _clean_word(self, word: str) -> str:
        """
        FIX #3: Extract clean word without ending punctuation
        Used for matching pause hints
        """
        import re
        # Remove trailing punctuation for matching
        match = re.match(r'^(.*?)([.,!?;:।]*)$', word)
        if match:
            return match.group(1)
        return word
    
    def _get_lang_code(self, target_language: str) -> str:
        """Get language code for SSML"""
        codes = {
            "eng_Latn": "en",
            "hin_Deva": "hi",
            "spa_Latn": "es",
            "fra_Latn": "fr",
        }
        return codes.get(target_language, "en")

class DualLaneErrorRecovery:
    """Handle errors in dual-lane processing"""
    
    @staticmethod
    def handle_path_error(error: Exception, path: str, call_id: str, speaker_id: str) -> Dict[str, Any]:
        """Handle error gracefully without crashing other path"""
        
        error_msg = f"Error in {path} for {speaker_id}: {str(error)}"
        
        # Import logger
        from common.logger import setup_mt_logger
        log = setup_mt_logger()
        log.error(f"[{call_id}] {error_msg}")
        
        return {
            "error": True,
            "path": path,
            "speaker_id": speaker_id,
            "call_id": call_id,
            "error_message": error_msg,
            "fallback": True
        }

class DualLanePerformanceManager:
    """Monitor and optimize dual-path performance"""
    
    def __init__(self):
        self.path_metrics = {
            "path_1": {"latencies": [], "confidences": [], "cache_hits": 0},
            "path_2": {"latencies": [], "confidences": [], "cache_hits": 0}
        }
        self.lock = RLock()
    
    def record_performance(self, path: str, latency_ms: float, confidence: float, cache_hit: bool):
        """Record metrics for each path"""
        with self.lock:
            self.path_metrics[path]["latencies"].append(latency_ms)
            self.path_metrics[path]["confidences"].append(confidence)
            if cache_hit:
                self.path_metrics[path]["cache_hits"] += 1
    
    def get_statistics(self, path: str) -> Dict[str, Any]:
        """Get performance statistics for path"""
        with self.lock:
            metrics = self.path_metrics[path]
            if not metrics["latencies"]:
                return {}
            
            return {
                "avg_latency_ms": np.mean(metrics["latencies"]),
                "avg_confidence": np.mean(metrics["confidences"]),
                "cache_hits": metrics["cache_hits"],
                "total_calls": len(metrics["latencies"])
            }