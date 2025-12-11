"""Main entry point for the MT system."""

from __future__ import annotations
import json
import sys
import time
import uuid
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock, RLock

# Import shared utilities
from common.utils import gen_id, safe_float_conversion, compute_rms
# from common.logger import setup_mt_logger

# Import config
from .core.config import *

# Import schemas
from .api.schemas import *

# Import services
from .services.engine import NLLBTranslationEngine
from .services.processing import (
    PathRouter,
    DualLaneContextManager,
    GrammarAnalyzer,
    HomophoneResolver,
    SynonymSelector,
    IdiomHandler,
    AlignmentExtractor,
    ConfidenceFusion,
    TTSPreparator,
    DualLaneErrorRecovery,
    DualLanePerformanceManager
)

# Import helpers
from .utils.helpers import ASRTokenProcessor

# Setup logger
from common.logger import setup_mt_logger
log = setup_mt_logger()

# Dual-Lane Orchestrator

class DualLaneOrchestrator:
    """Dual-lane MT orchestrator with concurrent processing"""
    
    def __init__(self, 
                 model_name: str = "facebook/nllb-200-1.3B",
                 device: str = "cpu",
                 max_workers: int = 2):
        
        log.info("Initializing Dual-Lane MT Pipeline v9.0 - ULTIMATE TOP-TIER")
        
        # Core components
        self.translation_engine = NLLBTranslationEngine(model_name, device)
        self.asr_processor = ASRTokenProcessor()
        self.idiom_handler = IdiomHandler()
        self.homophone_resolver = HomophoneResolver()
        self.synonym_selector = SynonymSelector()
        self.grammar_analyzer = GrammarAnalyzer()
        self.alignment_extractor = AlignmentExtractor()
        self.confidence_fusion = ConfidenceFusion()
        self.tts_preparator = TTSPreparator()
        self.path_router = PathRouter()
        
        # Dual-lane components
        self.context_manager = DualLaneContextManager()
        self.performance_manager = DualLanePerformanceManager()
        self.error_recovery = DualLaneErrorRecovery()
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.call_lock = RLock()
        
        log.info("All dual-lane components initialized")
    
    def translate_dual_lane_concurrent(self,
                                      text_A: str,
                                      text_B: str,
                                      source_lang_A: str,
                                      source_lang_B: str,
                                      target_lang_A: str,
                                      target_lang_B: str,
                                      call_id: str,
                                      asr_tokens_A: Optional[List[Dict[str, Any]]] = None,
                                      asr_tokens_B: Optional[List[Dict[str, Any]]] = None,
                                      speaker_embedding_A: Optional[List[float]] = None,
                                      speaker_embedding_B: Optional[List[float]] = None) -> Dict[str, TranslationResult]:
        """
        Process both paths CONCURRENTLY:
        - Path 1: Speaker A (text_A) -> translate to target_lang_A
        - Path 2: Speaker B (text_B) -> translate to target_lang_B
        """
        
        t0 = time.time()
        
        log.info(f"[{call_id}] Starting dual-lane concurrent processing")
        log.info(f"[{call_id}] Path 1: {source_lang_A} -> {target_lang_A}")
        log.info(f"[{call_id}] Path 2: {source_lang_B} -> {target_lang_B}")
        
        try:
            # Submit both paths concurrently
            future_path_1 = self.executor.submit(
                self._translate_single_path,
                text_A, source_lang_A, target_lang_A,
                call_id, "speaker_A", asr_tokens_A, "path_1", speaker_embedding_A
            )
            
            future_path_2 = self.executor.submit(
                self._translate_single_path,
                text_B, source_lang_B, target_lang_B,
                call_id, "speaker_B", asr_tokens_B, "path_2", speaker_embedding_B
            )
            
            # Wait for both to complete
            result_path_1 = future_path_1.result(timeout=30)
            result_path_2 = future_path_2.result(timeout=30)
            
            # Store in context
            if isinstance(result_path_1, TranslationResult):
                self.context_manager.add_context("speaker_A", text_A, result_path_1.translated_text)
            if isinstance(result_path_2, TranslationResult):
                self.context_manager.add_context("speaker_B", text_B, result_path_2.translated_text)
            
            total_time = (time.time() - t0) * 1000
            
            log.info(f"[{call_id}] Dual-lane processing complete | Total time: {total_time:.0f}ms")
            
            return {
                "path_1": result_path_1,
                "path_2": result_path_2,
                "total_time_ms": total_time,
                "concurrent": True
            }
        
        except Exception as e:
            log.error(f"[{call_id}] Dual-lane error: {e}")
            import traceback
            log.debug(traceback.format_exc())
            
            return {
                "path_1": None,
                "path_2": None,
                "error": str(e),
                "total_time_ms": (time.time() - t0) * 1000
            }
    
    def _translate_single_path(self,
                              text: str,
                              source_lang: str,
                              target_lang: str,
                              call_id: str,
                              speaker_id: str,
                              asr_tokens: Optional[List[Dict[str, Any]]],
                              processing_path: str,
                              speaker_embedding: Optional[List[float]] = None) -> TranslationResult:
        """Process single translation path with voice cloning support"""
        
        t0 = time.time()
        
        # Normalize languages
        source_lang = self._normalize_lang(source_lang)
        target_lang = self._normalize_lang(target_lang)
        
        log.debug(f"[{call_id}] {processing_path} processing started")
        
        try:
            # Process tokens
            processed_tokens = []
            if asr_tokens:
                processed_tokens = self.asr_processor.process_tokens(
                    asr_tokens, call_id, speaker_id, source_lang, processing_path
                )
            else:
                words = text.split()
                duration_ms = 2000
                per_word = duration_ms / len(words) if words else 100
                
                for i, word in enumerate(words):
                    processed_tokens.append(ProcessedToken(
                        text=word,
                        start_ms=i * per_word,
                        end_ms=(i + 1) * per_word,
                        confidence=0.95,
                        call_id=call_id,
                        speaker_id=speaker_id,
                        source_language=source_lang,
                        processing_path=processing_path,
                        source_words=[word]
                    ))
            
            # Idiom handling
            idiom_processed_text, detected_idioms = self.idiom_handler.detect_and_preserve_idioms(
                text, source_lang, speaker_id
            )
            
            # Get speaker context for homophone resolution
            speaker_context = self.context_manager.get_context(speaker_id)
            
            # Dual-lane routing
            routed_source_lang = self.path_router.route_language(source_lang, target_lang, processing_path)
            
            # Translation with context awareness
            translation_result = self.translation_engine.translate_with_context(
                idiom_processed_text,
                routed_source_lang,
                target_lang,
                speaker_context,
                speaker_id
            )
            
            # Post-processing
            translated_text = translation_result.get("translated_text", "")
            
            # Grammar analysis
            grammar_issues = self.grammar_analyzer.analyze_grammar(
                translated_text, target_lang, speaker_id
            )
            
            # Homophone resolution
            resolved_text = self.homophone_resolver.resolve_homophones(
                translated_text, target_lang, speaker_id
            )
            
            # Synonym selection
            synonym_text = self.synonym_selector.select_appropriate_synonyms(
                resolved_text, target_lang, speaker_id
            )
            
            # Idiom restoration
            final_text = self.idiom_handler.restore_idioms(
                synonym_text, detected_idioms, target_lang
            )
            
            # Update translation result
            translation_result["translated_text"] = final_text
            
            # Word alignment extraction
            target_words = final_text.split()
            alignments = self.alignment_extractor.extract_alignments(
                processed_tokens, target_words
            )
            
            # Confidence fusion
            fused_confidence = self.confidence_fusion.fuse_confidences(
                translation_result.get("confidence", 0.8),
                grammar_issues
            )
            
            # Prepare for TTS
            tts_result = self.tts_preparator.prepare_for_tts(
                translation_result, processed_tokens, target_words, alignments, speaker_embedding
            )
            
            # Create TranslationResult
            result = TranslationResult(
                source_text=text,
                translated_text=final_text,
                source_language=source_lang,
                target_language=target_lang,
                confidence=fused_confidence,
                processing_time_ms=(time.time() - t0) * 1000,
                call_id=call_id,
                speaker_id=speaker_id,
                tts_text=tts_result.tts_text,
                tts_timing=tts_result.tts_timing,
                word_alignments=tts_result.word_alignments,
                speaker_embedding=speaker_embedding
            )
            
            log.debug(f"[{call_id}] {processing_path} processing complete | Time: {(time.time() - t0) * 1000:.0f}ms")
            return result
            
        except Exception as e:
            log.error(f"[{call_id}] {processing_path} error: {e}")
            import traceback
            log.debug(traceback.format_exc())
            
            # Return error result
            return TranslationResult(
                source_text=text,
                translated_text=f"[ERROR] {str(e)}",
                source_language=source_lang,
                target_language=target_lang,
                confidence=0.0,
                processing_time_ms=(time.time() - t0) * 1000,
                call_id=call_id,
                speaker_id=speaker_id,
                tts_text=f"[ERROR] {str(e)}",
                tts_timing=[],
                word_alignments={},
                speaker_embedding=speaker_embedding
            )
    
    def _normalize_lang(self, lang: str) -> str:
        """Normalize language code to NLLB format"""
        return LANGUAGE_CODES.get(lang.lower(), lang)

# Interactive Demo

def interactive_dual_lane_demo():
    """Interactive demo for dual-lane MT"""
    
    log.info("DUAL-LANE MT PIPELINE v9.0 - ULTIMATE TOP-TIER 10/10")
    log.info("NLLB-1.3B | 4 Languages | Concurrent A<->B Translation | TTS-Ready")
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        orchestrator = DualLaneOrchestrator(device=device)
    except Exception as e:
        log.error(f"Initialization error: {e}")
        return
    
    log.info("Supported languages:")
    for code, name in LANGUAGE_NAMES.items():
        short_codes = [k for k, v in LANGUAGE_CODES.items() if v == code]
        log.info(f"  {short_codes[0]:3} -> {code:10} ({name})")
    
    call_id = f"meeting_{uuid.uuid4().hex[:8]}"
    
    while True:
        try:
            # Get path configuration
            try:
                src_A = input("Speaker A source language (en/hi/es/fr) or 'quit': ").strip().lower()
            except EOFError:
                log.debug("EOF when reading a line")
                break
            except KeyboardInterrupt:
                log.info("Demo interrupted by user.")
                break
                
            if src_A == "quit":
                break
            
            try:
                tgt_A = input("Speaker A target language (en/hi/es/fr): ").strip().lower()
            except EOFError:
                log.debug("EOF when reading a line")
                break
            except KeyboardInterrupt:
                log.info("Demo interrupted by user.")
                break
                
            try:
                src_B = input("Speaker B source language (en/hi/es/fr): ").strip().lower()
            except EOFError:
                log.debug("EOF when reading a line")
                break
            except KeyboardInterrupt:
                log.info("Demo interrupted by user.")
                break
                
            try:
                tgt_B = input("Speaker B target language (en/hi/es/fr): ").strip().lower()
            except EOFError:
                log.debug("EOF when reading a line")
                break
            except KeyboardInterrupt:
                log.info("Demo interrupted by user.")
                break
            
            # Validate
            src_A_norm = LANGUAGE_CODES.get(src_A, src_A)
            tgt_A_norm = LANGUAGE_CODES.get(tgt_A, tgt_A)
            src_B_norm = LANGUAGE_CODES.get(src_B, src_B)
            tgt_B_norm = LANGUAGE_CODES.get(tgt_B, tgt_B)
            
            if (src_A_norm not in LANGUAGE_NAMES or tgt_A_norm not in LANGUAGE_NAMES or
                src_B_norm not in LANGUAGE_NAMES or tgt_B_norm not in LANGUAGE_NAMES):
                log.warning("Invalid language codes")
                continue
            
            log.info(f"Speaker A ({LANGUAGE_NAMES[src_A_norm]} -> {LANGUAGE_NAMES[tgt_A_norm]}): ")
            try:
                text_A = input().strip()
            except EOFError:
                log.debug("EOF when reading a line")
                continue
            except KeyboardInterrupt:
                log.info("Demo interrupted by user.")
                break
            
            log.info(f"Speaker B ({LANGUAGE_NAMES[src_B_norm]} -> {LANGUAGE_NAMES[tgt_B_norm]}): ")
            try:
                text_B = input().strip()
            except EOFError:
                log.debug("EOF when reading a line")
                continue
            except KeyboardInterrupt:
                log.info("Demo interrupted by user.")
                break
            
            if not text_A or not text_B:
                continue
            
            # Process dual-lane
            results = orchestrator.translate_dual_lane_concurrent(
                text_A=text_A,
                text_B=text_B,
                source_lang_A=src_A,
                source_lang_B=src_B,
                target_lang_A=tgt_A,
                target_lang_B=tgt_B,
                call_id=call_id
            )
            
            # Extract results
            result_A = results.get("path_1")
            result_B = results.get("path_2")
            
            if result_A and isinstance(result_A, TranslationResult):
                log.info(f"PATH 1 - Speaker A")
                log.info(f"Source: {result_A.source_text}")
                log.info(f"Target: {result_A.translated_text}")
                log.info(f"TTS Text: {result_A.tts_text}")
                log.info(f"Confidence: {result_A.confidence:.3f}")
                log.info(f"Latency: {result_A.processing_time_ms:.0f}ms")
            
            if result_B and isinstance(result_B, TranslationResult):
                log.info(f"PATH 2 - Speaker B")
                log.info(f"Source: {result_B.source_text}")
                log.info(f"Target: {result_B.translated_text}")
                log.info(f"TTS Text: {result_B.tts_text}")
                log.info(f"Confidence: {result_B.confidence:.3f}")
                log.info(f"Latency: {result_B.processing_time_ms:.0f}ms")
            
            log.info(f"Total concurrent time: {results.get('total_time_ms', 0):.0f}ms")
        
        except Exception as e:
            log.error(f"Error: {e}")
            continue

if __name__ == "__main__":
    interactive_dual_lane_demo()
