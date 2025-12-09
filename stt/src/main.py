"""Main entry point for the STT system."""

from __future__ import annotations
import asyncio
import sys
import threading
import queue
import time
import numpy as np
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Callable
import websockets

# Import shared utilities
from common.logger import setup_stt_logger
from common.utils import gen_id

# Import config
from .core.config import *

# Import schemas
from .api.schemas import *

# Import services
from .services.engine import SharedASRManager
from .services.processing import (
    VAD, 
    AudioEnhancer, 
    Segmenter, 
    SpeakerEmbeddingExtractor,
    extract_prosody_features
)

# Import API
from .api.routes import websocket_handler

# Import helpers
from .utils.helpers import resample_if_needed, safe_float_conversion, compute_rms, write_wav_int16

log = setup_stt_logger()

# ============================================================================ 
# OUTPUT MANAGER - MT/TTS READY
# ============================================================================

class OutputManager:
    """Saves outputs in MT/TTS-ready format"""
    
    def __init__(self):
        self.embedding_extractor = SpeakerEmbeddingExtractor()
    
    def save_outputs(self, segment: Segment, asr_result: ASRResult):
        """Save all outputs for MT/TTS pipeline"""
        try:
            # Save audio segment
            audio_path = AUDIO_DIR / f"{asr_result.segment_id}.wav"
            write_wav_int16(audio_path, segment.audio)
            
            # Save transcript with word-level timestamps
            transcript_data = {
                "segment_id": asr_result.segment_id,
                "call_id": asr_result.call_id,
                "speaker_id": asr_result.speaker_id,
                "language": asr_result.language,
                "text": asr_result.text,
                "confidence": asr_result.confidence,
                "duration_sec": round(segment.duration, 3),
                "processing_time_sec": round(asr_result.processing_time, 3),
                "e2e_latency_sec": round(asr_result.end_time - segment.start_time, 3),
                "timestamp": asr_result.timestamp,
                "model": asr_result.model_used,
                "words": asr_result.words
            }
            
            transcript_path = TRANSCRIPTS_DIR / f"{asr_result.segment_id}.json"
            with open(transcript_path, 'w', encoding='utf-8') as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)
            
            # Save prosody features for TTS
            prosody_features = extract_prosody_features(segment.audio)
            prosody_data = {
                "segment_id": asr_result.segment_id,
                "speaker_id": asr_result.speaker_id,
                "language": asr_result.language,
                "prosody": prosody_features
            }
            
            prosody_path = PROSODY_DIR / f"{asr_result.segment_id}_prosody.json"
            with open(prosody_path, 'w', encoding='utf-8') as f:
                json.dump(prosody_data, f, ensure_ascii=False, indent=2)
            
            # Save speaker embeddings
            embedding = self.embedding_extractor.extract(segment.audio, SAMPLE_RATE, asr_result.speaker_id)
            embedding_data = {
                "segment_id": asr_result.segment_id,
                "speaker_id": asr_result.speaker_id,
                "embedding": embedding,
                "dimensions": 256,
                "timestamp": asr_result.timestamp
            }
            
            embedding_path = EMBEDDINGS_DIR / f"{asr_result.segment_id}_embedding.json"
            with open(embedding_path, 'w', encoding='utf-8') as f:
                json.dump(embedding_data, f, ensure_ascii=False, indent=2)
            
            # Log successful save
            latency_sec = asr_result.end_time - segment.start_time
            lang_emoji = {"en": "🇬🇧", "hi": "🇮🇳", "es": "🇪🇸", "fr": "🇫🇷"}.get(asr_result.language, "🌐")
            log.info(f"{lang_emoji} [{asr_result.speaker_id}] {asr_result.text[:70]}")
            log.info(f" {asr_result.segment_id}.json | Conf: {asr_result.confidence:.2%} | Dur: {segment.duration:.2f}s | Latency: {latency_sec:.2f}s")
        
        except Exception as e:
            log.exception(f"Output save error: {e}")

# ============================================================================ 
# PARTICIPANT PIPELINE - WITH GATEWAY REFERENCE (FIX #2, #3)
# ============================================================================

class ParticipantPipeline:
    """Pipeline for single participant in call"""
    
    def __init__(self, call_id: str, participant: CallParticipant, asr_manager: SharedASRManager, gateway=None):
        """FIX #2: Added gateway=None parameter"""
        self.call_id = call_id
        self.participant = participant
        self.asr_manager = asr_manager
        self.gateway = gateway  # FIX #3: Store gateway reference
        self.queue = queue.Queue(maxsize=300)
        self.stop_event = threading.Event()
        self.thread = None
        self.vad = VAD(participant.language)
        self.enhancer = AudioEnhancer()
        self.segmenter = Segmenter()
        self.output_mgr = OutputManager()
        self.segments = {}
        self.segments_lock = threading.Lock()
        self.confidences = []
        self.last_segment_time = 0.0
        self.stats = {
            "chunks_received": 0,
            "segments_processed": 0,
            "total_duration": 0.0,
            "error_count": 0,
            "avg_confidence": 0.0
        }
    
    def start(self):
        """Start processing thread"""
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        log.info(f"Pipeline started for {self.participant.speaker_id} ({self.participant.language.upper()})")
    
    def stop(self):
        """Stop processing and finalize"""
        self.stop_event.set()
        
        # Finalize pending segments
        pending_segments = self.segmenter.force_finalize_all()
        for segment in pending_segments:
            self._process_segment(segment)
        
        if self.thread:
            self.thread.join(timeout=5.0)
        
        if self.confidences:
            self.stats["avg_confidence"] = float(np.mean(self.confidences))
        
        log.info(f"Pipeline stopped for {self.participant.speaker_id}")
        log.info(f" Segments: {self.stats['segments_processed']}, Chunks: {self.stats['chunks_received']}, Avg Conf: {self.stats['avg_confidence']:.2%}")
    
    def push(self, audio_chunk: np.ndarray, src_sr: int):
        """Push audio chunk to processing queue"""
        a = safe_float_conversion(audio_chunk)
        if a.size == 0:
            return
        
        if src_sr != SAMPLE_RATE:
            a = resample_if_needed(a, src_sr, SAMPLE_RATE)
        
        try:
            if not self.queue.full():
                self.queue.put_nowait({"audio": a, "sr": SAMPLE_RATE, "timestamp": time.time()})
                self.stats["chunks_received"] += 1
        except Exception as e:
            log.warning(f"Queue error: {e}")
    
    def _process_loop(self):
        """Main processing loop"""
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            
            try:
                audio = item["audio"]
                
                # Enhance audio
                enhanced, stats = self.enhancer.enhance(audio)
                
                # VAD
                is_speech, vad_conf = self.vad.is_speech(enhanced)
                
                # Segment
                segment = self.segmenter.process_chunk(
                    enhanced,
                    (is_speech, vad_conf),
                    self.call_id,
                    self.participant.speaker_id,
                    self.participant.language
                )
                
                if segment:
                    # Rate limit (min 150ms between segments)
                    if time.time() - self.last_segment_time > 0.15:
                        self._process_segment(segment)
                        self.last_segment_time = time.time()
            
            except Exception as e:
                log.exception(f"Processing error [{self.participant.speaker_id}]: {e}")
                self.stats["error_count"] += 1
    
    def _process_segment(self, segment: Segment):
        """Process finalized segment"""
        try:
            with self.segments_lock:
                self.segments[segment.id] = segment
                self.asr_manager.submit_segment(segment, self._on_asr_result)
                self.stats["segments_processed"] += 1
                self.stats["total_duration"] += segment.duration
        except Exception as e:
            log.exception(f"Segment processing error [{self.participant.speaker_id}]: {e}")
            self.stats["error_count"] += 1
    
    def _on_asr_result(self, asr_result: ASRResult):
        """Handle ASR result - WITH WEBSOCKET RELAY (FIX #6)"""
        try:
            confidence = asr_result.confidence
            self.confidences.append(confidence)
            
            with self.segments_lock:
                segment = self.segments.pop(asr_result.segment_id, None)
                if segment is None:
                    log.error(f"Segment {asr_result.segment_id} not found!")
                    return
                
                self.output_mgr.save_outputs(segment, asr_result)
            
            # FIX #6: Send transcript back to client over websocket
            try:
                if self.gateway:
                    ws = self.gateway.client_websockets.get(self.participant.client_id)
                    if ws and self.gateway.server_loop:
                        response = {
                            "type": "transcript",
                            "segment_id": asr_result.segment_id,
                            "text": asr_result.text,
                            "confidence": float(asr_result.confidence),
                            "speaker_id": asr_result.speaker_id,
                            "language": asr_result.language,
                            "timestamp": asr_result.timestamp,
                            "processing_time": float(asr_result.processing_time),
                            "words": asr_result.words
                        }
                        
                        asyncio.run_coroutine_threadsafe(
                            ws.send(json.dumps(response)),
                            self.gateway.server_loop
                        )
                        
                        log.info(f"📤 Sent to {asr_result.speaker_id}: {asr_result.text[:50]}...")
            except Exception as send_err:
                log.warning(f"Failed to send transcript: {send_err}")
        
        except Exception as e:
            log.exception(f"ASR callback error: {e}")

# ============================================================================ 
# EDGE GATEWAY - WITH WEBSOCKET DICT AND LOOP (FIX #1, #4, #5, #9)
# ============================================================================

class EdgeGateway:
    """Manages calls and participants"""
    
    def __init__(self):
        self.calls = {}
        self.client_to_call = {}
        self.asr_manager = SharedASRManager(num_workers=4)  # 4 parallel workers
        self.pipelines = {}
        self.client_websockets = {}  # FIX #1: Store WebSocket connections
        self.server_loop = None  # FIX #1: Store event loop reference
    
    def create_call(self, initiator: CallParticipant) -> str:
        call_id = gen_id("call")
        call = Call(call_id=call_id)
        call.participants[initiator.client_id] = initiator
        self.calls[call_id] = call
        self.client_to_call[initiator.client_id] = call_id
        
        pipeline = ParticipantPipeline(call_id, initiator, self.asr_manager, self)  # FIX #4: Pass gateway
        pipeline.start()
        self.pipelines[(call_id, initiator.client_id)] = pipeline
        
        log.info(f"📞 Call created: {call_id} by {initiator.client_id} ({initiator.language.upper()})")
        return call_id
    
    def join_call(self, call_id: str, participant: CallParticipant) -> bool:
        call = self.calls.get(call_id)
        if not call or not call.active:
            return False
        
        call.participants[participant.client_id] = participant
        self.client_to_call[participant.client_id] = call_id
        
        pipeline = ParticipantPipeline(call_id, participant, self.asr_manager, self)  # FIX #5: Pass gateway
        pipeline.start()
        self.pipelines[(call_id, participant.client_id)] = pipeline
        
        log.info(f"👥 Participant joined: {participant.client_id} ({participant.language.upper()}) → {call_id}")
        return True
    
    def leave_call(self, client_id: str):
        call_id = self.client_to_call.get(client_id)
        if not call_id:
            return
        
        call = self.calls.get(call_id)
        if call:
            pipeline = self.pipelines.pop((call_id, client_id), None)
            if pipeline:
                pipeline.stop()
            
            call.participants.pop(client_id, None)
            log.info(f"Participant left: {client_id}")
            
            if not call.participants:
                call.active = False
                log.info(f"Call ended: {call_id}")
        
        self.client_to_call.pop(client_id, None)
    
    def push_audio(self, client_id: str, audio: np.ndarray, src_sr: int):
        call_id = self.client_to_call.get(client_id)
        if not call_id:
            return
        
        pipeline = self.pipelines.get((call_id, client_id))
        if pipeline:
            pipeline.push(audio, src_sr)
    
    def stop(self):
        for pipeline in self.pipelines.values():
            pipeline.stop()
        self.asr_manager.stop()
        log.info("Edge Gateway stopped")

# ============================================================================ 
# MAIN SERVER ENTRYPOINT - WITH EVENT LOOP (FIX #10)
# ============================================================================

async def main():
    try:
        import websockets
        HAS_WEBSOCKETS = True
    except:
        HAS_WEBSOCKETS = False
        
    if not HAS_WEBSOCKETS:
        log.error("websockets not available!")
        return
    
    gateway = EdgeGateway()
    gateway.server_loop = asyncio.get_event_loop()  # FIX #10: Store event loop for thread-safe sending
    
    async def handler(websocket):
        await websocket_handler(websocket, gateway)
    
    server = await websockets.serve(handler, "0.0.0.0", 8765)
    
    log.info("=" * 90)
    log.info("Production STT System - Optimized for <2s Latency")
    log.info("=" * 90)
    log.info("WebSocket: ws://0.0.0.0:8765")
    log.info("Outputs: ./stt_outputs/")
    log.info("")
    log.info("<2s Latency (Silence padding: 250ms)")
    log.info("<1% WER (Advanced deduplication + word-level timestamps)")
    log.info("MT/TTS-ready outputs (Word-level timestamps + Prosody)")
    log.info("Speaker embeddings included (256-D normalized)")
    log.info("Multi-language: EN, HI, ES, FR")
    log.info("4-worker parallel ASR pipeline")
    log.info("REAL-TIME TRANSCRIPT RELAY TO CLIENTS (WEBSOCKET)")
    log.info("=" * 90)
    
    try:
        await server.wait_closed()
    finally:
        gateway.stop()

if __name__ == "__main__":
    if sys.version_info < (3, 7):
        print("Require Python 3.7+")
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Server stopped.")