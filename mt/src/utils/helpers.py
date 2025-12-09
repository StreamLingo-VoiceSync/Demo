"""Utility functions for the MT system."""

from __future__ import annotations
import re
import numpy as np
from typing import Any, Dict, List

# Import shared utilities
from common.utils import gen_id, safe_float_conversion, compute_rms

# Import schemas
from ..api.schemas import ProcessedToken

class ASRTokenProcessor:
    """Process ASR tokens with timestamp interpolation"""    
    def __init__(self):
        self.filler_words = {
            "um", "uh", "hmm", "mm", "erm", "ah", "you know", "i mean", "like",
            "er", "uh-huh", "basically", "actually", "literally", "right"
        }
        self.artifact_pattern = re.compile(r"(\[.*?\]|<.*?>|\d{1,2}:\d{2}|[^\w\s।?!])")
    
    def process_tokens(self,
                      tokens: List[Dict[str, Any]],
                      call_id: str,
                      speaker_id: str,
                      source_language: str,
                      processing_path: str = "path_1") -> List[ProcessedToken]:
        """Process ASR tokens with artifact removal and timestamp interpolation"""
        
        processed = []
        cleaned_tokens = []
        
        for token in tokens:
            text = token.get("text", "").strip()
            text = self.artifact_pattern.sub("", text).strip()
            
            if not text or text.lower() in self.filler_words:
                continue
            
            cleaned_tokens.append({**token, "text": text})
        
        for i, token in enumerate(cleaned_tokens):
            start_ms = token.get("start_ms", None)
            end_ms = token.get("end_ms", None)
            
            if start_ms is None or end_ms is None:
                total_duration = 2000
                per_token = total_duration / len(cleaned_tokens) if cleaned_tokens else 100
                
                if start_ms is None:
                    start_ms = i * per_token
                if end_ms is None:
                    end_ms = (i + 1) * per_token
            
            processed_token = ProcessedToken(
                text=token["text"],
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=token.get("confidence", 0.95),
                call_id=call_id,
                speaker_id=speaker_id,
                source_language=source_language,
                processing_path=processing_path,
                source_words=token["text"].split()
            )
            processed.append(processed_token)
        
        # Import logger
        from common.logger import setup_mt_logger
        log = setup_mt_logger()
        log.debug(f"Processed {len(processed)} tokens for {processing_path}")
        return processed