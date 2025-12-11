"""API routes for the MT system."""

import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Import the orchestrator
from mt.src.main import DualLaneOrchestrator

# Import logger
from common.logger import setup_mt_logger
log = setup_mt_logger()

# Global orchestrator instance
orchestrator = None

class TranslationRequest(BaseModel):
    text: str
    source_lang: str
    target_lang: str
    speaker_embedding: Optional[List[float]] = None  # Add speaker embedding for voice cloning

class DualTranslationRequest(BaseModel):
    text_A: str
    text_B: str
    source_lang_A: str
    source_lang_B: str
    target_lang_A: str
    target_lang_B: str
    speaker_embedding_A: Optional[List[float]] = None  # Add speaker embedding for voice cloning
    speaker_embedding_B: Optional[List[float]] = None  # Add speaker embedding for voice cloning

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    global orchestrator
    
    app = FastAPI(
        title="VoiceSync MT Service",
        description="Machine Translation service for VoiceSync-Demo",
        version="1.0.0",
    )
    
    # Initialize orchestrator
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        orchestrator = DualLaneOrchestrator(device=device)
        log.info("MT Orchestrator initialized successfully")
    except Exception as e:
        log.error(f"Failed to initialize MT Orchestrator: {e}")
        raise
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "mt"}
    
    @app.post("/translate")
    async def translate_text(request: TranslationRequest):
        """Translate text from source language to target language with voice cloning support"""
        try:
            # For single translation, we'll use the dual-lane with both texts being the same
            # and both target languages being the same
            results = orchestrator.translate_dual_lane_concurrent(
                text_A=request.text,
                text_B=request.text,
                source_lang_A=request.source_lang,
                source_lang_B=request.source_lang,
                target_lang_A=request.target_lang,
                target_lang_B=request.target_lang,
                call_id="single_translation",
                speaker_embedding_A=request.speaker_embedding,  # Pass speaker embedding
                speaker_embedding_B=request.speaker_embedding   # Pass speaker embedding
            )
            
            # Extract the translation result
            path_1_result = results.get("path_1")
            if path_1_result and hasattr(path_1_result, 'translated_text'):
                return {
                    "translated_text": path_1_result.translated_text,
                    "confidence": path_1_result.confidence,
                    "speaker_embedding": path_1_result.speaker_embedding if hasattr(path_1_result, 'speaker_embedding') else (request.speaker_embedding if request.speaker_embedding else [])
                }
            else:
                # Fallback if there was an error
                return {
                    "translated_text": f"[Translated: {request.text}]",
                    "confidence": 0.5,
                    "speaker_embedding": request.speaker_embedding if request.speaker_embedding else []
                }
                
        except Exception as e:
            log.error(f"Translation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/translate/dual")
    async def translate_dual(request: DualTranslationRequest):
        """Translate two texts concurrently using dual-lane processing with voice cloning support"""
        try:
            results = orchestrator.translate_dual_lane_concurrent(
                text_A=request.text_A,
                text_B=request.text_B,
                source_lang_A=request.source_lang_A,
                source_lang_B=request.source_lang_B,
                target_lang_A=request.target_lang_A,
                target_lang_B=request.target_lang_B,
                call_id="dual_translation",
                speaker_embedding_A=request.speaker_embedding_A,  # Pass speaker embedding A
                speaker_embedding_B=request.speaker_embedding_B   # Pass speaker embedding B
            )
            
            # Format the response
            response = {
                "path_1": None,
                "path_2": None,
                "total_time_ms": results.get("total_time_ms", 0),
                "concurrent": results.get("concurrent", False)
            }
            
            # Process path 1 result
            path_1_result = results.get("path_1")
            if path_1_result and hasattr(path_1_result, 'translated_text'):
                response["path_1"] = {
                    "translated_text": path_1_result.translated_text,
                    "confidence": path_1_result.confidence,
                    "source_text": path_1_result.source_text,
                    "processing_time_ms": path_1_result.processing_time_ms,
                    "speaker_embedding": path_1_result.speaker_embedding if hasattr(path_1_result, 'speaker_embedding') else (request.speaker_embedding_A if request.speaker_embedding_A else [])
                }
            
            # Process path 2 result
            path_2_result = results.get("path_2")
            if path_2_result and hasattr(path_2_result, 'translated_text'):
                response["path_2"] = {
                    "translated_text": path_2_result.translated_text,
                    "confidence": path_2_result.confidence,
                    "source_text": path_2_result.source_text,
                    "processing_time_ms": path_2_result.processing_time_ms,
                    "speaker_embedding": path_2_result.speaker_embedding if hasattr(path_2_result, 'speaker_embedding') else (request.speaker_embedding_B if request.speaker_embedding_B else [])
                }
            
            return response
            
        except Exception as e:
            log.error(f"Dual translation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return app

def main():
    """Main entry point for the MT service"""
    app = create_app()
    
    # Get configuration from environment variables
    host = os.getenv("MT_HOST", "0.0.0.0")
    port = int(os.getenv("MT_PORT", "8766"))  # Changed to port 8766
    workers = int(os.getenv("MT_WORKERS", "1"))
    log_level = os.getenv("MT_LOG_LEVEL", "info")
    
    log.info(f"Starting MT service on {host}:{port} with {workers} workers")
    
    # For production, use uvicorn with multiple workers
    if workers > 1:
        uvicorn.run(
            "mt.src.api.routes:create_app",
            host=host,
            port=port,
            factory=True,
            workers=workers,
            log_level=log_level
        )
    else:
        uvicorn.run(
            "mt.src.api.routes:create_app",
            host=host,
            port=port,
            factory=True,
            reload=False,
            log_level=log_level
        )

if __name__ == "__main__":
    main()