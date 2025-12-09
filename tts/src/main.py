"""Main entry point for the TTS system."""

from __future__ import annotations
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import os

# Import shared utilities
from common.logger import setup_tts_logger

# Import config
from .core.config import TTS_OUTPUT_DIR

# Import API routes
from .api.routes import router as tts_router

log = setup_tts_logger()

def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="VoiceSync TTS Service",
        description="Text-to-Speech service for VoiceSync-Demo with voice cloning capabilities",
        version="1.0.0",
        # Add security
        docs_url="/docs" if os.getenv("ENABLE_DOCS", "false").lower() == "true" else None,
        redoc_url="/redoc" if os.getenv("ENABLE_REDOC", "false").lower() == "true" else None
    )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["*"]  # In production, restrict to known hosts
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict to known origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time"]
    )
    
    # Add custom middleware for request timing
    @app.middleware("http")
    async def add_process_time_header(request, call_next):
        import time
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Include API routes
    app.include_router(tts_router, prefix="/api/v1/tts", tags=["tts"])
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "tts"}
    
    @app.on_event("startup")
    async def startup_event():
        log.info("VoiceSync TTS Service Starting Up - Voice Cloning Enabled")
        log.info(f"Output directory: {TTS_OUTPUT_DIR}")
        log.info("Available endpoints:")
        log.info("  POST /api/v1/tts/synthesize - Synthesize speech with voice cloning")
        log.info("  POST /api/v1/tts/clone_voice - Clone voice and synthesize")
        log.info("  GET  /api/v1/tts/health - Health check")
        log.info("  GET  /api/v1/tts/stats - Engine statistics")
        log.info("  POST /api/v1/tts/clear_cache - Clear synthesis cache")
        log.info("  GET  /api/v1/tts/models - List available models")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        log.info("VoiceSync TTS Service Shutting Down")
    
    return app

def main():
    """Main entry point"""
    app = create_app()
    
    # Get configuration from environment variables
    host = os.getenv("TTS_HOST", "0.0.0.0")
    port = int(os.getenv("TTS_PORT", "8767"))
    workers = int(os.getenv("TTS_WORKERS", "1"))
    log_level = os.getenv("TTS_LOG_LEVEL", "info")
    
    log.info(f"Starting TTS service on {host}:{port} with {workers} workers")
    
    # For production, use uvicorn with multiple workers
    if workers > 1:
        import multiprocessing
        uvicorn.run(
            "tts.src.main:create_app",
            host=host,
            port=port,
            factory=True,
            workers=workers,
            log_level=log_level
        )
    else:
        uvicorn.run(
            "tts.src.main:create_app",
            host=host,
            port=port,
            factory=True,
            reload=False,
            log_level=log_level
        )

if __name__ == "__main__":
    main()