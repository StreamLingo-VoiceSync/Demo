"""Custom exceptions for the STT system."""

class STTError(Exception):
    """Base exception for STT system errors."""
    pass

class ModelLoadError(STTError):
    """Raised when model fails to load."""
    pass

class AudioProcessingError(STTError):
    """Raised when audio processing fails."""
    pass

class TranscriptionError(STTError):
    """Raised when transcription fails."""
    pass

class VADError(STTError):
    """Raised when VAD processing fails."""
    pass

class OutputSaveError(STTError):
    """Raised when saving output fails."""
    pass