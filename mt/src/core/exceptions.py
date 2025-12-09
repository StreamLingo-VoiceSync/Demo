"""Custom exceptions for the MT system."""

class MTError(Exception):
    """Base exception for MT system errors."""
    pass

class ModelLoadError(MTError):
    """Raised when model fails to load."""
    pass

class TranslationError(MTError):
    """Raised when translation fails."""
    pass

class AlignmentError(MTError):
    """Raised when alignment extraction fails."""
    pass

class TTSPreparationError(MTError):
    """Raised when TTS preparation fails."""
    pass

class ContextError(MTError):
    """Raised when context management fails."""
    pass