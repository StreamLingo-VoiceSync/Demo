"""Custom exceptions for the TTS system."""

class TTSError(Exception):
    """Base exception for TTS system errors."""
    pass

class ModelLoadError(TTSError):
    """Raised when model fails to load."""
    pass

class SynthesisError(TTSError):
    """Raised when audio synthesis fails."""
    pass

class LanguageNotSupportedError(TTSError):
    """Raised when unsupported language is requested."""
    pass

class InvalidInputError(TTSError):
    """Raised when input data is invalid."""
    pass

class VoiceCloningError(TTSError):
    """Raised when voice cloning fails."""
    pass

class CacheError(TTSError):
    """Raised when cache operations fail."""
    pass

class RateLimitError(TTSError):
    """Raised when rate limit is exceeded."""
    pass

class SecurityError(TTSError):
    """Raised when security validation fails."""
    pass