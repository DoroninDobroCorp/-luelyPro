"""CluelyPro - интеллектуальный голосовой ассистент для подготовки к интервью."""
__version__ = "0.2.0"

# Экспортируем только то, что уже создано
from cluely.models import VoiceProfile, QueuedSegment
from cluely.core.audio_utils import float_to_pcm16, median_spectral_flatness
from cluely.utils import setup_logging, extract_theses_from_text

__all__ = [
    "VoiceProfile",
    "QueuedSegment",
    "float_to_pcm16",
    "median_spectral_flatness",
    "setup_logging",
    "extract_theses_from_text",
]
