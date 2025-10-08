"""
Google Cloud Text-to-Speech модуль (БЫСТРЫЙ, высокое качество)
Ускорение в 5-10 раз по сравнению с Silero TTS (300-500мс вместо 2-5 сек)
"""
from __future__ import annotations

import os
from typing import Optional

from loguru import logger

try:
    from google.cloud import texttospeech
    GOOGLE_TTS_AVAILABLE = True
except ImportError:
    GOOGLE_TTS_AVAILABLE = False
    logger.warning("google-cloud-texttospeech не установлен. Установите: pip install google-cloud-texttospeech")


class GoogleTTS:
    """
    Google Cloud Text-to-Speech wrapper
    
    Требует:
    1. pip install google-cloud-texttospeech
    2. Настройка аутентификации (один из способов):
       - export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
       - gcloud auth application-default login
    
    Цена: $4 за 1M символов (~$0.004 за тезис из 100 символов)
    """
    
    def __init__(
        self,
        language: str = "ru-RU",
        voice_name: Optional[str] = None,
        speaking_rate: float = 1.0,
        pitch: float = 0.0,
    ):
        if not GOOGLE_TTS_AVAILABLE:
            raise RuntimeError(
                "google-cloud-texttospeech не установлен. "
                "Установите: pip install google-cloud-texttospeech"
            )
        
        self.language = language
        self.speaking_rate = speaking_rate
        self.pitch = pitch
        self.sample_rate = 24000  # Google TTS использует 24kHz
        
        # Лучшие голоса для русского:
        # - ru-RU-Wavenet-A (женский, естественный)
        # - ru-RU-Wavenet-B (мужской, естественный)
        # - ru-RU-Wavenet-C (женский, выразительный)
        # - ru-RU-Wavenet-D (мужской, выразительный) ⭐ РЕКОМЕНДУЮ
        self.voice_name = voice_name or "ru-RU-Wavenet-D"
        
        try:
            self.client = texttospeech.TextToSpeechClient()
            logger.info(
                f"GoogleTTS инициализирован: voice={self.voice_name}, "
                f"rate={self.speaking_rate}, pitch={self.pitch}"
            )
        except Exception as e:
            logger.error(
                f"Не удалось инициализировать Google TTS: {e}\n"
                f"Убедитесь что GOOGLE_APPLICATION_CREDENTIALS настроен или "
                f"выполнен 'gcloud auth application-default login'"
            )
            raise
    
    def synth(self, text: str) -> bytes:
        """
        Синтезирует текст в аудио (WAV, 24kHz)
        
        Returns:
            bytes: WAV audio data (LINEAR16, 24000 Hz)
        """
        if not text or not text.strip():
            return b""
        
        # Настройка голоса
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language,
            name=self.voice_name,
        )
        
        # Настройка аудио
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
            speaking_rate=self.speaking_rate,
            pitch=self.pitch,
        )
        
        # Синтез
        synthesis_input = texttospeech.SynthesisInput(text=text.strip())
        
        try:
            response = self.client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )
            
            logger.debug(f"Синтезировано {len(response.audio_content)} байт для текста: {text[:50]}...")
            return response.audio_content
            
        except Exception as e:
            logger.exception(f"Ошибка синтеза Google TTS: {e}")
            return b""
    
    def available_voices(self) -> list[str]:
        """Возвращает список доступных голосов для языка"""
        try:
            voices = self.client.list_voices(language_code=self.language)
            return [voice.name for voice in voices.voices]
        except Exception as e:
            logger.error(f"Не удалось получить список голосов: {e}")
            return []


__all__ = ["GoogleTTS", "GOOGLE_TTS_AVAILABLE"]
