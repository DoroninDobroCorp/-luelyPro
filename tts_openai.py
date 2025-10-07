"""
OpenAI Text-to-Speech модуль
⚡ БЫСТРО: 0.5-1 сек (в 3-5 раз быстрее Silero)
🎤 КАЧЕСТВО: Отличное (почти как Google TTS)
💰 ЦЕНА: ~$0.0015 за тезис
"""
from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
from loguru import logger

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai не установлен. Установите: pip install openai")


class OpenAITTS:
    """
    OpenAI Text-to-Speech wrapper
    
    Требует:
    1. pip install openai
    2. OPENAI_API_KEY в .env или переменной окружения
    
    Голоса:
    - alloy:   нейтральный (по умолчанию)
    - echo:    мужской
    - fable:   британский акцент
    - onyx:    глубокий мужской ⭐ РЕКОМЕНДУЮ для экзамена
    - nova:    женский
    - shimmer: женский, выразительный
    
    Модели:
    - tts-1:     быстрая (300-500мс), хорошее качество
    - tts-1-hd:  медленнее (800-1200мс), отличное качество
    
    Цена:
    - tts-1:    $15 / 1M символов
    - tts-1-hd: $30 / 1M символов
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "tts-1",
        voice: str = "onyx",
        speed: float = 1.0,
    ):
        if not OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai не установлен. Установите: pip install openai"
            )
        
        # Получаем API ключ
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY не найден! "
                "Добавьте в .env: OPENAI_API_KEY=your_key_here"
            )
        
        self.model = model
        self.voice = voice
        self.speed = speed
        self.sample_rate = 24000  # OpenAI TTS использует 24kHz
        
        # Инициализируем клиент
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info(
                f"OpenAI TTS инициализирован: model={self.model}, "
                f"voice={self.voice}, speed={self.speed}"
            )
        except Exception as e:
            logger.error(f"Не удалось инициализировать OpenAI TTS: {e}")
            raise
    
    def synth(self, text: str) -> np.ndarray:
        """
        Синтезирует текст в аудио
        
        Returns:
            np.ndarray: float32 audio в диапазоне [-1, 1], sample_rate=24000
        """
        if not text or not text.strip():
            return np.zeros((0,), dtype=np.float32)
        
        try:
            # Вызываем OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text.strip(),
                speed=self.speed,
                response_format="opus",  # Opus - сжатый формат (быстрее передача)
            )
            
            # Получаем байты аудио
            audio_bytes = response.content
            
            # Декодируем в numpy array
            audio_np = self._decode_audio(audio_bytes)
            
            logger.debug(
                f"Синтезировано {len(audio_np)} сэмплов "
                f"({len(audio_np)/self.sample_rate:.2f}с) для текста: {text[:50]}..."
            )
            
            return audio_np
            
        except Exception as e:
            logger.exception(f"Ошибка синтеза OpenAI TTS: {e}")
            return np.zeros((0,), dtype=np.float32)
    
    def _decode_audio(self, audio_bytes: bytes) -> np.ndarray:
        """
        Декодирует opus/mp3 в numpy array
        """
        try:
            # Пробуем использовать pydub (если есть)
            from pydub import AudioSegment
            
            # Загружаем аудио
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_bytes),
                format="opus"
            )
            
            # Конвертируем в numpy
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
            
            # Нормализуем в [-1, 1]
            samples = samples / 32768.0
            
            # Если стерео → моно
            if audio_segment.channels == 2:
                samples = samples.reshape((-1, 2)).mean(axis=1)
            
            return samples
            
        except ImportError:
            logger.warning(
                "pydub не установлен, используем fallback декодер. "
                "Для лучшей производительности: pip install pydub ffmpeg-python"
            )
            return self._decode_audio_fallback(audio_bytes)
    
    def _decode_audio_fallback(self, audio_bytes: bytes) -> np.ndarray:
        """
        Fallback декодер через ffmpeg напрямую
        """
        import subprocess
        import tempfile
        
        try:
            # Сохраняем во временный файл
            with tempfile.NamedTemporaryFile(suffix=".opus", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            
            # Декодируем через ffmpeg
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-i", tmp_path,
                    "-f", "f32le",
                    "-acodec", "pcm_f32le",
                    "-ar", str(self.sample_rate),
                    "-ac", "1",  # моно
                    "-"
                ],
                capture_output=True,
                check=True
            )
            
            # Удаляем временный файл
            os.unlink(tmp_path)
            
            # Конвертируем в numpy
            audio_np = np.frombuffer(result.stdout, dtype=np.float32)
            return audio_np
            
        except Exception as e:
            logger.error(f"Не удалось декодировать аудио: {e}")
            return np.zeros((0,), dtype=np.float32)


__all__ = ["OpenAITTS", "OPENAI_AVAILABLE"]
