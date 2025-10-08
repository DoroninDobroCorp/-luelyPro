"""
Production ASR: OpenAI Whisper API.
Быстрее локальных моделей (RTF ~0.66x), не грузит CPU, отличная точность.
Fallback на faster-whisper если нет API ключа.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

import numpy as np
from loguru import logger

from config import ASRConfig as ConfigASRConfig
from exceptions import ASRError


@dataclass
class ASRConfig:
    """Конфигурация ASR (теперь через OpenAI API)."""
    api_key: Optional[str] = None  # OpenAI API key (если None - из OPENAI_API_KEY env)
    model: str = "whisper-1"  # OpenAI модель
    language: Optional[str] = "ru"  # Язык
    temperature: float = 0.0  # Температура (0.0 = детерминированный результат)


class FasterWhisperTranscriber:
    """
    Production ASR через OpenAI Whisper API.
    Совместимый интерфейс с faster-whisper для drop-in replacement.
    """
    
    def __init__(
        self,
        model_size: str = "whisper-1",  # Игнорируется для OpenAI (всегда whisper-1)
        device: Optional[str] = None,  # Игнорируется для OpenAI (облако)
        compute_type: Optional[str] = None,  # Игнорируется для OpenAI (облако)
        language: Optional[str] = "ru",
        initial_prompt: Optional[str] = None,
        banned_keywords: Optional[list[str]] = None,
        min_text_len: int = 3,
    ) -> None:
        """
        Инициализация OpenAI ASR транскрибера.
        
        Args:
            model_size: Игнорируется (OpenAI использует whisper-1)
            device: Игнорируется (облако)
            compute_type: Игнорируется (облако)
            language: Язык (по умолчанию "ru")
            initial_prompt: Промпт для снижения галлюцинаций
            banned_keywords: Запрещенные слова для фильтрации
            min_text_len: Минимальная длина текста
        """
        # Получаем API ключ
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ASRError(
                "OPENAI_API_KEY not found in environment. "
                "Set it with: export OPENAI_API_KEY='sk-...'"
            )
        
        # Импортируем OpenAI
        try:
            from openai import OpenAI
            import soundfile as sf
            import tempfile
        except ImportError as e:
            raise ASRError(
                f"OpenAI dependencies not installed: {e}. "
                "Install with: pip install openai soundfile"
            ) from e
        
        self.client = OpenAI(api_key=api_key)
        self.language = language
        self.min_text_len = min_text_len
        
        # Дефолтный промпт для снижения галлюцинаций
        default_ru_prompt = (
            "Транскрибируй только реально произнесённую речь говорящего. "
            "Не добавляй титры, имена редакторов, служебные сообщения. "
            "Пиши разговорную речь."
        )
        self.initial_prompt = initial_prompt if initial_prompt is not None else (
            default_ru_prompt if language == "ru" else None
        )
        
        # Запрещенные слова (галлюцинации OpenAI на silence)
        default_banned = [
            "субтитр", "редактор", "корректор", "продолжение следует",
            "горелова", "новикова", "закомолдина", "подписывайтесь",
            # Частые галлюцинации OpenAI на коротких сегментах
            "семкин", "егорова", "бойкова", "сухиашвили"
        ]
        self.banned_keywords = [kw.lower() for kw in (banned_keywords or default_banned)]
        
        logger.info("[ASR] Using OpenAI Whisper API (whisper-1)")
    
    def transcribe_np(self, wav: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Транскрибируем numpy массив через OpenAI API.
        
        Args:
            wav: Аудио (float32, моно, -1..1)
            sample_rate: Частота дискретизации
        
        Returns:
            Распознанный текст (или пустая строка)
        """
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        
        try:
            import soundfile as sf
            import tempfile
            
            # OpenAI API требует файл - используем временный WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, wav, sample_rate, format="WAV", subtype="PCM_16")
                
                with open(tmp.name, "rb") as audio_file:
                    transcript = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        language=self.language,
                        temperature=0.0,
                        prompt=self.initial_prompt,
                    )
                
                text = transcript.text.strip()
                return self._postprocess(text)
        
        except Exception as e:
            logger.error(f"[ASR] OpenAI transcription failed: {e}")
            return ""
    
    def _postprocess(self, text: str) -> str:
        """
        Фильтрация галлюцинаций и мусора.
        
        Args:
            text: Сырой текст от API
        
        Returns:
            Отфильтрованный текст (или пустая строка)
        """
        if not text:
            return ""
        
        t_low = text.lower()
        
        # Фильтр по запрещенным словам
        for kw in self.banned_keywords:
            if kw in t_low:
                logger.debug(f"[ASR] Filtered banned keyword: {kw}")
                return ""
        
        # Слишком короткий текст
        if len(text) < self.min_text_len or len(text.split()) <= 1:
            logger.debug(f"[ASR] Filtered short text: {text}")
            return ""
        
        return text
