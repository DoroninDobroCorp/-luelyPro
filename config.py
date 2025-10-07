"""Централизованная конфигурация для CluelyPro."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class VADConfig:
    """Настройки Voice Activity Detection."""
    backend: str = "webrtc"  # "webrtc" | "silero"
    aggressiveness: int = 2  # 0..3 для WebRTC (выше = строже к шумам)
    min_consec_frames: int = 5  # Минимум подряд речевых кадров (20ms каждый)
    flatness_threshold: float = 0.60  # Порог спектральной плоскостности (> = шум)
    silero_threshold: float = 0.5  # Порог для Silero VAD (0..1)
    silero_window_ms: int = 100  # Длина окна Silero VAD (мс)


@dataclass
class ASRConfig:
    """Настройки Automatic Speech Recognition."""
    model: str = "tiny"  # tiny|base|small|medium|large-v3|large-v3-turbo
    language: str = "ru"
    device: Optional[str] = None  # None = auto (cuda если доступен, иначе cpu)
    compute_type: Optional[str] = None  # float16|int8|None = auto
    
    @classmethod
    def from_env(cls) -> "ASRConfig":
        """Создать конфиг из переменных окружения."""
        return cls(
            model=os.getenv("ASR_MODEL", "tiny"),
            language=os.getenv("ASR_LANG", "ru"),
            device=os.getenv("ASR_DEVICE"),
            compute_type=os.getenv("ASR_COMPUTE_TYPE"),
        )


@dataclass
class ThesisConfig:
    """Настройки тезисного помощника."""
    match_threshold: float = 0.6  # Порог совпадения по токенам (0..1)
    semantic_threshold: float = 0.55  # Порог семантической близости (0..1)
    gemini_min_conf: float = 0.60  # Мин. уверенность Gemini-судьи (0..1)
    repeat_interval_sec: float = 10.0  # Интервал повтора текущего тезиса
    autogen_batch_size: int = 3  # Размер пакета автогенерируемых тезисов
    enable_semantic: bool = True  # Включить семантическое сравнение
    enable_gemini: bool = True  # Включить Gemini-судью
    
    @classmethod
    def from_env(cls) -> "ThesisConfig":
        """Создать конфиг из переменных окружения."""
        return cls(
            repeat_interval_sec=float(os.getenv("THESIS_REPEAT_SEC", "10")),
            autogen_batch_size=int(os.getenv("THESIS_AUTOGEN_BATCH", "3")),
        )


@dataclass
class AudioConfig:
    """Настройки аудио потока."""
    sample_rate: int = 16000  # Гц (требуется для WebRTC VAD и ECAPA)
    frame_ms: int = 20  # Длина фрейма (мс) для VAD
    channels: int = 1  # Моно
    min_segment_ms: int = 500  # Минимальная длительность речевого сегмента
    max_silence_ms: int = 400  # Пауза для завершения сегмента
    pre_roll_ms: int = 160  # Предзахват до старта сегмента
    
    @property
    def frame_size(self) -> int:
        """Размер фрейма в сэмплах."""
        return int(self.sample_rate * self.frame_ms / 1000)


@dataclass
class LLMConfig:
    """Настройки Language Model."""
    model_id: str = "gemini-flash-latest"
    max_tokens: int = 320
    temperature: float = 0.3
    top_p: float = 0.9
    enable_history: bool = True
    history_max_turns: int = 8


@dataclass
class VerifierConfig:
    """Настройки LiveVoiceVerifier (свой/чужой голос)."""
    embedder_model: str = "speechbrain/spkrec-ecapa-voxceleb"
    distance_threshold: float = 0.75  # Косинусная дистанция (<= = мой голос)
    device: Optional[str] = None  # None = auto
    
    @classmethod
    def from_env(cls) -> "VerifierConfig":
        """Создать конфиг из переменных окружения."""
        return cls(
            distance_threshold=float(os.getenv("VOICE_THRESHOLD", "0.75")),
            device=os.getenv("EMBEDDER_DEVICE"),
        )


@dataclass
class SystemConfig:
    """Системные настройки."""
    log_level_console: str = "INFO"
    log_level_file: str = "DEBUG"
    log_dir: str = "logs"
    run_timeout_sec: float = 0.0  # 0 = бесконечно
    
    # Режимы работы
    ai_only_thesis: bool = True  # Только AI-генерация тезисов (без статических)
    commentary_mode: bool = False  # Режим комментариев (факты без явных вопросов)
    question_filter_mode: str = "gemini"  # "heuristic" | "gemini"
    question_min_len: int = 8  # Минимальная длина текста для обработки как вопроса
    append_question_to_answer: bool = False  # Добавлять вопрос к ответу LLM
    use_headphones: bool = True  # True = in-ear наушники (микрофон не слышит TTS)
    
    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Создать конфиг из переменных окружения."""
        return cls(
            log_level_console=os.getenv("CONSOLE_LOG_LEVEL", "INFO"),
            log_level_file=os.getenv("FILE_LOG_LEVEL", "DEBUG"),
            log_dir=os.getenv("LOG_DIR", "logs"),
            run_timeout_sec=float(os.getenv("RUN_SECONDS") or os.getenv("ASSISTANT_RUN_SECONDS") or "0"),
            ai_only_thesis=os.getenv("AI_ONLY_THESIS", "1").strip() not in ("0", "false", "False"),
            commentary_mode=os.getenv("COMMENTARY_MODE", "0").strip() not in ("0", "false", "False"),
            question_filter_mode=os.getenv("QUESTION_FILTER_MODE", "gemini").strip().lower(),
            question_min_len=int(os.getenv("QUESTION_MIN_LEN", "8")),
            append_question_to_answer=os.getenv("APPEND_QUESTION_TO_ANSWER", "0").strip().lower() in ("1", "true", "yes", "on"),
            use_headphones=os.getenv("USE_HEADPHONES", "1").strip() not in ("0", "false", "False"),
        )


@dataclass
class AppConfig:
    """Главный конфиг приложения."""
    vad: VADConfig
    asr: ASRConfig
    thesis: ThesisConfig
    audio: AudioConfig
    llm: LLMConfig
    verifier: VerifierConfig
    system: SystemConfig
    
    @classmethod
    def default(cls) -> "AppConfig":
        """Создать дефолтный конфиг."""
        return cls(
            vad=VADConfig(),
            asr=ASRConfig(),
            thesis=ThesisConfig(),
            audio=AudioConfig(),
            llm=LLMConfig(),
            verifier=VerifierConfig(),
            system=SystemConfig(),
        )
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Создать конфиг из переменных окружения."""
        return cls(
            vad=VADConfig(),
            asr=ASRConfig.from_env(),
            thesis=ThesisConfig.from_env(),
            audio=AudioConfig(),
            llm=LLMConfig(),
            verifier=VerifierConfig.from_env(),
            system=SystemConfig.from_env(),
        )


# Удобные константы для быстрого доступа
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 20
DEFAULT_CHANNELS = 1


__all__ = [
    "VADConfig",
    "ASRConfig",
    "ThesisConfig",
    "AudioConfig",
    "LLMConfig",
    "VerifierConfig",
    "SystemConfig",
    "AppConfig",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_FRAME_MS",
    "DEFAULT_CHANNELS",
]
