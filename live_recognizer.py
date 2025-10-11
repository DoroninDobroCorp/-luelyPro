from __future__ import annotations

import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Optional, List, Callable
from datetime import date

import numpy as np
import io
import wave
from loguru import logger

# Мягкие импорты тяжёлых/опциональных зависимостей, чтобы юнит-тесты работали оффлайн
try:  # sound I/O
    import sounddevice as sd  # type: ignore
except Exception:  # noqa: BLE001
    sd = None  # type: ignore

try:  # torch (может отсутствовать в окружении CI)
    import torch  # type: ignore
except Exception:  # noqa: BLE001
    torch = None  # type: ignore

try:  # WebRTC VAD
    import webrtcvad  # type: ignore
except Exception:  # noqa: BLE001
    webrtcvad = None  # type: ignore

try:  # эмбеддер голоса
    from pyannote.audio.pipelines.speaker_verification import (  # type: ignore
        PretrainedSpeakerEmbedding,
    )
except Exception:  # noqa: BLE001
    PretrainedSpeakerEmbedding = None  # type: ignore

try:  # ASR (опционально)
    from asr_transcriber import FasterWhisperTranscriber  # type: ignore
except Exception:  # noqa: BLE001
    FasterWhisperTranscriber = None  # type: ignore
try:  # LLM (опционально)
    from llm_answer import LLMResponder  # type: ignore
except Exception:  # noqa: BLE001
    LLMResponder = None  # type: ignore
try:  # OpenAI TTS (опционально)
    from tts_openai import OpenAITTS, OPENAI_AVAILABLE  # type: ignore
except Exception:  # noqa: BLE001
    OpenAITTS = None  # type: ignore
    OPENAI_AVAILABLE = False
try:  # Google TTS (опционально)
    from tts_google import GoogleTTS, GOOGLE_TTS_AVAILABLE  # type: ignore
except Exception:  # noqa: BLE001
    GoogleTTS = None  # type: ignore
    GOOGLE_TTS_AVAILABLE = False
try:  # Silero TTS (опционально)
    from tts_silero import SileroTTS  # type: ignore
except Exception:  # noqa: BLE001
    SileroTTS = None  # type: ignore
try:  # Silero VAD (опционально)
    from vad_silero import SileroVAD  # type: ignore
except Exception:  # noqa: BLE001
    SileroVAD = None  # type: ignore
try:
    from thesis_generator import GeminiThesisGenerator  # опционально
except Exception:  # noqa: BLE001
    GeminiThesisGenerator = None  # type: ignore


LOGGING_CONFIGURED = False
CURRENT_LOG_DIR: Optional[Path] = None


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: str = os.getenv("CONSOLE_LOG_LEVEL", "INFO"),
    file_level: str = os.getenv("FILE_LOG_LEVEL", "DEBUG"),
) -> Path:
    """Configure loguru sinks and return the file sink path.
    If log_dir is provided, reconfigure sinks to write there.
    """
    global LOGGING_CONFIGURED, CURRENT_LOG_DIR
    target_dir = Path(log_dir or os.getenv("LOG_DIR", "logs"))
    # Reconfigure if not configured yet OR explicit log_dir differs from current
    need_reconfigure = (not LOGGING_CONFIGURED) or (
        log_dir is not None and (CURRENT_LOG_DIR is None or target_dir != CURRENT_LOG_DIR)
    )
    if not need_reconfigure and CURRENT_LOG_DIR is not None:
        return CURRENT_LOG_DIR / "assistant.log"

    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "assistant.log"

    logger.remove()
    logger.add(sys.stderr, level=console_level.upper(), enqueue=True, backtrace=False)
    logger.add(
        log_file,
        level=file_level.upper(),
        enqueue=False,  # синхронная запись, чтобы файл гарантированно создавался в тестах
        mode="w",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )
    # гарантируем наличие файла независимо от поведения sink
    try:
        log_file.touch(exist_ok=True)
    except Exception:
        pass
    logger.info(f"Логи пишутся в {log_file.resolve()}")
    LOGGING_CONFIGURED = True
    CURRENT_LOG_DIR = target_dir
    return log_file


DEFAULT_ENROLL_PARAGRAPHS: list[str] = [
    (
        "Этот абзац нужен, чтобы система уверенно запомнила мой голос. "
        "Я говорю размеренно, держу одинаковое расстояние до микрофона и не спешу. "
        "В конце делаю короткую паузу, чтобы запись завершилась корректно."
    ),
    (
        "В рабочем дне мне часто приходится объяснять сложные идеи простым языком. "
        "Поэтому важно, чтобы ассистент чётко распознавал мои интонации и тембр. "
        "Я произношу слова ясно и уверенно, словно отвечаю на вопрос интервьюера."
    ),
    (
        "Чтобы профиль получился естественным, я читаю текст, похожий на разговор о проектах. "
        "Я кратко описываю задачи, решения и выводы, сохраняя живую и спокойную речь. "
        "Так алгоритм уловит мой реальный стиль общения."
    ),
]


def _split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\r", "").split("\n\n")]
    return [p for p in parts if p]


SAMPLE_RATE = 16000  # required by WebRTC VAD and recommended for embeddings
FRAME_MS = 20  # WebRTC VAD supports 10, 20, or 30 ms
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame
CHANNELS = 1


def float_to_pcm16(x: np.ndarray) -> bytes:
    """Convert float32 (-1..1) to 16-bit PCM bytes."""
    x = np.clip(x, -1.0, 1.0)
    x_int16 = (x * 32767.0).astype(np.int16)
    return x_int16.tobytes()


def median_spectral_flatness(
    wav: np.ndarray, sample_rate: int = SAMPLE_RATE, frame_len: int = 512, hop: int = 256
) -> float:
    """Оцениваем спектральную плоскостность (0..1), где ближе к 1 — шумоподобный сигнал.
    Возвращаем медиану по окнам.
    """
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if wav.size < frame_len:
        return 1.0  # слишком коротко — считаем как шум
    # Оконное представление
    n_frames = 1 + (wav.size - frame_len) // hop
    if n_frames <= 0:
        return 1.0
    window = np.hanning(frame_len).astype(np.float32)
    sf_values: list[float] = []
    eps = 1e-10
    for i in range(n_frames):
        start = i * hop
        frame = wav[start : start + frame_len]
        if frame.size < frame_len:
            break
        frame = frame * window
        spec = np.fft.rfft(frame)
        p = (spec.real ** 2 + spec.imag ** 2) + eps
        # Можно ограничить полосу [80..4000] Гц, чтобы фокусироваться на речи
        freqs = np.fft.rfftfreq(frame_len, d=1.0 / sample_rate)
        band = (freqs >= 80) & (freqs <= 4000)
        p_band = p[band]
        if p_band.size == 0:
            continue
        gm = np.exp(np.mean(np.log(p_band)))
        am = np.mean(p_band)
        sf = float(gm / (am + eps))
        sf_values.append(sf)
    if not sf_values:
        return 1.0
    return float(np.median(np.asarray(sf_values)))


@dataclass
class VoiceProfile:
    embedding: np.ndarray  # shape (d,)

    @staticmethod
    def load(path: Path) -> Optional["VoiceProfile"]:
        if not path.exists():
            return None
        data = np.load(path)
        return VoiceProfile(embedding=data["embedding"])  # type: ignore[index]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, embedding=self.embedding)


@dataclass
class QueuedSegment:
    kind: str
    audio: np.ndarray
    timestamp: float
    distance: float = 0.0


class ThesisManager:
    """
    Менеджер тезисов для управления озвучкой и асинхронным углублением.
    
    Логика:
    - При новом вопросе: генерируются 5 тезисов (или 1 для простого)
    - Каждый тезис озвучивается 2 раза
    - После 3-го тезиса (2-е повторение): запрос 5 дополнительных тезисов (параллельно)
    - Углубление до 7 итераций максимум
    - При новом вопросе с новыми тезисами: мгновенное прерывание старых
    
    ✅ ОПТИМИЗАЦИЯ 2B: Параллельная генерация TTS для тезисов
    - При start_new_question запускается параллельная генерация TTS для первых 2-3 тезисов
    - Пока озвучивается первый тезис, генерируются следующие
    - Ускорение 40-50% времени ожидания. См. OPTIMIZATION_TABLE.md - код 2B
    """
    
    def __init__(
        self,
        generator: Optional["GeminiThesisGenerator"],
        max_depth_iterations: int = 7,
        deeper_trigger_idx: int = 2,
        tts_engine: Optional[object] = None,  # TTS для предгенерации
    ):
        self.generator = generator
        self.max_depth_iterations = max_depth_iterations
        self.deeper_trigger_idx = deeper_trigger_idx
        self.tts_engine = tts_engine  # Ссылка на TTS
        
        # Текущее состояние
        self.theses: List[str] = []  # Все тезисы (первые + углубленные)
        self.current_question: str = ""  # Текущий вопрос
        self.context: Optional[str] = None  # Контекст диалога
        self.current_idx: int = 0  # Индекс текущего тезиса (0-based)
        self.current_repeat: int = 1  # Номер повторения (1 или 2)
        self.depth_iterations: int = 0  # Счетчик итераций углубления
        
        # Асинхронное углубление
        self.deeper_request_in_progress: bool = False
        self.deeper_request_thread: Optional[threading.Thread] = None
        self.pending_deeper_theses: List[str] = []  # Готовые доп.тезисы
        self.lock = threading.Lock()  # Защита от гонки
        
        # ✅ ОПТИМИЗАЦИЯ 2B: Кэш предгенерированных аудио (параллельная TTS)
        from concurrent.futures import ThreadPoolExecutor, Future
        self.audio_cache: dict[int, Future] = {}  # {индекс_тезиса: Future[audio_np]}
        self.tts_executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="tts-prefetch")
        self.prefetch_enabled = True  # Флаг для включения/выключения предгенерации
    
    def start_new_question(
        self, 
        question: str, 
        theses: List[str], 
        context: Optional[str] = None
    ) -> None:
        """Начать новый вопрос (сброс состояния, новые тезисы)"""
        with self.lock:
            self.current_question = question
            self.theses = theses.copy()
            self.context = context
            self.current_idx = 0
            self.current_repeat = 1
            self.depth_iterations = 0
            self.deeper_request_in_progress = False
            self.pending_deeper_theses = []
            
            # ✅ ОПТИМИЗАЦИЯ 2B: Очищаем старый кэш и запускаем предгенерацию TTS
            self.audio_cache.clear()
            if self.prefetch_enabled and self.tts_engine is not None and len(theses) > 0:
                # Предгенерируем TTS для первых 2-3 тезисов параллельно
                prefetch_count = min(3, len(theses))
                for i in range(prefetch_count):
                    future = self.tts_executor.submit(self._generate_tts_audio, theses[i])
                    self.audio_cache[i] = future
                logger.debug(f"🎵 Запущена предгенерация TTS для {prefetch_count} тезисов")
            
            logger.info(f"🎤 Новый вопрос: {len(theses)} тезисов")
    
    def get_next_thesis(self) -> Optional[str]:
        """Получить следующий тезис для озвучки"""
        with self.lock:
            if self.current_idx >= len(self.theses):
                # Проверяем есть ли готовые углубленные тезисы
                if self.pending_deeper_theses:
                    self.theses.extend(self.pending_deeper_theses)
                    self.pending_deeper_theses = []
                    logger.info(f"📚 Углубление {self.depth_iterations}/{self.max_depth_iterations}: добавлено {len(self.theses) - self.current_idx} доп.тезисов")
                else:
                    # Нет тезисов
                    return None
            
            if self.current_idx >= len(self.theses):
                return None
            
            return self.theses[self.current_idx]
    
    def advance(self) -> None:
        """Перейти к следующему повторению/тезису"""
        with self.lock:
            if self.current_repeat == 1:
                self.current_repeat = 2
            else:
                self.current_repeat = 1
                self.current_idx += 1
    
    def should_trigger_deeper(self) -> bool:
        """Проверить нужно ли запустить асинхронное углубление"""
        with self.lock:
            # Триггер: 3-й тезис (индекс 2), 2-е повторение
            return (
                self.current_idx == self.deeper_trigger_idx and
                self.current_repeat == 2 and
                self.depth_iterations < self.max_depth_iterations and
                not self.deeper_request_in_progress and
                self.generator is not None
            )
    
    def trigger_deeper_async(self) -> None:
        """Запустить асинхронный запрос углубления"""
        if not self.should_trigger_deeper():
            return
        
        with self.lock:
            self.deeper_request_in_progress = True
            self.depth_iterations += 1
        
        # Запускаем запрос в отдельном потоке
        def request_deeper():
            try:
                all_theses = self.theses.copy()
                question = self.current_question
                context = self.context
                
                logger.debug(f"📚 Углубление {self.depth_iterations}/{self.max_depth_iterations}: запрос 5 доп.тезисов...")
                
                deeper_theses = self.generator.generate_deeper(
                    previous_theses=all_theses,
                    question=question,
                    context=context,
                    n=5,
                    language="ru"
                )
                
                if deeper_theses:
                    with self.lock:
                        self.pending_deeper_theses = deeper_theses
                        logger.info(f"✅ Углубление {self.depth_iterations}: получено {len(deeper_theses)} доп.тезисов")
                        logger.info(f"Углубленные тезисы ({len(deeper_theses)}): {deeper_theses}")
                else:
                    logger.warning(f"⚠️ Углубление {self.depth_iterations}: пустой результат")
                    
            except Exception as e:
                logger.error(f"Ошибка асинхронного углубления: {e}")
            finally:
                with self.lock:
                    self.deeper_request_in_progress = False
        
        self.deeper_request_thread = threading.Thread(
            target=request_deeper,
            name=f"deeper-{self.depth_iterations}",
            daemon=True
        )
        self.deeper_request_thread.start()
    
    def _generate_tts_audio(self, text: str) -> Optional[np.ndarray]:
        """
        ✅ ОПТИМИЗАЦИЯ 2B: Генерация TTS аудио для одного тезиса (для ThreadPoolExecutor)
        """
        if not self.tts_engine or not text:
            return None
        
        try:
            tts_start = time.time()
            audio = self.tts_engine.synth(text)
            
            # Конвертируем bytes (Google TTS) в numpy array
            if isinstance(audio, bytes):
                if len(audio) == 0:
                    return None
                # Декодируем WAV bytes в numpy
                with wave.open(io.BytesIO(audio), 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            
            tts_elapsed = (time.time() - tts_start) * 1000
            logger.debug(f"✓ TTS предгенерация: {tts_elapsed:.0f}мс ({len(text)} символов)")
            return audio
        except Exception as e:
            logger.error(f"Ошибка предгенерации TTS: {e}")
            return None
    
    def get_cached_audio(self, index: int) -> Optional[np.ndarray]:
        """
        ✅ ОПТИМИЗАЦИЯ 2B: Получить готовое аудио из кэша (если есть)
        Возвращает None если кэш пустой или генерация еще не завершена
        """
        with self.lock:
            if index not in self.audio_cache:
                return None
            
            future = self.audio_cache[index]
            
        # Проверяем готовность (без блокировки)
        if not future.done():
            return None
        
        try:
            return future.result(timeout=0.1)
        except Exception as e:
            logger.debug(f"Ошибка получения кэшированного аудио: {e}")
            return None
    
    def has_more_theses(self) -> bool:
        """Проверить есть ли еще тезисы или ожидаются углубленные"""
        with self.lock:
            return (
                self.current_idx < len(self.theses) or
                len(self.pending_deeper_theses) > 0 or
                self.deeper_request_in_progress
            )


class LiveVoiceVerifier:
    """
    Минимальный лайв-модуль:
    - WebRTC VAD, чтобы отделять речь от шумов/тишины.
    - Эмбеддинги динамически (ECAPA) и сравнение с профилем по косинусной близости.
    - Вывод: "мой голос" / "незнакомый голос".
    """

    def __init__(
        self,
        model_id: str = "speechbrain/spkrec-ecapa-voxceleb",
        device: Optional[str] = None,
        embedder: Optional[PretrainedSpeakerEmbedding] = None,
        vad_aggressiveness: int = 2,
        vad_backend: str = "webrtc",  # "webrtc" | "silero"
        silero_vad_threshold: float = 0.5,
        silero_vad_window_ms: int = 100,
        threshold: float = 0.30,  # cosine distance threshold: <= is my voice
        min_consec_speech_frames: int = 5,  # >=5 * 20ms = 100ms подряд речи для старта сегмента
        flatness_reject_threshold: float = 0.60,  # > -> шум, сегмент отбрасываем
        # ASR настройки
        asr_enable: bool = False,
        asr_model_size: str = "large-v3-turbo",
        asr_language: Optional[str] = None,
        asr_device: Optional[str] = None,
        asr_compute_type: Optional[str] = None,
        # LLM настройки
        llm_enable: bool = False,
        # Thesis настройки (используем дефолты из ThesisConfig)
        thesis_match_threshold: float = 0.6,
        thesis_gemini_enable: bool = True,
        thesis_gemini_min_conf: float = 0.60,
        thesis_autogen_enable: bool = True,
        thesis_autogen_batch: int = 3,
    ) -> None:
        if device is None:
            # Без torch считаем, что доступен только CPU
            if 'cuda' in str(os.getenv('ASR_DEVICE', '')).lower():
                device = 'cuda'
            else:
                try:
                    device = "cuda" if (torch is not None and getattr(torch, "cuda", None) and torch.cuda.is_available()) else "cpu"
                except Exception:
                    device = "cpu"
        # Если torch недоступен — сохраняем строку, иначе torch.device
        self.device = torch.device(device) if torch is not None else device  # type: ignore[assignment]
        self._device_str = self.device.type
        self._embedder_model_id = model_id
        self._embedder: Optional[PretrainedSpeakerEmbedding] = embedder
        # VAD backend selection
        self.vad_backend = vad_backend.lower().strip()
        if self.vad_backend not in ("webrtc", "silero"):
            logger.warning(f"Неизвестный vad_backend={vad_backend}, используем 'webrtc'")
            self.vad_backend = "webrtc"
        if self.vad_backend == "webrtc":
            if webrtcvad is None:
                logger.warning("webrtcvad недоступен — VAD выключен для оффлайн-тестов")
                self._vad_fn = lambda frame: False
            else:
                self.vad_webrtc = webrtcvad.Vad(vad_aggressiveness)
                self._vad_fn = lambda frame: self.vad_webrtc.is_speech(
                    float_to_pcm16(frame), SAMPLE_RATE
                )
        else:
            if SileroVAD is None:
                logger.warning("SileroVAD недоступен — VAD выключен для оффлайн-тестов")
                self._vad_fn = lambda frame: False
            else:
                self.vad_silero = SileroVAD(
                    sample_rate=SAMPLE_RATE,
                    threshold=float(silero_vad_threshold),
                    window_ms=int(silero_vad_window_ms),
                    device=self._device_str,
                )
                self._vad_fn = lambda frame: self.vad_silero.is_speech(frame)
        self.threshold = threshold
        self.min_consec_speech_frames = max(1, int(min_consec_speech_frames))
        self.flatness_reject_threshold = float(flatness_reject_threshold)
        # ASR
        self.asr_enable = bool(asr_enable)
        self.asr_model_size = asr_model_size
        self.asr_language = asr_language
        self.asr_device = asr_device
        self.asr_compute_type = asr_compute_type
        self._asr: Optional[FasterWhisperTranscriber] = None
        # LLM
        self.llm_enable = bool(llm_enable)
        self._llm: Optional[LLMResponder] = None
        # TTS для озвучки ответа LLM
        self._tts: Optional[SileroTTS] = None
        self._suppress_until: float = 0.0  # подавляем обработку входа на время TTS
        self._is_announcing: bool = False  # флаг что сейчас озвучивается тезис
        self._tts_interrupt: threading.Event = threading.Event()  # флаг прерывания TTS
        self._tts_lock: threading.Lock = threading.Lock()  # защита от одновременного воспроизведения
        self._stop_requested: threading.Event = threading.Event()  # глобальный флаг остановки
        self._thesis_thread: Optional[threading.Thread] = None  # текущий поток озвучки тезисов
        self._tts_generation: int = 0  # счетчик поколений озвучки (для прерывания старых потоков)
        # Внешний получатель аудио (например, WebSocket-клиент). При наличии —
        # TTS будет отправляться туда, а не проигрываться локально через sounddevice
        self._audio_sink: Optional[Callable[[bytes, int], None]] = None

        # Генератор тезисов для подсказок
        self._thesis_generator: Optional[GeminiThesisGenerator] = None
        self._max_theses_history: int = 50  # Лимит для очистки старых тезисов
        self._last_announce_ts: float = 0.0
        # Контекст диалога за последние 30 секунд (для местоимений)
        self._dialogue_context: list[tuple[float, str]] = []  # [(timestamp, text), ...]
        self._context_window_sec: float = 30.0  # Окно контекста в секундах
        # ✅ ОПТИМИЗАЦИЯ 6C: Оптимизация очереди - maxsize 20 → 8
        # 2 воркера × 3 сегмента в обработке + 2 в очереди = 8 (вместо 20)
        # Снижает memory и latency при перегрузке. См. OPTIMIZATION_TABLE.md - код 6C
        self._segment_queue: "queue.Queue[QueuedSegment]" = queue.Queue(maxsize=8)
        self._segment_workers: List[threading.Thread] = []  # Список воркеров для параллельной обработки
        self._segment_stop = threading.Event()
        self._num_asr_workers: int = int(os.getenv("ASR_WORKERS", "2"))  # Количество параллельных воркеров
        # Фильтрация «не-вопросов»: heuristic | gemini (по умолчанию gemini)
        self._question_filter_mode: str = os.getenv("QUESTION_FILTER_MODE", "gemini").strip().lower()
        try:
            self._question_min_len: int = int(os.getenv("QUESTION_MIN_LEN", "8"))
        except Exception:
            self._question_min_len = 8
        # Управление добавлением исходного вопроса к ответу LLM (по умолчанию выкл)
        try:
            _aq = os.getenv("APPEND_QUESTION_TO_ANSWER", "0").strip().lower()
            self._append_question_to_answer: bool = _aq in ("1", "true", "yes", "on")
        except Exception:
            self._append_question_to_answer = False
        # Режим: исключительно ИИ-экстракция тезисов из чужой речи
        self._ai_only_thesis: bool = os.getenv("AI_ONLY_THESIS", "1").strip() not in ("0", "false", "False")
        # Новый режим комментариев: генерировать короткие факт-заметки даже без явных вопросов
        self._commentary_mode: bool = os.getenv("COMMENTARY_MODE", "0").strip() not in ("0", "false", "False", "no", "No")
        # Используем наушники (True = микрофон не слышит TTS, блокировка не нужна)
        self._use_headphones: bool = os.getenv("USE_HEADPHONES", "1").strip() not in ("0", "false", "False")
        # Инициализация генератора тезисов (всегда включен)
        if GeminiThesisGenerator is not None:
            try:
                self._thesis_generator = GeminiThesisGenerator()  # type: ignore
                logger.info("Автогенерация тезисов Gemini включена")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Не удалось инициализировать ThesisGenerator: {e}")
        else:
            logger.warning("Генератор тезисов недоступен: нет зависимости thesis_generator/google-genai")
        
        # Инициализация менеджера тезисов
        from config import ThesisConfig
        thesis_cfg = ThesisConfig()
        self._thesis_manager = ThesisManager(
            generator=self._thesis_generator,
            max_depth_iterations=thesis_cfg.max_depth_iterations,
            deeper_trigger_idx=thesis_cfg.deeper_trigger_idx,
        )
        logger.info(f"ThesisManager инициализирован: макс углубление={thesis_cfg.max_depth_iterations}")

        if self.asr_enable:
            try:
                self._ensure_asr()
            except Exception as e:  # noqa: BLE001
                logger.exception(f"ASR недоступен: {e}")

        if self.llm_enable and LLMResponder is not None:
            try:
                # Уменьшаем историю LLM с 8 до 4 пар для снижения нагрузки
                self._llm = LLMResponder(history_max_turns=4)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Не удалось инициализировать LLMResponder: {e}")

        # Инициализируем TTS (RU по умолчанию), если он нужен для LLM
        if self.llm_enable:
            # Выбор TTS движка через USE_TTS_ENGINE (openai | google | silero)
            tts_engine = os.getenv("USE_TTS_ENGINE", "silero").lower()
            
            # Скорость воспроизведения (0.25-4.0, 1.0 = нормальная)
            # 1.3-1.5 оптимально для тезисов - быстро, но разборчиво
            try:
                tts_speed = float(os.getenv("TTS_SPEED", "1.35"))
                tts_speed = max(0.25, min(4.0, tts_speed))  # Ограничиваем диапазон
            except ValueError:
                tts_speed = 1.35
            
            # ✅ ОПТИМИЗАЦИЯ 2A: OpenAI TTS - РЕКОМЕНДУЕТСЯ (ускорение 3-5x vs Silero)
            # OpenAI TTS в 3-5 раз быстрее Silero (300-800мс vs 2-5сек на тезис).
            # См. OPTIMIZATION_TABLE.md - код 2A
            if tts_engine == "openai" and OpenAITTS is not None and OPENAI_AVAILABLE:
                try:
                    self._tts = OpenAITTS(
                        model="tts-1",           # ✅ 2D: tts-1 (быстро) | tts-1-hd (качественно)
                        voice="onyx",            # onyx (мужской) | nova (женский)
                        speed=tts_speed,
                    )
                    logger.info(f"✅ Используется OpenAI TTS (скорость {tts_speed}x)")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"OpenAI TTS недоступен: {e}, переключаюсь на Silero TTS")
                    self._tts = None
            
            # Google TTS (опционально, если настроен Service Account)
            elif tts_engine == "google" and self._tts is None and GoogleTTS is not None and GOOGLE_TTS_AVAILABLE:
                try:
                    self._tts = GoogleTTS(
                        language="ru-RU",
                        voice_name="ru-RU-Wavenet-D",  # Мужской, выразительный
                        speaking_rate=tts_speed,  # Ускоренное воспроизведение без повышения тона
                    )
                    logger.info(f"✅ Используется Google TTS (скорость {tts_speed}x)")
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Google TTS недоступен: {e}, переключаюсь на Silero TTS")
                    self._tts = None
            
            # Silero TTS (fallback - работает из коробки, офлайн)
            if self._tts is None and SileroTTS is not None:
                try:
                    self._tts = SileroTTS(
                        language="ru", model_id="v4_ru", speaker="eugene", sample_rate=24000
                    )
                    if tts_engine in ("openai", "google"):
                        logger.info(f"⚠️ Используется Silero TTS ({tts_engine.upper()} TTS недоступен)")
                    else:
                        logger.info("✅ Используется Silero TTS")
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"Не удалось инициализировать TTS: {e}")
            
            if self._tts is None:
                logger.warning("❌ TTS не установлен, озвучка недоступна")
            
            # ✅ ОПТИМИЗАЦИЯ 2B: Обновляем ссылку на TTS в ThesisManager
            if self._tts is not None:
                self._thesis_manager.tts_engine = self._tts
                logger.debug("✓ ThesisManager подключен к TTS для предгенерации")

        logger.info(
            f"LiveVoiceVerifier initialized | model={model_id}, device={self.device}, threshold={threshold}, VAD={self.vad_backend}"
        )

        # Прогрев моделей, чтобы снизить задержку при первом обращении
        try:
            self._warmup_models()
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Warmup error (ignored): {e}")

    # ==== Внешний приёмник аудио (для гибридной архитектуры) ====
    def set_audio_sink(self, sink: Callable[[bytes, int], None]) -> None:
        """Задать внешний приёмник аудио.
        sink принимает (wav_bytes, sample_rate). Если задан, TTS будет
        отправляться туда вместо локального воспроизведения."""
        self._audio_sink = sink

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(1.0 - np.dot(a, b))

    def _ensure_embedder(self) -> PretrainedSpeakerEmbedding:
        if self._embedder is None:
            self._embedder = PretrainedSpeakerEmbedding(
                self._embedder_model_id,
                device=self.device,
            )
        return self._embedder

    def embedding_from_waveform(self, wav: np.ndarray) -> np.ndarray:
        """wav: mono float32 in [-1, 1], shape (n_samples,) at 16kHz"""
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        # Ensure tensor shape (batch=1, channels=1, samples)
        if torch is not None and hasattr(torch, "from_numpy"):
            tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # (1,1,n)
            ctx = getattr(torch, "inference_mode", None)
            if callable(ctx):
                cm = ctx()
            else:
                class _Dummy:
                    def __enter__(self):
                        return None
                    def __exit__(self, exc_type, exc, tb):
                        return False
                cm = _Dummy()
            with cm:  # type: ignore
                out = self._ensure_embedder()(tensor)  # expected shape (1, dim)
        else:
            # Фоллбэк: эмбеддеру достаточно объекта с shape, DummyEmbedder из тестов это поддерживает
            class _DummyTensor:
                def __init__(self, n: int):
                    self.shape = (1, 1, n)
            out = self._ensure_embedder()(_DummyTensor(wav.size))  # type: ignore
        # out can be numpy array or torch tensor depending on backend
        if isinstance(out, np.ndarray):
            return out[0]
        else:
            try:
                return out[0].detach().cpu().numpy()
            except Exception:
                # Если это уже np.ndarray или совместимый тип
                try:
                    return np.asarray(out)[0]
                except Exception:
                    # В крайнем случае — нулевой вектор фиксированного размера
                    return np.zeros((192,), dtype=np.float32)

    # ==== Асинхронная обработка сегментов ====
    def _start_segment_worker(self) -> None:
        # Проверяем есть ли уже живые воркеры
        if any(w.is_alive() for w in self._segment_workers):
            return
        
        self._segment_stop.clear()
        self._segment_workers = []
        
        # Запускаем N параллельных воркеров
        for i in range(self._num_asr_workers):
            worker = threading.Thread(
                target=self._segment_worker_loop,
                name=f"segment-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._segment_workers.append(worker)
        
        logger.info(f"🚀 Запущено {self._num_asr_workers} параллельных ASR воркеров")

    def _stop_segment_worker(self) -> None:
        self._segment_stop.set()
        self._stop_requested.set()  # Устанавливаем глобальный флаг остановки
        
        # Прерываем TTS и останавливаем sounddevice
        self._tts_interrupt.set()
        if sd is not None:
            try:
                sd.stop()
            except Exception:
                pass
        
        # Останавливаем все воркеры
        for worker in self._segment_workers:
            if worker is not None:
                worker.join(timeout=2.0)
                if worker.is_alive():
                    logger.warning(f"Worker {worker.name} не остановился за 2с")
        
        self._segment_workers = []
        
        # Очищаем очередь
        while not self._segment_queue.empty():
            try:
                self._segment_queue.get_nowait()
                self._segment_queue.task_done()
            except queue.Empty:
                break

    def _enqueue_segment(self, kind: str, audio: np.ndarray, distance: float = 0.0) -> None:
        if audio.size == 0:
            return
        segment = QueuedSegment(kind=kind, audio=audio, timestamp=time.time(), distance=distance)
        try:
            self._segment_queue.put_nowait(segment)
        except queue.Full:
            try:
                dropped = self._segment_queue.get_nowait()
                self._segment_queue.task_done()
                logger.warning(
                    f"Очередь сегментов переполнена, отбрасываю {dropped.kind} сегмент"
                )
            except queue.Empty:
                pass
            try:
                self._segment_queue.put_nowait(segment)
            except queue.Full:
                logger.error("Не удалось добавить сегмент в очередь — пропускаю сигнал")

    def _segment_worker_loop(self) -> None:
        # Создаём локальный экземпляр ASR для этого воркера (thread-safe)
        local_asr: Optional[FasterWhisperTranscriber] = None
        if self.asr_enable and FasterWhisperTranscriber is not None:
            try:
                local_asr = FasterWhisperTranscriber(
                    model_size=self.asr_model_size,
                    device=self.asr_device,
                    compute_type=self.asr_compute_type,
                    language=self.asr_language,
                )
                worker_name = threading.current_thread().name
                logger.info(f"✓ {worker_name}: локальный ASR инициализирован")
            except Exception as e:
                logger.warning(f"Не удалось создать локальный ASR: {e}")
        
        while not self._segment_stop.is_set() and not self._stop_requested.is_set():
            try:
                segment = self._segment_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            
            # ВАЖНО: Во время озвучки тезисов игнорируем СВОИ сегменты
            # (чтобы не распознавать собственное повторение тезисов)
            # ЧУЖИЕ сегменты обрабатываем (это могут быть новые вопросы или продолжение диалога)
            if self._is_announcing and segment.kind == "self":
                logger.debug("⏸️ Игнорируем свой голос во время озвучки тезисов")
                self._segment_queue.task_done()
                continue
            
            try:
                if segment.kind == "self":
                    # Свой голос просто логируем и игнорируем
                    logger.debug("мой голос (игнорируем)")
                else:
                    self._handle_foreign_segment_with_asr(segment, local_asr)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Ошибка обработки сегмента: {e}")
            finally:
                self._segment_queue.task_done()

    def _handle_foreign_segment_with_asr(self, segment: QueuedSegment, local_asr: Optional[FasterWhisperTranscriber]) -> None:
        """Обработка чужого сегмента с локальным ASR (для параллельных воркеров)"""
        segment_duration = segment.audio.size / SAMPLE_RATE
        logger.debug(f"📏 Длина сегмента: {segment_duration:.2f}с")
        
        if self.asr_enable and local_asr is not None:
            try:
                # ✅ ОПТИМИЗАЦИЯ 8A: Засекаем время ДО ASR для корректных метрик
                processing_start = time.time()
                
                asr_start = time.time()
                text = local_asr.transcribe_np(segment.audio, SAMPLE_RATE)
                asr_elapsed = (time.time() - asr_start) * 1000
                worker_name = threading.current_thread().name
                logger.debug(f"⏱️  [{worker_name}] ASR обработка: {asr_elapsed:.0f}мс (аудио {segment_duration:.2f}с)")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"ASR ошибка: {e}")
                return
            self._handle_foreign_text(text, asr_elapsed=asr_elapsed, processing_start=processing_start)
        else:
            logger.info("незнакомый голос")

    def _handle_foreign_text(self, text: Optional[str], asr_elapsed: float = 0.0, processing_start: Optional[float] = None) -> None:
        """Обработка чужого голоса: ASR → генерация тезисов → LLM ответ → TTS
        
        Args:
            text: Распознанный текст от ASR
            asr_elapsed: Время выполнения ASR в миллисекундах
            processing_start: Время начала обработки (до ASR), для корректных метрик
        """
        if processing_start is None:
            processing_start = time.time()
        
        t = (text or "").strip()
        if not t:
            logger.info("Чужой голос (ASR: пусто)")
            return
        
        # НЕ блокируем обработку чужих вопросов - они ВСЕГДА должны обрабатываться
        # Защита от самоподхвата происходит на уровне VAD (в основном цикле live_verify)
        # и через флаг _is_announcing (в segment_worker_loop)
        
        logger.info(f"Чужой голос (ASR): {t}")
        
        # Добавляем вопрос в контекст диалога
        now = time.time()
        self._dialogue_context.append((now, t))
        
        # Очищаем старый контекст (>30 сек)
        cutoff_time = now - self._context_window_sec
        self._dialogue_context = [(ts, txt) for ts, txt in self._dialogue_context if ts >= cutoff_time]
        
        # Формируем контекст (предыдущие реплики) и текущий вопрос ОТДЕЛЬНО
        if len(self._dialogue_context) > 1:
            # Есть предыдущие реплики - передаём их как контекст
            context_items = [txt for _, txt in self._dialogue_context[:-1]]
            context_text = "\n".join(context_items)
            current_question = t
        else:
            # Первый вопрос - контекста нет
            context_text = None
            current_question = t
        
        logger.debug(f"Контекст: {len(self._dialogue_context)-1} реплик, текущий вопрос: {current_question}")
        
        # ✅ ОПТИМИЗАЦИЯ 3C: Параллельная генерация тезисов + прогрев TTS
        # Запускаем Gemini и TTS warmup параллельно → ускорение 15-25%
        # См. OPTIMIZATION_TABLE.md - код 3C
        if self._thesis_generator is None and GeminiThesisGenerator is not None:
            try:
                self._thesis_generator = GeminiThesisGenerator()
            except Exception as e:
                logger.warning(f"Не удалось создать thesis_generator: {e}")
        
        if self._thesis_generator:
            try:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                gemini_start = time.time()
                
                # Функция генерации тезисов
                def generate_theses():
                    return self._thesis_generator.generate(
                        current_question, 
                        n=5, 
                        language="ru",
                        context=context_text
                    )
                
                # Функция прогрева TTS (сгенерировать короткое аудио)
                def warmup_tts():
                    if self._tts is not None:
                        try:
                            # Генерируем короткое аудио для прогрева модели
                            warmup_text = "тест"
                            warmup_audio = self._tts.synth(warmup_text)
                            
                            # Если bytes - декодируем
                            if isinstance(warmup_audio, bytes) and len(warmup_audio) > 0:
                                with wave.open(io.BytesIO(warmup_audio), 'rb') as wf:
                                    frames = wf.readframes(wf.getnframes())
                                    warmup_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                            
                            logger.debug("✓ TTS прогрет")
                            return True
                        except Exception as e:
                            logger.debug(f"TTS warmup ошибка: {e}")
                            return False
                    return False
                
                # Запускаем параллельно через ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen-warmup") as executor:
                    # Запускаем обе задачи
                    thesis_future = executor.submit(generate_theses)
                    warmup_future = executor.submit(warmup_tts)
                    
                    # Ждём результаты (Gemini обычно дольше, поэтому ждём его)
                    theses_raw = thesis_future.result(timeout=5.0)
                    # TTS warmup может завершиться быстрее, не блокируем
                    try:
                        warmup_future.result(timeout=0.1)
                    except Exception:
                        pass
                
                llm_elapsed = (time.time() - gemini_start) * 1000
                
                # Определяем какой движок использовался
                llm_name = "LLM"
                if hasattr(self._thesis_generator, 'primary_engine'):
                    if self._thesis_generator.primary_engine == "cerebras":
                        llm_name = "Cerebras"
                    elif self._thesis_generator.primary_engine == "gemini":
                        llm_name = "Gemini"
                
                logger.debug(f"⏱️  {llm_name} API (параллельно с TTS warmup): {llm_elapsed:.0f}мс")
                
                # Парсим тезисы - ожидаем список строк или одну строку с |||
                theses = []
                if isinstance(theses_raw, list):
                    # Если список - парсим каждую строку отдельно
                    for item in theses_raw:
                        if "|||" in item:
                            theses.extend([t.strip() for t in item.split("|||") if t.strip()])
                        else:
                            theses.append(item.strip())
                elif isinstance(theses_raw, str):
                    # Если строка - парсим по |||
                    theses = [t.strip() for t in theses_raw.split("|||") if t.strip()]
                
                if theses:
                    logger.info(f"Сгенерированы тезисы ({len(theses)}): {theses}")
                    
                    # Инициализируем ThesisManager новым вопросом
                    self._thesis_manager.start_new_question(
                        question=current_question,
                        theses=theses,
                        context=context_text
                    )
                    
                    # Прерываем предыдущую озвучку (новый вопрос с новыми тезисами) - мгновенно!
                    if self._is_announcing:
                        logger.info("🚨 Новый вопрос с новыми тезисами - прерываем старые тезисы")
                        
                        # Увеличиваем счетчик поколений - старый поток увидит что устарел
                        self._tts_generation += 1
                        
                        # Устанавливаем флаг прерывания для немедленной остановки
                        self._tts_interrupt.set()
                        
                        # Принудительно останавливаем воспроизведение (прерываем sd.wait())
                        if sd is not None:
                            try:
                                sd.stop()
                                logger.debug("sd.stop() вызван для прерывания озвучки")
                            except Exception as e:
                                logger.debug(f"sd.stop() ошибка: {e}")
                        
                        # Даем время старому потоку остановиться
                        if self._thesis_thread is not None and self._thesis_thread.is_alive():
                            logger.debug("Ждём завершения старого потока озвучки...")
                            self._thesis_thread.join(timeout=0.5)
                            
                            if self._thesis_thread.is_alive():
                                logger.warning("⚠️ Старый поток еще работает, но запускаем новый (версионность)")
                    
                    # Всегда очищаем флаг для нового потока
                    self._tts_interrupt.clear()
                    
                    # Запоминаем текущее поколение для нового потока
                    current_generation = self._tts_generation
                    
                    # ✅ ОПТИМИЗАЦИЯ 8A: Детализированные метрики обработки вопроса
                    total_elapsed = (time.time() - processing_start) * 1000
                    # Вычисляем долю каждого компонента
                    asr_pct = (asr_elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
                    llm_pct = (llm_elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
                    other_pct = 100 - asr_pct - llm_pct
                    
                    logger.info(
                        f"⏱️  МЕТРИКИ ОБРАБОТКИ ВОПРОСА:\n"
                        f"  • ASR:     {asr_elapsed:6.0f}мс ({asr_pct:5.1f}%)\n"
                        f"  • {llm_name:8s} {llm_elapsed:6.0f}мс ({llm_pct:5.1f}%)\n"
                        f"  • Другое:  {other_pct:5.1f}% (парсинг, контекст)\n"
                        f"  • ИТОГО:   {total_elapsed:6.0f}мс"
                    )
                    
                    # Озвучиваем тезисы в отдельном потоке чтобы не блокировать прием новых вопросов
                    def announce_theses():
                        my_generation = current_generation  # Запоминаем свое поколение
                        self._is_announcing = True
                        logger.debug(f"🎤 Начинаю озвучку тезисов (поколение {my_generation})")
                        try:
                            # Озвучиваем тезисы через ThesisManager
                            while self._thesis_manager.has_more_theses():
                                # Проверяем прерывание (новый вопрос с новыми тезисами)
                                if self._tts_generation > my_generation or self._tts_interrupt.is_set() or self._stop_requested.is_set():
                                    logger.info(f"⚠️ Озвучка тезисов прервана (gen {my_generation} -> {self._tts_generation})")
                                    break
                                
                                # Получаем текущий тезис
                                thesis = self._thesis_manager.get_next_thesis()
                                if not thesis:
                                    # Нет готовых тезисов, ждем углубленных
                                    time.sleep(0.1)
                                    continue
                                
                                # Получаем номера для логирования
                                with self._thesis_manager.lock:
                                    idx = self._thesis_manager.current_idx + 1
                                    repeat = self._thesis_manager.current_repeat
                                    total = len(self._thesis_manager.theses)
                                
                                logger.debug(f"🔊 Тезис {idx}/{total} ({repeat}/2): {thesis[:50]}...")
                                
                                # ✅ ОПТИМИЗАЦИЯ 2B: Озвучиваем тезис с индексом для кэша
                                self._speak_text(thesis, generation=my_generation, thesis_index=self._thesis_manager.current_idx)
                                
                                # Проверяем прерывание после озвучки
                                if self._tts_generation > my_generation or self._tts_interrupt.is_set() or self._stop_requested.is_set():
                                    logger.info(f"⚠️ Прервано после озвучки тезиса {idx} ({repeat}/2)")
                                    break
                                
                                # Триггер углубления (после 3-го тезиса, 2-е повторение)
                                if self._thesis_manager.should_trigger_deeper():
                                    logger.info(f"🚀 Триггер углубления после тезиса {idx} (repeat {repeat})")
                                    self._thesis_manager.trigger_deeper_async()
                                
                                # Переходим к следующему повторению/тезису
                                self._thesis_manager.advance()
                                
                                # Паузы между повторениями и тезисами
                                with self._thesis_manager.lock:
                                    next_repeat = self._thesis_manager.current_repeat
                                if next_repeat == 1:
                                    # Перешли к новому тезису - пауза длиннее
                                    time.sleep(0.3)
                                else:
                                    # Между повторениями - пауза короче
                                    time.sleep(0.15)
                            
                            if self._tts_generation == my_generation and not (self._tts_interrupt.is_set() or self._stop_requested.is_set()):
                                logger.debug(f"✅ Озвучка всех тезисов завершена (gen {my_generation})")
                        finally:
                            self._is_announcing = False
                            logger.debug(f"🎤 Озвучка тезисов остановлена (gen {my_generation})")
                    
                    self._thesis_thread = threading.Thread(target=announce_theses, name="thesis-announcer", daemon=True)
                    self._thesis_thread.start()
                else:
                    logger.warning("Тезисы не сгенерированы (пустой результат)")
            except Exception as e:
                logger.error(f"Ошибка генерации тезисов: {e}")

    @staticmethod
    def _should_ignore_non_question_text(text: str) -> bool:
        """Фильтр явных нерелевантных триггеров, которые не надо трактовать как вопросы.
        Примеры: "динамичная музыка", "динамическая музыка" и т.п.
        """
        if not text:
            return False
        import re
        s = text.strip().lower()
        patterns = [
            r"\bдинамич\w*\s+музык\w*\b",
        ]
        for p in patterns:
            try:
                if re.search(p, s):
                    return True
            except Exception:
                continue
        return False

    def _handle_self_transcript(self, transcript: Optional[str]) -> None:
        """Обработка своей речи - просто логируем и игнорируем."""
        t = (transcript or "").strip()
        if not t:
            return
        logger.info(f"Моя речь (фильтруем): {t}")

    def simulate_dialogue(self, events: List[tuple[str, str]]) -> None:
        """Прогоняет последовательность реплик без аудио.

        events: список ("self"|"other", текст), который позволяет прогнать
        конвейер с готовыми транскриптами. Полезно для автоматических тестов.
        """
        for role, content in events:
            kind = (role or "").strip().lower()
            if kind in {"self", "me", "candidate"}:
                self._handle_self_transcript(content)
            elif kind in {"other", "interviewer", "question"}:
                self._handle_foreign_text(content)

    # ==== Аудио предобработка: простой highpass + AGC для сегментов ====
    @staticmethod
    def _highpass_simple(wav: np.ndarray, sr: int = SAMPLE_RATE, cutoff_hz: int = 80) -> np.ndarray:
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        # Однополюсный HPF
        rc = 1.0 / (2 * np.pi * float(cutoff_hz))
        dt = 1.0 / float(sr)
        alpha = rc / (rc + dt)
        y = np.zeros_like(wav, dtype=np.float32)
        prev_y = 0.0
        prev_x = 0.0
        for i in range(wav.size):
            x = float(wav[i])
            hp = alpha * (prev_y + x - prev_x)
            y[i] = hp
            prev_y = hp
            prev_x = x
        return y

    @staticmethod
    def _apply_agc(wav: np.ndarray, target_rms: float = 0.03, max_gain: float = 10.0) -> np.ndarray:
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(wav)) + 1e-12))
        if rms <= 1e-6:
            return wav
        gain = min(max_gain, max(0.1, target_rms / rms))
        out = np.clip(wav * gain, -1.0, 1.0)
        return out.astype(np.float32)

    @staticmethod
    def _preprocess_segment(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
        y = LiveVoiceVerifier._highpass_simple(wav, sr=sr, cutoff_hz=80)
        y = LiveVoiceVerifier._apply_agc(y, target_rms=0.03)
        return y

    # ==== Устойчивый парсер тезисов из сырого ответа модели ====
    @staticmethod
    def _parse_theses_from_raw(raw: str, n_max: int = 3) -> List[str]:
        import json, re
        s = (raw or "").strip()
        if not s:
            return []
        # Уберём возможные обрамляющие кавычки, если внутри есть фигурные скобки
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            if '{' in s and '}' in s:
                s = s[1:-1].strip()

        def _from_dict(d: dict) -> List[str]:
            items = d.get("theses", []) if isinstance(d, dict) else []
            out: List[str] = []
            for it in items:
                if isinstance(it, str):
                    t = it.strip()
                    if t:
                        out.append(t)
            return out

        # Снимем возможные code fences ```...```
        if s.startswith("```"):
            # возьмём содержимое между первой и последней тройной кавычкой
            m = re.findall(r"```[a-zA-Z]*\n([\s\S]*?)\n```", s)
            if m:
                s = m[-1].strip()

        # Попытка 1: прямой JSON / или JSON-строка
        try:
            obj = json.loads(s)
            if isinstance(obj, str):
                # иногда приходит как строка JSON
                try:
                    obj2 = json.loads(obj)
                    if isinstance(obj2, dict):
                        vals = _from_dict(obj2)
                        if vals:
                            return vals[:n_max]
                except Exception:
                    pass
            elif isinstance(obj, dict):
                vals = _from_dict(obj)
                if vals:
                    return vals[:n_max]
            elif isinstance(obj, list):
                # иногда модель вернёт просто список строк
                out: List[str] = []
                for it in obj:
                    if isinstance(it, str):
                        t = it.strip()
                        if t:
                            out.append(t)
                if out:
                    return out[:n_max]
        except Exception:
            pass

        # Попытка 2: вытащить первый JSON-объект по фигурным скобкам
        try:
            start = s.find("{")
            end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                sub = s[start : end + 1]
                d = json.loads(sub)
                vals = _from_dict(d)
                if vals:
                    return vals[:n_max]
        except Exception:
            pass

        # Попытка 3: распарсить как обычный список строк
        parts: List[str] = []
        s2 = s.replace(";", "\n")
        for line in s2.splitlines():
            t = line.strip()
            # Уберём маркеры списков и нумерацию
            t = re.sub(r"^[\-•*\s]*", "", t)
            t = re.sub(r"^\d+[\).]\s*", "", t)
            # Отбросим явные JSON-структуры
            if not t or t in {"[", "]", "{", "}", "},", "],"}:
                continue
            # Пропустим строки, начинающиеся/заканчивающиеся скобками — это, вероятно, JSON
            if t.startswith("{") or t.startswith("[") or t.endswith("}") or t.endswith("]"):
                continue
            # Уберём завершающие запятые
            if t.endswith(","):
                t = t[:-1].strip()
            # Уберём внешние кавычки
            if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
                t = t[1:-1].strip()
            # Пропустим строки вида key: value (скорее всего это JSON-ключи, напр. "theses": [)
            if re.match(r'^"?[A-Za-z_][\w\s\-]*"?\s*:\s*', t):
                continue
            # Пропустим пустяковые скобки
            if t in {"[", "]"}:
                continue
            # Должен быть осмысленный текст c буквами
            if not any(ch.isalpha() for ch in t):
                continue
            parts.append(t)
        return parts[:n_max]

    def _collect_enrollment_audio(
        self,
        seconds: float = 5.0,
        min_voiced_seconds: float = 2.0,
        silence_stop_sec: float = 1.2,
    ) -> np.ndarray:
        """Собираем сигнал с микрофона, оставляя только озвученные (VAD) фреймы."""
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.warning(f"InputStream status: {status}")
            q.put(indata.copy())

        voiced_samples: List[np.ndarray] = []
        voiced_duration = 0.0
        silence_stop_sec = max(0.6, float(silence_stop_sec))
        target_end = time.time() + max(1.0, float(seconds))
        last_voiced_ts: Optional[float] = None

        with sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            blocksize=FRAME_SIZE,  # deliver 20ms blocks
            callback=callback,
        ):
            logger.info("Говорите для записи профиля...")
            while True:
                now = time.time()
                if now >= target_end:
                    logger.debug("Останавливаем запись профиля: достигнут лимит по времени")
                    break
                if (
                    voiced_duration >= min_voiced_seconds
                    and last_voiced_ts is not None
                    and (now - last_voiced_ts) >= silence_stop_sec
                ):
                    logger.debug("Останавливаем запись профиля: зафиксирована тишина после речи")
                    break
                try:
                    block = q.get(timeout=0.5)[:, 0]  # mono
                except queue.Empty:
                    continue
                # VAD on 20ms subframes
                # block may be larger than FRAME_SIZE if sounddevice batches; split it
                i = 0
                while i + FRAME_SIZE <= len(block):
                    frame = block[i : i + FRAME_SIZE]
                    i += FRAME_SIZE
                    is_speech = self._vad_fn(frame)
                    if is_speech:
                        voiced_samples.append(frame)
                        voiced_duration += FRAME_MS / 1000.0
                        last_voiced_ts = time.time()

        if voiced_duration < min_voiced_seconds:
            logger.warning(
                f"Недостаточно речи для профиля: {voiced_duration:.2f}s < {min_voiced_seconds:.2f}s"
            )

        if len(voiced_samples) == 0:
            raise RuntimeError("Не удалось захватить голос для энроллмента")

        logger.info(
            "Сегмент профиля собран: озвучено {dur:.2f}s, кадров {frames}",
            dur=voiced_duration,
            frames=len(voiced_samples),
        )

        return np.concatenate(voiced_samples, axis=0)

    def enroll(
        self,
        path: Path,
        seconds: float = 20.0,
        min_voiced_seconds: float = 2.0,
        silence_stop_sec: float = 1.2,
    ) -> VoiceProfile:
        wav = self._collect_enrollment_audio(
            seconds=seconds,
            min_voiced_seconds=min_voiced_seconds,
            silence_stop_sec=silence_stop_sec,
        )
        emb = self.embedding_from_waveform(wav)
        profile = VoiceProfile(embedding=emb)
        profile.save(path)
        logger.info(f"Профиль сохранён: {path}")
        return profile

    def live_verify(
        self,
        profile: VoiceProfile,
        min_segment_ms: int = 1500,  # минимальная длительность сегмента (1.5s = фильтр коротких шумов)
        max_silence_ms: int = 400,  # завершение сегмента после паузы
        pre_roll_ms: int = 160,     # предзахват аудио до старта сегмента
        run_seconds: float = 0.0,   # авто-остановка (0 = бесконечно)
    ) -> None:
        """Слушаем микрофон и детектируем речевые сегменты. Для каждого сегмента
        считаем эмбеддинг и решаем: мой голос или незнакомый голос.
        Печатаем вывод сразу в консоль.
        """

        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.warning(f"InputStream status: {status}")
            q.put(indata.copy())

        seg_audio: List[np.ndarray] = []
        pre_frames: List[np.ndarray] = []  # буфер для накапливания подряд речевых кадров до старта
        # Буфер предзахвата: последние pre_roll_ms до старта сегмента
        pre_roll_frames_cnt = max(0, int(np.ceil(pre_roll_ms / FRAME_MS)))
        pre_roll = deque(maxlen=pre_roll_frames_cnt)
        voiced_ms = 0
        silence_ms = 0
        in_speech = False
        consec_speech = 0

        logger.info(
            "Старт лайв-распознавания. Нажмите Ctrl+C для остановки. Говорите в микрофон."
        )
        # Не озвучиваем тезис при старте - ждём вопроса

        stop_at = time.time() + run_seconds if run_seconds and run_seconds > 0 else None
        stream = None
        try:
            self._start_segment_worker()
            stream = sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                blocksize=FRAME_SIZE,  # deliver 20ms blocks
                callback=callback,
            )
            with stream:
                while True:
                    if self._stop_requested.is_set():
                        logger.info("Остановка по флагу _stop_requested")
                        break
                    if stop_at is not None and time.time() >= stop_at:
                        logger.info("Авто-остановка по таймеру run_seconds")
                        break
                    try:
                        block = q.get(timeout=0.5)[:, 0]
                    except queue.Empty:
                        continue

                    # Подавление самоподхвата: блокируем ТОЛЬКО если используются динамики
                    # С наушниками микрофон не слышит TTS, блокировка мешает ловить новые вопросы
                    if not self._use_headphones and time.time() < self._suppress_until:
                        continue

                    i = 0
                    while i + FRAME_SIZE <= len(block):
                        frame = block[i : i + FRAME_SIZE]
                        # обновляем pre-roll всегда (в том числе в тишине)
                        if pre_roll_frames_cnt > 0:
                            pre_roll.append(frame.copy())
                        i += FRAME_SIZE
                        is_speech = self._vad_fn(frame)

                        if is_speech:
                            consec_speech += 1
                            if not in_speech:
                                # ещё не стартовали сегмент: копим pre_frames
                                pre_frames.append(frame)
                                if consec_speech >= self.min_consec_speech_frames:
                                    # старт сегмента: сначала добавляем предзахват, затем pre_frames
                                    if pre_roll_frames_cnt > 0 and len(pre_roll) > 0:
                                        seg_audio.extend(list(pre_roll))
                                        pre_roll.clear()
                                    seg_audio.extend(pre_frames)
                                    pre_frames.clear()
                                    in_speech = True
                                    voiced_ms = self.min_consec_speech_frames * FRAME_MS
                            else:
                                seg_audio.append(frame)
                                voiced_ms += FRAME_MS
                            silence_ms = 0
                        else:
                            consec_speech = 0
                            pre_frames.clear()
                            if in_speech:
                                silence_ms += FRAME_MS

                        # завершение сегмента
                        if in_speech and silence_ms >= max_silence_ms:
                            in_speech = False
                            total_ms = voiced_ms
                            wav = (
                                np.concatenate(seg_audio, axis=0)
                                if len(seg_audio) > 0
                                else np.array([], dtype=np.float32)
                            )
                            seg_audio.clear()
                            voiced_ms = 0
                            silence_ms = 0

                            if total_ms < min_segment_ms or wav.size < FRAME_SIZE:
                                # слишком коротко — игнорируем
                                continue

                            # Дополнительная отбраковка шумных сегментов по спектральной плоскостности
                            sf_med = median_spectral_flatness(wav, SAMPLE_RATE)
                            if sf_med >= self.flatness_reject_threshold:
                                # шумоподобный сегмент — игнорируем без вывода
                                continue

                            # Предобработка: highpass + AGC
                            wav = self._preprocess_segment(wav, sr=SAMPLE_RATE)

                            # считаем эмбеддинг и сравниваем
                            emb = self.embedding_from_waveform(wav)
                            dist = self.cosine_distance(emb, profile.embedding)
                            if dist <= self.threshold:
                                self._enqueue_segment("self", wav, dist)
                            else:
                                self._enqueue_segment("other", wav, dist)

        except KeyboardInterrupt:
            logger.info("⚠️ Остановлено пользователем (Ctrl+C)")
            self._stop_requested.set()
        finally:
            # Останавливаем stream явно
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception as e:
                    logger.debug(f"Ошибка остановки stream: {e}")
            
            # Останавливаем воркеры и очереди
            self._stop_segment_worker()
            
            # Финальная остановка sounddevice
            if sd is not None:
                try:
                    sd.stop()
                except Exception:
                    pass
            
            logger.info("✅ Остановка завершена")

    def _ensure_asr(self) -> FasterWhisperTranscriber:
        if self._asr is None:
            if FasterWhisperTranscriber is None:
                raise RuntimeError("ASR модуль не установлен")
            self._asr = FasterWhisperTranscriber(
                model_size=self.asr_model_size,
                device=self.asr_device,
                compute_type=self.asr_compute_type,
                language=self.asr_language,
            )
        return self._asr  # type: ignore[return-value]
    
    def _play_cached_audio(self, audio: np.ndarray, generation: Optional[int] = None) -> None:
        """
        ✅ ОПТИМИЗАЦИЯ 2B: Воспроизведение предгенерированного аудио из кэша
        """
        if audio.size <= 0 or self._tts is None:
            return
        
        # Проверяем прерывание
        if generation is not None and self._tts_generation > generation:
            logger.debug("Кэшированное аудио прервано (устаревшее поколение)")
            return
        
        if self._tts_interrupt.is_set() or self._stop_requested.is_set():
            logger.debug("Кэшированное аудио прервано")
            return
        
        duration = float(audio.shape[0]) / float(self._tts.sample_rate)
        
        # Блокируем микрофон ТОЛЬКО если динамики
        if not self._use_headphones:
            self._suppress_until = time.time() + duration + 0.2
            logger.debug(f"Блокировка микрофона на {duration + 0.2:.1f}с (кэш)")
        
        # Воспроизводим
        if sd is None:
            return
        
        with self._tts_lock:
            if self._tts_interrupt.is_set() or self._stop_requested.is_set():
                logger.debug("Прервано до воспроизведения (кэш)")
                try:
                    sd.stop()
                except Exception:
                    pass
                return
            
            sd.play(audio, samplerate=self._tts.sample_rate, device=None)
            sd.wait()
            
            if self._tts_interrupt.is_set() or self._stop_requested.is_set():
                logger.debug("Прервано после воспроизведения (кэш)")
                return

    def _speak_text(self, text: str, generation: Optional[int] = None, thesis_index: Optional[int] = None) -> None:
        # Если нет текста или TTS не инициализирован — выходим
        if not text or self._tts is None:
            return
        # Проверяем поколение - если устарели, выходим
        if generation is not None and self._tts_generation > generation:
            logger.debug(f"TTS прервано (устаревшее поколение: {generation} < {self._tts_generation})")
            return
        # ВАЖНО: Если текст пустой или только пробелы - не озвучиваем
        if not text.strip():
            logger.debug("Пустой ответ - не озвучиваем")
            return
        
        # ✅ ОПТИМИЗАЦИЯ 2B: Проверяем кэш предгенерированного аудио
        cached_audio = None
        if thesis_index is not None:
            cached_audio = self._thesis_manager.get_cached_audio(thesis_index)
            if cached_audio is not None:
                logger.debug(f"✓ Использую кэшированное аудио для тезиса {thesis_index}")
                # Воспроизводим кэшированное аудио напрямую
                self._play_cached_audio(cached_audio, generation)
                return
        try:
            # Базовая валидация: не озвучиваем JSON-подобные ключи и пустые конструкции
            s = (text or "").strip()
            if not s:
                return
            # Удалим внешние кавычки
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1].strip()
            # Специальное подавление: не озвучиваем служебное уведомление,
            # а только логируем его
            try:
                if "прямых вопросов пользователю не обнаружено" in s.lower():
                    logger.info("прямых вопросов пользователю не обнаружено")
                    return
            except Exception:
                # На всякий случай, даже если что-то пойдёт не так при проверке
                # строки — не блокируем основную логику
                pass
            # Пропускаем строки вида "theses": [], а также любые ключ: значение без нормального текста
            import re as _re
            if _re.match(r'^"?\w+"?\s*:\s*(\[.*\]|\{.*\}|".*"|\d+|true|false|null)?\s*$', s, flags=_re.IGNORECASE):
                return
            # Должны быть буквы (кириллица/латиница)
            if not any(ch.isalpha() for ch in s):
                return
            # Разбиваем длинный текст на фразы, чтобы TTS не обрезался и стартовал быстрее
            def _split_for_tts(t: str, max_len: int = 180) -> list[str]:
                parts: list[str] = []
                # делим по предложениям
                sent = _re.split(r'([\.\!\?]+\s+)', t)
                buf = ""
                for i in range(0, len(sent), 2):
                    chunk = sent[i]
                    sep = sent[i + 1] if i + 1 < len(sent) else ""
                    piece = (chunk + sep).strip()
                    if not piece:
                        continue
                    if len((buf + " " + piece).strip()) <= max_len:
                        buf = (buf + " " + piece).strip()
                    else:
                        if buf:
                            parts.append(buf)
                        if len(piece) > max_len:
                            words = piece.split()
                            cur = ""
                            for w in words:
                                if len((cur + " " + w).strip()) <= max_len:
                                    cur = (cur + " " + w).strip()
                                else:
                                    if cur:
                                        parts.append(cur)
                                    cur = w
                            if cur:
                                parts.append(cur)
                            buf = ""
                        else:
                            buf = piece
                if buf:
                    parts.append(buf)
                return parts or [t]

            chunks = _split_for_tts(s)

            # ✅ ОПТИМИЗАЦИЯ 8A: Метрики времени TTS
            tts_start = time.time()
            
            for part in chunks:
                # Проверяем прерывание ПЕРЕД синтезом (новый вопрос)
                if self._tts_interrupt.is_set() or self._stop_requested.is_set():
                    logger.debug("TTS прервано до синтеза")
                    return
                
                chunk_start = time.time()
                audio = self._tts.synth(part)
                chunk_elapsed = (time.time() - chunk_start) * 1000
                logger.debug(f"⏱️  TTS chunk: {chunk_elapsed:.0f}мс ({len(part)} символов)")
                
                # Конвертируем bytes (Google TTS) в numpy array
                if isinstance(audio, bytes):
                    if len(audio) == 0:
                        continue
                    # Декодируем WAV bytes в numpy
                    import io, wave
                    with wave.open(io.BytesIO(audio), 'rb') as wf:
                        frames = wf.readframes(wf.getnframes())
                        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                if audio.size <= 0:
                    continue
                duration = float(audio.shape[0]) / float(self._tts.sample_rate)
                # Блокируем микрофон ТОЛЬКО если динамики (не наушники)
                # С наушниками микрофон не слышит TTS, блокировка мешает ловить новые вопросы
                if not self._use_headphones:
                    # Для динамиков: блокируем на время воспроизведения + минимальный запас
                    self._suppress_until = time.time() + duration + 0.2
                    logger.debug(f"Блокировка микрофона на {duration + 0.2:.1f}с (динамики)")
                else:
                    # Для наушников: микрофон всегда открыт
                    logger.debug(f"Наушники - микрофон продолжает слушать (TTS {duration:.1f}с)")
                if self._audio_sink is not None:
                    import io, wave
                    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
                    buf = io.BytesIO()
                    with wave.open(buf, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(int(self._tts.sample_rate))
                        wf.writeframes(pcm16.tobytes())
                    wav_bytes = buf.getvalue()
                    try:
                        self._audio_sink(wav_bytes, int(self._tts.sample_rate))
                    except Exception as _e:  # noqa: F841, BLE001
                        logger.debug("Audio sink send failed", _e)
                else:
                    if sd is None:
                        continue
                    
                    # Используем Lock чтобы избежать конфликта PaMacCore
                    with self._tts_lock:
                        # Проверяем прерывание перед воспроизведением
                        if self._tts_interrupt.is_set() or self._stop_requested.is_set():
                            logger.debug("TTS прервано")
                            # Останавливаем аудио если прервано
                            try:
                                sd.stop()
                            except Exception:
                                pass
                            continue
                        
                        # Воспроизводим через системный вывод (наушники если подключены, иначе динамики)
                        sd.play(audio, samplerate=self._tts.sample_rate, device=None)
                        sd.wait()  # Ждём завершения воспроизведения
                        
                        # Проверяем прерывание ПОСЛЕ воспроизведения (на случай если sd.stop() сработал во время wait)
                        if self._tts_interrupt.is_set() or self._stop_requested.is_set():
                            logger.debug("TTS прервано после воспроизведения")
                            return
            
            # ✅ ОПТИМИЗАЦИЯ 8A: Итоговая метрика TTS
            tts_elapsed = (time.time() - tts_start) * 1000
            logger.debug(f"⏱️  TTS ИТОГО: {tts_elapsed:.0f}мс ({len(chunks)} chunks)")
            
        except Exception as e:  # noqa: BLE001
            logger.exception(f"TTS ошибка: {e}")

    # ==== Лайв из внешнего потока кадров (20мс) ====
    def live_verify_stream(
        self,
        profile: VoiceProfile,
        frame_queue: "queue.Queue[np.ndarray]",
        min_segment_ms: int = 1500,
        max_silence_ms: int = 400,
        pre_roll_ms: int = 160,
        run_seconds: float = 0.0,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Аналог live_verify, но вместо микрофона читает кадры из очереди frame_queue.
        В очередь должны поступать моно float32 фреймы длиной FRAME_SIZE (= 20мс @ 16кГц)."""
        logger.info("Старт лайв-распознавания (внешний поток кадров)")
        seg_audio: List[np.ndarray] = []
        pre_frames: List[np.ndarray] = []
        pre_roll_frames_cnt = max(0, int(np.ceil(pre_roll_ms / FRAME_MS)))
        pre_roll = deque(maxlen=pre_roll_frames_cnt)
        voiced_ms = 0
        silence_ms = 0
        in_speech = False
        consec_speech = 0

        stop_at = time.time() + run_seconds if run_seconds and run_seconds > 0 else None
        try:
            self._start_segment_worker()
            leftover: Optional[np.ndarray] = None
            while True:
                if stop_event is not None and stop_event.is_set():
                    logger.info("Остановка live_verify_stream по сигналу stop_event")
                    break
                if stop_at is not None and time.time() >= stop_at:
                    logger.info("Авто-остановка live_verify_stream по таймеру run_seconds")
                    break
                try:
                    block = frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Подавление самоподхвата: блокируем ТОЛЬКО если используются динамики
                # С наушниками микрофон не слышит TTS, блокировка мешает ловить новые вопросы
                if not self._use_headphones and time.time() < self._suppress_until:
                    continue

                # склеим с хвостом предыдущего блока, если был
                if leftover is not None and leftover.size > 0:
                    block = np.concatenate([leftover, block.astype(np.float32).reshape(-1)])
                    leftover = None
                else:
                    block = block.astype(np.float32).reshape(-1)

                i = 0
                n = block.shape[0]
                while i + FRAME_SIZE <= n:
                    frame = block[i : i + FRAME_SIZE]
                    i += FRAME_SIZE
                    if pre_roll_frames_cnt > 0:
                        pre_roll.append(frame.copy())
                    is_speech = self._vad_fn(frame)
                    if is_speech:
                        consec_speech += 1
                        if not in_speech:
                            pre_frames.append(frame)
                            if consec_speech >= self.min_consec_speech_frames:
                                if pre_roll_frames_cnt > 0 and len(pre_roll) > 0:
                                    seg_audio.extend(list(pre_roll))
                                    pre_roll.clear()
                                seg_audio.extend(pre_frames)
                                pre_frames.clear()
                                in_speech = True
                                voiced_ms = self.min_consec_speech_frames * FRAME_MS
                        else:
                            seg_audio.append(frame)
                            voiced_ms += FRAME_MS
                        silence_ms = 0
                    else:
                        consec_speech = 0
                        pre_frames.clear()
                        if in_speech:
                            silence_ms += FRAME_MS

                    if in_speech and silence_ms >= max_silence_ms:
                        in_speech = False
                        total_ms = voiced_ms
                        wav = (
                            np.concatenate(seg_audio, axis=0)
                            if len(seg_audio) > 0
                            else np.array([], dtype=np.float32)
                        )
                        seg_audio.clear()
                        voiced_ms = 0
                        silence_ms = 0
                        if total_ms < min_segment_ms or wav.size < FRAME_SIZE:
                            continue
                        sf_med = median_spectral_flatness(wav, SAMPLE_RATE)
                        if sf_med >= self.flatness_reject_threshold:
                            continue
                        wav = self._preprocess_segment(wav, sr=SAMPLE_RATE)
                        emb = self.embedding_from_waveform(wav)
                        dist = self.cosine_distance(emb, profile.embedding)
                        if dist <= self.threshold:
                            self._enqueue_segment("self", wav, dist)
                        else:
                            self._enqueue_segment("other", wav, dist)

                # сохраним хвост, если остался неполный кадр
                if i < n:
                    leftover = block[i:]
                else:
                    leftover = None
        except KeyboardInterrupt:
            logger.info("Остановлено пользователем")
        finally:
            self._stop_segment_worker()

    def _append_question_context(self, text: str) -> None:
        if not text:
            return
        if self._question_context:
            self._question_context += "\n"
        self._question_context += text.strip()
        # Ограничим окно контекста, чтобы держать только последние N символов
        if len(self._question_context) > self._max_question_context_chars:
            self._question_context = self._question_context[-self._max_question_context_chars :]
        # Очистка истории тезисов при переполнении (предотвращаем утечку памяти)
        if len(self._theses_history) > self._max_theses_history:
            # Сохраняем только последние N тезисов
            recent = list(self._theses_history)[-self._max_theses_history:]
            self._theses_history = set(recent)

    def _process_self_segment(self, wav: np.ndarray) -> None:
        if not self.asr_enable:
            return
        try:
            transcript = self._ensure_asr().transcribe_np(wav, SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"ASR ошибка при распознавании моего голоса: {e}")
            return
        self._handle_self_transcript(transcript)

    # ==== Warmup моделей ====
    def _warmup_models(self) -> None:
        # ASR: прогнать короткую тишину
        try:
            if self.asr_enable:
                _ = self._ensure_asr().transcribe_np(np.zeros((SAMPLE_RATE//2,), dtype=np.float32), SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"ASR warmup failed: {e}")
        # LLM: сгенерировать короткий ответ (не озвучивать)
        try:
            if self.llm_enable and self._llm is None and LLMResponder is not None:
                self._llm = LLMResponder()
            if self.llm_enable and self._llm is not None:
                _ = self._llm.generate("Привет. Проверь готовность.")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"LLM warmup failed: {e}")
        # TTS: синтезировать короткое аудио без воспроизведения
        try:
            if self._tts is not None:
                _ = self._tts.synth("Готов.")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"TTS warmup failed: {e}")

    # ==== Извлечение вопросов из текста собеседника ====
    @staticmethod
    def _extract_questions(text: str) -> List[str]:
        if not text:
            return []
        t = text.strip()
        # Разобьём на предложения по знакам препинания
        parts: List[str] = []
        buf = []
        for ch in t:
            buf.append(ch)
            if ch in "?!.\n":
                seg = "".join(buf).strip()
                if seg:
                    parts.append(seg)
                buf = []
        if buf:
            parts.append("".join(buf).strip())
        res: List[str] = []
        for p in parts:
            if LiveVoiceVerifier._is_question_ru(p):
                res.append(p)
        return res

    @staticmethod
    def _is_question_ru(s: str) -> bool:
        if not s:
            return False
        s2 = s.strip().lower()
        # Явный признак вопроса
        if s2.endswith("?"):
            return True
        # Вопросительные слова и шаблоны
        patterns = [
            r"\bкто\b", r"\bчто\b", r"\bкогда\b", r"\bгде\b", r"\bпочему\b",
            r"\bзачем\b", r"\bкак\b", r"\bкакой\b", r"\bкакова\b", r"\bкотор\w*\b",
            r"\bсколько\b", r"\bв каком году\b", r"\bправда ли\b", r"\bможно ли\b",
        ]
        for p in patterns:
            try:
                import re
                if re.search(p, s2):
                    return True
            except Exception:
                continue
        return False

    @staticmethod
    def _compute_age(year: int, month: int, day: int) -> int:
        """Вычисляет возраст в полных годах на сегодня."""
        today = date.today()
        age = today.year - year - (1 if (today.month, today.day) < (month, day) else 0)
        return int(age)

    @staticmethod
    def _format_age_ru(age: int) -> str:
        """Форматирует возраст с корректным склонением: 1 год, 2-4 года, 5+ лет."""
        n = abs(int(age))
        last_two = n % 100
        last = n % 10
        if 11 <= last_two <= 14:
            word = "лет"
        elif last == 1:
            word = "год"
        elif 2 <= last <= 4:
            word = "года"
        else:
            word = "лет"
        return f"{n} {word}"

    # ==== Фильтрация вопросов по важности ====
    def _filter_questions_by_importance(self, qs: List[str]) -> List[str]:
        if not qs:
            return []
        # Быстрый эвристический фильтр: минимальная длина и ключевые слова
        if self._question_filter_mode != "gemini":
            keep: List[str] = []
            import re
            keywords = [
                r"\bв каком году\b",
                r"\bгод(а|у|ом)?\b",
                r"\bсколько\b",
                r"\bкто\b",
                r"\bчто такое\b",
                r"\bопредели(те)?\b",
                r"\bпочему\b",
                r"\bкак(им|ой|ая|ое)? образом\b",
                r"\bперечисли(те)?\b",
            ]
            exclude_patterns = [
                r"\bсколько тебе лет\b",
                r"\bсколько лет тебе\b",
                r"\bкак тебя зовут\b",
                r"\bкто ты\b",
                r"\bтебя зовут\b",
                r"\bтвой возраст\b",
                r"\bчто делаешь\b",
                r"\bкак дела\b",
            ]
            for q in qs:
                q2 = q.strip()
                if len(q2) < self._question_min_len:
                    continue
                if any(re.search(p, q2.lower()) for p in exclude_patterns):
                    continue
                hit = any(re.search(p, q2.lower()) for p in keywords)
                if q2.endswith("?") or hit:
                    keep.append(q2)
            return keep

        # Режим Gemini: классифицируем список одним запросом
        try:
            import json
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            import os as _os
            key = _os.getenv("GEMINI_API_KEY")
            if not key:
                return qs  # без ключа вернём как есть
            client = genai.Client(api_key=key)
            sys_instr = (
                "Ты — фильтр вопросов для экзамена/собеседования. Оцени каждую строку:"
                " это ли содержательный вопрос по теме (история, факты, технарщина и т.п.)?"
                " Игнорируй бытовые реплики и обсуждения. Верни JSON вида {\"keep\": [0|1,...]}"
            )
            payload = {"items": [s.strip() for s in qs if s.strip()]}
            user_text = json.dumps(payload, ensure_ascii=False)
            cfg = types.GenerateContentConfig(
                system_instruction=sys_instr,
                max_output_tokens=128,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            resp = client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_text)])],
                config=cfg,
            )
            raw = (resp.text or "").strip()
            data = json.loads(raw)
            keep_mask = list(map(int, data.get("keep", [])))
            out: List[str] = []
            for s, k in zip(qs, keep_mask):
                if k:
                    out.append(s)
            return out if out else []
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Gemini question filter failed: {e}")
            return qs

    # ==== ИИ-экстракция тезисов напрямую из чужой реплики ====
    def _extract_theses_ai(self, text: str) -> List[str]:
        """Возвращает список тезисов для ответа, если в тексте есть содержательные вопросы.
        Если вопросов нет — возвращает пустой список.
        Использует Gemini с строгим JSON-контрактом.
        """
        t = (text or "").strip()
        if not t:
            return []
        try:
            import json
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
            key = os.getenv("GEMINI_API_KEY")
            if not key:
                return []
            client = genai.Client(api_key=key)

            def _call_and_parse(force_json_start: bool = False) -> List[str]:
                sys_instr = (
                    "Вопрос к кандидату? Верни 2-3 тезиса. Личное? Пустой список."
                    " JSON: {\"theses\": [\"...\"]}"
                )
                if force_json_start:
                    sys_instr += " Ответ ДОЛЖЕН начинаться с символа { и не содержать ничего кроме JSON."
                prompt = json.dumps({"transcript": t}, ensure_ascii=False)
                cfg = types.GenerateContentConfig(
                    system_instruction=sys_instr,
                    max_output_tokens=64,
                    temperature=0.0,
                    top_p=0.8,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                )
                resp = client.models.generate_content(
                    model="gemini-flash-lite-latest",
                    contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                    config=cfg,
                )
                raw = (resp.text or "").strip()
                return LiveVoiceVerifier._parse_theses_from_raw(raw, n_max=self._thesis_autogen_batch)

            out = _call_and_parse(False)
            if not out:
                out = _call_and_parse(True)
            return out
        except Exception as e:  # noqa: BLE001
            logger.debug(f"AI theses extraction failed: {e}")
            return []


def enroll_cli(
    profile_path: Path = Path("voice_profile.npz"),
    seconds: float = 20.0,
    min_voiced_seconds: float = 4.0,
    vad_aggr: int = 2,
    min_consec: int = 5,
    flatness_th: float = 0.60,
    # VAD backend
    vad_backend: str = "webrtc",
    silero_vad_threshold: float = 0.5,
    silero_vad_window_ms: int = 100,
    # ASR в энроллменте не используется, но оставим одинаковую сигнатуру для единообразия
    asr: bool = False,
    asr_model: str = "large-v3-turbo",
    asr_lang: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute: Optional[str] = None,
    # Текст для чтения при записи профиля
    read_script: Optional[str] = None,
    read_script_file: Optional[Path] = None,
) -> None:
    setup_logging()
    verifier = LiveVoiceVerifier(
        vad_backend=vad_backend,
        silero_vad_threshold=silero_vad_threshold,
        silero_vad_window_ms=silero_vad_window_ms,
        vad_aggressiveness=vad_aggr,
        min_consec_speech_frames=min_consec,
        flatness_reject_threshold=flatness_th,
    )

    # Подготовим текст для чтения (опционально)
    script_text: Optional[str] = None
    if read_script_file is not None:
        try:
            script_text = Path(read_script_file).read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Не удалось прочитать файл скрипта {read_script_file}: {e}")
            script_text = None
    if not script_text and read_script:
        script_text = read_script.strip()
    paragraphs = _split_paragraphs(script_text or "")
    if not paragraphs:
        paragraphs = DEFAULT_ENROLL_PARAGRAPHS.copy()

    selected_idx = 0
    if len(paragraphs) > 1:
        print("Выберите абзац для чтения (полный текст будет показан ниже):")
        for idx, para in enumerate(paragraphs, 1):
            preview = para.replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:87] + "…"
            print(f"  {idx}) {preview}")
        choice = input("Номер абзаца (по умолчанию 1): ").strip()
        try:
            parsed = int(choice)
            if 1 <= parsed <= len(paragraphs):
                selected_idx = parsed - 1
            else:
                logger.warning(f"Некорректный номер {choice}, используем абзац 1")
        except Exception:
            if choice:
                logger.warning(f"Не удалось распознать ввод '{choice}', используем абзац 1")

    script_to_read = paragraphs[selected_idx]
    logger.info(
        "Для профиля выбран абзац {idx} ({words} слов)",
        idx=selected_idx + 1,
        words=len(script_to_read.split()),
    )

    print("\nЧитайте абзац №{0}:\n{1}\n".format(selected_idx + 1, script_to_read))
    print("Совет: говорите размеренно и сделайте паузу около секунды после окончания чтения.")
    try:
        input("Нажмите Enter, чтобы начать запись…")
    except Exception:
        pass

    words = len(script_to_read.split())
    min_read_seconds = words / 2.5 + 2.0  # комфортный темп речи ~150 слов/мин
    record_seconds = max(seconds, min_read_seconds)
    logger.info(
        "Старт записи профиля: целевой лимит {record_seconds:.1f}s, минимум озвученной речи {min_voiced_seconds:.1f}s",
        record_seconds=record_seconds,
        min_voiced_seconds=min_voiced_seconds,
    )

    wav = verifier._collect_enrollment_audio(
        seconds=record_seconds,
        min_voiced_seconds=min_voiced_seconds,
        silence_stop_sec=1.4,
    )
    emb = verifier.embedding_from_waveform(wav)
    profile = VoiceProfile(embedding=emb)
    profile.save(profile_path)
    logger.info(f"Профиль сохранён: {profile_path}")



def live_cli(
    profile_path: Path = Path("voice_profile.npz"),
    threshold: float = 0.30,
    vad_aggr: int = 2,
    min_consec: int = 5,
    flatness_th: float = 0.60,
    min_segment_ms: int = 1500,
    max_silence_ms: int = 400,
    pre_roll_ms: int = 160,
    # VAD backend
    vad_backend: str = "webrtc",
    silero_vad_threshold: float = 0.5,
    silero_vad_window_ms: int = 100,
    # ASR настройки
    asr: bool = False,
    asr_model: str = "large-v3-turbo",
    asr_lang: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute: Optional[str] = None,
    # LLM
    llm: bool = False,
    run_seconds: float = 0.0,
) -> None:
    setup_logging()
    verifier = LiveVoiceVerifier(
        threshold=threshold,
        vad_backend=vad_backend,
        silero_vad_threshold=silero_vad_threshold,
        silero_vad_window_ms=silero_vad_window_ms,
        vad_aggressiveness=vad_aggr,
        min_consec_speech_frames=min_consec,
        flatness_reject_threshold=flatness_th,
        asr_enable=asr,
        asr_model_size=asr_model,
        asr_language=asr_lang,
        asr_device=asr_device,
        asr_compute_type=asr_compute,
        llm_enable=llm,
    )
    profile = VoiceProfile.load(profile_path)
    if profile is None:
        logger.error(
            f"Профиль не найден: {profile_path}. Сначала выполните команду enroll."
        )
        sys.exit(1)
    verifier.live_verify(
        profile,
        min_segment_ms=min_segment_ms,
        max_silence_ms=max_silence_ms,
        pre_roll_ms=pre_roll_ms,
        run_seconds=run_seconds,
    )


__all__ = [
    "LiveVoiceVerifier",
    "VoiceProfile",
    "enroll_cli",
    "live_cli",
]
