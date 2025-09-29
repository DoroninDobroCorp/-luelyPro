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
try:  # Silero TTS (опционально)
    from tts_silero import SileroTTS  # type: ignore
except Exception:  # noqa: BLE001
    SileroTTS = None  # type: ignore
try:  # Silero VAD (опционально)
    from vad_silero import SileroVAD  # type: ignore
except Exception:  # noqa: BLE001
    SileroVAD = None  # type: ignore
from thesis_prompter import ThesisPrompter
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
        # Тезисы
        theses_path: Optional[Path | str] = Path("theses.txt"),
        thesis_match_threshold: float = 0.6,
        thesis_semantic_threshold: float = 0.55,
        thesis_semantic_model: Optional[str] = None,
        thesis_semantic_enable: bool = True,
        # Gemini judge
        thesis_gemini_enable: bool = True,
        thesis_gemini_min_conf: float = 0.60,
        # Автогенерация тезисов на основе чужой речи
        thesis_autogen_enable: bool = True,
        thesis_autogen_batch: int = 4,
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
        # Внешний получатель аудио (например, WebSocket-клиент). При наличии —
        # TTS будет отправляться туда, а не проигрываться локально через sounddevice
        self._audio_sink: Optional[Callable[[bytes, int], None]] = None

        # Тезисный помощник
        self.thesis_prompter: Optional[ThesisPrompter] = None
        self._thesis_done_notified = False
        # Конфигурация для переинициализации помощника
        self._thesis_match_threshold = float(thesis_match_threshold)
        self._thesis_semantic_enable = bool(thesis_semantic_enable)
        self._thesis_semantic_threshold = float(thesis_semantic_threshold)
        self._thesis_semantic_model = thesis_semantic_model
        self._thesis_gemini_enable = bool(thesis_gemini_enable)
        self._thesis_gemini_min_conf = float(thesis_gemini_min_conf)
        # Автогенерация
        self._thesis_autogen_enable = bool(thesis_autogen_enable)
        self._thesis_autogen_batch = max(1, int(thesis_autogen_batch))
        self._thesis_generator: Optional[GeminiThesisGenerator] = None
        self._theses_history: set[str] = set()
        self._question_context: str = ""
        self._max_question_context_chars: int = 2000
        self._last_announce_ts: float = 0.0
        self._segment_queue: "queue.Queue[QueuedSegment]" = queue.Queue(maxsize=4)
        self._segment_worker: Optional[threading.Thread] = None
        self._segment_stop = threading.Event()
        # Как часто повторять текущий тезис (секунды), если он ещё не закрыт
        try:
            # Увеличиваем интервал повторения чтобы не мешать
            self._thesis_repeat_sec: float = float(os.getenv("THESIS_REPEAT_SEC", "30"))
        except Exception:
            self._thesis_repeat_sec = 30.0
        # Фильтрация «не-вопросов»: heuristic | gemini (по умолчанию gemini)
        self._question_filter_mode: str = os.getenv("QUESTION_FILTER_MODE", "gemini").strip().lower()
        try:
            self._question_min_len: int = int(os.getenv("QUESTION_MIN_LEN", "8"))
        except Exception:
            self._question_min_len = 8
        # Режим: исключительно ИИ-экстракция тезисов из чужой речи
        self._ai_only_thesis: bool = os.getenv("AI_ONLY_THESIS", "1").strip() not in ("0", "false", "False")
        if theses_path:
            theses_list = self._load_theses(Path(theses_path))
            if theses_list:
                self.thesis_prompter = ThesisPrompter(
                    theses=theses_list,
                    match_threshold=self._thesis_match_threshold,
                    enable_semantic=self._thesis_semantic_enable,
                    semantic_threshold=self._thesis_semantic_threshold,
                    semantic_model_id=self._thesis_semantic_model,
                    enable_gemini=self._thesis_gemini_enable,
                    gemini_min_conf=self._thesis_gemini_min_conf,
                )
                logger.info(
                    f"Тезисный помощник активирован: {len(theses_list)} пунктов"
                )
            else:
                logger.info(
                    f"Статические тезисы не найдены (файл {theses_path} отсутствует или пуст) — будет использоваться автогенерация"
                )
        # Инициализация генератора тезисов
        if self._thesis_autogen_enable and GeminiThesisGenerator is not None:
            try:
                self._thesis_generator = GeminiThesisGenerator()  # type: ignore
                logger.info("Автогенерация тезисов Gemini включена")
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Не удалось инициализировать ThesisGenerator: {e}")
                self._thesis_autogen_enable = False
        elif self._thesis_autogen_enable and GeminiThesisGenerator is None:
            logger.warning("Автогенерация тезисов отключена: нет зависимости thesis_generator/google-genai")
            self._thesis_autogen_enable = False

        if self.asr_enable:
            try:
                self._ensure_asr()
            except Exception as e:  # noqa: BLE001
                logger.exception(f"ASR недоступен: {e}")

        if self.llm_enable and LLMResponder is not None:
            try:
                self._llm = LLMResponder()
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Не удалось инициализировать LLMResponder: {e}")

        # Инициализируем TTS (RU по умолчанию), если он нужен для LLM или тезисов
        if self.llm_enable or self.thesis_prompter is not None:
            if SileroTTS is not None:
                try:
                    self._tts = SileroTTS(
                        language="ru", model_id="v4_ru", speaker="eugene", sample_rate=24000
                    )
                except Exception as e:  # noqa: BLE001
                    logger.exception(f"Не удалось инициализировать TTS: {e}")
            else:
                logger.debug("SileroTTS недоступен — озвучка отключена")

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
        if self._segment_worker and self._segment_worker.is_alive():
            return
        self._segment_stop.clear()
        self._segment_worker = threading.Thread(
            target=self._segment_worker_loop,
            name="segment-worker",
            daemon=True,
        )
        self._segment_worker.start()

    def _stop_segment_worker(self) -> None:
        self._segment_stop.set()
        if self._segment_worker is not None:
            self._segment_worker.join(timeout=2.0)
            self._segment_worker = None
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
        while not self._segment_stop.is_set() or not self._segment_queue.empty():
            try:
                segment = self._segment_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if segment.kind == "self":
                    self._handle_self_segment(segment)
                else:
                    self._handle_foreign_segment(segment)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Ошибка обработки сегмента: {e}")
            finally:
                self._segment_queue.task_done()

    def _handle_self_segment(self, segment: QueuedSegment) -> None:
        logger.info("мой голос")
        if not self.asr_enable:
            logger.debug("ASR отключён — пропускаю анализ собственной речи")
            return
        try:
            transcript = self._ensure_asr().transcribe_np(segment.audio, SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"ASR ошибка при распознавании моего голоса: {e}")
            return
        self._handle_self_transcript(transcript)

    def _handle_foreign_segment(self, segment: QueuedSegment) -> None:
        if self.asr_enable:
            try:
                text = self._ensure_asr().transcribe_np(segment.audio, SAMPLE_RATE)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"ASR ошибка: {e}")
                return
            self._handle_foreign_text(text)
        else:
            logger.info("незнакомый голос")

    def _handle_foreign_text(self, text: Optional[str]) -> None:
        t = (text or "").strip()
        if not t:
            logger.info("незнакомый голос (ASR: пусто)")
            return
        if (time.time() - self._suppress_until) < 0.4:
            logger.debug(f"Игнорирую распознанный TTS-хвост: {t}")
            return
        logger.info(f"незнакомый голос (ASR): {t}")
        # Быстрый хэндлер простых математических запросов
        try:
            math_ans = self._answer_math_if_any(t)
        except Exception:
            math_ans = None
        if math_ans:
            logger.info(f"Ответ: {math_ans}")
            self._speak_text(math_ans)
            return
        # Жёсткие исключения: некоторые триггеры не считаем вопросами и не обрабатываем
        try:
            if self._should_ignore_non_question_text(t):
                logger.debug("Игнорирую нерелевантный триггер (музыка и т.п.)")
                return
        except Exception:
            pass
        # Если включён LLM — генерируем краткий ответ сразу
        if self.llm_enable and self._llm is not None:
            try:
                prompt = t
                ans = (self._llm.generate(prompt) or "").strip()
                if ans:
                    try:
                        ans = self._enforce_answer_then_question(ans)
                    except Exception:
                        pass
                    logger.info(f"Ответ: {ans}")
                    self._speak_text(ans)
                    return
            except Exception as e:  # noqa: BLE001
                logger.exception(f"LLM ошибка при генерации ответа: {e}")
        # Быстрая обработка тезисов через AI
        if self._ai_only_thesis:
            theses = self._extract_theses_ai(t)
            if theses:
                # Ограничиваем количество тезисов для скорости
                limited_theses = theses[:min(3, len(theses))]
                self.thesis_prompter = ThesisPrompter(
                    theses=limited_theses,
                    match_threshold=self._thesis_match_threshold,
                    enable_semantic=False,  # Отключаем семантику для скорости
                    semantic_threshold=self._thesis_semantic_threshold,
                    semantic_model_id=self._thesis_semantic_model,
                    enable_gemini=False,  # Отключаем Gemini для скорости
                    gemini_min_conf=self._thesis_gemini_min_conf,
                )
                self._thesis_done_notified = False
                logger.info(f"Тезисы (новые): {', '.join(limited_theses)}")
                # Анонсируем через общий механизм, чтобы писалось в логи и работал автоповтор
                self._announce_thesis()
            else:
                logger.debug("ИИ не нашёл вопросов/тезисов в фрагменте — пропускаю")
            return
        # Обработка вопросов - упрощаем
        questions = self._extract_questions(t)
        questions = self._filter_questions_by_importance(questions)
        if questions:
            q_joined = " ".join(questions)
            self._append_question_context(q_joined)
            self._maybe_generate_theses()
        else:
            logger.debug("Пропускаю нерелевантный/не-вопрос текст собеседника")

    @staticmethod
    def _answer_math_if_any(text: str) -> Optional[str]:
        """Распознаёт простые математические фразы и строит ответ.
        Сейчас поддерживается шаблон: "<int> в степени <int>".
        Возвращает готовую фразу вида: "<результат> будет <основание> в степени <показатель>".
        """
        if not text:
            return None
        import re
        t = text.strip().lower()
        # Наиболее частотный и однозначный паттерн: цифры + "в степени" + цифры
        m = re.search(r"\b(\d{1,9})\s+в\s+степен[еи]\s+(\d{1,3})\b", t)
        if not m:
            return None
        try:
            base = int(m.group(1))
            exp = int(m.group(2))
        except Exception:
            return None
        # Защита от чрезмерно больших расчётов
        if exp > 1000:
            return None
        try:
            res = pow(base, exp)
        except Exception:
            return None
        # Сформируем человекочитаемую фразу: сначала ответ, затем пояснение
        base_w = None
        exp_w = None
        try:
            from num2words import num2words  # type: ignore
            base_w = num2words(base, lang="ru")
            exp_w = num2words(exp, lang="ru")
        except Exception:
            pass
        base_part = base_w if base_w else str(base)
        exp_part = exp_w if exp_w else str(exp)
        # Если число слишком длинное, не озвучиваем все цифры
        res_s = str(res)
        if len(res_s) > 30:
            # Дадим компактную оценку в научной форме
            try:
                import math as _math
                # mantissa * 10^k
                k = len(res_s) - 1
                mantissa = float(res_s[0] + "." + res_s[1: min(6, len(res_s))])
                approx = f"примерно {mantissa:.3f} на десять в степени {k}"
                return f"{approx}. {base_part} в степени {exp_part}."
            except Exception:
                return f"Число очень большое. {base_part} в степени {exp_part}."
        return f"{res_s} будет {base_part} в степени {exp_part}."

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

    @staticmethod
    def _enforce_answer_then_question(text: str) -> str:
        """Переставляет вопросительные предложения в конец, сначала оставляя ответ.
        Разделяет по предложениям, сохраняет порядок внутри групп.
        """
        if not text:
            return text
        import re
        s = text.strip()
        # Быстро нормализуем переносы
        s = re.sub(r"\s+", " ", s)
        # Разбиваем на предложения, сохранив разделители
        parts = re.split(r"([\.!?]+\s+)", s)
        # Сконструируем список предложений с их финальным пунктуационным знаком
        sentences = []
        for i in range(0, len(parts), 2):
            chunk = parts[i].strip()
            sep = parts[i+1] if i+1 < len(parts) else ""
            sent = (chunk + (sep or "")).strip()
            if sent:
                sentences.append(sent)
        if not sentences:
            return s
        declarative: list[str] = []
        questions: list[str] = []
        for sent in sentences:
            if sent.endswith("?"):
                questions.append(sent)
            else:
                declarative.append(sent)
        if not declarative:
            # Если все — вопросы, вернём как есть
            return s
        out = " ".join(declarative + questions).strip()
        return out

    def _handle_self_transcript(self, transcript: Optional[str]) -> None:
        t = (transcript or "").strip()
        if not t:
            return
        logger.info(f"Моя речь (ASR): {t}")
        if self.thesis_prompter is None:
            logger.debug("Тезисный помощник не активен — пропускаю самоанализ")
            return
        if self.thesis_prompter.consume_transcript(t):
            logger.info("Тезис закрыт")
            if not self.thesis_prompter.has_pending():
                self._maybe_generate_theses()
            else:
                self._announce_thesis()
            return
        try:
            cov = self.thesis_prompter.coverage_of_current()
            logger.info(f"Прогресс текущего тезиса: {int(cov*100)}%")
        except Exception:
            pass
        self._announce_thesis()

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
        if (
            self.thesis_prompter is not None
            and self.thesis_prompter.has_pending()
            and self.thesis_prompter.need_announce()
        ):
            self._announce_thesis()

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
        min_segment_ms: int = 500,  # минимальная длительность сегмента для эмбеддинга
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
        try:
            self._start_segment_worker()
            with sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                blocksize=FRAME_SIZE,  # deliver 20ms blocks
                callback=callback,
            ):
                while True:
                    if stop_at is not None and time.time() >= stop_at:
                        logger.info("Авто-остановка по таймеру run_seconds")
                        break
                    try:
                        block = q.get(timeout=0.5)[:, 0]
                    except queue.Empty:
                        continue

                    # если сейчас идёт воспроизведение TTS — глушим распознавание (избегаем самоподхвата)
                    if time.time() < self._suppress_until:
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

                        # Периодически повторяем текущий тезис, если ещё не закрыт
                        try:
                            if (
                                self.thesis_prompter is not None
                                and self.thesis_prompter.has_pending()
                                and (time.time() - self._last_announce_ts) >= self._thesis_repeat_sec
                                and time.time() >= self._suppress_until
                            ):
                                self.thesis_prompter.reset_announcement()
                                self._announce_thesis()
                        except Exception:
                            pass

        except KeyboardInterrupt:
            logger.info("Остановлено пользователем")
        finally:
            self._stop_segment_worker()

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

    def _speak_text(self, text: str) -> None:
        # Если нет текста или TTS не инициализирован — выходим
        if not text or self._tts is None:
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

            for part in chunks:
                audio = self._tts.synth(part)
                if audio.size <= 0:
                    continue
                duration = float(audio.shape[0]) / float(self._tts.sample_rate)
                self._suppress_until = time.time() + duration + 0.05
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
                    sd.stop()
                    sd.play(audio, samplerate=self._tts.sample_rate)
                    time.sleep(min(0.3, duration * 0.3))
        except Exception as e:  # noqa: BLE001
            logger.exception(f"TTS ошибка: {e}")

    # ==== Лайв из внешнего потока кадров (20мс) ====
    def live_verify_stream(
        self,
        profile: VoiceProfile,
        frame_queue: "queue.Queue[np.ndarray]",
        min_segment_ms: int = 500,
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

                # подавление самоподхвата во время озвучки
                if time.time() < self._suppress_until:
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

                    # периодический повтор тезиса
                    try:
                        if (
                            self.thesis_prompter is not None
                            and self.thesis_prompter.has_pending()
                            and (time.time() - self._last_announce_ts) >= self._thesis_repeat_sec
                            and time.time() >= self._suppress_until
                        ):
                            self.thesis_prompter.reset_announcement()
                            self._announce_thesis()
                    except Exception:
                        pass

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

    def _maybe_generate_theses(self) -> None:
        if not self._thesis_autogen_enable or self._thesis_generator is None:
            return
        need_new_batch = False
        if self.thesis_prompter is None:
            need_new_batch = True
        else:
            if not self.thesis_prompter.has_pending():
                need_new_batch = True
        if not need_new_batch:
            return
        qtext = self._question_context.strip()
        if not qtext:
            return
        try:
            # Генерируем меньше тезисов для скорости (2-3 вместо 4)
            candidates = self._thesis_generator.generate(qtext, n=min(3, self._thesis_autogen_batch), language="ru")
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Ошибка автогенерации тезисов: {e}")
            return
        # Быстрая фильтрация дублей 
        new_items: list[str] = []
        for c in candidates:
            key = c.strip().lower()
            if not key or key in self._theses_history:
                continue
            self._theses_history.add(key)
            new_items.append(c.strip())
            # Ограничиваем количество тезисов
            if len(new_items) >= 3:
                break
        if not new_items:
            return
        # Создаём упрощённый помощник без лишних проверок
        self.thesis_prompter = ThesisPrompter(
            theses=new_items,
            match_threshold=self._thesis_match_threshold,
            enable_semantic=False,  # Отключаем для скорости
            semantic_threshold=self._thesis_semantic_threshold,
            semantic_model_id=self._thesis_semantic_model,
            enable_gemini=False,  # Отключаем для скорости  
            gemini_min_conf=self._thesis_gemini_min_conf,
        )
        self._thesis_done_notified = False
        logger.info(f"Тезисы (новые): {', '.join(new_items)}")
        # Анонсируем через общий механизм, чтобы писалось в логи и работал автоповтор
        self._announce_thesis()

    def _announce_theses_batch(self, theses: list[str]) -> None:
        if not theses:
            return
        # Озвучиваем только сами тезисы, без лишних префиксов
        # Берём только первые 2-3 тезиса для краткости
        to_announce = theses[:min(3, len(theses))]
        for i, thesis in enumerate(to_announce, 1):
            # Краткое озвучивание: только номер и тезис
            text = f"{i}. {thesis}"
            self._speak_text(text)
            # Небольшая пауза между тезисами
            time.sleep(0.2)
        self._last_announce_ts = time.time()
        if self.thesis_prompter is not None:
            self.thesis_prompter.reset_announcement()

    def _announce_thesis(self) -> None:
        if self.thesis_prompter is None:
            return
        if not self.thesis_prompter.has_pending():
            if not self._thesis_done_notified:
                logger.info("Все тезисы пройдены")
                self._thesis_done_notified = True
            return
        if not self.thesis_prompter.need_announce():
            return
        text = self.thesis_prompter.current_text()
        if not text:
            return
        logger.info(f"Тезис: {text}")
        # Озвучиваем только сам тезис, без дополнительной информации
        self._speak_text(text)
        self.thesis_prompter.mark_announced()
        # Убираем озвучивание оставшихся тезисов - это избыточно
        self._last_announce_ts = time.time()

    def _process_self_segment(self, wav: np.ndarray) -> None:
        if not self.asr_enable:
            return
        try:
            transcript = self._ensure_asr().transcribe_np(wav, SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001
            logger.exception(f"ASR ошибка при распознавании моего голоса: {e}")
            return
        self._handle_self_transcript(transcript)

    @staticmethod
    def _load_theses(path: Path) -> List[str]:
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            logger.error(f"Не удалось прочитать файл тезисов {path}: {e}")
            return []
        items: List[str] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.lstrip("-•*0123456789. )\t")
            line = line.strip()
            if line:
                items.append(line)
        return items

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
                model="gemini-2.5-flash-lite",
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
                    model="gemini-2.5-flash-lite",
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


def extract_theses_from_text(text: str) -> List[str]:
    """Публичная функция: извлечь тезисы напрямую из текста чужой реплики/диалога.
    Возвращает [] если вопросов нет. Использует тот же JSON-контракт Gemini, что и _extract_theses_ai.
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
                "Вопрос к кандидату? Верни 2-3 тезиса ответа. Личное/бытовое? Пустой список."
                " JSON: {\"theses\": [\"...\"]}"
            )
            if force_json_start:
                sys_instr += " Ответ ДОЛЖЕН начинаться с символа { и не содержать ничего кроме JSON."
            prompt = json.dumps({"transcript": t}, ensure_ascii=False)
            cfg = types.GenerateContentConfig(
                system_instruction=sys_instr,
                max_output_tokens=128,
                temperature=0.0,
                top_p=0.8,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                response_mime_type="application/json",
            )
            resp = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=cfg,
            )
            raw = (resp.text or "").strip()
            return LiveVoiceVerifier._parse_theses_from_raw(raw, n_max=8)

        out = _call_and_parse(False)
        if not out:
            out = _call_and_parse(True)
        return out
    except Exception as e:  # noqa: BLE001
        logger.debug(f"extract_theses_from_text failed: {e}")
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
    min_segment_ms: int = 500,
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
    # Тезисы
    theses_path: Optional[Path] = None,
    thesis_match: float = 0.6,
    thesis_semantic: float = 0.55,
    thesis_semantic_model: Optional[str] = None,
    thesis_semantic_disable: bool = False,
    # Gemini judge
    thesis_gemini_conf: float = 0.60,
    thesis_gemini_disable: bool = False,
    # Автогенерация тезисов
    thesis_autogen_disable: bool = False,
    thesis_autogen_batch: int = 4,
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
        theses_path=theses_path,
        thesis_match_threshold=thesis_match,
        thesis_semantic_threshold=thesis_semantic,
        thesis_semantic_model=thesis_semantic_model,
        thesis_semantic_enable=not thesis_semantic_disable,
        thesis_gemini_enable=not thesis_gemini_disable,
        thesis_gemini_min_conf=thesis_gemini_conf,
        thesis_autogen_enable=not thesis_autogen_disable,
        thesis_autogen_batch=thesis_autogen_batch,
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
    "extract_theses_from_text",
]
