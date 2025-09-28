from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel
from loguru import logger
import re
import ctypes
from pathlib import Path


def _preload_nvidia_cuda_libs() -> None:
    """Предзагружаем cuDNN/cuBLAS из pip-пакетов nvidia-* внутри venv.
    Это позволяет CTranslate2 (faster-whisper) найти нужные .so даже если в системе нет
    системной установки cuDNN. Безопасно: просто пытаемся загрузить доступные .so.
    """
    try:
        import nvidia  # предоставляется пакетами вроде nvidia-cudnn-cu12, nvidia-cublas-cu12
    except Exception:
        return

    base = Path(getattr(nvidia, "__file__", "")).parent
    candidates = [
        base / "cudnn" / "lib" / "libcudnn.so.9",
        base / "cudnn" / "lib" / "libcudnn_cnn.so.9",
        base / "cudnn" / "lib" / "libcudnn_adv.so.9",
        base / "cublas" / "lib" / "libcublas.so.12",
        base / "cublas" / "lib" / "libcublasLt.so.12",
    ]

    for so in candidates:
        try:
            if so.is_file():
                ctypes.CDLL(str(so), mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # тихо продолжаем — другие библиотеки могут загрузиться, а часть не обязательна
            pass


@dataclass
class ASRConfig:
    model_size: str = "large-v3-turbo"  # e.g. "tiny", "base", "small", "medium", "large-v3"
    device: Optional[str] = None  # "cuda" | "cpu" (auto if None)
    compute_type: Optional[str] = None  # "float16" (GPU), "int8" (CPU), etc.
    language: Optional[str] = None  # e.g. "ru", or None for auto


class FasterWhisperTranscriber:
    def __init__(
        self,
        model_size: str = "large-v3-turbo",
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        banned_keywords: Optional[list[str]] = None,
        min_text_len: int = 3,
    ) -> None:
        # По умолчанию используем CPU для максимальной совместимости.
        # CUDA можно включить флагом CLI (--asr-device cuda). При ошибке cuDNN падаем на CPU.
        if device is None:
            device = "cpu"
        if compute_type is None:
            compute_type = "int8" if device == "cpu" else "float16"

        self.language = language
        # Базовый промпт для снижения галлюцинаций (RU). Можно переопределить через параметр.
        default_ru_prompt = (
            "Транскрибируй только реально произнесённую речь говорящего."
            " Не добавляй титры, имена редакторов и корректором, слова 'субтитры' тоже полностью исключи,"
            " служебные сообщения и лишние повторы. Пиши разговорную речь."
            " Не пиши только 'Продолжение следует.' или 'Редактор субтитров Т.Горелова'"
        )
        self.initial_prompt = initial_prompt if initial_prompt is not None else (
            default_ru_prompt if (self.language == "ru") else None
        )
        # Базовый список запрещённых ключевых слов/фраз (RU-кредиты и т.п.)
        default_banned = [
            "субтитр",  # покроет 'субтитры', 'субтитров'
            "редактор",
            "корректор",
            "продолжение следует",
            # наиболее часто встречавшиеся фамилии из логов
            "горелова",
            "новикова",
            "закомолдина",
            "подписывайтесь"
        ]
        self.banned_keywords = [kw.lower() for kw in (banned_keywords or default_banned)]
        self.min_text_len = max(0, int(min_text_len))

        # Для CUDA: заранее предзагрузим cuDNN/cuBLAS из установленных через pip пакетов nvidia-*
        # Это устраняет ошибку вида "Unable to load libcudnn_cnn.so.9 ..." на системах без глобального cuDNN
        if device == "cuda":
            try:
                _preload_nvidia_cuda_libs()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Не удалось предзагрузить NVIDIA .so библиотеки: {e}")

        def _init(device_: str, compute_type_: str):
            logger.info(
                f"Loading faster-whisper model={model_size} device={device_} compute_type={compute_type_} lang={language}"
            )
            return WhisperModel(model_size, device=device_, compute_type=compute_type_)

        # Первая попытка — как запросили
        try:
            self.model = _init(device, compute_type)
            self.device = device
            self.compute_type = compute_type
            return
        except Exception as e:  # noqa: BLE001
            # Если пробовали CUDA — делаем фоллбэк на CPU
            if device == "cuda":
                logger.warning(
                    f"ASR инициализация на CUDA не удалась ({e}). Переключаюсь на CPU/int8."
                )
                try:
                    self.model = _init("cpu", "int8")
                    self.device = "cpu"
                    self.compute_type = "int8"
                    return
                except Exception:
                    logger.exception("ASR инициализация на CPU также не удалась")
                    raise
            else:
                logger.exception("ASR инициализация не удалась")
                raise

    def transcribe_np(self, wav: np.ndarray, sample_rate: int = 16000) -> str:
        """Транскрибируем моно сигнал (float32 -1..1), 16 кГц.
        Возвращаем склеенный текст.
        """
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        audio = wav.astype(np.float32)

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            task="transcribe",
            vad_filter=False,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            temperature=0.0,
        )
        text_parts = []
        for seg in segments:
            # seg.text включает пробел в начале, удалим
            text_parts.append(seg.text.strip())
        text = " ".join([t for t in text_parts if t]).strip()
        return self._postprocess(text)

    def _postprocess(self, text: str) -> str:
        """Фильтрация выходного текста от галлюцинаций/кредитов.
        Возвращает пустую строку, если текст следует отбросить.
        """
        if not text:
            return ""
        t_low = text.lower()

        # Простая фильтрация по ключевым словам
        for kw in self.banned_keywords:
            if kw in t_low:
                return ""

        # Удалим повторяющиеся подряд слова типа "субтитры субтитры субтитры"
        tokens = re.findall(r"\w+[\w\-\.]*", text, flags=re.UNICODE)
        dedup_tokens: list[str] = []
        for tok in tokens:
            if not dedup_tokens or dedup_tokens[-1].lower() != tok.lower():
                dedup_tokens.append(tok)
        dedup_text = " ".join(dedup_tokens).strip()

        # Отсекаем слишком короткий результат (например, одна фамилия)
        if len(dedup_text) < self.min_text_len or len(dedup_text.split()) <= 1:
            return ""

        # Дополнительно: если после дедупликации всё ещё остались запрещённые слова — отбрасываем
        dlow = dedup_text.lower()
        if any(kw in dlow for kw in self.banned_keywords):
            return ""

        return dedup_text
