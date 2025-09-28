from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from loguru import logger


@dataclass
class SileroVADConfig:
    sample_rate: int = 16000
    threshold: float = 0.2            # порог вероятности речи
    window_ms: int = 100              # длина скользящего окна для оценки (мс)
    device: Optional[str] = None      # 'cpu' только для стабильности


class SileroVAD:
    """
    Простой VAD на базе Silero, ориентированный на онлайн-обработку:
    - Принимает короткие фреймы (например, 20 мс, float32 [-1..1]).
    - Поддерживает скользящее окно (по умолчанию 100 мс) для устойчивой оценки.
    - Возвращает булево is_speech по порогу вероятности.

    Использует torch.hub: snakers4/silero-vad.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        window_ms: int = 100,
        device: Optional[str] = None,
    ) -> None:
        # Silero VAD (TorchScript) наиболее стабилен на CPU, особенно в utils.
        # Принудительно используем CPU, даже если доступна CUDA.
        if device not in (None, "cpu"):
            logger.warning("SileroVAD: принудительно использую CPU (игнорирую device={})", device)
        device = "cpu"
        self.device = torch.device("cpu")

        self.cfg = SileroVADConfig(
            sample_rate=sample_rate,
            threshold=float(threshold),
            window_ms=int(window_ms),
            device=device,
        )

        # Загружаем модель Silero VAD
        logger.info(
            f"Загрузка Silero VAD: sr={sample_rate}, threshold={threshold}, window={window_ms}ms, device={self.device}"
        )
        # Пытаемся загрузить из pip-пакета (без интернета), иначе fallback на torch.hub
        _utils = None
        try:
            from silero_vad import load_silero_vad  # type: ignore

            self.model = load_silero_vad()
            # logger.info("Silero VAD: загружен через pip-пакет silero-vad")
        except Exception:  # noqa: BLE001
            logger.info("Silero VAD: fallback на torch.hub (требуется интернет при первой загрузке)")
            self.model, _utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
        # Модель оставляем на CPU для избежания конфликтов устройств внутри utils
        self.model.to(self.device)
        self.model.eval()
        # VADIterator и прочие утилиты не используем для простоты и стабильности

        # Скользящее окно аудио (float32 [-1..1])
        self.window_size = int(self.cfg.sample_rate * self.cfg.window_ms / 1000)
        self._buf = np.zeros((0,), dtype=np.float32)
        # Минимальная длина окна для стабильного инференса TorchScript VAD
        self._min_len = 512

    def reset(self) -> None:
        self._buf = np.zeros((0,), dtype=np.float32)

    def is_speech(self, frame: np.ndarray) -> bool:
        """
        frame: mono float32 [-1..1], shape (n_samples,) с частотой cfg.sample_rate
        Возвращает True, если в скользящем окне вероятность речи > threshold.
        """
        if frame.ndim != 1:
            frame = frame.reshape(-1)
        if frame.dtype != np.float32:
            frame = frame.astype(np.float32)

        # Обновляем буфер окна
        if self._buf.size == 0:
            self._buf = frame
        else:
            self._buf = np.concatenate([self._buf, frame], axis=0)
        if self._buf.size > self.window_size:
            self._buf = self._buf[-self.window_size :]

        # Используем РОВНО 512 семплов при 16 кГц, как требует TorchScript модель Silero VAD
        if self._buf.size < 512:
            return False
        chunk_np = self._buf[-512:]
        audio = torch.from_numpy(chunk_np).float().to(self.device)

        prob: float = 0.0
        try:
            with torch.inference_mode():
                out = self.model(audio.cpu(), self.cfg.sample_rate)
                try:
                    prob = float(out.item())
                except Exception:
                    prob = float(out)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Silero VAD: ошибка инференса: {e}")
            prob = 0.0

        return prob >= self.cfg.threshold


__all__ = ["SileroVAD", "SileroVADConfig"]
