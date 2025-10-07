from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger


@dataclass
class SileroConfig:
    language: str = "ru"                  # 'ru', 'en', 'de', 'es' (см. silero docs)
    model_id: str = "v4_ru"               # для русского языка
    speaker: str = "eugene"               # один из доступных голосов для ru
    sample_rate: int = 24000               # допустимые: 8000, 24000, 48000
    device: Optional[str] = None           # 'cuda' | 'cpu' (auto если None)


class SileroTTS:
    def __init__(
        self,
        language: str = "ru",
        model_id: str = "v4_ru",
        speaker: str = "eugene",
        sample_rate: int = 24000,
        speed: float = 1.5,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if sample_rate not in (8000, 24000, 48000):
            logger.warning(
                f"sample_rate {sample_rate} не поддерживается Silero, используем 24000"
            )
            sample_rate = 24000

        self.language = language
        self.model_id = model_id
        self.speaker = speaker
        self.sample_rate = sample_rate
        self.speed = speed
        self.device = torch.device(device)

        logger.info(
            f"Загрузка Silero TTS: lang={language}, model_id={model_id}, speaker={speaker}, sr={sample_rate}, device={self.device}"
        )
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language=self.language,
            speaker=self.model_id,
        )
        self.model.to(self.device)

    def synth(self, text: str) -> np.ndarray:
        """Синтез речи. Принимает обычный текст или SSML-строку.
        Возвращает float32 numpy-массив в диапазоне [-1, 1] с sample_rate=self.sample_rate.
        """
        if not text or not text.strip():
            return np.zeros((0,), dtype=np.float32)

        with torch.inference_mode():
            audio: torch.Tensor = self.model.apply_tts(
                text=text,
                speaker=self.speaker,
                sample_rate=self.sample_rate,
            )
        audio_np = audio.detach().cpu().numpy().astype(np.float32)
        return audio_np

    @staticmethod
    def save_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
        """Сохранение WAV без внешних зависимостей (wave + int16).
        """
        import wave

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        # нормируем и конвертируем в int16
        audio = np.clip(audio, -1.0, 1.0)
        int16 = (audio * 32767.0).astype(np.int16)

        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(int16.tobytes())


def tts_cli(
    text: Optional[str] = None,
    infile: Optional[Path] = None,
    outfile: Path = Path("silero_out.wav"),
    language: str = "ru",
    model_id: str = "v4_ru",
    speaker: str = "eugene",
    sample_rate: int = 24000,
    device: Optional[str] = None,
) -> None:
    """Озвучка текста. Если text не задан, а infile указан — читаем текст из файла.
    Если оба не заданы — завершаем с предупреждением.
    """
    if (text is None or not text.strip()) and infile is not None:
        if infile.exists():
            text = infile.read_text(encoding="utf-8")
    if text is None or not text.strip():
        logger.error("Не задан текст для озвучивания (ни --text, ни --file)")
        return

    tts = SileroTTS(
        language=language,
        model_id=model_id,
        speaker=speaker,
        sample_rate=sample_rate,
        device=device,
    )
    audio = tts.synth(text)
    if audio.size == 0:
        logger.warning("Пустой результат TTS")
        return

    tts.save_wav(outfile, audio, tts.sample_rate)
    logger.info(f"TTS: WAV сохранён в {outfile}")
