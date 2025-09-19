from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
import torch
from loguru import logger
import webrtcvad

from pyannote.audio.pipelines.speaker_verification import (
    PretrainedSpeakerEmbedding,
)
from asr_transcriber import FasterWhisperTranscriber
from llm_answer import LLMResponder
from tts_silero import SileroTTS


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
        vad_aggressiveness: int = 2,
        threshold: float = 0.30,  # cosine distance threshold: <= is my voice
        min_consec_speech_frames: int = 5,  # >=5 * 20ms = 100ms подряд речи для старта сегмента
        flatness_reject_threshold: float = 0.60,  # > -> шум, сегмент отбрасываем
        # ASR настройки
        asr_enable: bool = False,
        asr_model_size: str = "small",
        asr_language: Optional[str] = None,
        asr_device: Optional[str] = None,
        asr_compute_type: Optional[str] = None,
        # LLM настройки
        llm_enable: bool = False,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.embedder = PretrainedSpeakerEmbedding(model_id, device=self.device)
        self.vad = webrtcvad.Vad(vad_aggressiveness)
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

        if self.asr_enable:
            self._asr = FasterWhisperTranscriber(
                model_size=self.asr_model_size,
                device=self.asr_device,
                compute_type=self.asr_compute_type,
                language=self.asr_language,
            )

        # Инициализируем LLM сразу при старте, если включен флаг
        if self.llm_enable:
            self._llm = LLMResponder()
            # Инициализируем TTS (RU по умолчанию)
            try:
                self._tts = SileroTTS(language="ru", model_id="v4_ru", speaker="eugene", sample_rate=24000)
            except Exception as e:  # noqa: BLE001
                logger.exception(f"Не удалось инициализировать TTS: {e}")

        logger.info(
            f"LiveVoiceVerifier initialized | model={model_id}, device={self.device}, threshold={threshold}"
        )

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        a = a / (np.linalg.norm(a) + 1e-8)
        b = b / (np.linalg.norm(b) + 1e-8)
        return float(1.0 - np.dot(a, b))

    def embedding_from_waveform(self, wav: np.ndarray) -> np.ndarray:
        """wav: mono float32 in [-1, 1], shape (n_samples,) at 16kHz"""
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        # Ensure torch tensor shape (batch=1, channels=1, samples)
        tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)  # (1, 1, n)
        with torch.inference_mode():
            out = self.embedder(tensor)  # expected shape (1, dim)
        # out can be numpy array or torch tensor depending on backend
        if isinstance(out, np.ndarray):
            return out[0]
        else:
            return out[0].detach().cpu().numpy()

    def _collect_enrollment_audio(
        self, seconds: float = 5.0, min_voiced_seconds: float = 2.0
    ) -> np.ndarray:
        """Собираем сигнал с микрофона, оставляя только озвученные (VAD) фреймы."""
        q: "queue.Queue[np.ndarray]" = queue.Queue()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                logger.warning(f"InputStream status: {status}")
            q.put(indata.copy())

        voiced_samples: List[np.ndarray] = []
        voiced_duration = 0.0
        target_end = time.time() + seconds

        with sd.InputStream(
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            dtype="float32",
            blocksize=FRAME_SIZE,  # deliver 20ms blocks
            callback=callback,
        ):
            logger.info("Говорите для записи профиля...")
            while time.time() < target_end and voiced_duration < min_voiced_seconds:
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
                    is_speech = self.vad.is_speech(float_to_pcm16(frame), SAMPLE_RATE)
                    if is_speech:
                        voiced_samples.append(frame)
                        voiced_duration += FRAME_MS / 1000.0

        if voiced_duration < min_voiced_seconds:
            logger.warning(
                f"Недостаточно речи для профиля: {voiced_duration:.2f}s < {min_voiced_seconds:.2f}s"
            )

        if len(voiced_samples) == 0:
            raise RuntimeError("Не удалось захватить голос для энроллмента")

        return np.concatenate(voiced_samples, axis=0)

    def enroll(self, path: Path, seconds: float = 5.0) -> VoiceProfile:
        wav = self._collect_enrollment_audio(seconds=seconds)
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
        voiced_ms = 0
        silence_ms = 0
        in_speech = False
        consec_speech = 0

        logger.info(
            "Старт лайв-распознавания. Нажмите Ctrl+C для остановки. Говорите в микрофон."
        )

        try:
            with sd.InputStream(
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype="float32",
                blocksize=FRAME_SIZE,  # deliver 20ms blocks
                callback=callback,
            ):
                while True:
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
                        i += FRAME_SIZE
                        is_speech = self.vad.is_speech(
                            float_to_pcm16(frame), SAMPLE_RATE
                        )

                        if is_speech:
                            consec_speech += 1
                            if not in_speech:
                                # ещё не стартовали сегмент: копим pre_frames
                                pre_frames.append(frame)
                                if consec_speech >= self.min_consec_speech_frames:
                                    # старт сегмента: переносим pre_frames в seg_audio
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

                            # считаем эмбеддинг и сравниваем
                            emb = self.embedding_from_waveform(wav)
                            dist = self.cosine_distance(emb, profile.embedding)
                            if dist <= self.threshold:
                                logger.info("мой голос")
                            else:
                                logger.debug("whisper call")
                                # Незнакомый голос: при включенном ASR — транскрибируем
                                if self.asr_enable:
                                    if self._asr is None:
                                        self._asr = FasterWhisperTranscriber(
                                            model_size=self.asr_model_size,
                                            device=self.asr_device,
                                            compute_type=self.asr_compute_type,
                                            language=self.asr_language,
                                        )
                                    try:
                                        text = self._asr.transcribe_np(wav, SAMPLE_RATE)
                                        if text:
                                            logger.info(f"незнакомый голос (ASR): {text}")
                                            # При включенном LLM — запрашиваем краткий ответ
                                            if self.llm_enable:
                                                try:
                                                    if self._llm is None:
                                                        self._llm = LLMResponder()
                                                    llm_answer = self._llm.generate(text)
                                                    if llm_answer:
                                                        logger.info(f"LLM: {llm_answer}")
                                                        # Озвучиваем ответ, если доступен TTS
                                                        if self._tts is not None:
                                                            try:
                                                                audio = self._tts.synth(llm_answer)
                                                                if audio.size > 0:
                                                                    duration = float(audio.shape[0]) / float(self._tts.sample_rate)
                                                                    self._suppress_until = time.time() + duration + 0.1
                                                                    sd.stop()
                                                                    sd.play(audio, samplerate=self._tts.sample_rate)
                                                            except Exception as e:  # noqa: BLE001
                                                                logger.exception(f"TTS ошибка: {e}")
                                                except Exception as e:  # noqa: BLE001
                                                    logger.exception(f"LLM ошибка: {e}")
                                        else:
                                            logger.info("незнакомый голос (ASR: пусто)")
                                    except Exception as e:  # noqa: BLE001
                                        logger.exception(f"ASR ошибка: {e}")
                                else:
                                    logger.info("незнакомый голос")

        except KeyboardInterrupt:
            logger.info("Остановлено пользователем")


def enroll_cli(
    profile_path: Path = Path("voice_profile.npz"),
    seconds: float = 5.0,
    vad_aggr: int = 2,
    min_consec: int = 5,
    flatness_th: float = 0.60,
    # ASR в энроллменте не используется, но оставим одинаковую сигнатуру для единообразия
    asr: bool = False,
    asr_model: str = "small",
    asr_lang: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute: Optional[str] = None,
) -> None:
    verifier = LiveVoiceVerifier(
        vad_aggressiveness=vad_aggr,
        min_consec_speech_frames=min_consec,
        flatness_reject_threshold=flatness_th,
        asr_enable=asr,
        asr_model_size=asr_model,
        asr_language=asr_lang,
        asr_device=asr_device,
        asr_compute_type=asr_compute,
    )
    verifier.enroll(profile_path, seconds=seconds)


def live_cli(
    profile_path: Path = Path("voice_profile.npz"),
    threshold: float = 0.30,
    vad_aggr: int = 2,
    min_consec: int = 5,
    flatness_th: float = 0.60,
    min_segment_ms: int = 500,
    max_silence_ms: int = 400,
    # ASR настройки
    asr: bool = False,
    asr_model: str = "small",
    asr_lang: Optional[str] = None,
    asr_device: Optional[str] = None,
    asr_compute: Optional[str] = None,
    # LLM
    llm: bool = False,
) -> None:
    verifier = LiveVoiceVerifier(
        threshold=threshold,
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
    )


__all__ = [
    "LiveVoiceVerifier",
    "VoiceProfile",
    "enroll_cli",
    "live_cli",
]
