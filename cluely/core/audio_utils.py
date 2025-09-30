"""Утилиты для работы с аудио."""
from __future__ import annotations

import numpy as np
from config import DEFAULT_SAMPLE_RATE


def float_to_pcm16(x: np.ndarray) -> bytes:
    """Конвертация float32 (-1..1) в 16-bit PCM bytes.
    
    Args:
        x: Массив аудио данных в формате float32
        
    Returns:
        PCM16 байты
    """
    x = np.clip(x, -1.0, 1.0)
    x_int16 = (x * 32767.0).astype(np.int16)
    return x_int16.tobytes()


def median_spectral_flatness(
    wav: np.ndarray, 
    sample_rate: int = DEFAULT_SAMPLE_RATE, 
    frame_len: int = 512, 
    hop: int = 256
) -> float:
    """Оценка спектральной плоскостности (0..1), где ближе к 1 — шумоподобный сигнал.
    
    Возвращает медиану по окнам. Используется для фильтрации шумов.
    
    Args:
        wav: Аудио массив (mono, float32)
        sample_rate: Частота дискретизации
        frame_len: Длина окна (сэмплы)
        hop: Шаг окна (сэмплы)
        
    Returns:
        Медианная спектральная плоскостность (0..1)
    """
    if wav.ndim != 1:
        wav = wav.reshape(-1)
    if wav.size < frame_len:
        return 1.0  # слишком коротко — считаем как шум
    
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
        
        # Ограничиваем полосу [80..4000] Гц для фокуса на речи
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


__all__ = ["float_to_pcm16", "median_spectral_flatness"]
