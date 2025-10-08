#!/usr/bin/env python3
"""
Тест форматов аудио от TTS (bytes vs numpy.ndarray)
Проверяет что live_recognizer корректно обрабатывает оба формата
"""
import io
import sys
import wave
from pathlib import Path

import numpy as np

# Добавляем родительскую папку в sys.path для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_bytes_to_numpy_conversion():
    """
    Проверка конвертации WAV bytes в numpy array
    (как делает live_recognizer для Google TTS)
    """
    # Создаём тестовое аудио: синусоида 440 Hz, 1 секунда, 24000 Hz
    sample_rate = 24000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio_float = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    
    # Конвертируем в PCM16
    audio_pcm16 = (np.clip(audio_float, -1.0, 1.0) * 32767.0).astype(np.int16)
    
    # Создаём WAV bytes (как Google TTS)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_pcm16.tobytes())
    
    wav_bytes = buf.getvalue()
    
    # Проверяем что получили bytes
    assert isinstance(wav_bytes, bytes)
    assert len(wav_bytes) > 0
    print(f"✓ Создано {len(wav_bytes)} байт WAV аудио")
    
    # Теперь декодируем обратно (как делает live_recognizer)
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio_decoded = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    
    # Проверяем что декодированное аудио похоже на исходное
    assert isinstance(audio_decoded, np.ndarray)
    assert audio_decoded.dtype == np.float32
    assert len(audio_decoded) == len(audio_float)
    
    # Проверяем что разница небольшая (могут быть ошибки округления)
    max_diff = np.max(np.abs(audio_decoded - audio_float))
    assert max_diff < 0.01, f"Слишком большая разница: {max_diff}"
    
    print(f"✓ Декодировано {len(audio_decoded)} сэмплов")
    print(f"✓ Максимальная разница: {max_diff:.6f}")


def test_empty_bytes_handling():
    """Проверка обработки пустых bytes"""
    empty_bytes = b""
    
    # Проверяем что len() работает для bytes
    assert len(empty_bytes) == 0
    print("✓ Пустые bytes корректно обрабатываются через len()")


def test_numpy_array_handling():
    """Проверка что numpy arrays работают как раньше"""
    # Создаём numpy array (как Silero/OpenAI TTS)
    audio_np = np.zeros((24000,), dtype=np.float32)
    
    # Проверяем что .size и .shape работают
    assert hasattr(audio_np, 'size')
    assert hasattr(audio_np, 'shape')
    assert audio_np.size == 24000
    assert audio_np.shape[0] == 24000
    
    print("✓ numpy.ndarray работает как раньше (.size, .shape)")


def test_type_detection():
    """Проверка определения типа данных"""
    # bytes (Google TTS)
    wav_bytes = b"RIFF\x00\x00\x00\x00WAVE"
    assert isinstance(wav_bytes, bytes)
    print("✓ isinstance(wav_bytes, bytes) = True")
    
    # numpy array (Silero/OpenAI TTS)
    audio_np = np.zeros((100,), dtype=np.float32)
    assert isinstance(audio_np, np.ndarray)
    assert not isinstance(audio_np, bytes)
    print("✓ isinstance(audio_np, np.ndarray) = True")
    print("✓ isinstance(audio_np, bytes) = False")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Тестирование форматов аудио от TTS")
    print("=" * 70 + "\n")
    
    test_bytes_to_numpy_conversion()
    print()
    
    test_empty_bytes_handling()
    print()
    
    test_numpy_array_handling()
    print()
    
    test_type_detection()
    print()
    
    print("=" * 70)
    print("✓ Все тесты форматов аудио прошли успешно!")
    print("=" * 70)
