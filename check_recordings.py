#!/usr/bin/env python3
"""Проверяем что записалось в тестовые файлы"""
import sys
import soundfile as sf
import numpy as np
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

from asr_transcriber import FasterWhisperTranscriber

AUDIO_DIR = Path("test_audio")

logger.info("=" * 80)
logger.info("ПРОВЕРКА ЗАПИСАННЫХ ФАЙЛОВ")
logger.info("=" * 80)
logger.info("")

# Загружаем ASR с русским языком
logger.info("Загружаем ASR (lang=ru)...")
asr = FasterWhisperTranscriber(model_size="large-v3-turbo", device="cpu", compute_type="int8", language="ru")
logger.info("")

# Проверяем каждый файл
for filename in ["question.wav", "answer.wav", "question2.wav"]:
    path = AUDIO_DIR / filename
    
    if not path.exists():
        logger.error(f"❌ {filename} не найден!")
        continue
    
    logger.info(f"📁 {filename}")
    logger.info("=" * 60)
    
    # Загружаем
    audio, sr = sf.read(path, dtype='float32')
    
    # Если стерео
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Статистика
    duration = len(audio) / sr
    max_amp = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio ** 2))
    
    logger.info(f"  Длина: {duration:.2f}с")
    logger.info(f"  Sample rate: {sr}Hz")
    logger.info(f"  Max амплитуда: {max_amp:.4f}")
    logger.info(f"  RMS (средняя громкость): {rms:.4f}")
    
    # Проверяем тишину
    if max_amp < 0.001:
        logger.error("  ❌ ТИШИНА! Ничего не записалось")
    elif max_amp < 0.01:
        logger.warning("  ⚠️  Очень тихо (может не распознаться)")
    else:
        logger.info("  ✅ Громкость OK")
    
    # Пробуем распознать
    logger.info("  🎤 Распознаём через ASR...")
    
    # Ресемпл если нужно
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
        audio = audio.astype(np.float32)
    
    text = asr.transcribe_np(audio)
    
    if text:
        logger.success(f"  ✅ ASR: '{text}'")
    else:
        logger.error("  ❌ ASR ничего не распознал (пустая строка)")
        logger.error("     Это значит:")
        logger.error("     - В файле только шум/тишина")
        logger.error("     - Или микрофон не работал при записи")
    
    logger.info("")

logger.info("=" * 80)
logger.info("ИТОГИ")
logger.info("=" * 80)
logger.info("")
logger.info("Если ASR ничего не распознал:")
logger.info("  1. Перезапусти запись: .venv/bin/python record_test_audio.py")
logger.info("  2. ГРОМКО говори в микрофон")
logger.info("  3. Проверь что микрофон не заглушен в системе")
logger.info("")
logger.info("Если ASR распознал текст:")
logger.info("  ✅ Записи OK! Тест должен работать")
