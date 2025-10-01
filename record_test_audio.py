#!/usr/bin/env python3
"""
Записываем тестовое аудио ОДИН РАЗ для автотестов.
Потом можно запускать автотесты без участия.
"""
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

# Папка для записей
AUDIO_DIR = Path("test_audio")
AUDIO_DIR.mkdir(exist_ok=True)

logger.info("=" * 80)
logger.info("ЗАПИСЬ ТЕСТОВОГО АУДИО")
logger.info("=" * 80)
logger.info("")
logger.info("Сейчас запишем 3 файла для автотестов:")
logger.info("  1. question.wav - вопрос от экзаменатора (можешь включить видео)")
logger.info("  2. answer.wav - твой ответ на тезис")
logger.info("  3. question2.wav - второй вопрос (для проверки что не глохнет)")
logger.info("")
logger.info("Эти файлы будут использоваться в автотестах!")
logger.info("")

# Импортируем микрофон
try:
    import sounddevice as sd
    
    # Проверяем микрофон
    devices = sd.query_devices()
    default_input = sd.query_devices(kind='input')
    logger.info(f"Используем микрофон: {default_input['name']}")
    
except Exception as e:
    logger.error(f"Ошибка инициализации микрофона: {e}")
    sys.exit(1)

SAMPLE_RATE = 16000

def record_audio(duration: int, filename: str) -> np.ndarray:
    """Записывает аудио с микрофона"""
    logger.info(f"🎙️  ЗАПИСЬ {duration} секунд...")
    logger.info("   Говорите СЕЙЧАС!")
    
    # Обратный отсчёт
    for i in range(duration, 0, -1):
        if i <= 3:
            logger.info(f"   ...{i}")
        time.sleep(1)
    
    # Записываем
    try:
        audio = sd.rec(
            int(duration * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Сохраняем
        path = AUDIO_DIR / filename
        sf.write(path, audio, SAMPLE_RATE)
        
        logger.success(f"✅ Сохранено: {path}")
        logger.info(f"   Длина: {len(audio)/SAMPLE_RATE:.1f}с")
        logger.info("")
        
        return audio
        
    except Exception as e:
        logger.error(f"❌ Ошибка записи: {e}")
        return None

# ========== ЗАПИСЬ 1: ВОПРОС ==========
logger.info("=" * 80)
logger.info("ЗАПИСЬ 1: ВОПРОС ОТ ЭКЗАМЕНАТОРА")
logger.info("=" * 80)
logger.info("")
logger.info("Включи видео с вопросом или сам задай вопрос другим голосом.")
logger.info("Например: 'Когда отменили рабство в США?'")
logger.info("")
input("Нажми Enter когда готов... ")

record_audio(5, "question.wav")

# ========== ЗАПИСЬ 2: ТВОЙ ОТВЕТ ==========
logger.info("=" * 80)
logger.info("ЗАПИСЬ 2: ТВОЙ ОТВЕТ")
logger.info("=" * 80)
logger.info("")
logger.info("Скажи короткий ответ СВОИМ голосом.")
logger.info("Например: 'В тысяча восемьсот шестьдесят пятом году'")
logger.info("Или: 'В 1865 году'")
logger.info("")
input("Нажми Enter когда готов... ")

record_audio(5, "answer.wav")

# ========== ЗАПИСЬ 3: ВТОРОЙ ВОПРОС ==========
logger.info("=" * 80)
logger.info("ЗАПИСЬ 3: ВТОРОЙ ВОПРОС")
logger.info("=" * 80)
logger.info("")
logger.info("Ещё один вопрос (для проверки что система не глохнет).")
logger.info("Например: 'Кто был президентом США во время отмены рабства?'")
logger.info("")
input("Нажми Enter когда готов... ")

record_audio(5, "question2.wav")

# ========== ИТОГИ ==========
logger.info("")
logger.info("=" * 80)
logger.info("✅ ВСЕ ЗАПИСИ ГОТОВЫ!")
logger.info("=" * 80)
logger.info("")
logger.info(f"Файлы сохранены в: {AUDIO_DIR}/")
logger.info("")

# Показываем что записали
for f in ["question.wav", "answer.wav", "question2.wav"]:
    path = AUDIO_DIR / f
    if path.exists():
        info = sf.info(path)
        logger.info(f"  ✅ {f}: {info.duration:.1f}с, {info.samplerate}Hz")
    else:
        logger.warning(f"  ⚠️  {f}: не найден")

logger.info("")
logger.success("Теперь можно запускать: .venv/bin/python test_with_real_audio.py")
logger.success("Автотест будет использовать эти записи!")
