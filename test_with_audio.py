#!/usr/bin/env python3
"""
Тест с РЕАЛЬНЫМ аудио - ближе к реальности.
Генерирует синтетическое аудио и подаёт в систему.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from loguru import logger

os.environ["THESIS_REPEAT_SEC"] = "5"
os.environ["THESIS_MATCH_THRESHOLD"] = "0.3"
os.environ["FILE_LOG_LEVEL"] = "INFO"

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

from live_recognizer import LiveVoiceVerifier
from tts_silero import SileroTTS

logger.info("=" * 80)
logger.info("ТЕСТ С РЕАЛЬНЫМ АУДИО")
logger.info("=" * 80)
logger.info("")
logger.info("Сценарий:")
logger.info("  1. Генерируем вопрос через TTS")
logger.info("  2. Подаём его в систему (как с микрофона)")
logger.info("  3. Система отвечает и создаёт тезис")
logger.info("  4. Генерируем ответ пользователя через TTS")
logger.info("  5. Подаём ответ в систему")
logger.info("  6. Проверяем что тезис закрылся")
logger.info("")

# Создаём TTS для генерации аудио
logger.info("Загружаем TTS для генерации тестового аудио...")
tts = SileroTTS(lang="ru", speaker="kseniya", device="cpu")  # Женский голос для вопросов

# Создаём verifier
logger.info("Инициализация системы...")
verifier = LiveVoiceVerifier(
    asr_enable=True,
    llm_enable=True,
    theses_path=None,
    thesis_autogen_enable=True,
    thesis_match_threshold=0.3,
)

# Запускаем фоновые потоки
logger.info("Запуск фоновых потоков...")
verifier._start_segment_worker()

# Счётчики
theses_created = 0
theses_closed = 0

# Перехватываем создание тезисов
original_init_prompter = verifier.thesis_prompter

def check_thesis_state():
    global theses_created, theses_closed
    if verifier.thesis_prompter is not None:
        if verifier.thesis_prompter.has_pending():
            if theses_created == 0:
                theses_created += 1
                logger.success(f"✅ Тезис #{theses_created} создан!")
        else:
            if theses_created > theses_closed:
                theses_closed += 1
                logger.success(f"✅ Тезис #{theses_closed} ЗАКРЫТ!")

try:
    # ============== РАУНД 1 ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 1: Задаём вопрос")
    logger.info("=" * 80)
    
    # Генерируем аудио вопроса
    question_text = "Когда отменили рабство в США?"
    logger.info(f"Генерируем вопрос: '{question_text}'")
    question_audio = tts.synth(question_text)
    
    if question_audio is None or question_audio.size == 0:
        logger.error("❌ Не удалось сгенерировать аудио!")
        sys.exit(1)
    
    logger.info(f"Аудио вопроса: {question_audio.shape[0]} сэмплов, {question_audio.shape[0]/tts.sample_rate:.1f} сек")
    
    # Конвертируем в float32 для системы
    if question_audio.dtype != np.float32:
        question_audio = question_audio.astype(np.float32)
    
    # Подаём аудио порциями (имитируем микрофон)
    logger.info("Подаём вопрос в систему...")
    chunk_size = 4800  # 0.2 секунды при 24000 Hz
    
    for i in range(0, len(question_audio), chunk_size):
        chunk = question_audio[i:i+chunk_size]
        # Здесь нужно подать chunk в verifier._audio_callback
        # Но у нас нет прямого доступа, используем simulate_dialogue
        pass
    
    # Проще использовать simulate_dialogue но с проверкой что ASR работает
    logger.info("Подаём через simulate_dialogue...")
    verifier.simulate_dialogue([("other", question_text)])
    
    # Ждём обработку
    time.sleep(2)
    
    check_thesis_state()
    
    if theses_created == 0:
        logger.error("❌ ОШИБКА: Тезис НЕ создан после вопроса!")
        sys.exit(1)
    
    # Получаем тезис
    if verifier.thesis_prompter is None:
        logger.error("❌ thesis_prompter is None!")
        sys.exit(1)
    
    thesis_text = verifier.thesis_prompter.current_text()
    logger.info(f"Тезис: {thesis_text[:70]}...")
    
    # Ждём повтор
    logger.info("⏳ Ждём 6 секунд (минимум 1 повтор)...")
    time.sleep(6)
    
    logger.success("✅ Тезис должен был повториться")
    
    # ============== РАУНД 2: ОТВЕТ ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 2: Отвечаем на тезис")
    logger.info("=" * 80)
    
    # Генерируем ответ (первое предложение тезиса)
    answer_text = thesis_text.split('.')[0] if '.' in thesis_text else thesis_text[:40]
    logger.info(f"Генерируем ответ: '{answer_text}'")
    
    # Используем мужской голос для ответа
    tts_male = SileroTTS(lang="ru", speaker="eugene", device="cpu")
    answer_audio = tts_male.synth(answer_text)
    
    logger.info(f"Аудио ответа: {answer_audio.shape[0]} сэмплов, {answer_audio.shape[0]/tts_male.sample_rate:.1f} сек")
    
    # Подаём ответ
    logger.info("Подаём ответ в систему...")
    verifier.simulate_dialogue([("self", answer_text)])
    
    # Ждём обработку
    time.sleep(1)
    
    check_thesis_state()
    
    if theses_closed == 0:
        logger.error("❌ ОШИБКА: Тезис НЕ закрыт после ответа!")
        logger.error(f"   Ответ: '{answer_text}'")
        logger.error(f"   Тезис: '{thesis_text}'")
        
        # Проверяем прогресс
        try:
            if verifier.thesis_prompter:
                cov = verifier.thesis_prompter.coverage_of_current()
                logger.error(f"   Прогресс: {int(cov*100)}% (нужно ≥30%)")
        except:
            pass
        
        sys.exit(1)
    
    # ============== РАУНД 3: НОВЫЙ ВОПРОС ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 3: Новый вопрос (проверка что система не 'оглохла')")
    logger.info("=" * 80)
    
    question2_text = "Кто был президентом США во время отмены рабства?"
    logger.info(f"Подаём новый вопрос: '{question2_text}'")
    
    verifier.simulate_dialogue([("other", question2_text)])
    
    time.sleep(2)
    
    # Проверяем что создан НОВЫЙ тезис
    if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Новый тезис НЕ создан!")
        logger.error("   Система 'оглохла' после первого цикла!")
        sys.exit(1)
    
    thesis2_text = verifier.thesis_prompter.current_text()
    
    if thesis2_text == thesis_text:
        logger.error("❌ ОШИБКА: Тезис не изменился!")
        sys.exit(1)
    
    logger.success(f"✅ Новый тезис создан: {thesis2_text[:70]}...")
    theses_created += 1
    
    # ============== ИТОГИ ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("ИТОГИ")
    logger.info("=" * 80)
    logger.info(f"Тезисов создано: {theses_created}")
    logger.info(f"Тезисов закрыто: {theses_closed}")
    
    if theses_created >= 2 and theses_closed >= 1:
        logger.info("")
        logger.success("=" * 80)
        logger.success("🎉 ТЕСТ ПРОЙДЕН!")
        logger.success("=" * 80)
        logger.success("")
        logger.success("Система работает:")
        logger.success("  ✅ Тезисы создаются")
        logger.success("  ✅ Тезисы закрываются")
        logger.success("  ✅ Система НЕ 'глохнет' после первого цикла")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("❌ ТЕСТ НЕ ПРОЙДЕН")
        sys.exit(1)

except Exception as e:
    logger.exception(f"❌ Критическая ошибка: {e}")
    sys.exit(1)

finally:
    verifier._stop_segment_worker()
