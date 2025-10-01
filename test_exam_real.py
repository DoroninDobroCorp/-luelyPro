#!/usr/bin/env python3
"""
НАСТОЯЩИЙ автоматический тест режима экзамена.
Использует simulate_dialogue() - имитирует голоса БЕЗ микрофона!
"""
import os
import sys
import time
from loguru import logger

# Настройка
os.environ["THESIS_REPEAT_SEC"] = "3"  # Быстрый повтор для теста
os.environ["FILE_LOG_LEVEL"] = "INFO"
os.environ["CONSOLE_LOG_LEVEL"] = "INFO"
os.environ["THESIS_MATCH_THRESHOLD"] = "0.3"  # Порог 30%

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

from live_recognizer import LiveVoiceVerifier

logger.info("=" * 80)
logger.info("АВТОМАТИЧЕСКИЙ ТЕСТ РЕЖИМА ЭКЗАМЕНА")
logger.info("=" * 80)
logger.info("")
logger.info("Тестируем полный цикл БЕЗ микрофона:")
logger.info("  1. Вопрос → LLM отвечает → Тезис создан")
logger.info("  2. Повтор тезиса (каждые 3 сек)")
logger.info("  3. Ответ пользователя → Тезис закрыт")
logger.info("  4. Новый вопрос → Новый тезис")
logger.info("")

# Счётчики
announced_count = 0
closed_count = 0
theses_created = []

# Создаём verifier
logger.info("Инициализация...")
verifier = LiveVoiceVerifier(
    asr_enable=False,  # Отключаем ASR - используем готовые транскрипты
    llm_enable=True,   # Включаем LLM для ответов
    theses_path=None,
    thesis_autogen_enable=True,
    thesis_match_threshold=0.3,
)

# Подменяем методы для отслеживания
original_speak = verifier._speak_text
original_announce = verifier._announce_thesis

def tracked_speak(text):
    logger.info(f"🔊 TTS: {text[:60]}...")
    # НЕ вызываем original_speak - без реального звука

def tracked_announce():
    global announced_count
    announced_count += 1
    if verifier.thesis_prompter:
        text = verifier.thesis_prompter.current_text()
        logger.info(f"📢 ОБЪЯВЛЕНИЕ #{announced_count}: {text[:60]}...")
        theses_created.append(text)
    # НЕ вызываем original_announce - без TTS

verifier._speak_text = tracked_speak
verifier._announce_thesis = tracked_announce

logger.info("Запуск фоновых потоков...")
verifier._start_segment_worker()

try:
    # ============== РАУНД 1 ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 1: Первый вопрос")
    logger.info("=" * 80)
    
    # Вопрос от экзаменатора
    question1 = "Когда отменили рабство в США?"
    logger.info(f"❓ ВОПРОС (чужой голос): {question1}")
    verifier.simulate_dialogue([("other", question1)])
    
    time.sleep(1)  # Ждём LLM
    
    # Проверяем что тезис создан
    if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Тезис НЕ создан после вопроса!")
        sys.exit(1)
    
    thesis1 = verifier.thesis_prompter.current_text()
    logger.success(f"✅ Тезис создан: {thesis1[:70]}...")
    
    # Ждём несколько повторов
    logger.info("⏳ Ждём 7 секунд (должно быть ≥2 повтора при интервале 3 сек)...")
    time.sleep(7)
    
    repeats = announced_count
    logger.info(f"Объявлений тезиса: {repeats}")
    
    if repeats < 2:
        logger.error(f"❌ ОШИБКА: Мало повторов! Ожидалось ≥2, получено {repeats}")
        sys.exit(1)
    
    logger.success(f"✅ Повтор работает ({repeats} раз)")
    
    # Отвечаем на тезис (используем первое предложение)
    answer1 = thesis1.split('.')[0] if '.' in thesis1 else thesis1[:30]
    logger.info("")
    logger.info(f"💬 МОЙ ОТВЕТ (свой голос): {answer1}")
    verifier.simulate_dialogue([("self", answer1)])
    
    time.sleep(0.5)
    
    # Проверяем что тезис закрыт
    if verifier.thesis_prompter and verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Тезис НЕ закрыт после ответа!")
        logger.error(f"   Ответ: '{answer1}'")
        logger.error(f"   Тезис: '{thesis1}'")
        # Проверим прогресс
        try:
            cov = verifier.thesis_prompter.coverage_of_current()
            logger.error(f"   Прогресс: {int(cov*100)}% (нужно ≥30%)")
        except:
            pass
        sys.exit(1)
    
    closed_count += 1
    logger.success("✅ Тезис #1 ЗАКРЫТ!")
    
    # ============== РАУНД 2 ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 2: Второй вопрос (проверка продолжения цикла)")
    logger.info("=" * 80)
    
    # Новый вопрос
    question2 = "Кто был президентом США во время отмены рабства?"
    logger.info(f"❓ ВОПРОС (чужой голос): {question2}")
    verifier.simulate_dialogue([("other", question2)])
    
    time.sleep(1)
    
    # Проверяем что создан НОВЫЙ тезис
    if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Новый тезис НЕ создан!")
        sys.exit(1)
    
    thesis2 = verifier.thesis_prompter.current_text()
    
    if thesis2 == thesis1:
        logger.error("❌ ОШИБКА: Тезис не изменился (тот же самый)!")
        sys.exit(1)
    
    logger.success(f"✅ Новый тезис создан: {thesis2[:70]}...")
    
    # Ждём повтор
    logger.info("⏳ Ждём 4 секунды (минимум 1 повтор)...")
    repeats_before = announced_count
    time.sleep(4)
    repeats2 = announced_count - repeats_before
    
    logger.info(f"Повторов нового тезиса: {repeats2}")
    
    if repeats2 < 1:
        logger.error("❌ ОШИБКА: Новый тезис НЕ повторяется!")
        sys.exit(1)
    
    logger.success(f"✅ Новый тезис повторяется ({repeats2} раз)")
    
    # Отвечаем на второй тезис
    answer2 = thesis2.split('.')[0] if '.' in thesis2 else thesis2[:30]
    logger.info("")
    logger.info(f"💬 МОЙ ОТВЕТ (свой голос): {answer2}")
    verifier.simulate_dialogue([("self", answer2)])
    
    time.sleep(0.5)
    
    # Проверяем что второй тезис закрыт
    if verifier.thesis_prompter and verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Второй тезис НЕ закрыт!")
        sys.exit(1)
    
    closed_count += 1
    logger.success("✅ Тезис #2 ЗАКРЫТ!")
    
    # ============== ИТОГИ ==============
    logger.info("")
    logger.info("=" * 80)
    logger.info("ИТОГИ ТЕСТА")
    logger.info("=" * 80)
    logger.info(f"Всего объявлений: {announced_count}")
    logger.info(f"Закрыто тезисов: {closed_count}")
    logger.info(f"Уникальных тезисов: {len(set(theses_created))}")
    
    logger.info("")
    logger.info("История тезисов:")
    for i, t in enumerate(set(theses_created), 1):
        logger.info(f"  {i}. {t[:70]}...")
    
    # Финальные проверки
    logger.info("")
    checks = [
        (announced_count >= 4, f"Объявлений достаточно (≥4): {announced_count}"),
        (closed_count >= 2, f"Закрыто тезисов (≥2): {closed_count}"),
        (len(set(theses_created)) >= 2, f"Разных тезисов (≥2): {len(set(theses_created))}"),
    ]
    
    all_ok = True
    for passed, msg in checks:
        if passed:
            logger.success(f"✅ {msg}")
        else:
            logger.error(f"❌ {msg}")
            all_ok = False
    
    if all_ok:
        logger.info("")
        logger.success("=" * 80)
        logger.success("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        logger.success("=" * 80)
        logger.success("")
        logger.success("Режим экзамена работает правильно:")
        logger.success("  ✅ Тезисы создаются из ответов LLM")
        logger.success("  ✅ Тезисы повторяются периодически")
        logger.success("  ✅ Тезисы закрываются ответами пользователя")
        logger.success("  ✅ Цикл продолжается с новыми вопросами")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("=" * 80)
        logger.error("❌ ТЕСТЫ НЕ ПРОЙДЕНЫ")
        logger.error("=" * 80)
        sys.exit(1)

finally:
    verifier._stop_segment_worker()
