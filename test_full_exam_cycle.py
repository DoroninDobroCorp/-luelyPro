#!/usr/bin/env python3
"""
Комплексный тест полного цикла экзамена.
Проверяет:
1. Вопрос → LLM отвечает → Тезис создаётся
2. Повтор тезиса работает
3. Ответ пользователя → Тезис закрывается
4. Новый вопрос → Новый тезис создаётся
5. Цикл продолжается
"""
import os
import sys
import time
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:8}</level> | {message}")

# Быстрый повтор для теста
os.environ["THESIS_REPEAT_SEC"] = "3"
os.environ["FILE_LOG_LEVEL"] = "INFO"
os.environ["CONSOLE_LOG_LEVEL"] = "INFO"

from live_recognizer import LiveVoiceVerifier


def test_full_exam_cycle():
    """Полный цикл экзамена без реального аудио"""
    
    logger.info("=" * 80)
    logger.info("КОМПЛЕКСНЫЙ ТЕСТ: Полный цикл экзамена")
    logger.info("=" * 80)
    
    # Счётчики для проверки
    announce_count = 0
    closed_count = 0
    theses_history = []
    
    def mock_speak(text):
        nonlocal announce_count
        announce_count += 1
        logger.info(f"🔊 ОБЪЯВЛЕНИЕ #{announce_count}: {text[:50]}...")
    
    # Создаём verifier с LLM но БЕЗ ASR (используем simulate_dialogue)
    logger.info("Инициализация системы...")
    verifier = LiveVoiceVerifier(
        asr_enable=False,  # Отключаем ASR - используем имитацию
        llm_enable=True,   # Включаем LLM для ответов
        theses_path=None,
        thesis_autogen_enable=True,
        thesis_match_threshold=0.3,  # Снижаем порог до 30%
    )
    
    # Подменяем _speak_text
    verifier._speak_text = mock_speak
    
    # Запускаем фоновые потоки
    logger.info("Запуск фоновых потоков...")
    verifier._start_segment_worker()
    
    # Сохраняем исходный _handle_self_transcript для подсчёта закрытий
    original_handle = verifier._handle_self_transcript
    
    def tracked_handle(transcript):
        nonlocal closed_count
        result = original_handle(transcript)
        # Проверяем был ли тезис закрыт
        if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
            closed_count += 1
            logger.success(f"✅ Тезис #{closed_count} закрыт!")
        return result
    
    verifier._handle_self_transcript = tracked_handle
    
    try:
        # ================== РАУНД 1 ==================
        logger.info("")
        logger.info("=" * 80)
        logger.info("РАУНД 1: Первый вопрос")
        logger.info("=" * 80)
        
        # 1. Вопрос экзаменатора
        question1 = "Когда отменили рабство в США?"
        logger.info(f"❓ Вопрос: {question1}")
        verifier.simulate_dialogue([("other", question1)])
        
        # Ждём LLM ответ и создание тезиса
        time.sleep(1)
        
        # Проверяем что тезис создан
        if verifier.thesis_prompter is None:
            logger.error("❌ ОШИБКА: Тезис НЕ создан после вопроса!")
            return False
        
        thesis1 = verifier.thesis_prompter.current_text()
        logger.success(f"✅ Тезис создан: {thesis1[:50]}...")
        theses_history.append(thesis1)
        announce_count_before = announce_count
        
        # 2. Ждём несколько повторов (6 сек = 2 повтора при интервале 3 сек)
        logger.info("")
        logger.info("⏳ Ожидаем 6 секунд (должно быть минимум 2 повтора)...")
        time.sleep(6)
        
        repeats = announce_count - announce_count_before
        logger.info(f"Повторов: {repeats}")
        
        if repeats < 2:
            logger.error(f"❌ ОШИБКА: Мало повторов! Ожидалось ≥2, получено {repeats}")
            return False
        
        logger.success(f"✅ Повтор работает ({repeats} раз)")
        
        # 3. Отвечаем на тезис
        logger.info("")
        logger.info("💬 Отвечаем на тезис...")
        # Используем начало тезиса как ответ (чтобы точно совпало)
        answer1 = thesis1.split('.')[0]  # Первое предложение
        logger.info(f"📝 МОЙ ОТВЕТ: {answer1}")
        verifier.simulate_dialogue([("self", answer1)])
        
        # Ждём обработку
        time.sleep(0.5)
        
        # Проверяем что тезис закрыт
        if closed_count < 1:
            logger.error("❌ ОШИБКА: Тезис НЕ закрыт после ответа!")
            logger.error(f"   Ответ: '{answer1}'")
            logger.error(f"   Тезис: '{thesis1}'")
            return False
        
        logger.success("✅ Тезис закрыт после ответа!")
        
        # ================== РАУНД 2 ==================
        logger.info("")
        logger.info("=" * 80)
        logger.info("РАУНД 2: Второй вопрос (проверка продолжения цикла)")
        logger.info("=" * 80)
        
        # 4. Новый вопрос экзаменатора
        question2 = "Кто был президентом США во время отмены рабства?"
        logger.info(f"❓ Вопрос: {question2}")
        verifier.simulate_dialogue([("other", question2)])
        
        # Ждём LLM ответ и создание нового тезиса
        time.sleep(1)
        
        # Проверяем что создан НОВЫЙ тезис
        if verifier.thesis_prompter is None:
            logger.error("❌ ОШИБКА: Новый тезис НЕ создан!")
            return False
        
        thesis2 = verifier.thesis_prompter.current_text()
        
        if thesis2 == thesis1:
            logger.error("❌ ОШИБКА: Тезис НЕ изменился (тот же самый)!")
            return False
        
        logger.success(f"✅ Новый тезис создан: {thesis2[:50]}...")
        theses_history.append(thesis2)
        
        # 5. Ждём повтор нового тезиса
        logger.info("")
        logger.info("⏳ Ожидаем 4 секунды (должен быть минимум 1 повтор)...")
        announce_count_before = announce_count
        time.sleep(4)
        
        repeats2 = announce_count - announce_count_before
        logger.info(f"Повторов нового тезиса: {repeats2}")
        
        if repeats2 < 1:
            logger.error(f"❌ ОШИБКА: Новый тезис НЕ повторяется!")
            return False
        
        logger.success(f"✅ Новый тезис повторяется ({repeats2} раз)")
        
        # 6. Отвечаем на второй тезис
        logger.info("")
        logger.info("💬 Отвечаем на второй тезис...")
        # Используем начало тезиса как ответ
        answer2 = thesis2.split('.')[0]
        logger.info(f"📝 МОЙ ОТВЕТ: {answer2}")
        verifier.simulate_dialogue([("self", answer2)])
        
        # Ждём обработку
        time.sleep(0.5)
        
        # Проверяем что второй тезис закрыт
        if closed_count < 2:
            logger.error("❌ ОШИБКА: Второй тезис НЕ закрыт!")
            return False
        
        logger.success("✅ Второй тезис закрыт!")
        
        # ================== ИТОГИ ==================
        logger.info("")
        logger.info("=" * 80)
        logger.info("ИТОГИ ТЕСТА")
        logger.info("=" * 80)
        logger.info(f"Всего объявлений: {announce_count}")
        logger.info(f"Закрыто тезисов: {closed_count}")
        logger.info(f"Уникальных тезисов: {len(set(theses_history))}")
        
        logger.info("")
        logger.info("История тезисов:")
        for i, t in enumerate(theses_history, 1):
            logger.info(f"  {i}. {t[:60]}...")
        
        # Проверки
        logger.info("")
        checks = [
            (announce_count >= 5, f"Объявлений достаточно (≥5): {announce_count}"),
            (closed_count >= 2, f"Закрыто тезисов (≥2): {closed_count}"),
            (len(set(theses_history)) >= 2, f"Разных тезисов (≥2): {len(set(theses_history))}"),
        ]
        
        all_passed = True
        for passed, msg in checks:
            if passed:
                logger.success(f"✅ {msg}")
            else:
                logger.error(f"❌ {msg}")
                all_passed = False
        
        if all_passed:
            logger.info("")
            logger.success("=" * 80)
            logger.success("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
            logger.success("=" * 80)
            return True
        else:
            logger.error("")
            logger.error("=" * 80)
            logger.error("❌ ТЕСТЫ НЕ ПРОЙДЕНЫ")
            logger.error("=" * 80)
            return False
            
    finally:
        # Останавливаем потоки
        verifier._stop_segment_worker()


def main():
    logger.info("🚀 Запуск комплексного теста полного цикла экзамена")
    logger.info("")
    
    try:
        result = test_full_exam_cycle()
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.exception(f"❌ Критическая ошибка в тесте: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
