#!/usr/bin/env python3
"""
Тест автогенерации тезисов с проверкой повтора.
Имитирует вопрос → проверяет что тезисы создаются → проверяет повтор.
"""
import os
import sys
import time
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:8}</level> | {message}")

# Быстрый повтор для теста
os.environ["THESIS_REPEAT_SEC"] = "2"
os.environ["FILE_LOG_LEVEL"] = "DEBUG"
os.environ["CONSOLE_LOG_LEVEL"] = "DEBUG"

from live_recognizer import LiveVoiceVerifier


def test_autogen_with_repeat():
    """Тест: вопрос → автогенерация тезисов → повтор"""
    logger.info("=" * 70)
    logger.info("ТЕСТ: Автогенерация тезисов + повтор")
    logger.info("=" * 70)
    
    announce_count = 0
    announce_times = []
    announced_texts = []
    
    def mock_speak(text):
        nonlocal announce_count
        announce_count += 1
        announce_times.append(time.time())
        announced_texts.append(text)
        logger.info(f"🔊 ОБЪЯВЛЕНИЕ #{announce_count}: {text}")
    
    # Создаём verifier БЕЗ LLM чтобы не было конфликта с генерацией тезисов
    verifier = LiveVoiceVerifier(
        asr_enable=False,
        llm_enable=False,  # ❌ Отключаем LLM чтобы дойти до автогенерации
        theses_path=None,  # Не грузим статические
        thesis_autogen_enable=True,  # ✅ Включаем автогенерацию
    )
    
    # Подменяем _speak_text
    verifier._speak_text = mock_speak
    
    # Запускаем фоновые потоки
    logger.info("Запуск фоновых потоков...")
    verifier._start_segment_worker()
    
    # Имитируем вопрос
    logger.info("")
    logger.info("📝 Имитируем вопрос: 'Когда создали Python?'")
    question = "Когда создали Python?"
    
    # Добавляем в контекст и запускаем генерацию
    verifier._handle_foreign_text(question)
    
    # Небольшая пауза для генерации тезисов через Gemini
    logger.info("⏳ Ждём генерацию тезисов (1 сек)...")
    time.sleep(1)
    
    # Проверяем что тезисы созданы
    if verifier.thesis_prompter is None:
        logger.error("❌ Тезисы НЕ СОЗДАНЫ!")
        logger.error("Возможные причины:")
        logger.error("  1. AI_ONLY_THESIS=0 и код не дошел до _maybe_generate_theses")
        logger.error("  2. _question_context пустой")
        logger.error("  3. Ошибка в thesis_generator.generate()")
        verifier._stop_segment_worker()
        return False
    
    theses_count = len(verifier.thesis_prompter.theses)
    logger.success(f"✅ Тезисы созданы: {theses_count} шт.")
    for i, t in enumerate(verifier.thesis_prompter.theses, 1):
        logger.info(f"  {i}. {t}")
    
    # Ждём повторы
    test_duration = 6
    logger.info("")
    logger.info(f"⏳ Ожидаем {test_duration} секунд и считаем повторы...")
    
    start_time = time.time()
    while time.time() - start_time < test_duration:
        time.sleep(0.5)
    
    # Останавливаем потоки
    verifier._stop_segment_worker()
    
    # Анализ
    logger.info("")
    logger.info("=" * 70)
    logger.info("РЕЗУЛЬТАТЫ")
    logger.info("=" * 70)
    logger.info(f"Тезисов создано: {theses_count}")
    logger.info(f"Всего объявлений: {announce_count}")
    logger.info(f"Ожидалось: минимум 3 (первое + 2 повтора за 6 сек с интервалом 2 сек)")
    
    if len(announce_times) > 1:
        intervals = [announce_times[i] - announce_times[i-1] 
                    for i in range(1, len(announce_times))]
        logger.info(f"Интервалы: {[f'{x:.1f}с' for x in intervals]}")
    
    logger.info("")
    logger.info("Объявленные тезисы:")
    for i, text in enumerate(announced_texts, 1):
        logger.info(f"  {i}. {text}")
    
    # Проверка
    logger.info("")
    if theses_count > 0 and announce_count >= 3:
        logger.success("✅ ТЕСТ ПРОЙДЕН!")
        logger.success("  ✓ Тезисы автоматически сгенерированы")
        logger.success("  ✓ Повтор работает")
        return True
    else:
        logger.error(f"❌ ТЕСТ НЕ ПРОЙДЕН!")
        if theses_count == 0:
            logger.error("  ✗ Тезисы НЕ созданы")
        if announce_count < 3:
            logger.error(f"  ✗ Повтор НЕ работает ({announce_count} объявлений)")
        return False


def main():
    logger.info("🚀 Проверка автогенерации тезисов с повтором")
    logger.info("")
    
    try:
        result = test_autogen_with_repeat()
        sys.exit(0 if result else 1)
    except Exception as e:
        logger.exception(f"❌ Ошибка в тесте: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
