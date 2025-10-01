#!/usr/bin/env python3
"""
Автоматический тест повтора тезисов.
Запускается БЕЗ реального аудио, проверяет что тезисы повторяются.
"""
import os
import sys
import time
from pathlib import Path
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="DEBUG", format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level:8}</level> | {message}")

# Настройки для теста
os.environ["THESIS_REPEAT_SEC"] = "2"  # Быстрый повтор для теста
os.environ["FILE_LOG_LEVEL"] = "DEBUG"
os.environ["CONSOLE_LOG_LEVEL"] = "DEBUG"

from live_recognizer import LiveVoiceVerifier
from thesis_prompter import ThesisPrompter


class ThesisRepeatTester:
    """Тестер повтора тезисов"""
    
    def __init__(self):
        self.announce_count = 0
        self.announce_times = []
        self.announced_texts = []
        
    def test_repeat_without_audio(self):
        """Тест 1: Повтор тезисов без аудио (имитация)"""
        logger.info("=" * 70)
        logger.info("ТЕСТ 1: Повтор тезисов через фоновый поток")
        logger.info("=" * 70)
        
        # Создаём тестовые тезисы
        test_theses = [
            "Python создали в 1991 году",
            "JavaScript изобрели в 1995 году",
            "Java выпустили в 1995 году"
        ]
        
        # Создаём LiveVoiceVerifier с отключенными тяжёлыми зависимостями
        verifier = LiveVoiceVerifier(
            asr_enable=False,
            llm_enable=False,
            theses_path=None,  # Не грузим из файла
            thesis_autogen_enable=False,
        )
        
        # Вручную устанавливаем тезисный помощник
        verifier.thesis_prompter = ThesisPrompter(
            theses=test_theses,
            match_threshold=0.6,
            enable_semantic=False,
            enable_gemini=False,
        )
        
        # Подменяем метод _speak_text для отслеживания вызовов
        original_speak = verifier._speak_text
        def mock_speak(text):
            self.announce_count += 1
            self.announce_times.append(time.time())
            self.announced_texts.append(text)
            logger.info(f"🔊 ОБЪЯВЛЕНИЕ #{self.announce_count}: {text}")
        
        verifier._speak_text = mock_speak
        
        # Запускаем фоновые потоки
        logger.info("Запуск фоновых потоков...")
        verifier._start_segment_worker()
        
        # Ждём несколько секунд и проверяем количество повторов
        test_duration = 7  # секунд
        logger.info(f"Ожидаем {test_duration} секунд и считаем повторы...")
        logger.info(f"Интервал повтора: {verifier._thesis_repeat_sec} сек")
        
        # Первое объявление вручную
        verifier._announce_thesis()
        time.sleep(0.1)
        
        start_time = time.time()
        while time.time() - start_time < test_duration:
            time.sleep(0.5)
            elapsed = time.time() - start_time
            logger.debug(f"Прошло {elapsed:.1f}с, объявлений: {self.announce_count}")
        
        # Останавливаем потоки
        verifier._stop_segment_worker()
        
        # Анализ результатов
        logger.info("")
        logger.info("=" * 70)
        logger.info("РЕЗУЛЬТАТЫ ТЕСТА")
        logger.info("=" * 70)
        logger.info(f"Всего объявлений: {self.announce_count}")
        logger.info(f"Ожидалось: минимум 3 (первое + 2 повтора за 7 сек с интервалом 2 сек)")
        
        if len(self.announce_times) > 1:
            intervals = [self.announce_times[i] - self.announce_times[i-1] 
                        for i in range(1, len(self.announce_times))]
            logger.info(f"Интервалы между объявлениями: {[f'{x:.1f}с' for x in intervals]}")
        
        logger.info("")
        logger.info("Объявленные тексты:")
        for i, text in enumerate(self.announced_texts, 1):
            logger.info(f"  {i}. {text}")
        
        # Проверка успешности
        logger.info("")
        if self.announce_count >= 3:
            logger.success("✅ ТЕСТ ПРОЙДЕН: Тезисы повторяются!")
            return True
        else:
            logger.error(f"❌ ТЕСТ НЕ ПРОЙДЕН: Только {self.announce_count} объявлений вместо минимум 3")
            logger.error("Возможные причины:")
            logger.error("  1. Фоновый поток _thesis_repeater_loop не запустился")
            logger.error("  2. Условия в _thesis_repeater_loop не выполняются")
            logger.error("  3. _last_announce_ts не обновляется корректно")
            return False
    
    def test_repeat_with_dialogue(self):
        """Тест 2: Повтор в контексте диалога"""
        logger.info("")
        logger.info("=" * 70)
        logger.info("ТЕСТ 2: Повтор с имитацией диалога")
        logger.info("=" * 70)
        
        self.announce_count = 0
        self.announce_times = []
        self.announced_texts = []
        
        test_theses = ["Тестовый тезис для проверки повтора"]
        
        verifier = LiveVoiceVerifier(
            asr_enable=False,
            llm_enable=False,
            theses_path=None,
            thesis_autogen_enable=False,
        )
        
        verifier.thesis_prompter = ThesisPrompter(
            theses=test_theses,
            match_threshold=0.9,  # Высокий порог, чтобы не закрывался случайно
            enable_semantic=False,
            enable_gemini=False,
        )
        
        def mock_speak(text):
            self.announce_count += 1
            self.announce_times.append(time.time())
            self.announced_texts.append(text)
            logger.info(f"🔊 ОБЪЯВЛЕНИЕ #{self.announce_count}: {text}")
        
        verifier._speak_text = mock_speak
        
        # Запускаем фоновые потоки
        verifier._start_segment_worker()
        
        # Имитируем диалог
        logger.info("Имитируем вопрос интервьюера...")
        verifier.simulate_dialogue([
            ("other", "Расскажи про тестовый тезис")
        ])
        
        # Ждём повторы
        test_duration = 5
        logger.info(f"Ожидаем {test_duration} секунд...")
        time.sleep(test_duration)
        
        verifier._stop_segment_worker()
        
        logger.info("")
        logger.info(f"Результат: {self.announce_count} объявлений")
        
        if self.announce_count >= 2:
            logger.success("✅ ТЕСТ 2 ПРОЙДЕН: Повтор работает в диалоге!")
            return True
        else:
            logger.error(f"❌ ТЕСТ 2 НЕ ПРОЙДЕН: Только {self.announce_count} объявлений")
            return False


def main():
    """Запуск всех тестов"""
    logger.info("🚀 Запуск автоматических тестов повтора тезисов")
    logger.info("")
    
    tester = ThesisRepeatTester()
    
    results = []
    
    # Тест 1
    try:
        result1 = tester.test_repeat_without_audio()
        results.append(("Тест 1: Фоновый повтор", result1))
    except Exception as e:
        logger.exception(f"Ошибка в Тесте 1: {e}")
        results.append(("Тест 1: Фоновый повтор", False))
    
    # Тест 2
    try:
        result2 = tester.test_repeat_with_dialogue()
        results.append(("Тест 2: Повтор в диалоге", result2))
    except Exception as e:
        logger.exception(f"Ошибка в Тесте 2: {e}")
        results.append(("Тест 2: Повтор в диалоге", False))
    
    # Итоговый отчёт
    logger.info("")
    logger.info("=" * 70)
    logger.info("ИТОГОВЫЙ ОТЧЁТ")
    logger.info("=" * 70)
    
    for test_name, passed in results:
        status = "✅ ПРОЙДЕН" if passed else "❌ НЕ ПРОЙДЕН"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(r[1] for r in results)
    
    logger.info("")
    if all_passed:
        logger.success("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        sys.exit(0)
    else:
        logger.error("❌ ЕСТЬ ПРОВАЛЕННЫЕ ТЕСТЫ")
        sys.exit(1)


if __name__ == "__main__":
    main()
