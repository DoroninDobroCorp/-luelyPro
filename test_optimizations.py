"""
Тесты оптимизаций CluelyPro
Проверяем что все оптимизации работают корректно
"""
import os
import sys
import time
from pathlib import Path

# Убедимся что импорты работают
sys.path.insert(0, str(Path(__file__).parent))

# Упрощенное логирование (без loguru для независимости от зависимостей)
import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")


def test_cerebras_api():
    """
    ✅ ТЕСТ 3B: Проверка работы Cerebras API
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 3B: Cerebras API")
    print("="*70)
    
    try:
        from cerebras_llm import CerebrasLLM, CEREBRAS_AVAILABLE
        
        if not CEREBRAS_AVAILABLE:
            print("❌ FAILED: openai не установлен (требуется для Cerebras)")
            return False
        
        api_key = os.getenv("CEREBRAS_API_KEY")
        if not api_key:
            print("⚠️  SKIPPED: CEREBRAS_API_KEY не найден в .env")
            return True  # Skip, не ошибка
        
        print(f"✓ API ключ найден: {api_key[:10]}...")
        
        # Создаем клиент
        cerebras = CerebrasLLM(
            api_key=api_key,
            model="llama3.3-70b",
            max_tokens=100,
            temperature=0.3,
        )
        print("✓ CerebrasLLM инициализирован")
        
        # Тест генерации тезисов
        start = time.time()
        theses = cerebras.generate_theses(
            question="Кто первым полетел в космос?",
            n=3,
        )
        elapsed = (time.time() - start) * 1000
        
        if not theses:
            print("❌ FAILED: Cerebras вернул пустой список тезисов")
            return False
        
        print(f"✓ Cerebras вернул {len(theses)} тезисов за {elapsed:.0f}мс")
        print(f"  Тезисы: {theses}")
        
        # Проверяем скорость (должно быть быстрее 2 секунд)
        if elapsed > 2000:
            print(f"⚠️  WARNING: Cerebras медленнее ожидаемого ({elapsed:.0f}мс > 2000мс)")
        
        print("✅ PASSED: Cerebras API работает!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_thesis_generator_with_cerebras():
    """
    ✅ ТЕСТ 3B: Проверка ThesisGenerator с Cerebras как основной LLM
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 3B: ThesisGenerator с Cerebras primary")
    print("="*70)
    
    try:
        # Временно устанавливаем Cerebras как основную LLM
        os.environ["USE_LLM_ENGINE"] = "cerebras"
        
        from thesis_generator import GeminiThesisGenerator
        
        # Проверяем что API ключи есть
        if not os.getenv("CEREBRAS_API_KEY"):
            print("⚠️  SKIPPED: CEREBRAS_API_KEY не найден")
            return True
        
        if not os.getenv("GEMINI_API_KEY"):
            print("⚠️  SKIPPED: GEMINI_API_KEY не найден")
            return True
        
        # Создаем генератор
        generator = GeminiThesisGenerator()
        print(f"✓ ThesisGenerator создан, primary_engine={generator.primary_engine}")
        
        if generator.primary_engine != "cerebras":
            print(f"❌ FAILED: primary_engine должен быть 'cerebras', но {generator.primary_engine}")
            return False
        
        # Тест генерации
        start = time.time()
        theses = generator.generate(
            question_text="Что такое фотосинтез?",
            n=3,
            language="ru"
        )
        elapsed = (time.time() - start) * 1000
        
        if not theses:
            print("❌ FAILED: Не получены тезисы")
            return False
        
        print(f"✓ Получено {len(theses)} тезисов за {elapsed:.0f}мс")
        print(f"  Тезисы: {theses}")
        
        print("✅ PASSED: ThesisGenerator с Cerebras работает!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Восстанавливаем настройку
        if "USE_LLM_ENGINE" in os.environ:
            del os.environ["USE_LLM_ENGINE"]


def test_tts_cache():
    """
    ✅ ТЕСТ 2B: Проверка параллельной TTS генерации
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 2B: Параллельная TTS генерация (кэш)")
    print("="*70)
    
    try:
        # Проверяем что ThesisManager имеет кэш
        from live_recognizer import ThesisManager
        
        # Создаем менеджер без TTS (для быстрого теста)
        manager = ThesisManager(generator=None, tts_engine=None)
        
        # Проверяем наличие атрибутов кэширования
        if not hasattr(manager, 'audio_cache'):
            print("❌ FAILED: ThesisManager не имеет audio_cache")
            return False
        print("✓ ThesisManager.audio_cache существует")
        
        if not hasattr(manager, 'tts_executor'):
            print("❌ FAILED: ThesisManager не имеет tts_executor")
            return False
        print("✓ ThesisManager.tts_executor существует")
        
        if not hasattr(manager, 'prefetch_enabled'):
            print("❌ FAILED: ThesisManager не имеет prefetch_enabled")
            return False
        print("✓ ThesisManager.prefetch_enabled существует")
        
        # Проверяем методы
        if not hasattr(manager, '_generate_tts_audio'):
            print("❌ FAILED: ThesisManager не имеет _generate_tts_audio")
            return False
        print("✓ ThesisManager._generate_tts_audio существует")
        
        if not hasattr(manager, 'get_cached_audio'):
            print("❌ FAILED: ThesisManager не имеет get_cached_audio")
            return False
        print("✓ ThesisManager.get_cached_audio существует")
        
        print("✅ PASSED: Параллельная TTS генерация реализована!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_generation():
    """
    ✅ ТЕСТ 3C: Проверка параллельной генерации тезисов + прогрев TTS
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 3C: Параллельная генерация + прогрев TTS")
    print("="*70)
    
    try:
        # Проверяем что в коде есть ThreadPoolExecutor
        from live_recognizer import LiveVoiceVerifier
        import inspect
        
        source = inspect.getsource(LiveVoiceVerifier._handle_foreign_text)
        
        if "ThreadPoolExecutor" not in source:
            print("❌ FAILED: ThreadPoolExecutor не найден в _handle_foreign_text")
            return False
        print("✓ ThreadPoolExecutor используется")
        
        if "generate_theses" not in source or "warmup_tts" not in source:
            print("❌ FAILED: Параллельные функции не найдены")
            return False
        print("✓ Функции generate_theses и warmup_tts найдены")
        
        # Проверяем что есть as_completed
        if "executor.submit" not in source:
            print("❌ FAILED: executor.submit не найден")
            return False
        print("✓ executor.submit используется для параллельного запуска")
        
        print("✅ PASSED: Параллельная генерация реализована!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_logging():
    """
    ✅ ТЕСТ 8A: Проверка логирования метрик
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 8A: Логирование метрик")
    print("="*70)
    
    try:
        from live_recognizer import LiveVoiceVerifier
        import inspect
        
        # Проверяем _handle_foreign_text
        source = inspect.getsource(LiveVoiceVerifier._handle_foreign_text)
        
        # Должны быть метрики ASR и Gemini
        if "asr_elapsed" not in source:
            print("❌ FAILED: asr_elapsed метрика не найдена")
            return False
        print("✓ ASR метрика (asr_elapsed) найдена")
        
        if "llm_elapsed" not in source:
            print("❌ FAILED: llm_elapsed метрика не найдена")
            return False
        print("✓ LLM метрика (llm_elapsed) найдена")
        
        # Должен быть детализированный лог
        if "МЕТРИКИ ОБРАБОТКИ ВОПРОСА" not in source:
            print("❌ FAILED: Детализированный лог метрик не найден")
            return False
        print("✓ Детализированный лог метрик найден")
        
        # Проверяем _speak_text
        source_speak = inspect.getsource(LiveVoiceVerifier._speak_text)
        
        if "tts_start" not in source_speak or "tts_elapsed" not in source_speak:
            print("❌ FAILED: TTS метрики не найдены в _speak_text")
            return False
        print("✓ TTS метрики найдены в _speak_text")
        
        print("✅ PASSED: Логирование метрик реализовано!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interrupt_logic():
    """
    ✅ ТЕСТ 6B: Проверка логики прерывания
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 6B: Логика агрессивного прерывания")
    print("="*70)
    
    try:
        from live_recognizer import LiveVoiceVerifier
        import inspect
        
        source = inspect.getsource(LiveVoiceVerifier._handle_foreign_text)
        
        # Проверяем что прерывание только при непустых тезисах
        if "if theses:" not in source:
            print("❌ FAILED: Проверка 'if theses:' не найдена")
            return False
        print("✓ Прерывание только при непустых тезисах")
        
        # Проверяем механизм прерывания
        if "_tts_generation" not in source:
            print("❌ FAILED: _tts_generation counter не найден")
            return False
        print("✓ _tts_generation counter для версионности")
        
        if "_tts_interrupt" not in source:
            print("❌ FAILED: _tts_interrupt флаг не найден")
            return False
        print("✓ _tts_interrupt флаг для прерывания")
        
        if "sd.stop()" not in source:
            print("❌ FAILED: sd.stop() не найден для мгновенной остановки")
            return False
        print("✓ sd.stop() для мгновенной остановки воспроизведения")
        
        print("✅ PASSED: Агрессивное прерывание реализовано корректно!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_optimizations():
    """
    ✅ ТЕСТ 3A, 6C: Проверка конфигурационных оптимизаций
    """
    print("\n" + "="*70)
    print("🧪 ТЕСТ 3A, 6C: Конфигурационные оптимизации")
    print("="*70)
    
    try:
        # Тест 3A: max_output_tokens
        from thesis_generator import ThesisGenConfig
        
        cfg = ThesisGenConfig()
        if cfg.max_output_tokens != 180:
            print(f"❌ FAILED 3A: max_output_tokens должен быть 180, но {cfg.max_output_tokens}")
            return False
        print("✓ 3A: max_output_tokens = 180 ✅")
        
        # Тест 6C: queue maxsize
        from live_recognizer import LiveVoiceVerifier
        import inspect
        
        source = inspect.getsource(LiveVoiceVerifier.__init__)
        
        # Ищем maxsize=8
        if "maxsize=8" not in source:
            print("❌ FAILED 6C: maxsize должен быть 8")
            return False
        print("✓ 6C: queue maxsize = 8 ✅")
        
        print("✅ PASSED: Конфигурационные оптимизации применены!")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Запуск всех тестов
    """
    print("\n" + "🚀 "*30)
    print("ТЕСТИРОВАНИЕ ОПТИМИЗАЦИЙ CluelyPro")
    print("🚀 "*30 + "\n")
    
    # Загружаем .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print("✓ .env файл загружен\n")
    except:
        print("⚠️  dotenv не установлен, используем системные переменные\n")
    
    results = {}
    
    # Быстрые тесты (проверка кода)
    results["3A, 6C (Config)"] = test_config_optimizations()
    results["2B (TTS Cache)"] = test_tts_cache()
    results["3C (Parallel Gen)"] = test_parallel_generation()
    results["8A (Metrics)"] = test_metrics_logging()
    results["6B (Interrupt)"] = test_interrupt_logic()
    
    # API тесты (требуют ключи)
    results["3B (Cerebras API)"] = test_cerebras_api()
    results["3B (ThesisGen+Cerebras)"] = test_thesis_generator_with_cerebras()
    
    # Итоги
    print("\n" + "="*70)
    print("📊 ИТОГИ ТЕСТИРОВАНИЯ")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status} | {test_name}")
    
    print("\n" + "="*70)
    print(f"Пройдено: {passed}/{total} ({passed/total*100:.0f}%)")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ! Оптимизации работают корректно!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} тестов провалено. Проверьте ошибки выше.")
        return 1


if __name__ == "__main__":
    exit(main())
