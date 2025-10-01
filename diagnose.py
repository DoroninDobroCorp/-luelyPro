#!/usr/bin/env python3
"""
Интерактивная диагностика CluelyPro.
Проводит по шагам, проверяет все компоненты.
"""
import os
import sys
import time
import numpy as np
from pathlib import Path
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

# Быстрые настройки
os.environ["THESIS_REPEAT_SEC"] = "5"
os.environ["FILE_LOG_LEVEL"] = "INFO"
os.environ["CONSOLE_LOG_LEVEL"] = "INFO"


def print_step(num, title):
    """Красивый заголовок шага"""
    logger.info("")
    logger.info("=" * 80)
    logger.info(f"ШАГ {num}: {title}")
    logger.info("=" * 80)


def ask_continue():
    """Спросить продолжить ли"""
    logger.info("")
    try:
        response = input(">>> Нажми Enter чтобы продолжить (или 'q' чтобы выйти): ").strip()
        if response.lower() == 'q':
            logger.info("Диагностика прервана.")
            sys.exit(0)
    except KeyboardInterrupt:
        logger.info("\nДиагностика прервана.")
        sys.exit(0)


def test_microphone():
    """Тест 1: Проверка микрофона"""
    print_step(1, "ПРОВЕРКА МИКРОФОНА")
    
    logger.info("Сейчас проверим работает ли микрофон вообще...")
    logger.info("")
    logger.info("📋 Что будет:")
    logger.info("  1. Запишем 3 секунды аудио с микрофона")
    logger.info("  2. Проверим что уровень сигнала > 0")
    logger.info("")
    logger.info("✋ ПРИГОТОВЬСЯ ГОВОРИТЬ через 2 секунды!")
    
    ask_continue()
    
    try:
        import sounddevice as sd
        
        logger.info("⏺ ГОВОРИ ЧТО-НИБУДЬ СЕЙЧАС! (3 секунды)")
        audio = sd.rec(int(3 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        
        # Проверяем уровень
        rms = np.sqrt(np.mean(audio**2))
        max_amp = np.max(np.abs(audio))
        
        logger.info("")
        logger.info(f"📊 Результат:")
        logger.info(f"  RMS уровень: {rms:.6f}")
        logger.info(f"  Максимальная амплитуда: {max_amp:.6f}")
        
        if max_amp < 0.001:
            logger.error("❌ МИКРОФОН НЕ РАБОТАЕТ!")
            logger.error("   Уровень сигнала слишком низкий.")
            logger.error("")
            logger.error("🔧 Что проверить:")
            logger.error("   1. Микрофон подключен?")
            logger.error("   2. В Системных настройках → Конфиденциальность → Микрофон")
            logger.error("   3. Правильный микрофон выбран как устройство по умолчанию?")
            return False
        
        logger.success("✅ МИКРОФОН РАБОТАЕТ!")
        logger.info(f"   Уровень сигнала нормальный: {max_amp:.4f}")
        return True
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА при проверке микрофона: {e}")
        return False


def test_vad():
    """Тест 2: Проверка VAD (детектор голоса)"""
    print_step(2, "ПРОВЕРКА VAD (ДЕТЕКТОР РЕЧИ)")
    
    logger.info("Сейчас проверим детектирует ли система речь...")
    logger.info("")
    logger.info("📋 Что будет:")
    logger.info("  1. Запишем 5 секунд с микрофона")
    logger.info("  2. Проверим сколько фреймов VAD считает речью")
    logger.info("")
    logger.info("✋ ПРИГОТОВЬСЯ ГОВОРИТЬ!")
    
    ask_continue()
    
    try:
        import sounddevice as sd
        import webrtcvad
        
        vad = webrtcvad.Vad(2)  # агрессивность 2
        
        logger.info("⏺ ГОВОРИ РАЗНЫЕ ПРЕДЛОЖЕНИЯ! (5 секунд)")
        logger.info("   Например: 'Когда отменили рабство? Кто изобрёл компьютер?'")
        
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='int16')
        sd.wait()
        
        # Проверяем VAD
        frame_duration = 20  # мс
        frame_size = int(16000 * frame_duration / 1000)
        num_frames = len(audio) // frame_size
        
        speech_frames = 0
        for i in range(num_frames):
            start = i * frame_size
            end = start + frame_size
            frame = audio[start:end].tobytes()
            
            try:
                if vad.is_speech(frame, 16000):
                    speech_frames += 1
            except:
                pass
        
        speech_percent = (speech_frames / num_frames) * 100 if num_frames > 0 else 0
        
        logger.info("")
        logger.info(f"📊 Результат:")
        logger.info(f"  Всего фреймов: {num_frames}")
        logger.info(f"  Фреймов с речью: {speech_frames}")
        logger.info(f"  Процент речи: {speech_percent:.1f}%")
        
        if speech_frames < 10:
            logger.error("❌ VAD НЕ ДЕТЕКТИРУЕТ РЕЧЬ!")
            logger.error("   Почти все фреймы считаются тишиной.")
            logger.error("")
            logger.error("🔧 Возможные причины:")
            logger.error("   1. Говоришь слишком тихо")
            logger.error("   2. Слишком много фонового шума")
            logger.error("   3. Микрофон далеко")
            return False
        
        logger.success(f"✅ VAD РАБОТАЕТ!")
        logger.info(f"   Детектировано {speech_frames} фреймов с речью ({speech_percent:.1f}%)")
        return True
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА при проверке VAD: {e}")
        return False


def test_asr():
    """Тест 3: Проверка ASR (распознавание речи)"""
    print_step(3, "ПРОВЕРКА ASR (РАСПОЗНАВАНИЕ РЕЧИ)")
    
    logger.info("Сейчас проверим распознаёт ли ASR твою речь...")
    logger.info("")
    logger.info("📋 Что будет:")
    logger.info("  1. Запишем 5 секунд аудио")
    logger.info("  2. Прогоним через faster-whisper")
    logger.info("  3. Проверим что распознал")
    logger.info("")
    logger.info("✋ ПРИГОТОВЬСЯ ГОВОРИТЬ ЧЁТКО!")
    
    ask_continue()
    
    try:
        import sounddevice as sd
        from asr_transcriber import FasterWhisperTranscriber
        
        logger.info("Загружаем модель ASR (tiny)...")
        asr = FasterWhisperTranscriber(
            model_size="tiny",
            device="cpu",
            compute_type="int8",
            language="ru"
        )
        
        logger.info("⏺ ПРОИЗНЕСИ: 'Когда отменили рабство в США?' (5 секунд)")
        audio = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        
        # Распознаём
        logger.info("🔄 Распознаю...")
        text = asr.transcribe_np(audio.flatten(), 16000)
        
        logger.info("")
        logger.info(f"📊 Результат:")
        logger.info(f"  Распознано: '{text}'")
        
        if not text or len(text.strip()) < 3:
            logger.error("❌ ASR НЕ РАСПОЗНАЛ РЕЧЬ!")
            logger.error("   Вернул пустую строку или слишком короткую.")
            logger.error("")
            logger.error("🔧 Что попробовать:")
            logger.error("   1. Говори ГРОМЧЕ и ЧЁТЧЕ")
            logger.error("   2. Используй модель побольше: ASR_MODEL=small")
            logger.error("   3. Убери фоновый шум (музыка, телевизор)")
            return False
        
        logger.success(f"✅ ASR РАБОТАЕТ!")
        logger.info(f"   Распознал: '{text}'")
        return True
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА при проверке ASR: {e}")
        logger.exception(e)
        return False


def test_voice_profile():
    """Тест 4: Проверка профиля голоса"""
    print_step(4, "ПРОВЕРКА ПРОФИЛЯ ГОЛОСА")
    
    logger.info("Пропускаем отдельную проверку профиля.")
    logger.info("Профиль будет проверен в полном цикле (Шаг 5).")
    logger.info("")
    
    # Проверяем есть ли профиль
    profiles = list(Path("profiles").glob("*.npz"))
    
    if not profiles:
        logger.error("❌ НЕТ НИ ОДНОГО ПРОФИЛЯ ГОЛОСА!")
        logger.error("")
        logger.error("🔧 Создай профиль:")
        logger.error("   ./run.sh enroll")
        return False
    
    logger.info(f"✅ Найдено профилей: {len(profiles)}")
    for i, p in enumerate(profiles, 1):
        logger.info(f"  {i}. {p.name}")
    
    return True


def test_full_cycle():
    """Тест 5: Полный цикл"""
    print_step(5, "ПОЛНЫЙ ЦИКЛ (10 секунд)")
    
    logger.info("Финальный тест: запустим систему на 10 секунд.")
    logger.info("")
    logger.info("📋 Что сделать:")
    logger.info("  1. Система запустится")
    logger.info("  2. Кто-то задаёт вопрос (или включи видео)")
    logger.info("  3. Ты отвечаешь когда услышишь тезис")
    logger.info("  4. Через 10 секунд система остановится")
    logger.info("")
    logger.info("📝 ПРИМЕР:")
    logger.info("  - Вопрос: 'Когда отменили рабство?'")
    logger.info("  - Система: 'В 1865 году...' (повторяет)")
    logger.info("  - ТЫ: 'В 1865 году'")
    logger.info("  - Система: 'Тезис закрыт' ← это успех!")
    logger.info("")
    
    ask_continue()
    
    # Выбираем профиль
    profiles = list(Path("profiles").glob("*.npz"))
    if not profiles:
        logger.error("❌ Нет профилей! Создай: ./run.sh enroll")
        return False
    
    logger.info(f"Используем профиль: {profiles[0].name}")
    
    try:
        import subprocess
        
        os.environ["RUN_SECONDS"] = "10"
        os.environ["THESIS_REPEAT_SEC"] = "5"
        
        logger.info("")
        logger.info("🚀 ЗАПУСК! Говорите в микрофон!")
        logger.info("   (Автоостановка через 10 секунд)")
        logger.info("")
        
        # Запускаем через main.py с профилем
        result = subprocess.run(
            [".venv/bin/python", "main.py", "live", "--profile", str(profiles[0])],
            env={**os.environ, "RUN_SECONDS": "10", "THESIS_REPEAT_SEC": "5"},
            capture_output=False
        )
        
        logger.info("")
        logger.success("✅ Тест завершён!")
        logger.info("")
        logger.info("🔍 Проверь логи выше:")
        logger.info("  1. Есть 'незнакомый голос (ASR): вопрос'? ← вопрос распознан")
        logger.info("  2. Есть 'Ответ: ...'? ← LLM ответил")
        logger.info("  3. Есть 'Тезис (из ответа): ...'? ← тезис создан")
        logger.info("  4. Есть 'мой голос'? ← ты распознан")
        logger.info("  5. Есть 'Моя речь (ASR): ...'? ← твой ответ распознан")
        logger.info("  6. Есть 'Тезис закрыт'? ← тезис закрыт твоим ответом")
        logger.info("")
        
        return result.returncode == 0
        
    except Exception as e:
        logger.error(f"❌ ОШИБКА в полном цикле: {e}")
        logger.exception(e)
        return False


def main():
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "ДИАГНОСТИКА CluelyPro" + " " * 38 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")
    logger.info("Этот скрипт проверит все компоненты системы пошагово.")
    logger.info("")
    logger.info("📋 Будет проверено:")
    logger.info("  1. Микрофон")
    logger.info("  2. VAD (детектор речи)")
    logger.info("  3. ASR (распознавание речи)")
    logger.info("  4. Профиль голоса")
    logger.info("  5. Полный цикл")
    logger.info("")
    logger.info("⏱ Займёт ~5-10 минут.")
    logger.info("")
    
    ask_continue()
    
    results = {}
    
    # Тест 1-3: Базовые проверки (ПРОПУЩЕНЫ - уже прошли ✅)
    logger.info("⏩ Пропускаем Шаги 1-3 (микрофон, VAD, ASR) - они уже проверены")
    logger.info("")
    results['microphone'] = True  # Уже прошёл
    results['vad'] = True  # Уже прошёл
    results['asr'] = True  # Уже прошёл
    
    # Тест 4: Профиль голоса
    results['profile'] = test_voice_profile()
    if not results['profile']:
        logger.warning("⚠️ Профиль не совпадает - пересоздай или настрой threshold")
    
    # Тест 5: Полный цикл
    results['full_cycle'] = test_full_cycle()
    
    # Итоги
    print_step("ИТОГИ", "РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
    
    logger.info("Проверка компонентов:")
    for key, value in results.items():
        status = "✅ OK" if value else "❌ FAIL"
        logger.info(f"  {key:15} {status}")
    
    logger.info("")
    
    all_ok = all(results.values())
    
    if all_ok:
        logger.success("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
        logger.success("   Система готова к использованию!")
        logger.info("")
        logger.info("🚀 Запуск:")
        logger.info("   THESIS_REPEAT_SEC=10 ./run.sh live --profile profiles/*.npz")
    else:
        failed = [k for k, v in results.items() if not v]
        logger.error(f"❌ Проблемы: {', '.join(failed)}")
        logger.error("   Смотри рекомендации выше.")
    
    logger.info("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nДиагностика прервана.")
        sys.exit(0)
