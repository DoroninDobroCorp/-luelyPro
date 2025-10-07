#!/usr/bin/env python3
"""
Базовый тест CluelyPro - проверка импортов и основных компонентов
"""
import sys
from pathlib import Path

# Добавляем родительскую папку в sys.path для импорта модулей
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Проверка что все модули импортируются"""
    modules_to_test = [
        ("main", True),  # (module_name, critical)
        ("config", True),
        ("exceptions", True),
        ("asr_transcriber", False),  # Зависит от faster_whisper
        ("llm_answer", False),  # Зависит от google.genai
        ("tts_silero", False),  # Зависит от torch
        ("vad_silero", False),  # Зависит от torch
        ("thesis_generator", False),  # Зависит от google.genai
    ]
    
    failed_critical = False
    for module_name, is_critical in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}.py импортируется")
        except Exception as e:
            if is_critical:
                print(f"✗ {module_name}.py КРИТИЧЕСКАЯ ошибка: {e}")
                failed_critical = True
            else:
                print(f"⚠ {module_name}.py пропущен (отсутствуют зависимости): {e}")
    
    if failed_critical:
        raise RuntimeError("Критические модули не импортируются!")


def test_config():
    """Проверка конфигурации"""
    from config import AppConfig
    
    # Дефолтный конфиг
    cfg = AppConfig.default()
    assert cfg.vad.backend in ["webrtc", "silero"], "Неверный VAD backend"
    assert cfg.asr.model in ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"], "Неверная ASR модель"
    print("✓ AppConfig.default() работает")
    
    # Конфиг из env
    cfg_env = AppConfig.from_env()
    assert cfg_env.asr.model, "ASR модель не задана"
    print("✓ AppConfig.from_env() работает")


def test_exceptions():
    """Проверка кастомных исключений"""
    from exceptions import (
        CluelyProException,
        AudioDeviceError,
        ASRError,
        LLMError,
        TTSError,
        VADError,
    )
    
    # Проверяем иерархию
    assert issubclass(AudioDeviceError, CluelyProException)
    assert issubclass(ASRError, CluelyProException)
    assert issubclass(LLMError, CluelyProException)
    assert issubclass(TTSError, CluelyProException)
    assert issubclass(VADError, CluelyProException)
    print("✓ Иерархия исключений корректна")


def test_thesis_generator():
    """Проверка генератора тезисов (без реального API вызова)"""
    try:
        from thesis_generator import ThesisGenConfig
        
        cfg = ThesisGenConfig()
        assert cfg.model_id == "gemini-flash-lite-latest"
        assert cfg.temperature == 0.3
        assert cfg.n_theses == 8
        print("✓ ThesisGenConfig работает")
    except ModuleNotFoundError as e:
        print(f"⚠ test_thesis_generator пропущен: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("Запуск базовых тестов CluelyPro")
    print("=" * 50)
    print()
    
    test_imports()
    print()
    
    test_config()
    print()
    
    test_exceptions()
    print()
    
    test_thesis_generator()
    print()
    
    print("=" * 50)
    print("✓ Все тесты пройдены!")
    print("=" * 50)
