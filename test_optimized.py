#!/usr/bin/env python3
"""Тест оптимизированной программы"""

import sys
import time
from pathlib import Path
from live_recognizer import extract_theses_from_text

def test_thesis_extraction():
    """Тестируем извлечение тезисов"""
    test_cases = [
        "В каком году началась Вторая мировая война?",
        "Расскажите, как работает протокол TCP?",
        "Привет, как дела?",
        "Почему железо ржавеет?",
    ]
    
    print("="*50)
    print("ТЕСТ ИЗВЛЕЧЕНИЯ ТЕЗИСОВ")
    print("="*50)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\nТест {i}: {text}")
        start = time.time()
        theses = extract_theses_from_text(text)
        elapsed = time.time() - start
        
        if theses:
            print(f"Время обработки: {elapsed:.2f} сек")
            for j, t in enumerate(theses, 1):
                print(f"  {j}. {t}")
        else:
            print(f"Время обработки: {elapsed:.2f} сек")
            print("  (не найдено тезисов)")
    
    print("\n" + "="*50)
    print("ТЕСТ ЗАВЕРШЕН")
    print("="*50)

def test_tts_speed():
    """Тестируем скорость TTS"""
    try:
        from tts_silero import SileroTTS
        
        print("\n" + "="*50)
        print("ТЕСТ СКОРОСТИ TTS")
        print("="*50)
        
        tts = SileroTTS()
        texts = [
            "Тезис номер один",
            "Второй тезис ответа",
            "Третий и финальный тезис",
        ]
        
        for text in texts:
            start = time.time()
            audio = tts.synth(text)
            elapsed = time.time() - start
            duration = len(audio) / tts.sample_rate if len(audio) > 0 else 0
            print(f"'{text}': синтез {elapsed:.3f}с, длина {duration:.2f}с")
        
        print("="*50)
    except Exception as e:
        print(f"TTS тест пропущен: {e}")

if __name__ == "__main__":
    print("\n🚀 ЗАПУСКАЕМ ТЕСТЫ ОПТИМИЗИРОВАННОЙ СИСТЕМЫ\n")
    
    # Тестируем извлечение тезисов
    test_thesis_extraction()
    
    # Тестируем скорость TTS
    test_tts_speed()
    
    print("\n✅ ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ")
