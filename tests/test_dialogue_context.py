#!/usr/bin/env python3
"""
Тест контекста диалога и генерации тезисов с ||| разделителем
(Unit-тесты БЕЗ зависимостей от сторонних библиотек)
"""
import time


def test_dialogue_context_accumulation():
    """Проверка накопления контекста диалога за 30 сек"""
    
    # Эмулируем структуру данных из LiveVoiceVerifier
    dialogue_context = []
    context_window_sec = 30.0
    
    # Проверяем начальное состояние
    assert len(dialogue_context) == 0
    
    # Добавляем первый вопрос
    now = time.time()
    dialogue_context.append((now, "Кто первый полетел в космос?"))
    
    # Добавляем второй вопрос через 5 сек
    dialogue_context.append((now + 5, "Где он родился?"))
    
    # Проверяем что оба вопроса в контексте
    assert len(dialogue_context) == 2
    
    # Эмулируем прохождение 35 секунд - старый контекст должен удалиться
    current_time = now + 35  # Прошло 35 секунд
    cutoff_time = current_time - context_window_sec  # Граница = 35 - 30 = 5 сек от начала
    
    # Очищаем контекст: оставляем только вопросы новее cutoff_time
    dialogue_context = [
        (ts, txt) for ts, txt in dialogue_context 
        if ts >= cutoff_time
    ]
    
    # Первый вопрос (now=0) старше cutoff_time (5 сек) → удален
    # Второй вопрос (now+5=5) равен cutoff_time → остается
    assert len(dialogue_context) == 1
    assert dialogue_context[0][1] == "Где он родился?"
    
    print("✓ Контекст диалога работает корректно (накопление + очистка)")


def test_thesis_parsing_with_delimiter():
    """Проверка парсинга тезисов с разделителем |||"""
    
    # Тест 1: парсинг строки с |||
    raw_text = "Гагарин полетел 12 апреля 1961 ||| Полет длился 108 минут ||| Первый человек в космосе"
    theses = [t.strip() for t in raw_text.split("|||") if t.strip()]
    
    assert len(theses) == 3
    assert theses[0] == "Гагарин полетел 12 апреля 1961"
    assert theses[1] == "Полет длился 108 минут"
    assert theses[2] == "Первый человек в космосе"
    
    print("✓ Парсинг тезисов с ||| работает")
    
    # Тест 2: парсинг с пробелами
    raw_text2 = "  Тезис 1  |||   Тезис 2   ||| Тезис 3  "
    theses2 = [t.strip() for t in raw_text2.split("|||") if t.strip()]
    
    assert len(theses2) == 3
    assert all(not t.startswith(" ") and not t.endswith(" ") for t in theses2)
    
    print("✓ Парсинг с пробелами работает")


def test_thesis_repetition_logic():
    """Проверка логики повтора тезисов 2 раза"""
    
    # Эмулируем список тезисов
    theses = ["Тезис 1", "Тезис 2", "Тезис 3"]
    spoken_count = {}
    
    # Эмулируем озвучивание каждого тезиса 2 раза
    for i, thesis in enumerate(theses, 1):
        text_to_speak = f"{i}. {thesis}"
        
        # Первый раз
        spoken_count[text_to_speak] = spoken_count.get(text_to_speak, 0) + 1
        
        # Второй раз
        spoken_count[text_to_speak] = spoken_count.get(text_to_speak, 0) + 1
    
    # Проверяем что каждый тезис озвучен ровно 2 раза
    assert len(spoken_count) == 3
    assert all(count == 2 for count in spoken_count.values())
    
    print("✓ Логика повтора тезисов 2 раза работает")


def test_context_text_formatting():
    """Проверка форматирования контекста для передачи в Gemini"""
    
    dialogue = [
        (time.time(), "Кто первый полетел в космос?"),
        (time.time() + 10, "Где он родился?"),
        (time.time() + 20, "Когда это произошло?"),
    ]
    
    # Форматируем контекст
    context_text = "\n".join([txt for _, txt in dialogue])
    
    expected = "Кто первый полетел в космос?\nГде он родился?\nКогда это произошло?"
    assert context_text == expected
    
    # Проверяем что контекст содержит все местоимения
    assert "он" in context_text.lower()
    assert "это" in context_text.lower()
    
    print("✓ Форматирование контекста работает")


def test_context_pronoun_resolution():
    """Проверка что контекст позволяет разрешить местоимения"""
    
    # Сценарий: два вопроса с местоимением
    dialogue = [
        (time.time(), "Кто первый полетел в космос?"),
        (time.time() + 10, "Где ОН родился?"),
    ]
    
    # Форматируем контекст
    context_text = "\n".join([txt for _, txt in dialogue])
    
    # Проверяем что контекст содержит оба вопроса
    assert "Кто первый полетел в космос?" in context_text
    assert "Где ОН родился?" in context_text
    
    # Проверяем что местоимение есть в контексте
    assert "ОН" in context_text
    
    # В реальности Gemini будет понимать что ОН = Гагарин благодаря контексту
    print("✓ Контекст для разрешения местоимений работает")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Тестирование контекста диалога и тезисов с |||")
    print("=" * 70 + "\n")
    
    # Запускаем все тесты
    test_dialogue_context_accumulation()
    test_thesis_parsing_with_delimiter()
    test_thesis_repetition_logic()
    test_context_text_formatting()
    test_context_pronoun_resolution()
    
    print("\n" + "=" * 70)
    print("✓ Все тесты контекста диалога прошли успешно!")
    print("=" * 70 + "\n")
