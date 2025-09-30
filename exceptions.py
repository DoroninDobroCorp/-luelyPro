"""Кастомные исключения для CluelyPro."""
from __future__ import annotations


class CluelyProException(Exception):
    """Базовое исключение для всех ошибок CluelyPro."""
    pass


class AudioDeviceError(CluelyProException):
    """Ошибка работы с микрофоном или аудио устройствами.
    
    Примеры:
        - Микрофон не найден или недоступен
        - Ошибка при записи/воспроизведении аудио
        - Несовместимые параметры аудио (sample rate, channels)
    """
    pass


class ASRError(CluelyProException):
    """Ошибка транскрибации речи (Automatic Speech Recognition).
    
    Примеры:
        - Модель ASR не загрузилась
        - Ошибка при обработке аудио
        - Таймаут распознавания
    """
    pass


class LLMError(CluelyProException):
    """Ошибка генерации текста через Language Model.
    
    Примеры:
        - API ключ недействителен или отсутствует
        - Ошибка сети при обращении к API
        - Лимит запросов исчерпан
        - Некорректный ответ от модели
    """
    pass


class ThesisGenerationError(CluelyProException):
    """Ошибка генерации тезисов.
    
    Примеры:
        - Не удалось сгенерировать тезисы из вопроса
        - Пустой или некорректный ответ от генератора
        - API недоступен
    """
    pass


class ThesisJudgeError(CluelyProException):
    """Ошибка судьи тезисов (проверка раскрытия).
    
    Примеры:
        - Судья недоступен (нет API ключа)
        - Ошибка при оценке раскрытия тезиса
    """
    pass


class TTSError(CluelyProException):
    """Ошибка синтеза речи (Text-to-Speech).
    
    Примеры:
        - Модель TTS не загрузилась
        - Ошибка при генерации аудио
        - Аудио устройство недоступно для воспроизведения
    """
    pass


class VADError(CluelyProException):
    """Ошибка Voice Activity Detection.
    
    Примеры:
        - VAD модель не инициализирована
        - Некорректные параметры VAD
        - Ошибка при обработке аудио фрейма
    """
    pass


class EmbeddingError(CluelyProException):
    """Ошибка вычисления эмбеддингов (голосовых или текстовых).
    
    Примеры:
        - Модель эмбеддингов не загрузилась
        - Ошибка при вычислении эмбеддинга
        - Несовместимые размерности эмбеддингов
    """
    pass


class ProfileError(CluelyProException):
    """Ошибка работы с голосовыми профилями.
    
    Примеры:
        - Профиль не найден или повреждён
        - Ошибка при сохранении профиля
        - Некорректный формат профиля
    """
    pass


class ConfigError(CluelyProException):
    """Ошибка конфигурации приложения.
    
    Примеры:
        - Отсутствуют обязательные параметры
        - Некорректные значения параметров
        - Конфликтующие настройки
    """
    pass


class WebSocketError(CluelyProException):
    """Ошибка WebSocket соединения (веб-режим).
    
    Примеры:
        - Соединение разорвано
        - Ошибка при отправке/приёме данных
        - Таймаут соединения
    """
    pass


__all__ = [
    "CluelyProException",
    "AudioDeviceError",
    "ASRError",
    "LLMError",
    "ThesisGenerationError",
    "ThesisJudgeError",
    "TTSError",
    "VADError",
    "EmbeddingError",
    "ProfileError",
    "ConfigError",
    "WebSocketError",
]
