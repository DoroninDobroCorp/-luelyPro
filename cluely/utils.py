"""Утилиты для CluelyPro."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

from loguru import logger
from config import SystemConfig


# Глобальное состояние логирования
_LOGGING_CONFIGURED = False
_CURRENT_LOG_DIR: Optional[Path] = None


def setup_logging(
    log_dir: Optional[Path] = None,
    console_level: Optional[str] = None,
    file_level: Optional[str] = None,
) -> Path:
    """Настройка логирования через loguru.
    
    Args:
        log_dir: Директория для лог-файлов (по умолчанию из config)
        console_level: Уровень логов в консоль (DEBUG/INFO/WARNING/ERROR)
        file_level: Уровень логов в файл
        
    Returns:
        Путь к лог-файлу
    """
    global _LOGGING_CONFIGURED, _CURRENT_LOG_DIR
    
    cfg = SystemConfig.from_env()
    target_dir = Path(log_dir or cfg.log_dir)
    console_lvl = console_level or cfg.log_level_console
    file_lvl = file_level or cfg.log_level_file
    
    # Переконфигурируем только если нужно
    need_reconfigure = (not _LOGGING_CONFIGURED) or (
        log_dir is not None and (_CURRENT_LOG_DIR is None or target_dir != _CURRENT_LOG_DIR)
    )
    
    if not need_reconfigure and _CURRENT_LOG_DIR is not None:
        return _CURRENT_LOG_DIR / "assistant.log"
    
    target_dir.mkdir(parents=True, exist_ok=True)
    log_file = target_dir / "assistant.log"
    
    logger.remove()
    logger.add(
        sys.stderr, 
        level=console_lvl.upper(), 
        enqueue=True, 
        backtrace=False,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    )
    logger.add(
        log_file,
        level=file_lvl.upper(),
        enqueue=False,  # синхронная запись для тестов
        mode="w",
        encoding="utf-8",
        backtrace=True,
        diagnose=True,
    )
    
    # Гарантируем создание файла
    log_file.touch(exist_ok=True)
    logger.info(f"Логи пишутся в {log_file.resolve()}")
    
    _LOGGING_CONFIGURED = True
    _CURRENT_LOG_DIR = target_dir
    
    return log_file


# Дефолтные параграфы для записи профиля
DEFAULT_ENROLL_PARAGRAPHS: list[str] = [
    (
        "Этот абзац нужен, чтобы система уверенно запомнила мой голос. "
        "Я говорю размеренно, держу одинаковое расстояние до микрофона и не спешу. "
        "В конце делаю короткую паузу, чтобы запись завершилась корректно."
    ),
    (
        "В рабочем дне мне часто приходится объяснять сложные идеи простым языком. "
        "Поэтому важно, чтобы ассистент чётко распознавал мои интонации и тембр. "
        "Я произношу слова ясно и уверенно, словно отвечаю на вопрос интервьюера."
    ),
    (
        "Чтобы профиль получился естественным, я читаю текст, похожий на разговор о проектах. "
        "Я кратко описываю задачи, решения и выводы, сохраняя живую и спокойную речь. "
        "Так алгоритм уловит мой реальный стиль общения."
    ),
]


def split_paragraphs(text: str) -> list[str]:
    """Разбить текст на параграфы (разделитель: двойной перенос строки).
    
    Args:
        text: Исходный текст
        
    Returns:
        Список параграфов
    """
    if not text:
        return []
    parts = [p.strip() for p in text.replace("\r", "").split("\n\n")]
    return [p for p in parts if p]


def extract_theses_from_text(text: str) -> List[str]:
    """Извлечь тезисы из текста через AI (публичная функция для API).
    
    Args:
        text: Текст чужой реплики/диалога
        
    Returns:
        Список тезисов или [] если вопросов нет
    """
    try:
        from thesis_generator import GeminiThesisGenerator
        generator = GeminiThesisGenerator()
        return generator.generate_from_foreign_text(text)
    except Exception:
        return []


__all__ = [
    "setup_logging",
    "DEFAULT_ENROLL_PARAGRAPHS",
    "split_paragraphs",
    "extract_theses_from_text",
]
