#!/usr/bin/env python3
"""
Быстрый тест на 10 секунд - БЕЗ интерактива.
Просто запускает систему и показывает что проверить.
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

# Настройка логирования
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

logger.info("=" * 80)
logger.info("БЫСТРЫЙ ТЕСТ: 20 секунд работы системы")
logger.info("=" * 80)
logger.info("")

# Проверяем профили
profiles = list(Path("profiles").glob("*.npz"))

if not profiles:
    logger.error("❌ НЕТ ПРОФИЛЕЙ!")
    logger.error("   Создай: ./run.sh enroll")
    sys.exit(1)

logger.info(f"Используем профиль: {profiles[0].name}")
logger.info("")

logger.info("📋 ЧТО ДЕЛАТЬ:")
logger.info("  1. Система запустится на 20 секунд")
logger.info("  2. Пусть кто-то задаст вопрос (или включи видео)")
logger.info("  3. Дождись тезиса")
logger.info("  4. ОТВЕТЬ ГРОМКО когда услышишь повтор")
logger.info("")

logger.info("📝 ПРИМЕР:")
logger.info("  Вопрос: 'Когда отменили рабство?'")
logger.info("  Система: 'В 1865 году...' (повторяет)")
logger.info("  ТЫ: 'В 1865 году'")
logger.info("  Ожидаем: 'Тезис закрыт' ✅")
logger.info("")

logger.info("🚀 ЗАПУСК ЧЕРЕЗ 3 СЕКУНДЫ...")
import time
for i in range(3, 0, -1):
    logger.info(f"   {i}...")
    time.sleep(1)

logger.info("")
logger.info("▶ ЗАПУЩЕНО! Говорите в микрофон!")
logger.info("")

# Запускаем с ASR и LLM
result = subprocess.run(
    [".venv/bin/python", "main.py", "live", 
     "--profile", str(profiles[0]),
     "--asr",  # Включаем ASR!
     "--llm"   # Включаем LLM!
    ],
    env={**os.environ, "RUN_SECONDS": "20", "THESIS_REPEAT_SEC": "5"}
)

logger.info("")
logger.info("=" * 80)
logger.info("ТЕСТ ЗАВЕРШЁН")
logger.info("=" * 80)
logger.info("")

logger.info("🔍 ПРОВЕРЬ ЛОГИ ВЫШЕ - должно быть:")
logger.info("")
logger.info("  ✅ 'незнакомый голос (ASR): вопрос' - вопрос распознан")
logger.info("  ✅ 'Ответ: ...' - LLM ответил")
logger.info("  ✅ 'Тезис (из ответа): ...' - тезис создан (если не 'не ясен')")
logger.info("  ✅ 'Тезис: ...' - тезис объявлен")
logger.info("")
logger.info("  🎯 ГЛАВНОЕ:")
logger.info("  ✅ 'мой голос' - ТЫ распознан (когда отвечал)")
logger.info("  ✅ 'Моя речь (ASR): твой ответ' - твой ответ распознан")
logger.info("  ✅ 'Прогресс текущего тезиса: XX%' - показан прогресс")
logger.info("  ✅ 'Тезис закрыт' - тезис закрыт!")
logger.info("")

logger.info("❌ ЕСЛИ НЕТ 'мой голос':")
logger.info("   → Профиль не подходит или threshold слишком строгий")
logger.info("   → Пересоздай: ./run.sh enroll")
logger.info("   → Или увеличь threshold в .env: VOICE_THRESHOLD=0.85")
logger.info("")

logger.info("❌ ЕСЛИ НЕТ 'Тезис закрыт':")
logger.info("   → Порог слишком строгий или мало слов совпало")
logger.info("   → Снизь порог в .env: THESIS_MATCH_THRESHOLD=0.2")
logger.info("   → Или отвечай БОЛЬШЕ слов из тезиса")
logger.info("")

sys.exit(result.returncode)
