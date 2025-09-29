import time
from live_recognizer import setup_logging, LiveVoiceVerifier
from loguru import logger

# Настраиваем логирование в logs/assistant.log
setup_logging()

# Инициализируем без тяжёлых зависимостей
v = LiveVoiceVerifier(asr_enable=False, llm_enable=False, thesis_autogen_enable=False)
# Не использовать AI-экстракцию, чтобы не требовался ключ
v._ai_only_thesis = False
# Ускоряем интервал автоповтора для теста
v._thesis_repeat_sec = 3.0

logger.info("[SMOKE] Triggering foreign text handler: 'Сколько лет Путину?'")
v._handle_foreign_text("Сколько лет Путину?")

# Два ручных повторa через общий механизм
for _ in range(2):
    time.sleep(v._thesis_repeat_sec + 0.3)
    if v.thesis_prompter is not None:
        v.thesis_prompter.reset_announcement()
        v._announce_thesis()

# Доп. ожидание, чтобы суммарно вышло ~10 секунд
remaining = max(0.0, 10 - 2*(v._thesis_repeat_sec + 0.3))
time.sleep(remaining)
logger.info("[SMOKE] Done")
