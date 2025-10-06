#!/usr/bin/env python3
"""
ТЕСТ: Вопрос - Вопрос - Ответ
Проверяет накопление тезисов и закрытие только одного при ответе.
"""
import os
import sys
import time
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

os.environ["THESIS_REPEAT_SEC"] = "5"
os.environ["FILE_LOG_LEVEL"] = "DEBUG"
os.environ["CONSOLE_LOG_LEVEL"] = "INFO"

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

from live_recognizer import LiveVoiceVerifier, VoiceProfile

AUDIO_DIR = Path("test_audio")

logger.info("=" * 80)
logger.info("ТЕСТ: Вопрос - Вопрос - Ответ")
logger.info("=" * 80)
logger.info("")

# Проверяем что записи есть
required_files = ["question.wav", "question2.wav", "answer.wav"]
missing = []
for f in required_files:
    path = AUDIO_DIR / f
    if not path.exists():
        missing.append(f)

if missing:
    logger.error(f"❌ Не найдены файлы: {missing}")
    logger.error("Сначала запусти: tests/record_test_audio.py")
    sys.exit(1)

logger.info("✅ Найдены все записи:")
for f in required_files:
    info = sf.info(AUDIO_DIR / f)
    logger.info(f"   {f}: {info.duration:.1f}с")
logger.info("")

# Загружаем аудио
def load_audio(filename: str) -> np.ndarray:
    audio, sr = sf.read(AUDIO_DIR / filename, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    if sr != 16000:
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    max_amp = np.abs(audio).max()
    if max_amp > 0 and max_amp < 0.1:
        gain = 0.5 / max_amp
        audio = audio * gain
    return audio.astype(np.float32)

question1_audio = load_audio("question.wav")  # Первый человек в космосе
question2_audio = load_audio("question2.wav")  # Первая женщина в космосе
answer_audio = load_audio("answer.wav")  # Ответ про Гагарина

logger.info("Аудио загружено")
logger.info("")

# Создаём систему
logger.info("Инициализация системы...")

verifier = LiveVoiceVerifier(
    asr_enable=True,
    asr_language="ru",
    llm_enable=True,
)

# Загружаем профиль
profiles = list(Path("profiles").glob("*.npz"))
voice_profile = None
if profiles:
    logger.info(f"Загружаем профиль: {profiles[0].name}")
    voice_profile = VoiceProfile.load(profiles[0])
    if voice_profile is None:
        logger.error(f"❌ Не удалось загрузить профиль {profiles[0]}")
        sys.exit(1)
else:
    logger.warning("⚠️  Профиль не найден")
    sys.exit(1)

logger.info("Система готова")
logger.info("")

# Счётчики
theses_created = 0
theses_closed = 0
announces = 0
announced_theses = {}  # {тезис: количество_объявлений}

# Перехватываем объявления
_original_announce = verifier._announce_thesis

def tracked_announce(thesis_index=None):
    global announces
    announces += 1
    if verifier.thesis_prompter:
        # Получаем текст который будет озвучен
        if thesis_index is not None:
            theses = getattr(verifier.thesis_prompter, "theses", [])
            if 0 <= thesis_index < len(theses):
                text = theses[thesis_index]
            else:
                text = None
        else:
            text = verifier.thesis_prompter.current_text()
        # Отслеживаем какой тезис объявляется
        if text:
            announced_theses[text] = announced_theses.get(text, 0) + 1
            logger.info(f"📢 ОБЪЯВЛЕНИЕ #{announces}: {text[:60]}... (раз {announced_theses[text]})")
    _original_announce(thesis_index=thesis_index)

verifier._announce_thesis = tracked_announce

# Функция подачи аудио
def inject_audio_via_asr(audio: np.ndarray, label: str, is_self: bool):
    logger.info(f"🎙️  Обрабатываем {label}: {len(audio)/16000:.1f}с")
    if not verifier._asr:
        logger.error("❌ ASR не инициализирован!")
        return
    text = verifier._asr.transcribe_np(audio)
    logger.info(f"   ASR распознал: '{text}'")
    kind = "self" if is_self else "other"
    verifier.simulate_dialogue([(kind, text)])
    logger.info(f"   ✅ Текст передан")
    time.sleep(0.5)

# Запускаем фоновые потоки
logger.info("Запуск системы...")
verifier._start_segment_worker()

try:
    # ========== РАУНД 1: ПЕРВЫЙ ВОПРОС ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 1: ПЕРВЫЙ ВОПРОС (Первый человек в космосе)")
    logger.info("=" * 80)
    
    inject_audio_via_asr(question1_audio, "ВОПРОС #1", is_self=False)
    
    logger.info("⏳ Ждём 5 сек (LLM должен создать тезис)...")
    time.sleep(5)
    
    if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Тезис #1 НЕ создан!")
        sys.exit(1)
    
    thesis1 = verifier.thesis_prompter.current_text()
    logger.success(f"✅ Тезис #1 создан: {thesis1[:70]}...")
    theses_created += 1
    
    # ========== РАУНД 2: ВТОРОЙ ВОПРОС ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 2: ВТОРОЙ ВОПРОС (Первая женщина в космосе)")
    logger.info("=" * 80)
    
    inject_audio_via_asr(question2_audio, "ВОПРОС #2", is_self=False)
    
    logger.info("⏳ Ждём 5 сек (LLM должен создать второй тезис)...")
    time.sleep(5)
    
    if verifier.thesis_prompter is None:
        logger.error("❌ ОШИБКА: ThesisPrompter потерян!")
        sys.exit(1)
    
    # Проверяем что есть 2 тезиса
    total_theses = len(verifier.thesis_prompter.theses)
    if total_theses < 2:
        logger.error(f"❌ ОШИБКА: Ожидалось 2 тезиса, есть {total_theses}")
        logger.error(f"   Тезисы: {verifier.thesis_prompter.theses}")
        sys.exit(1)
    
    logger.success(f"✅ Накоплено {total_theses} тезисов")
    for i, t in enumerate(verifier.thesis_prompter.theses, 1):
        logger.info(f"   {i}) {t[:60]}...")
    theses_created = total_theses
    
    # Проверяем что текущий тезис всё ещё первый
    current_thesis = verifier.thesis_prompter.current_text()
    if current_thesis != thesis1:
        logger.error(f"❌ ОШИБКА: Текущий тезис сменился!")
        logger.error(f"   Было: {thesis1[:40]}...")
        logger.error(f"   Стало: {current_thesis[:40]}...")
        sys.exit(1)
    
    logger.info("⏳ Ждём 2 сек (должны быть повторы первого тезиса)...")
    time.sleep(2)
    
    # ========== РАУНД 3: ОТВЕТ НА ПЕРВЫЙ ВОПРОС ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 3: ОТВЕТ на первый вопрос (про Гагарина)")
    logger.info("=" * 80)
    
    inject_audio_via_asr(answer_audio, "ОТВЕТ", is_self=True)
    
    logger.info("⏳ Ждём 3 сек (первый тезис должен закрыться, второй остаться)...")
    time.sleep(3)
    
    if verifier.thesis_prompter is None:
        logger.error("❌ ОШИБКА: ThesisPrompter потерян!")
        sys.exit(1)
    
    # Проверяем что первый тезис закрылся
    current_thesis_now = verifier.thesis_prompter.current_text()
    
    if current_thesis_now == thesis1:
        logger.error("❌ ОШИБКА: Первый тезис НЕ закрылся!")
        logger.error(f"   Тезис: {thesis1[:60]}...")
        
        # Проверяем контекст диалога
        if hasattr(verifier.thesis_prompter, '_dialogue_context'):
            logger.error(f"   Контекст диалога: {verifier.thesis_prompter._dialogue_context}")
        
        sys.exit(1)
    
    logger.success("✅ Первый тезис закрыт!")
    theses_closed += 1
    
    # Проверяем что второй тезис активен
    if not verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Второй тезис пропал!")
        sys.exit(1)
    
    thesis2_text = verifier.thesis_prompter.current_text()
    logger.success(f"✅ Второй тезис активен: {thesis2_text[:70]}...")
    
    # Проверяем что это именно второй тезис
    if verifier.thesis_prompter._index != 1:
        logger.error(f"❌ ОШИБКА: Индекс тезиса неправильный: {verifier.thesis_prompter._index}, ожидался 1")
        sys.exit(1)
    
    # ========== ИТОГИ ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("ИТОГИ ТЕСТА")
    logger.info("=" * 80)
    logger.info(f"Тезисов создано: {theses_created}")
    logger.info(f"Тезисов закрыто: {theses_closed}")
    logger.info(f"Всего объявлений: {announces}")
    logger.info("")
    
    # Проверяем что оба тезиса объявлялись
    logger.info("Объявленные тезисы:")
    for thesis_text, count in announced_theses.items():
        logger.info(f"  - {thesis_text[:50]}... : {count} раз")
    
    if len(announced_theses) < 2:
        logger.error(f"❌ ОШИБКА: Объявлялся только {len(announced_theses)} тезис из 2!")
        logger.error("   Второй тезис не был проговорен")
        sys.exit(1)
    
    logger.success("✅ Оба тезиса были объявлены")
    logger.info("")
    
    if theses_created >= 2 and theses_closed >= 1:
        logger.success("=" * 80)
        logger.success("🎉 ТЕСТ ПРОЙДЕН!")
        logger.success("=" * 80)
        logger.success("")
        logger.success("Система работает корректно:")
        logger.success("  ✅ Накапливает тезисы от нескольких вопросов")
        logger.success("  ✅ Закрывает только подходящий тезис при ответе")
        logger.success("  ✅ Сохраняет остальные тезисы активными")
        logger.success("")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("❌ ТЕСТ НЕ ПРОЙДЕН")
        sys.exit(1)

except KeyboardInterrupt:
    logger.warning("\n⚠️  Тест прерван")
    sys.exit(130)

except Exception as e:
    logger.exception(f"❌ Критическая ошибка: {e}")
    sys.exit(1)

finally:
    verifier._stop_segment_worker()
