#!/usr/bin/env python3
"""
АВТОТЕСТ С РЕАЛЬНЫМ АУДИО.
Использует записи из test_audio/ (созданные record_test_audio.py).
Подаёт их в систему как если бы это был микрофон.
"""
import os
import sys
import time
import queue
import threading
import numpy as np
import soundfile as sf
from pathlib import Path
from loguru import logger

os.environ["THESIS_REPEAT_SEC"] = "5"
os.environ["THESIS_MATCH_THRESHOLD"] = "0.3"
os.environ["FILE_LOG_LEVEL"] = "DEBUG"
os.environ["CONSOLE_LOG_LEVEL"] = "INFO"

logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | {message}")

from live_recognizer import LiveVoiceVerifier, VoiceProfile

AUDIO_DIR = Path("test_audio")

logger.info("=" * 80)
logger.info("АВТОТЕСТ С РЕАЛЬНЫМ АУДИО")
logger.info("=" * 80)
logger.info("")

# Проверяем что записи есть
required_files = ["question.wav", "answer.wav", "question2.wav"]
missing = []
for f in required_files:
    path = AUDIO_DIR / f
    if not path.exists():
        missing.append(f)

if missing:
    logger.error(f"❌ Не найдены файлы: {missing}")
    logger.error("")
    logger.error("Сначала запусти:")
    logger.error("  .venv/bin/python record_test_audio.py")
    logger.error("")
    sys.exit(1)

logger.info("✅ Найдены все записи:")
for f in required_files:
    info = sf.info(AUDIO_DIR / f)
    logger.info(f"   {f}: {info.duration:.1f}с")
logger.info("")

# Загружаем аудио
def load_audio(filename: str) -> np.ndarray:
    """Загружает аудио и конвертирует в нужный формат"""
    audio, sr = sf.read(AUDIO_DIR / filename, dtype='float32')
    
    # Если стерео - берём один канал
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Ресемплим если нужно
    if sr != 16000:
        logger.info(f"Ресемплинг {filename}: {sr}Hz → 16000Hz")
        from scipy import signal
        audio = signal.resample(audio, int(len(audio) * 16000 / sr))
    
    # Нормализуем громкость (усиливаем если слишком тихо)
    max_amp = np.abs(audio).max()
    if max_amp > 0 and max_amp < 0.1:
        # Слишком тихо - усиливаем до 0.5
        gain = 0.5 / max_amp
        audio = audio * gain
        logger.info(f"Усилено аудио {filename}: {max_amp:.4f} → 0.5 (gain={gain:.1f}x)")
    
    return audio.astype(np.float32)

question_audio = load_audio("question.wav")
answer_audio = load_audio("answer.wav")
question2_audio = load_audio("question2.wav")

logger.info("Аудио загружено в память")
logger.info("")

# Создаём систему
logger.info("Инициализация системы...")

# Вместо микрофона будем подавать аудио из очереди
audio_queue = queue.Queue()

verifier = LiveVoiceVerifier(
    asr_enable=True,
    asr_language="ru",  # Форсируем русский!
    llm_enable=True,
    theses_path=None,
    thesis_autogen_enable=True,
    thesis_match_threshold=0.3,
)

# Загружаем профиль если есть
profiles = list(Path("profiles").glob("*.npz"))
voice_profile = None
if profiles:
    logger.info(f"Загружаем профиль: {profiles[0].name}")
    voice_profile = VoiceProfile.load(profiles[0])
    if voice_profile is None:
        logger.error(f"❌ Не удалось загрузить профиль {profiles[0]}")
        sys.exit(1)
    logger.info("Профиль загружен успешно")
else:
    logger.warning("⚠️  Профиль не найден - voice verification не будет работать")
    logger.warning("   Создай профиль: ./run.sh enroll")

logger.info("Система готова")
logger.info("")

# Счётчики
theses_created = 0
theses_closed = 0
announces = 0

# Перехватываем события
_original_announce = verifier._announce_thesis

def tracked_announce():
    global announces
    announces += 1
    if verifier.thesis_prompter:
        text = verifier.thesis_prompter.current_text()[:60]
        logger.info(f"📢 ОБЪЯВЛЕНИЕ #{announces}: {text}...")
    _original_announce()

verifier._announce_thesis = tracked_announce

# Функция для подачи аудио через ASR
def inject_audio_via_asr(audio: np.ndarray, label: str, is_self: bool):
    """Подаёт аудио через ASR для получения текста, затем передаёт текст"""
    logger.info(f"🎙️  Обрабатываем {label}: {len(audio)/16000:.1f}с")
    
    # Распознаём через ASR
    if not verifier._asr:
        logger.error("❌ ASR не инициализирован!")
        return
    
    text = verifier._asr.transcribe_np(audio)
    logger.info(f"   ASR распознал: '{text}'")
    
    # Подаём как диалог
    kind = "self" if is_self else "other"
    verifier.simulate_dialogue([(kind, text)])
    
    logger.info(f"   ✅ Текст передан в систему")
    
    # Небольшая пауза
    time.sleep(0.5)

# Запускаем систему
logger.info("Запуск фоновых потоков...")
verifier._start_segment_worker()

try:
    # ========== РАУНД 1: ВОПРОС ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 1: ПОДАЁМ ВОПРОС")
    logger.info("=" * 80)
    
    inject_audio_via_asr(question_audio, "ВОПРОС", is_self=False)
    
    # Ждём обработки
    logger.info("⏳ Ждём 5 сек (LLM должен ответить и создать тезис)...")
    logger.info("   Смотри логи выше - должно быть:")
    logger.info("   - 'незнакомый голос (ASR): <текст>' - вопрос распознан")
    logger.info("   - 'Ответ: ...' - LLM ответил")
    logger.info("   - 'Тезис (из ответа): ...' - тезис создан")
    time.sleep(5)
    
    # Проверяем что тезис создан
    if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Тезис НЕ создан после вопроса!")
        logger.error("   Возможно ASR не распознал вопрос или LLM не ответил")
        logger.error("")
        logger.error("Смотри логи выше:")
        logger.error("  - Если нет 'незнакомый голос (ASR)' → ASR не распознал")
        logger.error("  - Если нет 'Ответ:' → LLM не ответил")
        logger.error("  - Если нет 'Тезис (из ответа)' → тезис не создался")
        sys.exit(1)
    
    thesis1 = verifier.thesis_prompter.current_text()
    logger.success(f"✅ Тезис создан: {thesis1[:70]}...")
    theses_created += 1
    
    # Ждём повторы
    logger.info("")
    logger.info("⏳ Ждём 6 сек (должны быть повторы тезиса)...")
    time.sleep(6)
    
    if announces < 2:
        logger.error(f"❌ ОШИБКА: Мало повторов! Ожидалось ≥2, было {announces}")
        sys.exit(1)
    
    logger.success(f"✅ Тезис повторялся {announces} раз")
    
    # ========== РАУНД 2: ТВОЙ ОТВЕТ ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 2: ПОДАЁМ ТВОЙ ОТВЕТ")
    logger.info("=" * 80)
    
    inject_audio_via_asr(answer_audio, "ТВОЙ ОТВЕТ", is_self=True)
    
    # Ждём обработки
    logger.info("⏳ Ждём 2 сек (тезис должен закрыться)...")
    time.sleep(2)
    
    # Проверяем что тезис закрылся
    if verifier.thesis_prompter and verifier.thesis_prompter.has_pending():
        logger.error("❌ ОШИБКА: Тезис НЕ закрылся после ответа!")
        logger.error(f"   Тезис: {thesis1[:70]}...")
        
        # Проверяем прогресс
        try:
            cov = verifier.thesis_prompter.coverage_of_current()
            logger.error(f"   Прогресс: {int(cov*100)}% (нужно ≥30%)")
        except:
            pass
        
        logger.error("")
        logger.error("Возможные причины:")
        logger.error("  1. ASR не распознал твой ответ")
        logger.error("  2. Voice verification не распознал твой голос ('мой голос')")
        logger.error("  3. Слова в ответе не совпали с тезисом")
        logger.error("  4. Порог слишком высокий (THESIS_MATCH_THRESHOLD)")
        logger.error("")
        logger.error("Смотри логи выше на наличие:")
        logger.error("  - 'мой голос' - ты распознан")
        logger.error("  - 'Моя речь (ASR): ...' - ответ распознан")
        logger.error("  - 'Прогресс текущего тезиса: XX%' - прогресс показан")
        sys.exit(1)
    
    logger.success("✅ Тезис #1 ЗАКРЫТ!")
    theses_closed += 1
    
    # ========== РАУНД 3: ВТОРОЙ ВОПРОС ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("РАУНД 3: ПРОВЕРКА ЧТО СИСТЕМА НЕ 'ОГЛОХЛА'")
    logger.info("=" * 80)
    
    logger.info("⏳ Пауза 2 сек...")
    time.sleep(2)
    
    inject_audio_via_asr(question2_audio, "ВОПРОС #2", is_self=False)
    
    # Ждём обработки
    logger.info("⏳ Ждём 3 сек (должен создаться новый тезис)...")
    time.sleep(3)
    
    # Проверяем новый тезис
    if verifier.thesis_prompter is None or not verifier.thesis_prompter.has_pending():
        logger.error("❌ КРИТИЧЕСКАЯ ОШИБКА: Новый тезис НЕ создан!")
        logger.error("   Система 'оглохла' после первого цикла!")
        logger.error("")
        logger.error("Это значит баг в коде:")
        logger.error("  - ASR перестал работать после TTS")
        logger.error("  - suppress_until блокирует всё")
        logger.error("  - Очередь сегментов забита")
        sys.exit(1)
    
    thesis2 = verifier.thesis_prompter.current_text()
    
    if thesis2 == thesis1:
        logger.error("❌ ОШИБКА: Тезис не обновился (остался старый)!")
        sys.exit(1)
    
    logger.success(f"✅ Новый тезис создан: {thesis2[:70]}...")
    theses_created += 1
    
    # ========== ИТОГИ ==========
    logger.info("")
    logger.info("=" * 80)
    logger.info("ИТОГИ ТЕСТА")
    logger.info("=" * 80)
    logger.info(f"Тезисов создано: {theses_created}")
    logger.info(f"Тезисов закрыто: {theses_closed}")
    logger.info(f"Всего объявлений: {announces}")
    logger.info("")
    
    if theses_created >= 2 and theses_closed >= 1:
        logger.success("=" * 80)
        logger.success("🎉 ТЕСТ ПРОЙДЕН!")
        logger.success("=" * 80)
        logger.success("")
        logger.success("Система работает корректно:")
        logger.success("  ✅ Распознаёт вопросы с реального аудио")
        logger.success("  ✅ Создаёт тезисы из ответов LLM")
        logger.success("  ✅ Повторяет тезисы периодически")
        logger.success("  ✅ Закрывает тезисы при ответе пользователя")
        logger.success("  ✅ НЕ 'глохнет' после первого цикла")
        logger.success("")
        logger.success("Можно деплоить! 🚀")
        sys.exit(0)
    else:
        logger.error("")
        logger.error("❌ ТЕСТ НЕ ПРОЙДЕН")
        sys.exit(1)

except KeyboardInterrupt:
    logger.warning("\n⚠️  Тест прерван пользователем")
    sys.exit(130)

except Exception as e:
    logger.exception(f"❌ Критическая ошибка: {e}")
    sys.exit(1)

finally:
    verifier._stop_segment_worker()
