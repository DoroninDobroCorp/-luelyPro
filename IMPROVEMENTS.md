# Рекомендации по улучшению CluelyPro

## 🔴 Критические (Безопасность)

### 1. API ключ в репозитории
**Проблема:** Файл `.env` с реальным API ключом Gemini попал в git.

**Действия:**
```bash
# 1. Ротировать ключ в Google AI Studio: https://aistudio.google.com/app/apikey
# 2. Удалить .env из истории git
git rm --cached .env
git commit -m "Remove .env from tracking"
# 3. Обновить .env с новым ключом (локально)
```

**Статус:** ✅ Исправлено - `.gitignore` обновлён

---

## 🟡 Высокий приоритет

### 2. Производительность: размер файла live_recognizer.py (2148 строк)
**Проблема:** Монолитный файл сложно поддерживать.

**Решение:** Разбить на модули:
```
live_recognizer/
  __init__.py          # Экспорт основных классов
  verifier.py          # LiveVoiceVerifier
  audio_processor.py   # VAD, сегментация, эмбеддинги
  segment_worker.py    # Обработка сегментов (асинхронная)
  thesis_manager.py    # Управление тезисами и повторами
  utils.py             # setup_logging, утилиты
  enroll.py            # enroll_cli
  live.py              # live_cli
```

### 3. Латентность ASR
**Проблема:** Модель `large-v3-turbo` тяжёлая для реалтайм режима.

**Решение:**
- По умолчанию использовать `tiny` или `base` 
- В `.env.example` добавить рекомендацию:
```bash
# Быстрая модель для низкой задержки (рекомендуется)
ASR_MODEL=tiny
# Точная модель для высокого качества (медленнее)
# ASR_MODEL=large-v3-turbo
```

### 4. Magic numbers в коде
**Проблема:** Пороги и параметры разбросаны по коду.

**Решение:** Создать `config.py`:
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class VADConfig:
    backend: str = "webrtc"
    aggressiveness: int = 2
    min_consec_frames: int = 5
    flatness_threshold: float = 0.60
    silero_threshold: float = 0.5
    silero_window_ms: int = 100

@dataclass
class ASRConfig:
    model: str = "tiny"
    language: str = "ru"
    device: Optional[str] = None
    compute_type: Optional[str] = None

@dataclass
class ThesisConfig:
    match_threshold: float = 0.6
    semantic_threshold: float = 0.55
    gemini_min_conf: float = 0.60
    repeat_interval_sec: float = 10.0
    autogen_batch_size: int = 3

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    frame_ms: int = 20
    channels: int = 1
    min_segment_ms: int = 500
    max_silence_ms: int = 400
    pre_roll_ms: int = 160
```

---

## 🟢 Средний приоритет

### 5. UX улучшения веб-интерфейса

**Добавить:**
- Визуальную индикацию текущего тезиса
- Прогресс-бар покрытия тезиса
- Историю диалога
- Кнопки управления (пауза, пропустить тезис, регулировка громкости)

**Пример `index.html` улучшений:**
```html
<div id="thesis-panel">
  <h3>Текущий тезис:</h3>
  <p id="current-thesis"></p>
  <div class="progress-bar">
    <div id="thesis-progress" style="width: 0%"></div>
  </div>
  <button id="skip-thesis">Пропустить</button>
</div>

<div id="history">
  <h3>История:</h3>
  <ul id="conversation-log"></ul>
</div>
```

### 6. Обработка ошибок
**Проблема:** Много общих `except Exception` блоков.

**Решение:** Создать кастомные исключения:
```python
# exceptions.py
class CluelyProException(Exception):
    """Базовое исключение проекта"""
    pass

class AudioDeviceError(CluelyProException):
    """Ошибка работы с микрофоном/аудио"""
    pass

class ASRError(CluelyProException):
    """Ошибка транскрибации"""
    pass

class LLMError(CluelyProException):
    """Ошибка генерации LLM"""
    pass

class ThesisGenerationError(CluelyProException):
    """Ошибка генерации тезисов"""
    pass
```

### 7. Кэширование эмбеддингов
**Проблема:** Повторные вычисления эмбеддингов для одинаковых фраз.

**Решение:**
```python
from functools import lru_cache
import hashlib

class SemanticMatcher:
    @lru_cache(maxsize=128)
    def _get_embedding_cached(self, text_hash: str, text: str):
        return self._compute_embedding(text)
    
    def get_embedding(self, text: str):
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self._get_embedding_cached(text_hash, text)
```

### 8. Метрики и мониторинг
**Добавить сбор метрик:**
```python
# metrics.py
from dataclasses import dataclass
from datetime import datetime
from typing import List

@dataclass
class SessionMetrics:
    start_time: datetime
    end_time: datetime
    total_questions: int
    theses_generated: int
    theses_completed: int
    avg_thesis_time_sec: float
    asr_errors: int
    llm_errors: int

class MetricsCollector:
    def __init__(self):
        self.sessions: List[SessionMetrics] = []
    
    def save_to_json(self, path: str):
        # Сохранить метрики для анализа
        pass
```

---

## 🔵 Низкий приоритет

### 9. Тестирование
**Добавить:**
- Интеграционные тесты для end-to-end флоу
- Mock'и для внешних API (Gemini)
- Бенчмарки производительности

```python
# tests/test_e2e.py
def test_full_flow_with_mock_audio():
    # Тест полного цикла с синтетическим аудио
    pass

def test_thesis_generation_and_verification():
    # Тест генерации и проверки тезисов
    pass
```

### 10. Документация API
**Добавить docstrings в формате Google/NumPy:**
```python
def transcribe_np(self, audio: np.ndarray, sample_rate: int) -> Optional[str]:
    """Транскрибирует аудио в текст.
    
    Args:
        audio: Массив аудио данных (mono, float32)
        sample_rate: Частота дискретизации (Гц)
    
    Returns:
        Распознанный текст или None в случае ошибки
    
    Raises:
        ASRError: Если модель недоступна или данные некорректны
    
    Examples:
        >>> asr = FasterWhisperTranscriber(model_size="tiny")
        >>> text = asr.transcribe_np(audio_data, 16000)
        >>> print(text)
        'Привет, как дела?'
    """
```

### 11. CI/CD
**Добавить:**
- GitHub Actions для автотестов
- Pre-commit hooks (black, flake8, mypy)
- Автоматическая проверка на утечку ключей

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv && uv sync
      - run: uv run pytest
```

---

## 📊 Приоритизация

| Задача | Приоритет | Сложность | Влияние |
|--------|-----------|-----------|---------|
| Ротация API ключа | 🔴 Критичный | Низкая | Безопасность |
| Разбить live_recognizer.py | 🟡 Высокий | Средняя | Поддержка кода |
| Оптимизация ASR модели | 🟡 Высокий | Низкая | UX (латентность) |
| Config файл | 🟡 Высокий | Низкая | Поддержка |
| UX веб-интерфейса | 🟢 Средний | Средняя | UX |
| Кастомные исключения | 🟢 Средний | Низкая | Отладка |
| Кэширование | 🟢 Средний | Средняя | Производительность |
| Метрики | 🔵 Низкий | Средняя | Аналитика |
| Тесты | 🔵 Низкий | Высокая | Надёжность |
| CI/CD | 🔵 Низкий | Средняя | DevOps |

---

## 🎯 Quick Wins (быстрые победы)

Что можно сделать за 30 минут:

1. ✅ Исправлен `.gitignore`
2. Изменить дефолтную ASR модель на `tiny` в `main.py` (строка 468)
3. Добавить `.env.example` с комментариями
4. Создать простой `config.py` с основными параметрами
5. Добавить метку версии в README

---

## 💡 Идеи для новых фич

1. **Режим практики**: записывать сессии и давать feedback по времени ответа, полноте раскрытия
2. **Адаптивные тезисы**: подстраивать сложность под производительность
3. **Мультиязычность**: поддержка английского для практики интервью
4. **Экспорт в PDF**: генерировать отчёт о сессии с тезисами и транскриптами
5. **Мобильное приложение**: React Native обёртка над веб-версией

---

## 📝 Заключение

**Проект в целом хорошо структурирован**, основные фичи работают. Главные точки роста:
- Безопасность (ротация ключа)
- Модульность кода
- UX веб-интерфейса
- Производительность в режиме реального времени

Рекомендую начать с **Quick Wins** и **критичных задач**, затем последовательно двигаться по приоритетам.
