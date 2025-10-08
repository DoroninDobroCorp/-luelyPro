# CluelyPro - Анализ Производительности

> **Автор:** AI Performance Analysis  
> **Дата:** 2024  
> **Проект:** CluelyPro - Голосовой Ассистент для Экзаменов

---

## 📋 НАВИГАЦИЯ ПО ДОКУМЕНТУ

### ⚡ БЫСТРЫЙ СТАРТ:
🔥 **[СВОДНАЯ ТАБЛИЦА ВСЕХ ОПТИМИЗАЦИЙ](./OPTIMIZATION_TABLE.md)** 🔥
> Все 23 оптимизации с кодами, статусами и оценками в одном файле

### Основные компоненты:
- [1. ASR - Распознавание речи](#1-asr---распознавание-речи) | Оптимизации: **1A-1E**
- [2. TTS - Синтез речи](#2-tts---синтез-речи) | Оптимизации: **2A-2D**
- [3. LLM - Языковая модель](#3-llm---языковая-модель) | Оптимизации: **3A-3D**
- [4. VAD - Детектор активности](#4-vad---детектор-активности) | Оптимизации: -
- [5. Voice Embeddings - Верификация](#5-voice-embeddings---верификация-голоса) | Оптимизации: **5A**

### Дополнительные аспекты:
- [6. Общая архитектура](#6-общая-архитектура) | Оптимизации: **6A-6D**
- [7. Специфика экзаменов](#7-специфика-экзаменов) | Оптимизации: **7A-7C**
- [8. Профилирование и мониторинг](#8-профилирование-и-мониторинг) | Оптимизации: **8A-8C**

### Сводные разделы:
- [📊 Таблица всех оптимизаций](#-таблица-всех-оптимизаций-по-приоритетам)
- [⚡ Ожидаемое ускорение](#-ожидаемое-ускорение)
- [🎯 Быстрые wins](#-быстрые-wins-можно-сделать-прямо-сейчас)
- [📈 Долгосрочные улучшения](#-долгосрочные-улучшения)

---

## 🎯 ОСНОВНЫЕ КОМПОНЕНТЫ

### 1. ASR - Распознавание речи
**OpenAI Whisper API (production)**

#### Текущее состояние:
- **Используется:** OpenAI Whisper API (whisper-1)
- **Альтернатива:** faster-whisper для локальной обработки
- Модели faster-whisper: tiny/base/small/medium/large-v3/large-v3-turbo
- По умолчанию (если OpenAI недоступен): `small`
- Поддержка CPU/CUDA с автофаллбэком
- 2 параллельных воркера (`ASR_WORKERS=2`)
- Файл: `asr_transcriber.py`

#### Метрики:
- **OpenAI API:** 200-400мс (RTF ~0.66x) ⭐ ИСПОЛЬЗУЕТСЯ
- **faster-whisper small:** 500-1500мс (RTF ~0.70x)
- **faster-whisper tiny:** 200-400мс (RTF ~0.30x)
- Загрузка модели: **2-5 секунд** (только для локальных)
- CUDA cuDNN issues → fallback на CPU (3-5x медленнее)

#### Вклад в латентность: 
⚡⚡⚡ **КРИТИЧНЫЙ** (30-40% общего времени)

---

### 2. TTS - Синтез речи
**3 варианта: Silero (default), OpenAI, Google**

#### Текущее состояние:
- **Silero TTS** (по умолчанию): 2-5 сек/тезис, offline, CPU-only
- **OpenAI TTS** (tts-1): 300-800мс/тезис, требует API key
- **Google TTS** (Wavenet-D): 300-500мс/тезис, требует service account ⭐ ИСПОЛЬЗУЕТСЯ
- Переключение через `USE_TTS_ENGINE=openai|google|silero`
- Файлы: `tts_silero.py`, `tts_openai.py`, `tts_google.py`

#### Метрики:
- Silero: **2-5 секунд** на 10-15 слов (медленно!)
- OpenAI: **300-800мс** + сетевая задержка
- Google: **300-500мс** + сетевая задержка ⭐ ИСПОЛЬЗУЕТСЯ

#### Вклад в латентность:
⚡⚡⚡ **КРИТИЧНЫЙ** (40-50% времени до озвучки)

---

### 3. LLM - Языковая модель
**Google Gemini Flash + OpenAI fallback**

#### Текущее состояние:
- Модель тезисов: `gemini-flash-lite-latest`
- Модель ответов: `gemini-flash-latest`
- Fallback: `gpt-4o-mini` (при 503 ошибках)
- Streaming для ответов (снижает TTFT)
- Файлы: `thesis_generator.py`, `llm_answer.py`

#### Метрики:
- Генерация тезисов: **300-800мс**
- Генерация ответа: **500-1500мс**
- 503 ошибки → OpenAI fallback (~+200мс)

#### Вклад в латентность:
⚡⚡ **СРЕДНИЙ** (20-30% общего времени)

---

### 4. VAD - Детектор активности
**WebRTC VAD / Silero VAD**

#### Текущее состояние:
- **WebRTC VAD** (по умолчанию): ~1-2мс/фрейм
- **Silero VAD** (опция): ~5-10мс/фрейм, но точнее
- Файл: `vad_silero.py`

#### Метрики:
- Обработка фрейма: **1-10мс**
- Spectral flatness check: +5-10мс

#### Вклад в латентность:
⚡ **НИЗКИЙ** (<5% общего времени)

#### Примечание:
VAD уже достаточно быстрый, оптимизации не требуются.

---

### 5. Voice Embeddings - Верификация голоса
**ECAPA-TDNN (pyannote.audio)**

#### Текущее состояние:
- Модель: `speechbrain/spkrec-ecapa-voxceleb`
- Определяет свой/чужой голос
- Работает на CPU/CUDA
- Файл: `live_recognizer.py` (класс `VoiceProfile`)

#### Метрики:
- Генерация эмбеддинга: **50-150мс/сегмент**
- Загрузка модели: **3-5 секунд**

#### Вклад в латентность:
⚡ **НИЗКИЙ** (5-10% времени обработки сегмента)

---

## 💡 ОПТИМИЗАЦИИ ПО КОМПОНЕНТАМ

### 1E. Облачный ASR (OpenAI Whisper API) ✅ РЕАЛИЗОВАНО
**Компонент:** 1. ASR  
**Статус:** ✅ РЕАЛИЗОВАНО  
**Ускорение:** 2-3x + качество выше  
**Сложность:** Средняя (1 день)

**Решение: OpenAI Whisper API**

```python
# asr_transcriber.py - уже реализовано!
class FasterWhisperTranscriber:
    def __init__(self, ...):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ASRError("OPENAI_API_KEY not found")
        
        self.client = OpenAI(api_key=api_key)
```

**Характеристики OpenAI Whisper API:**
- ⚡ **Скорость:** 200-400мс (RTF ~0.66x) - в 1.5 раза быстрее реального времени!
- 🎯 **Качество:** Используют large-v3 (выше чем local small)
- 💰 **Цена:** $0.006 за минуту аудио (~$0.36 за час интервью)
- 🌐 **Интернет:** Обязателен
- 🔒 **Приватность:** Данные отправляются в OpenAI

**Результат:** ⭐ **ИСПОЛЬЗУЕТСЯ В ПРОДЕ** - работает отлично!

---

### 2A. Переключиться на OpenAI TTS как основной
**Компонент:** 2. TTS  
**Приоритет:** 🔴 КРИТИЧНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** 3-5x  
**Сложность:** Низкая (5 минут настройки)

```bash
# .env
USE_TTS_ENGINE=openai  # сейчас: google
OPENAI_API_KEY=sk-proj-...  # уже есть!
TTS_SPEED=1.35
```

**Плюсы:**
- **В 3-5 раз быстрее** Silero (300-800мс vs 2-5сек)
- Отличное качество (почти как Google Wavenet)
- Простая интеграция (уже реализована в `tts_openai.py`)
- Дешево: ~$0.0015 за тезис ($15/1M символов)

**Минусы:**
- Требует API ключ и интернет (уже есть!)
- Платно (но очень дешево)

**Текущее состояние:** Используется Google TTS (тоже быстрый!)

---

### 2B. Параллельная генерация TTS для тезисов
**Компонент:** 2. TTS  
**Приоритет:** 🔴 КРИТИЧНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** 40-50% времени ожидания  
**Сложность:** Средняя (1 день)

```python
from concurrent.futures import ThreadPoolExecutor

class ThesisManager:
    def start_new_question(self, question, theses, context=None):
        # Запускаем параллельную генерацию аудио для всех тезисов
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Генерируем аудио для первых 3 тезисов параллельно
            audio_futures = [
                executor.submit(self._tts.synth, thesis) 
                for thesis in theses[:3]
            ]
            
            # Сохраняем futures для последовательного воспроизведения
            self._audio_cache = {
                i: future for i, future in enumerate(audio_futures)
            }
        
        # ... остальная логика
    
    def _speak_text(self, text, generation=None):
        # Проверяем кэш готового аудио
        idx = self.current_idx
        if idx in self._audio_cache:
            audio_np = self._audio_cache[idx].result()  # Получаем готовый результат
            self._play_audio(audio_np)
        else:
            # Fallback на синхронную генерацию
            audio_np = self._tts.synth(text)
            self._play_audio(audio_np)
```

**Применимость:** Особенно эффективно с OpenAI/Google TTS (сетевые запросы параллелятся)

---

### 3A. Уменьшить max_output_tokens для тезисов
**Компонент:** 3. LLM  
**Приоритет:** 🟡 ВАЖНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** 20-30%  
**Сложность:** Низкая (1 минута)

```python
# thesis_generator.py, строка ~23
@dataclass
class ThesisGenConfig:
    model_id: str = "gemini-flash-lite-latest"
    max_output_tokens: int = 180  # было 256 - избыточно для 5 тезисов
    temperature: float = 0.3
    n_theses: int = 8
    language: str = "ru"
```

**Обоснование:**
- 5 тезисов × 10-15 слов = ~75-100 слов
- 100 слов × 1.3 (токенов на слово RU) = ~130 токенов
- 180 токенов с запасом (было 256 - избыточно)

**Эффект:** Gemini генерирует меньше токенов → быстрее завершение

---

### 3B. Найти самую быструю LLM модель
**Компонент:** 3. LLM  
**Приоритет:** 🟡 ВАЖНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** 20-30%  
**Сложность:** Низкая (1 минута)

**Проблема:**
- Текущая модель (`gemini-flash-lite-latest`) не самая быстрая
- Новые модели выходят постоянно, конкретные рекомендации быстро устаревают

**Решение:**
```python
# thesis_generator.py
@dataclass
class ThesisGenConfig:
    # Регулярно проверяйте документацию Google AI для самой быстрой модели:
    # https://ai.google.dev/gemini-api/docs/models
    model_id: str = "найти-самую-быструю-модель"
    # Критерии выбора:
    # 1. Latency < 500мс (скорость важнее качества)
    # 2. Поддержка русского языка
    # 3. Output до 200 токенов
```

**Плюсы:**
- Можно получить 20-30% ускорения
- Гибкость при появлении новых моделей

**Минусы:**
- Требует регулярного мониторинга новых релизов
- Нужно тестировать качество на русском языке

---

### 3C. Параллельная генерация тезисов и прогрев TTS
**Компонент:** 3. LLM + 2. TTS  
**Приоритет:** 🟡 ВАЖНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** 15-25%  
**Сложность:** Средняя (3-4 часа)

```python
def _handle_foreign_text(self, text):
    # Запускаем генерацию тезисов в отдельном потоке
    import threading
    
    theses_result = []
    def generate_theses():
        theses_result.extend(
            self._thesis_generator.generate(text, n=5, language="ru")
        )
    
    thesis_thread = threading.Thread(target=generate_theses, daemon=True)
    thesis_thread.start()
    
    # Пока генерируются тезисы - прогреваем TTS (dummy запрос)
    if self._tts and not self._tts_warmed:
        self._tts.synth("прогрев")  # Быстрый прогрев модели
        self._tts_warmed = True
    
    # Ждём тезисы
    thesis_thread.join(timeout=2.0)
    
    # ... дальше работа с theses_result
```

**Применимость:** Полезно на первом запросе (прогрев) и при медленной сети

---

### 5A. Изучить более быстрые технологии для embeddings
**Компонент:** 5. Voice Embeddings  
**Приоритет:** 🟡 ВАЖНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** TBD (зависит от найденной технологии)  
**Сложность:** Средняя (исследование + внедрение)

**Текущее состояние:**
- ECAPA-TDNN: 50-150мс/сегмент
- Загрузка модели: 3-5 секунд

**Направления исследований:**
1. Более легкие модели (меньше параметров)
2. Квантизация ECAPA-TDNN (int8/fp16)
3. Альтернативные архитектуры (x-vectors, ResNet-based)
4. Кэширование embeddings (см. старый 5A)

**Цель:** Сократить время генерации эмбеддинга до 20-50мс

---

## 6. Общая архитектура

### 6A. Pipeline parallelism
**Приоритет:** 🟡 ВАЖНО  
**Статус:** ⏳ В ОЖИДАНИИ  
**Ускорение:** 15-25%  
**Сложность:** Высокая (3-5 дней)

```python
# Сейчас: ASR → Gemini → TTS (последовательно)
# Можно: ASR → [Gemini + TTS предгрев] параллельно

def _handle_foreign_text_parallel(self, text):
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Запускаем параллельно:
        # 1. Генерация тезисов через Gemini
        theses_future = executor.submit(
            self._thesis_generator.generate, text, n=5
        )
        
        # 2. Прогрев TTS (dummy запрос)
        tts_warmup_future = executor.submit(
            self._tts.synth, "test"
        )
        
        # 3. Обновление контекста диалога
        context_future = executor.submit(
            self._update_dialogue_context, text
        )
        
        # Ждём результаты
        theses = theses_future.result()
        tts_warmup_future.result()  # Игнорируем результат
        context_future.result()
    
    # Дальше озвучка тезисов...
```

**Схема:**
```
[Аудио сегмент] → [ASR] → [Текст вопроса]
                              ↓
                    ┌─────────┼─────────┐
                    ↓         ↓         ↓
              [Gemini]  [TTS warm] [Context]  ← Параллельно
                    ↓         ↓         ↓
                    └─────────┼─────────┘
                              ↓
                      [TTS озвучка]
```

---

### 6B. Агрессивное прерывание старых задач
**Приоритет:** 🟡 ВАЖНО  
**Статус:** ⏳ В ОЖИДАНИИ  
**Ускорение:** Снижает latency при новых вопросах  
**Сложность:** Средняя (1-2 дня)

**⚠️ ВАЖНОЕ ТРЕБОВАНИЕ:** Прерывание должно срабатывать только когда есть **непустой ответ от ИИ**!

```python
# Уже реализовано через _tts_generation counter
# Можно усилить:

class LiveVoiceVerifier:
    def _interrupt_all_processing(self):
        """Прерывает все активные задачи при новом вопросе с непустым ответом"""
        # 1. Прерываем TTS
        self._tts_generation += 1
        self._tts_interrupt.set()
        
        # 2. Прерываем ASR воркеры (новое!)
        for worker in self._segment_workers:
            # Добавить флаг прерывания для каждого воркера
            worker.interrupt_flag.set()
        
        # 3. Очищаем очередь сегментов (новое!)
        while not self._segment_queue.empty():
            try:
                self._segment_queue.get_nowait()
                self._segment_queue.task_done()
            except queue.Empty:
                break
        
        # 4. Прерываем Gemini API (через timeout)
        # Уже реализовано через fallback на OpenAI при 503
```

**Применимость:** Критично для быстрой реакции на новые вопросы

---

### 6C. Оптимизация очереди сегментов
**Приоритет:** 🟢 ПОЛЕЗНО  
**Статус:** ⏳ В ОЖИДАНИИ  
**Ускорение:** Снижает memory + latency при перегрузке  
**Сложность:** Низкая (5 минут)

```python
# live_recognizer.py, строка ~519
# Было: maxsize=20 (избыточно для 2 воркеров)
self._segment_queue = queue.Queue(maxsize=8)  # 3-4 сегмента на воркера

# Также уменьшить количество воркеров если CPU слабый:
self._num_asr_workers = int(os.getenv("ASR_WORKERS", "2"))  # Было правильно
```

**Обоснование:**
- 2 воркера × 3 сегмента в обработке = 6
- +2 в очереди = 8 (вместо 20)

---

### 6D. Lazy loading тяжелых зависимостей
**Приоритет:** 🟢 ПОЛЕЗНО  
**Статус:** ⏳ В ОЖИДАНИИ  
**Ускорение:** Быстрый старт приложения  
**Сложность:** Средняя (1-2 дня)

**⚠️ ВАЖНОЕ ЗАМЕЧАНИЕ:** Лучше долго загружаться, но потом быстро работать без пауз!

```python
# Уже частично реализовано (try/except импорты)
# Можно усилить:

class LiveVoiceVerifier:
    def __init__(self):
        # НЕ загружаем тяжёлые модели сразу
        self._asr = None  # Lazy load
        self._llm = None  # Lazy load
        self._tts = None  # Lazy load
        self._embedder = None  # Lazy load
    
    def _ensure_asr(self):
        """Загружаем ASR только при первом использовании"""
        if self._asr is None:
            self._asr = FasterWhisperTranscriber(...)
        return self._asr
    
    # Аналогично для остальных компонентов
```

**Дополнительно:**
- Кэшировать модели между запусками через `model.save_pretrained()`
- Использовать HuggingFace cache для offline режима

**Примечание:** Это может замедлить первый запрос, но ускорить старт приложения.

---

## 8. Профилирование и мониторинг

### 8A. Подробные метрики времени
**Приоритет:** 🟡 ВАЖНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** Помогает выявить узкие места  
**Сложность:** Низкая (2-3 часа)

```python
import time
from dataclasses import dataclass
from typing import Dict

@dataclass
class PerformanceMetrics:
    asr_time_ms: float = 0.0
    gemini_time_ms: float = 0.0
    tts_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    def to_dict(self):
        return {
            "asr": self.asr_time_ms,
            "gemini": self.gemini_time_ms,
            "tts": self.tts_time_ms,
            "embedding": self.embedding_time_ms,
            "total": self.total_time_ms,
        }

class LiveVoiceVerifier:
    def __init__(self):
        self.metrics = PerformanceMetrics()
    
    def _handle_foreign_text(self, text):
        t_start = time.time()
        
        # ASR (уже измеряется)
        asr_start = time.time()
        # ... ASR code
        self.metrics.asr_time_ms = (time.time() - asr_start) * 1000
        
        # Gemini
        gemini_start = time.time()
        theses = self._thesis_generator.generate(text, ...)
        self.metrics.gemini_time_ms = (time.time() - gemini_start) * 1000
        
        # TTS
        tts_start = time.time()
        self._speak_text(theses[0])
        self.metrics.tts_time_ms = (time.time() - tts_start) * 1000
        
        # Total
        self.metrics.total_time_ms = (time.time() - t_start) * 1000
        
        # Логируем
        logger.info(f"⏱️  Metrics: {self.metrics.to_dict()}")
        
        # Опционально: отправляем в Prometheus/Grafana
        # prometheus_client.metrics.observe(self.metrics.total_time_ms)
```

**Дополнительно:**
- Экспорт метрик в Prometheus для визуализации
- Алерты при превышении порогов (например, >5 сек total_time)

---

### 8B. Использовать async/await вместо threading
**Приоритет:** 🟢 ПОЛЕЗНО  
**Статус:** 🔄 В РАБОТЕ  
**Ускорение:** 10-15%  
**Сложность:** Высокая (1-2 недели рефакторинга)

```python
import asyncio
import aiohttp

class AsyncLiveVoiceVerifier:
    async def _handle_foreign_text_async(self, text):
        # Параллельный запуск через asyncio
        tasks = [
            self._asr_async(audio),
            self._thesis_generator_async(text),
            self._update_context_async(text),
        ]
        
        asr_result, theses, _ = await asyncio.gather(*tasks)
        
        # Озвучка тезисов
        for thesis in theses:
            await self._tts_async(thesis)
    
    async def _tts_async(self, text):
        """Асинхронный TTS через streaming API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": "tts-1", "input": text, "voice": "onyx"}
            ) as resp:
                audio_bytes = await resp.read()
                return self._decode_audio(audio_bytes)
```

**Плюсы:**
- Меньше overhead чем threading (~10-15%)
- Лучше масштабируется (тысячи задач vs сотни потоков)
- Современный подход

**Минусы:**
- Требует полного рефакторинга кодабазы
- Сложнее отладка

---

## ⚡ ОЖИДАЕМОЕ УСКОРЕНИЕ

### Текущие тайминги (с OpenAI ASR):

**Вопрос → Первый тезис: ~1.5-3 секунды**
```
ASR:    0.2-0.4с  (OpenAI API, RTF 0.66x) ✅
Gemini: 0.3-0.8с  (генерация тезисов)
TTS:    0.3-0.5с  (Google TTS) ✅
────────────────
ИТОГО:  0.8-1.7с  → УЖЕ БЫСТРО! ⚡⚡
```

---

### После критичных оптимизаций (2B+3A+3B+3C):

**Вопрос → Первый тезис: ~0.6-1.2 секунды**
```
ASR:    0.2-0.4с  (OpenAI, без изменений)
Gemini: 0.2-0.5с  (max_tokens↓, быстрая модель)
TTS:    0.2-0.3с  (Google + параллельная генерация!)
────────────────
ИТОГО:  0.6-1.2с  → ускорение ещё на 30-40% ⚡⚡⚡
```

---

### После долгосрочных (+ 8B + 6A):

**Вопрос → Первый тезис: ~0.5-1.0 секунд**
```
Параллельно: ASR + Gemini + TTS warmup
Async/await: -10-15% overhead
────────────────
ИТОГО:       0.5-1.0с  → почти мгновенно! ⚡⚡⚡⚡
```

---

### Итоговая таблица:

| Конфигурация | Время (сек) | Ускорение vs старый baseline | Что сделано |
|--------------|-------------|-------------------------------|-------------|
| **Baseline старый** | 3.0-6.0 | 1x | Silero TTS + faster-whisper small |
| **Текущий (OpenAI ASR + Google TTS)** | 0.8-1.7 | **2-4x** | ✅ OpenAI ASR + Google TTS |
| **+ Критичные оптимизации** | 0.6-1.2 | **3-5x** | +2B+3A+3B+3C (1 неделя) |
| **+ Pipeline + async** | 0.5-1.0 | **4-6x** | +6A+8B (1 месяц) |

---

## 📊 АКТУАЛЬНАЯ СТАТИСТИКА

### Что уже реализовано:

✅ **1E - OpenAI Whisper API** (2-3x быстрее + качество выше!)  
✅ **1D - Квантизация int8** (частично для CPU)  
✅ **2D - tts-1 модель** (Google TTS используется)  
✅ **3D - thinking_budget=0** (отключен)  
✅ **MIN_SEGMENT_MS = 1500ms** (фильтр коротких шумов)

### В работе (7 задач):

🔄 **2A** - OpenAI TTS вместо Google (альтернатива)  
🔄 **2B** - Параллельная TTS генерация  
🔄 **3A** - max_tokens → 180  
🔄 **3B** - Найти быструю LLM модель  
🔄 **3C** - Параллельная ген+прогрев  
🔄 **5A** - Изучить быстрые технологии (embeddings)  
🔄 **8A** - Подробные метрики  
🔄 **8B** - Async/await рефакторинг

### В ожидании (4 задачи):

⏳ **6A** - Pipeline parallelism  
⏳ **6B** - Агрессивное прерывание (⚠️ только когда есть непустой ответ!)  
⏳ **6C** - Оптимизация очереди  
⏳ **6D** - Lazy loading (⚠️ лучше долго загружаться, но быстро работать!)

### Отклонено (8 задач):

❌ **1A, 1B, 1C** - ASR варианты (не нужны с OpenAI)  
❌ **2C** - Кэш TTS (нет пользы в реальных условиях)  
❌ **7A, 7B, 7C** - Специфика экзаменов (решение владельца)  
❌ **8C** - GPU profiling (CPU у нас)

---

## 🎯 РЕКОМЕНДУЕМЫЙ ПЛАН ДЕЙСТВИЙ

### Фаза 1: Критичные оптимизации (1 неделя)
- [ ] **2B**: Параллельная генерация TTS для тезисов
- [ ] **3A**: Уменьшить `max_output_tokens` до 180
- [ ] **3B**: Найти самую быструю LLM модель
- [ ] **3C**: Параллельная генерация + прогрев TTS
- [ ] **8A**: Добавить подробные метрики времени

**Результат:** Ускорение ещё на 30-40% (с текущих 0.8-1.7с до 0.6-1.2с) ⚡⚡⚡

---

### Фаза 2: Архитектурные улучшения (2-3 недели)
- [ ] **6A**: Pipeline parallelism (ASR + Gemini + TTS)
- [ ] **6B**: Усилить прерывание (только при непустом ответе!)
- [ ] **5A**: Внедрить более быстрые embeddings
- [ ] Нагрузочное тестирование

**Результат:** Ускорение до 0.5-1.0с ⚡⚡⚡⚡

---

### Фаза 3: Долгосрочные (1-2 месяца, опционально)
- [ ] **8B**: Рефакторинг на async/await
- [ ] **6C**, **6D**: Оптимизация очереди и lazy loading
- [ ] Интеграция с Prometheus/Grafana

**Результат:** Финальная оптимизация ⚡⚡⚡⚡⚡

---

**Последнее обновление:** 08.10.2025  
**Версия документа:** 3.0 (актуализирована с OpenAI ASR)  
**Статус:** ✅ Готов к использованию

---

## 📞 ОБСУЖДЕНИЕ ЗАДАЧ

Для обсуждения конкретной оптимизации используйте код:

**Примеры:**
- "Давай обсудим **2B** (Параллельная TTS)"
- "Реализуем **6A** (pipeline parallelism)"
- "Вопрос по **3A** (max_tokens)"

Это упростит коммуникацию! 🚀
