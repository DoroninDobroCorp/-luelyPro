# 🚀 Быстрый старт после улучшений

**Версия:** 0.1.1  
**Дата:** 2025-09-29

---

## ✅ Что было улучшено

1. ✅ **Безопасность**: `.env` теперь корректно игнорируется git
2. ✅ **Конфигурация**: Новый модуль `config.py` с централизованными настройками
3. ✅ **Обработка ошибок**: Модуль `exceptions.py` с 11 типами исключений
4. ✅ **Документация**: `.env.example` расширен до 120 строк с подробными комментариями

---

## ⚠️ КРИТИЧНО: Первые шаги

### 1. Ротировать API ключ

Текущий ключ в `.env` **попал в git** и должен быть заменён!

```bash
# Шаг 1: Получить новый ключ
open https://aistudio.google.com/app/apikey

# Шаг 2: Обновить .env с новым ключом
nano .env  # или vim .env

# Шаг 3: Удалить старый ключ из истории git
# ВАРИАНТ A: filter-repo (рекомендуется)
pip install git-filter-repo
git filter-repo --path .env --invert-paths --force

# ВАРИАНТ Б: BFG Repo Cleaner (проще)
# https://rtyley.github.io/bfg-repo-cleaner/
# java -jar bfg.jar --delete-files .env

# Шаг 4: Force push (ОСТОРОЖНО! Перепишет историю)
git push origin main --force
```

### 2. Проверить .gitignore

```bash
git status  # .env НЕ должен быть в списке изменений
cat .gitignore | grep "^\.env$"  # должно вывести: .env
```

---

## 📦 Установка и запуск

### Вариант 1: Обычный запуск (рекомендуется)

```bash
# 1. Скопировать пример конфига
cp .env.example .env

# 2. Заполнить .env своим API ключом
nano .env  # или vim .env
# Минимум: GEMINI_API_KEY=ваш_ключ

# 3. Установить зависимости (если ещё не установлены)
uv sync

# 4. Запустить ассистента
python main.py
# или через uv:
uv run python main.py
```

### Вариант 2: Быстрый тест (10 секунд)

```bash
# Дымовой тест с авто-остановкой
RUN_SECONDS=10 ASR_MODEL=tiny python main.py
```

---

## 🎛️ Основные настройки (.env)

### Минимальная конфигурация

```bash
# Только обязательный ключ
GEMINI_API_KEY=your_key_here
```

### Рекомендуемая конфигурация

```bash
# API ключ (обязательно)
GEMINI_API_KEY=your_key_here

# Быстрая модель ASR для низкой задержки
ASR_MODEL=tiny

# Интервал повтора тезиса (секунды)
THESIS_REPEAT_SEC=10

# Режимы
AI_ONLY_THESIS=1
COMMENTARY_MODE=0
```

### Полная конфигурация

См. подробности в `.env.example` (120 строк с комментариями)

---

## 🧪 Проверка работоспособности

### 1. Проверить новые модули

```bash
# В venv
.venv/bin/python -c "
from config import AppConfig
from exceptions import ASRError
cfg = AppConfig.from_env()
print(f'✓ ASR модель: {cfg.asr.model}')
print(f'✓ Порог: {cfg.verifier.distance_threshold}')
"
```

**Ожидаемый вывод:**
```
✓ ASR модель: tiny
✓ Порог: 0.75
```

### 2. Проверить CLI

```bash
python main.py --help
```

**Ожидаемый вывод:**
```
usage: diarization [-h] {enroll,live,test,profiles} ...
...
```

### 3. Запустить короткий тест

```bash
# Тест без аудио (извлечение тезисов из текста)
python main.py test --text "Расскажи о своём опыте работы с Python"
```

---

## 📖 Использование новых модулей

### config.py - Централизованная конфигурация

```python
from config import AppConfig, ASRConfig

# Загрузка из ENV
cfg = AppConfig.from_env()

# Доступ к настройкам
print(cfg.asr.model)              # "tiny"
print(cfg.vad.backend)            # "webrtc"
print(cfg.thesis.repeat_interval_sec)  # 10.0

# Создание кастомного конфига
asr = ASRConfig(model="small", language="en")
```

### exceptions.py - Специфичные исключения

```python
from exceptions import ASRError, LLMError, ProfileError

try:
    text = asr.transcribe(audio)
except ASRError as e:
    logger.error(f"Ошибка транскрибации: {e}")
    # Откат на fallback модель
except ProfileError as e:
    logger.error(f"Профиль повреждён: {e}")
    # Запросить создание нового профиля
```

---

## 🎯 Типичные сценарии

### Сценарий 1: Первый запуск

```bash
# 1. Настроить .env
cp .env.example .env
nano .env  # вставить GEMINI_API_KEY

# 2. Запустить
python main.py

# 3. Система предложит:
#    - Выбрать существующий профиль ИЛИ
#    - Создать новый (запись ~6 секунд)

# 4. Начать практику!
```

### Сценарий 2: Быстрая практика (без интерактива)

```bash
# Указать профиль явно
python main.py live --profile profiles/my_voice.npz --asr --llm

# Или с авто-созданием, если профиля нет
python main.py live --profile profiles/test.npz --auto-enroll-if-missing --asr --llm
```

### Сценарий 3: Веб-режим (удалённый микрофон)

```bash
# Запустить WebSocket сервер
./run.sh web

# Открыть в браузере
open http://127.0.0.1:8000/index.html
```

### Сценарий 4: Режим комментариев (эксперимент)

```bash
# Генерация коротких фактов даже без явных вопросов
COMMENTARY_MODE=1 THESIS_REPEAT_SEC=8 ./run_commentary.sh
```

---

## 🐛 Решение проблем

### Проблема: `.env` всё ещё отслеживается git

```bash
# Удалить из отслеживания
git rm --cached .env
git commit -m "Remove .env from tracking"

# Проверить
git status  # .env не должен быть виден
```

### Проблема: Модули config/exceptions не найдены

```bash
# Переустановить в editable режиме
pip install -e .

# Или через uv
uv sync
```

### Проблема: ASR слишком медленный

```bash
# В .env изменить модель
ASR_MODEL=tiny  # вместо large-v3-turbo

# Латентность: tiny ~0.5с, small ~1-2с, large ~3-5с
```

### Проблема: CUDA ошибки при ASR

```bash
# Принудительно использовать CPU
ASR_DEVICE=cpu python main.py live --asr --llm
```

---

## 📚 Документация

### Основные файлы

- **README.md** - Полная документация проекта
- **IMPROVEMENTS.md** - Список всех рекомендаций (приоритизировано)
- **CHANGES.md** - Выполненные улучшения с инструкциями
- **SUMMARY.md** - Итоговый отчёт по улучшениям
- **.env.example** - Документированный пример конфигурации
- **QUICK_START.md** (этот файл) - Быстрый старт

### Модули

- **config.py** - Централизованная конфигурация (8 классов настроек)
- **exceptions.py** - Кастомные исключения (11 типов)
- **live_recognizer.py** - Основной движок (VAD, ASR, LLM, TTS, тезисы)
- **main.py** - CLI точка входа

---

## 🎓 Дополнительно

### Примеры команд

```bash
# Список профилей
python main.py profiles list

# Удалить профиль
python main.py profiles delete --name my_voice

# Создать новый профиль
python main.py enroll --profile profiles/new_voice.npz --seconds 12

# Тест извлечения тезисов
python main.py test --text "Какие у вас сильные стороны?"

# Тест из файла
python main.py test --file tests/examples.txt --json
```

### Переменные окружения (полный список)

См. `.env.example` для подробностей. Основные:

- `GEMINI_API_KEY` - API ключ (обязательно)
- `ASR_MODEL` - Модель ASR (tiny|small|large-v3-turbo)
- `THESIS_REPEAT_SEC` - Интервал повтора (секунды)
- `AI_ONLY_THESIS` - Режим AI-only (0|1)
- `COMMENTARY_MODE` - Режим комментариев (0|1)
- `RUN_SECONDS` - Авто-остановка (0=бесконечно)

---

## 🆘 Поддержка

Если что-то не работает:

1. Проверить логи: `logs/assistant.log`
2. Убедиться, что `.env` настроен корректно
3. Проверить, что venv активирован: `.venv/bin/python`
4. Просмотреть IMPROVEMENTS.md для известных проблем

---

**Готово! Начинайте практику! 🎉**
