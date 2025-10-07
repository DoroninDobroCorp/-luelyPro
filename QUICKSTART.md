# 🚀 Быстрый Старт CluelyPro

## За 3 шага запустите систему!

### Шаг 1: Установите зависимости

```bash
pip install -r requirements.txt
```

### Шаг 2: Настройте API ключ Gemini

```bash
# Создайте файл .env
cp .env.example .env

# Получите ключ: https://aistudio.google.com/app/apikey
# Откройте .env и вставьте свой ключ:
nano .env  # или любой редактор
```

В файле `.env`:
```bash
GEMINI_API_KEY=ваш_ключ_здесь
```

### Шаг 3: Создайте голосовой профиль и запустите

```bash
# Создайте профиль (первый раз)
python3 main.py enroll

# Запустите систему
./run.sh
```

## ✅ Готово!

Система работает с **Silero TTS из коробки** - никакой дополнительной настройки!

---

## 🎤 Как работает:

1. **Ваш голос** → система фильтрует (не отвечает)
2. **Чужой голос** → система:
   - Распознает речь (ASR)
   - Генерирует тезисы через Gemini
   - Озвучивает каждый тезис **2 раза**
   - Всё работает **асинхронно** (микрофон не останавливается)

---

## ⚡ ОПЦИОНАЛЬНО: Ускорение с Google TTS (в 5-10 раз)

**Если хотите ускорить озвучку** (300-500мс вместо 2-5 сек):

1. Установите:
   ```bash
   pip install google-cloud-texttospeech
   ```

2. Настройте ключ (см. `GOOGLE_TTS_SETUP.md`)

3. Добавьте в `.env`:
   ```bash
   USE_GOOGLE_TTS=true
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
   ```

4. Запустите:
   ```bash
   ./run.sh
   ```

**НО ЭТО НЕ ОБЯЗАТЕЛЬНО!** Система работает без этого.

---

## 🧪 Тесты

```bash
# Базовые тесты
python3 tests/test_basic.py

# Тесты контекста диалога
python3 tests/test_dialogue_context.py
```

---

## 📋 Режимы ASR

```bash
./run.sh          # small модель (по умолчанию, баланс)
./run.sh test     # tiny модель (быстро, для теста)
./run.sh fast     # base модель (средне)
./run.sh quality  # large-v3 (максимальное качество)
```

---

## 🔧 Решение проблем

### Ошибка: "GEMINI_API_KEY не найден"

Создайте файл `.env` и добавьте ключ:
```bash
GEMINI_API_KEY=your_key_here
```

Получить ключ: https://aistudio.google.com/app/apikey

### Ошибка: "ModuleNotFoundError"

Установите зависимости:
```bash
pip install -r requirements.txt
```

### Ошибка: "No profile selected"

Создайте голосовой профиль:
```bash
python3 main.py enroll
```

---

## 📚 Дополнительно

- **Анализ стека**: `STACK_ANALYSIS.md`
- **Google TTS (опционально)**: `GOOGLE_TTS_SETUP.md`
- **Полная документация**: `README.md`

---

## 🎯 Всё готово!

Запустите:
```bash
./run.sh
```

Говорите в микрофон → система отвечает тезисами! ✨
