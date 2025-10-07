# 🚀 Настройка Google Cloud Text-to-Speech

## Зачем?

Google TTS дает:
- ⚡ **Ускорение в 5-10 раз** (300-500мс вместо 2-5 сек на Silero)
- 🎤 **Отличное качество голоса** (Wavenet)
- 🔄 **Streaming** (можно прерывать озвучку)
- 💰 **Копеечная цена** (~$0.004 за тезис)

---

## 📦 Шаг 1: Установка библиотеки

```bash
pip install google-cloud-texttospeech
```

---

## 🔑 Шаг 2: Получение API ключа (Service Account)

### Вариант A: Через Google Cloud Console (рекомендуется)

1. Перейдите: https://console.cloud.google.com/
2. Создайте новый проект (или выберите существующий)
3. Включите **Cloud Text-to-Speech API**:
   - https://console.cloud.google.com/apis/library/texttospeech.googleapis.com
   - Нажмите "ENABLE"
4. Создайте Service Account:
   - https://console.cloud.google.com/iam-admin/serviceaccounts
   - Нажмите "CREATE SERVICE ACCOUNT"
   - Имя: `cluely-tts` (любое)
   - Роль: **Cloud Text-to-Speech User** (минимальная)
5. Создайте ключ:
   - Нажмите на созданный аккаунт
   - Вкладка "KEYS" → "ADD KEY" → "Create new key"
   - Тип: JSON
   - Скачайте файл (например, `cluely-tts-key.json`)

### Вариант B: Через gcloud CLI

```bash
# Установите gcloud CLI (если еще нет)
# https://cloud.google.com/sdk/docs/install

# Аутентификация
gcloud auth application-default login

# ИЛИ создайте service account
gcloud iam service-accounts create cluely-tts \
    --display-name="CluelyPro TTS"

gcloud iam service-accounts keys create ~/cluely-tts-key.json \
    --iam-account=cluely-tts@PROJECT_ID.iam.gserviceaccount.com

# Выдайте права
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:cluely-tts@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudtexttospeech.user"
```

---

## ⚙️ Шаг 3: Настройка переменной окружения

### Linux/macOS

```bash
# Добавьте в ~/.bashrc или ~/.zshrc
export GOOGLE_APPLICATION_CREDENTIALS="/полный/путь/к/cluely-tts-key.json"

# Перезагрузите shell
source ~/.bashrc  # или source ~/.zshrc
```

### Windows (PowerShell)

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\cluely-tts-key.json"

# Постоянно (для всех сессий)
[System.Environment]::SetEnvironmentVariable(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "C:\path\to\cluely-tts-key.json",
    "User"
)
```

### В .env файле (для проекта)

```bash
# Добавьте в .env
GOOGLE_APPLICATION_CREDENTIALS="/полный/путь/к/cluely-tts-key.json"
```

---

## ✅ Шаг 4: Проверка

```python
# Тест в Python
from google.cloud import texttospeech

client = texttospeech.TextToSpeechClient()
print("✅ Google TTS работает!")
```

Если ошибок нет → всё настроено правильно!

---

## 🚀 Запуск CluelyPro

Теперь при запуске CluelyPro автоматически использует Google TTS:

```bash
./run.sh
# ИЛИ
python3 main.py live
```

В логах увидите:
```
✅ Используется Google TTS (быстрый, качественный)
```

Если Google TTS недоступен, система автоматически переключится на Silero TTS:
```
⚠️ Используется Silero TTS (медленный, fallback)
```

---

## 💰 Цены (очень дешево!)

- **Wavenet голоса**: $16 за 1M символов
- **Standard голоса**: $4 за 1M символов (мы используем Wavenet)

**Пример:**
- 1 тезис = ~100 символов
- 1000 тезисов = $0.0016 (менее 2 центов!)
- Первые 1M символов в месяц = **БЕСПЛАТНО** ✨

Подробнее: https://cloud.google.com/text-to-speech/pricing

---

## 🎤 Доступные голоса для русского

В коде используется `ru-RU-Wavenet-D` (мужской, выразительный).

Другие голоса:
- `ru-RU-Wavenet-A` - женский, естественный
- `ru-RU-Wavenet-B` - мужской, естественный
- `ru-RU-Wavenet-C` - женский, выразительный
- `ru-RU-Wavenet-E` - женский, спокойный

Чтобы изменить голос, отредактируйте `live_recognizer.py`:
```python
self._tts = GoogleTTS(
    voice_name="ru-RU-Wavenet-A",  # Измените здесь
    speaking_rate=1.0,
)
```

---

## 🔧 Решение проблем

### Ошибка: "google.auth.exceptions.DefaultCredentialsError"

**Причина:** Переменная `GOOGLE_APPLICATION_CREDENTIALS` не настроена.

**Решение:**
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/полный/путь/к/ключу.json"
```

### Ошибка: "PERMISSION_DENIED: Cloud Text-to-Speech API has not been used"

**Причина:** API не включен в проекте.

**Решение:** Включите API:
https://console.cloud.google.com/apis/library/texttospeech.googleapis.com

### Ошибка: "Service account does not have required permissions"

**Причина:** У service account нет прав.

**Решение:** Выдайте роль `Cloud Text-to-Speech User`:
```bash
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:EMAIL" \
    --role="roles/cloudtexttospeech.user"
```

---

## 🎯 Итог

После настройки Google TTS ваша система будет:
- ⚡ **В 5-10 раз быстрее** озвучивать тезисы
- 🎤 **Качественнее** звучать
- 💰 **Копеечная** цена использования

**Настройка занимает 5-10 минут, но ускоряет систему навсегда!** 🚀
