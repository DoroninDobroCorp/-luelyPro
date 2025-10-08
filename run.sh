#!/bin/bash
# Скрипт запуска CluelyPro

set -e  # Остановка при ошибке

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================${NC}"
echo -e "${BLUE}   CluelyPro - Голосой Ассистент    ${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Проверка .env файла
if [ ! -f .env ]; then
    echo -e "${RED}❌ Файл .env не найден!${NC}"
    echo -e "${YELLOW}Создайте .env файл на основе .env.example:${NC}"
    echo -e "   cp .env.example .env"
    echo -e "   # Затем отредактируйте .env и добавьте GEMINI_API_KEY"
    exit 1
fi

# Проверка GEMINI_API_KEY
if ! grep -q "GEMINI_API_KEY=.*[a-zA-Z0-9]" .env; then
    echo -e "${RED}❌ GEMINI_API_KEY не задан в .env файле!${NC}"
    echo -e "${YELLOW}Добавьте ключ в .env:${NC}"
    echo -e "   GEMINI_API_KEY=your_key_here"
    exit 1
fi

# Проверка виртуального окружения
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠️  Виртуальное окружение не найдено${NC}"
    echo -e "${BLUE}Создаю .venv...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Виртуальное окружение создано${NC}"
fi

# Активация виртуального окружения
echo -e "${BLUE}Активация виртуального окружения...${NC}"
source .venv/bin/activate

# Проверка зависимостей
if ! python -c "import faster_whisper" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Зависимости не установлены${NC}"
    echo -e "${BLUE}Установка зависимостей...${NC}"
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Зависимости установлены${NC}"
fi

# Настройка Google TTS (если есть credentials файл)
GOOGLE_CREDS="cluely-474414-7d78d7856935.json"
if [ -f "$GOOGLE_CREDS" ]; then
    export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/$GOOGLE_CREDS"
    echo -e "${GREEN}✓ Google TTS credentials настроены${NC}"
    
    # Проверка/установка google-cloud-texttospeech
    if ! python -c "from google.cloud import texttospeech" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  google-cloud-texttospeech не установлен${NC}"
        echo -e "${BLUE}Установка google-cloud-texttospeech...${NC}"
        
        # Убедимся что pip установлен в venv
        if ! python -m pip --version &>/dev/null; then
            echo -e "${BLUE}Установка pip...${NC}"
            python -m ensurepip --default-pip &>/dev/null || true
        fi
        
        python -m pip install -q google-cloud-texttospeech
        echo -e "${GREEN}✓ Google TTS установлен${NC}"
    else
        echo -e "${GREEN}✓ Google TTS уже установлен${NC}"
    fi
    
    # Использовать Google TTS по умолчанию (если не переопределено)
    if [ -z "$USE_TTS_ENGINE" ]; then
        export USE_TTS_ENGINE=google
    fi
else
    echo -e "${YELLOW}⚠️  Google TTS credentials не найдены (используется Silero TTS)${NC}"
fi

# Создание необходимых папок
mkdir -p profiles logs

# Проверка аргументов
MODE="${1:-default}"

case "$MODE" in
    "test")
        echo -e "${BLUE}🧪 Запуск в тестовом режиме (10 секунд, tiny модель)${NC}"
        echo ""
        RUN_SECONDS=10 ASR_MODEL=tiny python main.py
        ;;
    "fast")
        echo -e "${BLUE}⚡ Запуск в быстром режиме (модель base)${NC}"
        echo ""
        ASR_MODEL=base python main.py
        ;;
    "quality")
        echo -e "${BLUE}🎯 Запуск в режиме качества (модель large-v3-turbo)${NC}"
        echo ""
        ASR_MODEL=large-v3-turbo python main.py
        ;;
    "help"|"-h"|"--help")
        echo -e "${GREEN}Использование:${NC}"
        echo -e "  ./run.sh [режим]"
        echo ""
        echo -e "${GREEN}Режимы:${NC}"
        echo -e "  ${BLUE}default${NC}  - Обычный запуск (модель small)"
        echo -e "  ${BLUE}test${NC}     - Тестовый режим (10 сек, модель tiny)"
        echo -e "  ${BLUE}fast${NC}     - Быстрый режим (модель base)"
        echo -e "  ${BLUE}quality${NC}  - Режим качества (модель large-v3-turbo)"
        echo -e "  ${BLUE}help${NC}     - Показать эту справку"
        echo ""
        echo -e "${GREEN}Примеры:${NC}"
        echo -e "  ./run.sh              # Обычный запуск"
        echo -e "  ./run.sh test         # Быстрый тест"
        echo -e "  ./run.sh quality      # Лучшее качество ASR"
        exit 0
        ;;
    "default"|*)
        echo -e "${BLUE}🚀 Запуск в обычном режиме (модель small)${NC}"
        echo ""
        python main.py
        ;;
esac

echo ""
echo -e "${GREEN}✓ Завершено${NC}"
