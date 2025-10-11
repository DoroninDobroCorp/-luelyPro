"""
Cerebras LLM модуль (ОЧЕНЬ БЫСТРЫЙ)
⚡ СКОРОСТЬ: 1800+ tokens/sec (в 5-10 раз быстрее Gemini/GPT)
🎯 КАЧЕСТВО: llama3.3-70b - отличное качество для русского
💰 ЦЕНА: Бесплатный tier + очень дешево

✅ ОПТИМИЗАЦИЯ 3B: Интеграция Cerebras как быстрой LLM модели
Cerebras в 5-10 раз быстрее Gemini Flash благодаря специализированному железу.
Используется для генерации тезисов и ответов. См. OPTIMIZATION_TABLE.md - код 3B
"""
from __future__ import annotations

import os
from typing import Optional, List

from loguru import logger

try:
    from openai import OpenAI  # Cerebras использует OpenAI-совместимый API
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False
    logger.warning("openai не установлен, Cerebras недоступен")


class CerebrasLLM:
    """
    Cerebras LLM wrapper (OpenAI-compatible API)
    
    Требует:
    1. pip install openai
    2. CEREBRAS_API_KEY в .env или переменной окружения
    
    Модели:
    - llama3.3-70b: ⭐ РЕКОМЕНДУЮ - самая быстрая, отличное качество
    - llama-3.1-70b: альтернатива
    
    Скорость: 1800+ tokens/sec (в 5-10 раз быстрее Gemini Flash!)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama3.3-70b",
        base_url: str = "https://api.cerebras.ai/v1",
        max_tokens: int = 256,
        temperature: float = 0.3,
    ):
        if not CEREBRAS_AVAILABLE:
            raise RuntimeError(
                "openai не установлен. Установите: pip install openai"
            )
        
        self.api_key = api_key or os.getenv("CEREBRAS_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "CEREBRAS_API_KEY не найден! "
                "Получите ключ на https://cloud.cerebras.ai/ и добавьте в .env"
            )
        
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Инициализируем клиент (OpenAI-compatible)
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
            logger.info(f"Cerebras LLM инициализирован: model={self.model}")
        except Exception as e:
            logger.error(f"Не удалось инициализировать Cerebras: {e}")
            raise
    
    def generate_theses(
        self,
        question: str,
        context: Optional[str] = None,
        n: int = 5,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """
        Генерация тезисов через Cerebras
        
        Returns:
            List[str]: Список тезисов (разделенных |||)
        """
        if not question or not question.strip():
            return []
        
        # Формируем промпт (аналогично ThesisGenerator)
        context_section = ""
        if context and context.strip():
            context_section = (
                f"📚 КОНТЕКСТ (предыдущие вопросы):\n{context.strip()}\n\n"
                f"⚠️ КОНТЕКСТ ТОЛЬКО для понимания местоимений!\n\n"
            )
        
        user_prompt = (
            f"{context_section}"
            f"❓ ТЕКУЩИЙ ВОПРОС (отвечай ТОЛЬКО на него!):\n{question.strip()}\n\n"
            f"ЗАДАЧА: Сгенерируй {n} тезисов.\n\n"
            f"⚠️⚠️⚠️ КРИТИЧНО: ОТВЕТ ВСЕГДА ПЕРВЫМ СЛОВОМ!\n\n"
            f"ПРИМЕРЫ ПРАВИЛЬНОЙ СТРУКТУРЫ:\n"
            f"Вопрос: 'Кто первым полетел в космос?'\n"
            f"✅ 'Юрий Гагарин - первый космонавт'\n"
            f"✅ 'Двенадцатого апреля тысяча девятьсот шестьдесят первого года - дата полёта'\n"
            f"✅ 'Восток один - название космического корабля'\n\n"
            f"Вопрос: 'Что такое фотосинтез?'\n"
            f"✅ 'Фотосинтез - процесс преобразования света в энергию растениями'\n"
            f"✅ 'Хлорофилл - зелёный пигмент обеспечивающий фотосинтез'\n"
            f"✅ 'Углекислый газ и вода - исходные вещества фотосинтеза'\n\n"
            f"ФОРМАТ: тезис1 ||| тезис2 ||| тезис3\n"
            f"ВСЕ ЦИФРЫ ПРОПИСЬЮ! ОТВЕТ ПЕРВЫМ! БЕЗ КОММЕНТАРИЕВ!"
        )
        
        if system_prompt is None:
            system_prompt = (
                "Ты генератор тезисов для устных экзаменов. "
                "СТРОГИЕ ПРАВИЛА:\n"
                " 0. ⚠️ ФИЛЬТР: ПУСТАЯ СТРОКА только для ЯВНЫХ не-вопросов:\n"
                "    ❌ Игнорируй (пустая строка):\n"
                "    • Команды системе: 'пиши разговорную речь', 'говори громче', 'остановись', 'пиши разговорную', 'говори быстрее', 'говори медленнее'\n"
                "    • Чистый бытовой разговор: 'спасибо', 'пока', 'привет', 'до свидания'\n"
                "    ✅ Отвечай на ВСЁ остальное (вопросы, просьбы, математику)\n"
                " 1. ⚠️⚠️⚠️ СТРУКТУРА ТЕЗИСА - ОТВЕТ ВСЕГДА ПЕРВЫМ СЛОВОМ:\n"
                "    - КТО? → ИМЯ первым: 'Юрий Гагарин - первый космонавт'\n"
                "    - КОГДА? → ДАТА первой: 'Двенадцатого апреля тысяча девятьсот шестьдесят первого года - полёт Гагарина'\n"
                "    - ЧТО? → КЛЮЧЕВОЕ СЛОВО первым: 'Кератин - основной компонент волос'\n"
                "    - ГДЕ? → МЕСТО первым: 'Париж - место расположения Эйфелевой башни'\n"
                "    - СКОЛЬКО? → ЧИСЛО первым: 'Двести двадцать пять - пятнадцать в квадрате'\n"
                "    - ❌ НЕ ПИШИ: 'Основной компонент волос - кератин' (неправильно!)\n"
                "    - ✅ ПИШИ: 'Кератин - основной компонент волос' (правильно!)\n"
                " 2. Каждый тезис - ПОЛНОЕ предложение (5-15 слов): ОТВЕТ - ПОЯСНЕНИЕ\n"
                " 3. ТОЛЬКО конкретные факты, без воды\n"
                " 4. ВСЕ ЦИФРЫ ПРОПИСЬЮ: '1961'→'тысяча девятьсот шестьдесят первый'\n"
                " 5. ФОРМАТ: тезис1 ||| тезис2 ||| тезис3\n"
            )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            text = response.choices[0].message.content or ""
            text = text.strip()
            
            # Парсим по разделителю |||
            if "|||" in text:
                theses = [t.strip() for t in text.split("|||") if t.strip()]
                return theses[:n]
            
            # Fallback: построчный парсинг
            lines = [ln.strip("-•* \t") for ln in text.splitlines()]
            theses = [ln for ln in lines if ln]
            return theses[:n]
            
        except Exception as e:
            logger.error(f"Cerebras theses generation error: {e}")
            return []
    
    def generate_answer(
        self,
        question: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[tuple[str, str]]] = None,
    ) -> str:
        """
        Генерация ответа через Cerebras
        
        Returns:
            str: Ответ на вопрос
        """
        if not question or not question.strip():
            return ""
        
        if system_prompt is None:
            system_prompt = (
                "⚠️ ОТВЕЧАЙ ТОЛЬКО ФАКТАМИ! НЕ отвечай мнениями, советами, отказами. "
                "Ты — ассистент на экзамене. Отвечай СТРОГО по формату:\n"
                "1) Сначала ТОЛЬКО факт (дата/имя/число/место) - одно короткое предложение\n"
                "2) Затем краткое пояснение (если нужно) - одно предложение\n\n"
                "⚠️ ВСЕ ЦИФРЫ И ГОДЫ ПИШИ ПРОПИСЬЮ! "
                "Не повторяй вопрос в ответе. "
                "Начинай СРАЗУ с факта: года, имени, места. "
                "Все числа ПРОПИСЬЮ (не цифрами). "
                "Никакой латиницы - только русская транслитерация."
            )
        
        # Формируем сообщения с историей
        messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            for role, text in history[-8:]:  # Последние 8 пар
                openai_role = "assistant" if role == "model" else "user"
                messages.append({"role": openai_role, "content": text})
        
        messages.append({"role": "user", "content": question.strip()})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            
            text = response.choices[0].message.content or ""
            return text.strip()
            
        except Exception as e:
            logger.error(f"Cerebras answer generation error: {e}")
            return ""


__all__ = ["CerebrasLLM", "CEREBRAS_AVAILABLE"]
