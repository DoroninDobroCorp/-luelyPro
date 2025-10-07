from __future__ import annotations

from typing import Optional, List
import re
import os

from loguru import logger
from google import genai
from google.genai import types
from dotenv import load_dotenv

from config import LLMConfig as ConfigLLMConfig
from exceptions import LLMError

# OpenAI fallback (опционально)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None  # type: ignore

load_dotenv()


class LLMResponder:
    """
    Обёртка над Google GenAI (gemini) с синхронной генерацией коротких ответов.
    Использует стриминг (generate_content_stream), аккумулируя текст в строку.
    Интерфейс совместим с прежним LLMResponder: generate(user_text) -> str.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[ConfigLLMConfig] = None,
        enable_history: bool = True,
        history_max_turns: int = 8,
        use_openai_fallback: bool = True,  # Включить OpenAI fallback
    ) -> None:
        # Используем config из config.py
        cfg = config or ConfigLLMConfig()
        self.model_id = model_id or cfg.model_id
        self.max_new_tokens = cfg.max_tokens
        self.temperature = cfg.temperature
        self.top_p = cfg.top_p
        self.enable_history = bool(enable_history)
        self.history_max_turns = int(history_max_turns)
        self.history: List[tuple[str, str]] = []
        self.use_openai_fallback = use_openai_fallback

        # Ключ берём из параметра или окружения
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise LLMError(
                "GEMINI_API_KEY не найден. Установите переменную окружения или передайте api_key."
            )

        # Инициализация Gemini клиента
        try:
            self.client = genai.Client(api_key=key)
            logger.info(f"LLM (Gemini) инициализирован: model={self.model_id}")
        except Exception as e:
            raise LLMError(f"Не удалось инициализировать Gemini client: {e}") from e
        
        # Инициализация OpenAI fallback (опционально)
        self.openai_client = None
        if use_openai_fallback and OPENAI_AVAILABLE:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                try:
                    self.openai_client = OpenAI(api_key=openai_key)
                    logger.info("OpenAI fallback включен: model=gpt-4o-mini")
                except Exception as e:
                    logger.warning(f"OpenAI fallback недоступен: {e}")
            else:
                logger.debug("OPENAI_API_KEY не найден, OpenAI fallback отключен")

    def _build_contents(self, user_text: str) -> List[types.Content]:
        contents: List[types.Content] = []
        if self.enable_history and self.history:
            # Добавим последние turn'ы (user/model) перед текущим запросом
            for role, text in self.history[-(self.history_max_turns * 2) :]:
                mapped_role = "model" if role == "model" else "user"
                contents.append(
                    types.Content(role=mapped_role, parts=[types.Part.from_text(text=text)])
                )
        # Текущий пользовательский ввод
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )
        return contents

    def _build_config(
        self,
        extra_instruction: str = "",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> types.GenerateContentConfig:
        # Минимизируем задержку, отключая "thinking"
        sys_instr = self._SYSTEM_PROMPT + (" " + extra_instruction if extra_instruction else "")
        return types.GenerateContentConfig(
            system_instruction=sys_instr,
            max_output_tokens=int(max_tokens if max_tokens is not None else self.max_new_tokens),
            temperature=float(temperature if temperature is not None else self.temperature),
            top_p=float(self.top_p),
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )

    @staticmethod
    def _classify_intent(text: str) -> str:
        """Грубая классификация запроса: short | normal | extended."""
        t = text.strip().lower()
        # Простые служебные/ритуальные фразы
        simple_patterns = [
            r"\bготов\b", r"начать", r"старт", r"поехали", r"давайте начн[её]м",
            r"вы готовы", r"можем начинать", r"привет", r"здравствуй",
        ]
        for p in simple_patterns:
            if re.search(p, t):
                return "short"
        # Расширенные запросы (попросили подробно/с примерами)
        extended_patterns = [r"подробн", r"пример", r"архитектур", r"как устроено", r"объясни"]
        for p in extended_patterns:
            if re.search(p, t):
                return "extended"
        # Нормальные содержательные вопросы
        normal_patterns = [r"какие", r"перечисл", r"опыт", r"цели", r"почему", r"как", r"что"]
        for p in normal_patterns:
            if re.search(p, t):
                return "normal"
        return "normal"

    # Системный промпт для Gemini
    _SYSTEM_PROMPT = (
        "⚠️ ОТВЕЧАЙ ТОЛЬКО ФАКТАМИ! НЕ отвечай мнениями, советами, отказами. Только факты!\n\n"
        "Ты — ассистент на экзамене. Отвечай СТРОГО по формату:\n"
        "1) Сначала ТОЛЬКО факт (дата/имя/число/место) - одно короткое предложение\n"
        "2) Затем краткое пояснение (если нужно) - одно предложение\n\n"
        "⚠️ ВСЕ ЦИФРЫ И ГОДЫ ПИШИ ПРОПИСЬЮ! Примеры: 1945 → тысяча девятьсот сорок пятый, 20 → двадцать\n\n"
        "ПРАВИЛА:\n"
        "- НЕ повторяй вопрос в ответе\n"
        "- НЕ используй вводные слова типа 'Первый компьютер', 'Столица находится'\n"
        "- Начинай СРАЗУ с факта: года, имени, места\n"
        "- Все числа ПРОПИСЬЮ (не цифрами)\n"
        "- Никакой латиницы - только русская транслитерация\n"
        "- Не говори от первого лица\n\n"
        "⚠️ ASR ОШИБКИ: Распознавание речи часто ошибается! Примеры:\n"
        "- 'гармонастресаа' → вопрос о 'гормоне стресса' (кортизоле)\n"
        "- 'питон язык' → 'Python язык программирования'\n"
        "- 'юрий гагарин' → 'Юрий Гагарин'\n"
        "Пытайся РАЗУМНО понять вопрос через контекст, но БЕЗ ПЕРЕГИБОВ!\n\n"
        "ПРИМЕРЫ ОТВЕТОВ:\n"
        "Q: Кто первым полетел в космос?\n"
        "A: Юрий Гагарин. Полёт состоялся двенадцатого апреля тысяча девятьсот шестьдесят первого года.\n\n"
        "Q: Где находится Эйфелева башня?\n"
        "A: В Париже. Башню построили к Всемирной выставке тысяча восемьсот восемьдесят девятого года.\n\n"
        "Q: Что такое фотосинтез?\n"
        "A: Процесс образования органических веществ из углекислого газа и воды на свету. Происходит в хлоропластах растений.\n\n"
        "⚠️ ПУСТОЙ ОТВЕТ: Если вопрос неразборчив, нет фактов или тема незнакома - НЕ отвечай!\n"
        "Примеры когда НЕ отвечать:\n"
        "- Вопрос полностью искажён ASR и непонятен\n"
        "- Спрашивают мнение/совет (не факты)\n"
        "- Тема слишком специфична и нет проверенных фактов\n"
        "В таких случаях лучше молчать!"
    )
    
    @staticmethod
    def _extra_instruction_for(intent: str) -> tuple[str, int]:
        if intent == "short":
            return ("Отвечай очень кратко, одно короткое предложение, по делу.", 60)
        if intent == "extended":
            return ("Дай содержательный, но компактный ответ (до 4–5 предложений).", 280)
        return ("Ответь ёмко и по делу, 2–3 предложения.", 160)

    @staticmethod
    def _numbers_to_words_ru(text: str) -> str:
        try:
            from num2words import num2words  # type: ignore
        except Exception:
            # Если библиотека не установлена — возвращаем текст как есть
            return text

        def repl(m: re.Match[str]) -> str:
            s = m.group(0)
            # Пробуем как целое число
            try:
                n = int(s)
                return num2words(n, lang="ru")
            except Exception:
                return s

        # Заменяем изолированные группы цифр
        return re.sub(r"\b\d+\b", repl, text)

    @staticmethod
    def _has_latin(text: str) -> bool:
        return re.search(r"[A-Za-z]", text) is not None

    def _rewrite_without_latin(self, text: str) -> str:
        """Одноразовый перефраз с запретом латиницы и требованием чисел прописью."""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "Перепиши текст по правилам: 1) строго по-русски;"
                                " 2) никаких латинских букв — англоязычные слова передай русской транслитерацией;"
                                " 3) все числа прописью; 4) сохрани смысл и тех.точность; 5) 2–5 предложений.\n\n" + text
                            )
                        )
                    ],
                )
            ]
            config = types.GenerateContentConfig(
                system_instruction=(
                    "Ты — редактор-локализатор. Делай текст полностью русским, без латиницы,"
                    " числа прописью, сохраняя точность."
                ),
                max_output_tokens=int(self.max_new_tokens),
                temperature=float(self.temperature),
                top_p=float(self.top_p),
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )
            return (resp.text or "").strip() or text
        except Exception:
            return text

    def _generate_openai(self, user_text: str, intent: str) -> str:
        """Генерация через OpenAI (fallback)"""
        if not self.openai_client:
            return ""
        
        try:
            # Формируем историю для OpenAI
            messages = []
            
            # Системный промпт
            messages.append({
                "role": "system",
                "content": self._SYSTEM_PROMPT
            })
            
            # История диалога
            if self.enable_history and self.history:
                for role, text in self.history[-(self.history_max_turns * 2):]:
                    openai_role = "assistant" if role == "model" else "user"
                    messages.append({"role": openai_role, "content": text})
            
            # Текущий запрос
            extra, max_toks = self._extra_instruction_for(intent)
            user_content = user_text.strip()
            if extra:
                user_content = f"{extra}\n\n{user_content}"
            
            messages.append({"role": "user", "content": user_content})
            
            # Вызываем OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Самая дешевая и быстрая модель
                messages=messages,
                max_tokens=max_toks or self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            
            text = response.choices[0].message.content or ""
            text = text.strip()
            
            logger.debug(f"OpenAI fallback вернул {len(text)} символов")
            return text
            
        except Exception as e:
            logger.error(f"OpenAI fallback ошибка: {e}")
            return ""

    def generate(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""

        intent = self._classify_intent(user_text)
        
        # Сначала пробуем Gemini
        try:
            contents = self._build_contents(user_text)
            extra, max_toks = self._extra_instruction_for(intent)
            config = self._build_config(extra_instruction=extra, max_tokens=max_toks)
            
            # Стримим текст и аккумулируем
            out: List[str] = []
            for chunk in self.client.models.generate_content_stream(
                model=self.model_id,
                contents=contents,
                config=config,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    out.append(chunk.text)

            text = "".join(out).strip()
            if text:
                # Пост-обработка: числа прописью
                text = self._numbers_to_words_ru(text)
                # Если осталась латиница — попробуем перефраз без латиницы
                if self._has_latin(text):
                    text2 = self._rewrite_without_latin(text)
                    if text2:
                        text = text2
                # Обновим историю
                if self.enable_history:
                    self.history.append(("user", user_text.strip()))
                    self.history.append(("model", text))
                    if len(self.history) > self.history_max_turns * 2:
                        self.history = self.history[-(self.history_max_turns * 2) :]
                return text

            # Фоллбэк на нестиминговую генерацию, если пришло пусто
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )
            text = (resp.text or "").strip()
            if text:
                text = self._numbers_to_words_ru(text)
                if self._has_latin(text):
                    text2 = self._rewrite_without_latin(text)
                    if text2:
                        text = text2
                if self.enable_history:
                    self.history.append(("user", user_text.strip()))
                    self.history.append(("model", text))
                    if len(self.history) > self.history_max_turns * 2:
                        self.history = self.history[-(self.history_max_turns * 2) :]
            return text
            
        except Exception as e:
            error_str = str(e)
            
            # Проверяем что это 503 (перегрузка) или 504 (timeout)
            is_server_overload = (
                "503" in error_str or 
                "504" in error_str or
                "overloaded" in error_str.lower() or
                "unavailable" in error_str.lower()
            )
            
            logger.error(f"LLM (Gemini) ошибка: {e}")
            
            # Если сервер перегружен И OpenAI fallback доступен - пробуем OpenAI
            if is_server_overload and self.openai_client:
                logger.warning("Gemini перегружен (503), переключаемся на OpenAI fallback")
                text = self._generate_openai(user_text, intent)
                if text:
                    # Пост-обработка
                    text = self._numbers_to_words_ru(text)
                    if self._has_latin(text):
                        text2 = self._rewrite_without_latin(text)
                        if text2:
                            text = text2
                    # Обновляем историю
                    if self.enable_history:
                        self.history.append(("user", user_text.strip()))
                        self.history.append(("model", text))
                        if len(self.history) > self.history_max_turns * 2:
                            self.history = self.history[-(self.history_max_turns * 2) :]
                    logger.info("✅ OpenAI fallback успешно вернул ответ")
                    return text
                else:
                    logger.error("OpenAI fallback тоже не сработал")
            
            # Если fallback не помог или недоступен - бросаем исключение
            raise LLMError(f"Ошибка генерации ответа: {e}") from e


__all__ = ["LLMResponder"]
