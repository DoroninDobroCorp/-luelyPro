from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List
import re
import os

from loguru import logger
from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()


@dataclass
class LLMConfig:
    # Быстрая облачная модель по умолчанию
    model_id: str = "gemini-2.5-flash-lite"
    max_new_tokens: int = 320
    temperature: float = 0.3
    top_p: float = 0.9
    system_prompt: str = (
        "Ты — ассистент на экзамене/собеседовании. По входному тексту формируй краткие, фактологичные"
        " подсказки. Если во входе нет явного вопроса к кандидату — верни пустой ответ. Пиши по-русски,"
        " коротко и по делу (1–3 предложения), без воды и без ответов от первого лица. Строго не повторяй"
        " формулировку вопроса и не начинай ответ с пересказа вопроса — сразу давай ответ, затем при"
        " необходимости одно короткое пояснение. Никакой латиницы: англоязычные названия — русской"
        " транслитерацией; числа — прописью."
    )


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
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        enable_history: bool = True,
        history_max_turns: int = 8,
    ) -> None:
        cfg = LLMConfig()
        self.model_id = model_id or cfg.model_id
        self.max_new_tokens = max_new_tokens or cfg.max_new_tokens
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.top_p = top_p if top_p is not None else cfg.top_p
        self.system_prompt = system_prompt or cfg.system_prompt
        # Диалоговая история (последние N обменов): [("user"|"model"), text]
        self.enable_history = bool(enable_history)
        self.history_max_turns = int(history_max_turns)
        self.history: List[tuple[str, str]] = []

        # Ключ берём из параметра или окружения
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY не найден. Установите переменную окружения или передайте api_key в LLMResponder."
            )

        # Инициализация клиента
        self.client = genai.Client(api_key=key)
        logger.info(f"LLM (Gemini) инициализирован: model={self.model_id}")

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
        sys_instr = self.system_prompt + (" " + extra_instruction if extra_instruction else "")
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

    def generate(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""

        contents = self._build_contents(user_text)
        intent = self._classify_intent(user_text)
        extra, max_toks = self._extra_instruction_for(intent)
        config = self._build_config(extra_instruction=extra, max_tokens=max_toks)

        try:
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
        except Exception as e:  # noqa: BLE001
            logger.exception(f"LLM (Gemini) ошибка генерации: {e}")
            raise


__all__ = ["LLMResponder", "LLMConfig"]
