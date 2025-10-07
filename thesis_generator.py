from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


@dataclass
class ThesisGenConfig:
    model_id: str = "gemini-flash-lite-latest"
    max_output_tokens: int = 256
    temperature: float = 0.3
    n_theses: int = 8
    language: str = "ru"


class GeminiThesisGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        cfg = ThesisGenConfig()
        self.model_id = model_id or cfg.model_id
        self.max_output_tokens = int(max_output_tokens if max_output_tokens is not None else cfg.max_output_tokens)
        self.temperature = float(temperature if temperature is not None else cfg.temperature)

        import os
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY не найден для ThesisGenerator")
        self.client = genai.Client(api_key=key)
        logger.info(f"ThesisGenerator инициализирован: model={self.model_id}")

    def generate(self, question_text: str, n: int = 8, language: str = "ru") -> List[str]:
        if not question_text:
            return []
        n = max(1, int(n))
        sys_instr = (
            "Ты генератор тезисов для устных экзаменов."
            " СТРОГИЕ ПРАВИЛА:"
            " 1. Каждый тезис - короткое предложение (до 15 слов)"
            " 2. ТОЛЬКО конкретные факты, без воды"
            " 3. Учитывай местоимения из контекста (ОН, ЭТО, ТАМ, ЕГО)"
            " 4. ФОРМАТ ОТВЕТА: тезис1 ||| тезис2 ||| тезис3"
            " 5. НЕ ЗНАЕШЬ ТЕМУ -> пустая строка БЕЗ ОБЪЯСНЕНИЙ"
            " 6. ЗАПРЕЩЕНЫ объяснения типа 'не указан', 'нет информации', 'не знаю'"
            " 7. ВСЕ ЦИФРЫ ПИШИ СЛОВАМИ (не '1961', а 'тысяча девятьсот шестьдесят первый')"
            " 8. ДАТЫ ПИШИ СЛОВАМИ: 'двенадцатого апреля', 'первого января'"
            " 9. ЧИСЛА ПИШИ СЛОВАМИ: 'сто восемь минут', 'пять километров'"
        )
        user_prompt = (
            f"Контекст диалога (последние 30 сек):\n{question_text.strip()}\n\n"
            f"Дай {n} тезисов в формате: тезис1 ||| тезис2 ||| тезис3\n"
            f"ВАЖНО: ВСЕ ЦИФРЫ И ДАТЫ ПИШИ ТОЛЬКО СЛОВАМИ!\n"
            f"Пример: НЕ '1961', а 'тысяча девятьсот шестьдесят первый'\n"
            f"Пример: НЕ '12 апреля', а 'двенадцатого апреля'\n"
            f"Если тема непонятна -> пустая строка (БЕЗ объяснений)"
        )
        cfg = types.GenerateContentConfig(
            system_instruction=sys_instr,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=0.9,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        try:
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=[types.Content(role="user", parts=[types.Part.from_text(text=user_prompt)])],
                config=cfg,
            )
            raw = (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.exception(f"ThesisGenerator ошибка: {e}")
            return []

        # Парсинг ответа: ожидаем формат "тезис1 ||| тезис2 ||| тезис3"
        import json
        
        # Попытка 1: парсим по разделителю |||
        if "|||" in raw:
            theses = [t.strip() for t in raw.split("|||") if t.strip()]
            return theses[:n]
        
        # Попытка 2: парсим как JSON (fallback для старого формата)
        try:
            data = json.loads(raw)
            items = data.get("theses", [])
            out: List[str] = []
            for it in items:
                if not isinstance(it, str):
                    continue
                t = it.strip()
                if not t:
                    continue
                out.append(t)
            return out[:n]
        except Exception:
            pass
        
        # Попытка 3: построчный парсинг (fallback)
        lines = [ln.strip("-•* \t") for ln in raw.splitlines()]
        out: List[str] = [ln for ln in lines if ln]
        return out[:n]


__all__ = ["GeminiThesisGenerator", "ThesisGenConfig"]
