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
    model_id: str = "gemini-2.5-flash-lite"
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
            "Ты — помощник-методолог для экзамена/собеседования. На основе вопроса/темы"
            " сформируй краткие тезисы ответа кандидата. Если в тексте нет содержательных вопросов к кандидату,"
            " верни пустой список. Строго верни JSON вида {\"theses\": [\"тезис1\", \"тезис2\", ...]}"
            " без пояснений и лишнего текста. Язык тезисов: " + language + "."
            " Каждый тезис — максимум пятнадцать слов, по делу и без воды."
        )
        user_prompt = (
            f"Вопрос/тема: \n{question_text.strip()}\n\n"
            f"Сформируй до {n} тезисов для ответа кандидата. Если вопроса по сути нет — верни пустой массив."
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

        import json
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
            return out
        except Exception:
            # fallback: иногда модель вернёт список как текст с переносами строк
            lines = [ln.strip("-•* \t") for ln in raw.splitlines()]
            out: List[str] = [ln for ln in lines if ln]
            return out[:n]


__all__ = ["GeminiThesisGenerator", "ThesisGenConfig"]
