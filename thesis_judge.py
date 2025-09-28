from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


@dataclass
class GeminiJudgeConfig:
    model_id: str = "gemini-2.5-flash-lite"
    max_output_tokens: int = 64
    temperature: float = 0.0


class GeminiJudge:
    """
    Проверяет, покрывает ли речь пользователя заданный тезис (да/нет + уверенность 0..1),
    используя Google GenAI (Gemini Flash Lite).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        cfg = GeminiJudgeConfig()
        self.model_id = model_id or cfg.model_id
        self.max_output_tokens = int(max_output_tokens if max_output_tokens is not None else cfg.max_output_tokens)
        self.temperature = float(temperature if temperature is not None else cfg.temperature)

        import os
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY не найден для GeminiJudge")
        self.client = genai.Client(api_key=key)
        logger.info(f"GeminiJudge инициализирован: model={self.model_id}")

    def judge(self, thesis: str, spoken_text: str) -> Tuple[bool, float]:
        """Возвращает (covered, confidence[0..1])."""
        if not thesis or not spoken_text:
            return (False, 0.0)

        sys_instr = (
            "Ты — строгий экзаменатор на собеседовании/экзамене. Оцени, отражает ли речь кандидата"
            " ключевую идею тезиса. Ответ строго в JSON и только JSON:"
            " {\"covered\": true|false, \"confidence\": число 0..1}. Если речи недостаточно — covered=false."
        )
        user_prompt = (
            "Тезис:\n" + thesis.strip() + "\n\n" +
            "Речь пользователя:\n" + spoken_text.strip() + "\n\n" +
            "Оцени, передана ли суть тезиса, даже если формулировки отличаются."
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
            logger.exception(f"GeminiJudge ошибка: {e}")
            return (False, 0.0)

        import json
        try:
            data = json.loads(raw)
            covered = bool(data.get("covered", False))
            conf = float(data.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            return (covered, conf)
        except Exception:
            # Если не удалось распарсить — ужесточаем поведение
            logger.debug(f"GeminiJudge unparsable response: {raw}")
            return (False, 0.0)


__all__ = ["GeminiJudge", "GeminiJudgeConfig"]
