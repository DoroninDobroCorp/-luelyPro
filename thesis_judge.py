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

    def judge(self, thesis: str, dialogue_context) -> Tuple[bool, float]:
        """
        Возвращает (covered, confidence[0..1]).
        dialogue_context: либо строка (старый API), либо list[(role, text)]
        """
        if not thesis:
            return (False, 0.0)
        
        # Формируем контекст диалога
        if isinstance(dialogue_context, str):
            # Старый API - просто текст
            dialogue_text = dialogue_context.strip()
        elif isinstance(dialogue_context, list):
            # Новый API - список (роль, текст)
            lines = []
            for role, text in dialogue_context:
                if role == "студент":
                    lines.append(f"Студент: {text}")
                else:
                    lines.append(f"Экзаменатор: {text}")
            dialogue_text = "\n".join(lines)
        else:
            return (False, 0.0)
        
        if not dialogue_text:
            return (False, 0.0)

        sys_instr = (
            "Ты мягкий помощник на экзамене. Оцени, покрыл ли студент тезис-подсказку в своём ответе. "
            "ВАЖНО: будь снисходительным - учитывай опечатки распознавания речи (ASR), синонимы, перефразировки. "
            "Если студент передал СУТЬ тезиса (даже другими словами) - считай covered=true. "
            "Если разговор ушёл в другую тему совсем - covered=true (тема сменилась, тезис больше не актуален). "
            "Ответ строго в JSON: {\"covered\": true|false, \"confidence\": 0..1}"
        )
        user_prompt = (
            "Тезис-подсказка:\n" + thesis.strip() + "\n\n" +
            "Диалог:\n" + dialogue_text + "\n\n" +
            "Покрыл ли студент тезис (передал суть) или разговор ушёл в другую тему? Будь мягким к опечаткам ASR."
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
        import re
        try:
            # Убираем markdown код-блоки если есть
            cleaned = raw
            if '```json' in cleaned:
                match = re.search(r'```json\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            elif '```' in cleaned:
                match = re.search(r'```\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1)
            
            data = json.loads(cleaned)
            covered = bool(data.get("covered", False))
            conf = float(data.get("confidence", 0.0))
            conf = max(0.0, min(1.0, conf))
            return (covered, conf)
        except Exception:
            # Если не удалось распарсить — ужесточаем поведение
            logger.debug(f"GeminiJudge unparsable response: {raw}")
            return (False, 0.0)


__all__ = ["GeminiJudge", "GeminiJudgeConfig"]
