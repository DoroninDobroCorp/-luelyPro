from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from loguru import logger


@dataclass
class LLMConfig:
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 64
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    system_prompt: str = (
        "Ты — помощник, который отвечает очень кратко и по делу, одной-двумя фразами."
        " Дай короткое пояснение на русском языке."
    )


class LLMResponder:
    def __init__(
        self,
        model_id: str | None = None,
        device: Optional[str] = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        cfg = LLMConfig()
        self.model_id = model_id or cfg.model_id
        self.max_new_tokens = max_new_tokens or cfg.max_new_tokens
        self.temperature = temperature if temperature is not None else cfg.temperature
        self.top_p = top_p if top_p is not None else cfg.top_p
        self.repetition_penalty = (
            repetition_penalty if repetition_penalty is not None else cfg.repetition_penalty
        )
        self.system_prompt = system_prompt or cfg.system_prompt

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        try:
            logger.info(f"Загрузка LLM: {self.model_id} на {self.device} …")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
            )
            if self.device == "cpu":
                self.model = self.model.to("cpu")
            self.model.eval()
        except Exception as e:  # noqa: BLE001
            
            logger.warning("Ошибка:", e)
            # Фоллбэк на меньшую модель Gemma E2B
            # fallback_id = "google/gemma-3n-E2B"
            # if self.model_id != fallback_id:
            #     logger.warning(
            #         f"Не удалось загрузить {self.model_id} ({e}). Пытаюсь {fallback_id}."
            #     )
            #     self.model_id = fallback_id
            #     self._load_model()
            #     return
            logger.exception("Не удалось загрузить модель LLM")
            raise

    def _build_prompt(self, user_text: str) -> Dict[str, Any]:
        assert self.tokenizer is not None
        # Попробуем chat-template, если он доступен в токенайзере
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_text},
        ]
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback простой промпт
            prompt = (
                f"Система: {self.system_prompt}\n"
                f"Пользователь: {user_text}\n"
                f"Ассистент:"
            )
        return {"text": prompt}

    def generate(self, user_text: str) -> str:
        if not user_text or not user_text.strip():
            return ""
        prompt_dict = self._build_prompt(user_text)
        prompt_text = prompt_dict["text"]
        assert self.tokenizer is not None and self.model is not None

        inputs = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=(self.temperature > 0.0),
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = output[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text.strip()


__all__ = ["LLMResponder", "LLMConfig"]
