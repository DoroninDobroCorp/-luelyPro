from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer


@dataclass
class SemanticMatcherConfig:
    model_id: str = "intfloat/multilingual-e5-small"
    max_length: int = 128
    device: Optional[str] = None


class SemanticMatcher:
    """Оценка семантической близости между предложениями через трансформерные эмбеддинги."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        max_length: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        cfg = SemanticMatcherConfig()
        self.model_id = model_id or cfg.model_id
        self.max_length = int(max_length if max_length is not None else cfg.max_length)
        device = device or cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"SemanticMatcher: загрузка модели {self.model_id} на {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, text: str) -> torch.Tensor:
        if not text:
            return torch.zeros(1, dtype=torch.float32, device=self.device)
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        outputs = self.model(**tokens)
        hidden = outputs.last_hidden_state  # (1, seq, dim)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(hidden.shape[:2], device=self.device)
        mask = attention_mask.unsqueeze(-1)  # (1, seq, 1)
        # mean pooling
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        embedding = summed / counts
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.squeeze(0)

    @torch.inference_mode()
    def score(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        emb_a = self.encode(text_a)
        emb_b = self.encode(text_b)
        sim = torch.dot(emb_a, emb_b).item()
        # нормируем в диапазон [0,1]
        return max(0.0, min(1.0, (sim + 1.0) * 0.5))


__all__ = ["SemanticMatcher", "SemanticMatcherConfig"]
