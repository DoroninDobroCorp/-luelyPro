from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set

from loguru import logger

try:
    from semantic_matcher import SemanticMatcher  # опционально, как фоллбэк
except Exception:  # noqa: BLE001
    SemanticMatcher = None  # type: ignore

try:
    from thesis_judge import GeminiJudge  # основной судья
except Exception:  # noqa: BLE001
    GeminiJudge = None  # type: ignore


def _default_stopwords() -> Set[str]:
    return {
        "и",
        "в",
        "во",
        "на",
        "но",
        "а",
        "к",
        "ко",
        "что",
        "это",
        "как",
        "же",
        "из",
        "за",
        "для",
        "по",
        "с",
        "со",
        "у",
        "мы",
        "вы",
        "они",
        "он",
        "она",
        "оно",
        "когда",
        "где",
        "там",
        "тут",
        "тогда",
        "тоже",
        "так",
        "также",
        "если",
        "чтобы",
        "при",
        "от",
        "до",
        "бы",
        "будет",
        "буду",
        "будем",
        "есть",
        "нет",
    }


@dataclass
class ThesisPrompter:
    """Простое управление списком тезисов с отслеживанием произнесённых пунктов."""

    theses: List[str]
    match_threshold: float = 0.6
    min_token_len: int = 3
    stopwords: Set[str] = field(default_factory=_default_stopwords)
    enable_semantic: bool = True
    semantic_threshold: float = 0.55
    semantic_model_id: Optional[str] = None
    # Gemini-судья
    enable_gemini: bool = True
    gemini_min_conf: float = 0.60

    def __post_init__(self) -> None:
        self.theses = [t.strip() for t in self.theses if t and t.strip()]
        self.match_threshold = max(0.0, min(1.0, self.match_threshold))
        self.semantic_threshold = max(0.0, min(1.0, self.semantic_threshold))
        self.gemini_min_conf = max(0.0, min(1.0, self.gemini_min_conf))
        self._index = 0
        self._announced = False
        self._current_tokens: Set[str] = set()
        self._thesis_tokens: List[Set[str]] = [self._extract_tokens(t) for t in self.theses]
        self._current_text: str = ""
        # Семантика (фоллбэк)
        self._semantic_enabled = bool(self.enable_semantic) and (SemanticMatcher is not None)
        self._semantic_matcher: Optional[SemanticMatcher] = None
        if self._semantic_enabled:
            try:
                self._semantic_matcher = SemanticMatcher(model_id=self.semantic_model_id)  # type: ignore
            except Exception as e:  # noqa: BLE001
                logger.exception(f"SemanticMatcher недоступен: {e}")
                self._semantic_enabled = False
        # Gemini-судья (основной путь)
        self._gemini_enabled = bool(self.enable_gemini) and (GeminiJudge is not None)
        self._gemini_judge: Optional[GeminiJudge] = None
        if self._gemini_enabled:
            try:
                self._gemini_judge = GeminiJudge()  # type: ignore
            except Exception as e:  # noqa: BLE001
                logger.exception(f"GeminiJudge недоступен: {e}")
                self._gemini_enabled = False

    def has_pending(self) -> bool:
        return self._index < len(self.theses)

    def need_announce(self) -> bool:
        return self.has_pending() and not self._announced

    def current_text(self) -> Optional[str]:
        if not self.has_pending():
            return None
        return self.theses[self._index]

    def mark_announced(self) -> None:
        if self.has_pending():
            self._announced = True

    def consume_transcript(self, transcript: str) -> bool:
        if not transcript:
            return False
        tokens = self._extract_tokens(transcript)
        if not tokens and not transcript.strip():
            return False
        self._current_tokens.update(tokens)
        if transcript.strip():
            if self._current_text:
                self._current_text += " "
            self._current_text += transcript.strip()
        if not self.has_pending():
            return False
        thesis_tokens = self._thesis_tokens[self._index]
        if not thesis_tokens:
            # пустой тезис — сразу считаем пройденным
            self._advance()
            return True
        coverage = self._coverage(self._current_tokens, thesis_tokens)
        logger.debug(f"coverage={coverage:.2f} for current thesis {self._index+1}/{len(self.theses)}")
        if coverage >= self.match_threshold:
            self._advance()
            return True
        # Основной судья: Gemini
        if self._gemini_enabled and self._gemini_judge is not None and self._current_text:
            thesis_text = self.theses[self._index]
            covered, conf = self._gemini_judge.judge(thesis_text, self._current_text)
            logger.debug(
                f"GeminiJudge для тезиса '{thesis_text[:40]}...': covered={covered}, conf={conf:.3f}"
            )
            if covered and conf >= self.gemini_min_conf:
                self._advance()
                return True
        # Семантика (фоллбэк)
        if self._semantic_enabled and self._semantic_matcher is not None:
            thesis_text = self.theses[self._index]
            score = self._semantic_matcher.score(thesis_text, self._current_text)  # type: ignore
            logger.debug(
                f"semantic match для тезиса '{thesis_text[:40]}...': score={score:.3f}"
            )
            if score >= self.semantic_threshold:
                self._advance()
                return True
        return False

    def _advance(self) -> None:
        self._index += 1
        self._announced = False
        self._current_tokens.clear()
        self._current_text = ""

    # пустая строка перед следующим методом
    def _extract_tokens(self, text: str) -> Set[str]:
        tokens = re.findall(r"[\w-]+", text.lower(), flags=re.UNICODE)
        cleaned = {
            tok
            for tok in tokens
            if len(tok) >= self.min_token_len and tok not in self.stopwords
        }
        return cleaned

    @staticmethod
    def _coverage(spoken: Set[str], thesis_tokens: Set[str]) -> float:
        if not thesis_tokens:
            return 1.0
        matched = len(spoken & thesis_tokens)
        return matched / float(len(thesis_tokens))

    # Новые утилиты для логов и озвучки прогресса
    def remaining_count(self) -> int:
        return max(0, len(self.theses) - self._index)

    def coverage_of_current(self) -> float:
        if not self.has_pending():
            return 1.0
        thesis_tokens = self._thesis_tokens[self._index]
        return self._coverage(self._current_tokens, thesis_tokens)

    def remaining_list(self, limit: int = 3) -> List[str]:
        if self._index >= len(self.theses):
            return []
        return self.theses[self._index : self._index + max(0, int(limit))]

    def remaining_announcement(self, limit: int = 3) -> str:
        items = self.remaining_list(limit=limit)
        if not items:
            return ""
        nums = []
        for i, t in enumerate(items, 1):
            nums.append(f"{i}) {t}")
        return "Осталось: " + "; ".join(nums)


__all__ = ["ThesisPrompter"]
