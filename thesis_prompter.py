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
        self._dialogue_context: List[tuple[str, str]] = []  # (role, text) - роль: "экзаменатор" или "студент"
        # Tracking для Pro upgrades
        self._pro_snapshots = {}  # thesis -> {"index": int, "context_size": int, "timestamp": float}
        # Семантика (фоллбэк)
        self._semantic_enabled = bool(self.enable_semantic) and (SemanticMatcher is not None)
        self._semantic_matcher: Optional[SemanticMatcher] = None
        if self._semantic_enabled:
            try:
                self._semantic_matcher = SemanticMatcher(model_id=self.semantic_model_id)  # type: ignore
            except Exception as e:  # noqa: BLE001
                logger.exception(f"SemanticMatcher недоступен: {e}")
                self._semantic_enabled = False
        # Gemini-судья (основной путь) - ОТКЛЮЧАЕМ Pro для скорости
        self._gemini_enabled = bool(self.enable_gemini) and (GeminiJudge is not None)
        self._gemini_judge: Optional[GeminiJudge] = None
        if self._gemini_enabled:
            try:
                # УСКОРЕНИЕ: enable_pro=False - убираем параллельный Pro вызов (~500мс экономии)
                self._gemini_judge = GeminiJudge(enable_pro=False)  # type: ignore
                # Устанавливаем callback для обновлений от Pro
                self._gemini_judge.set_upgrade_callback(self._on_pro_upgrade)
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

    def reset_announcement(self) -> None:
        if self.has_pending():
            self._announced = False

    def consume_transcript(self, transcript: str, role: str = "студент") -> bool:
        """
        Обрабатываем реплику диалога.
        role: "студент" (мой ответ) или "экзаменатор" (вопрос)
        """
        if not transcript:
            return False
        text = transcript.strip()
        if not text:
            return False
        
        # Добавляем в контекст диалога С ЛИМИТОМ (макс 10 реплик = 5 пар вопрос-ответ)
        self._dialogue_context.append((role, text))
        if len(self._dialogue_context) > 10:
            self._dialogue_context = self._dialogue_context[-10:]
        
        # Обновляем накопленный текст (для старых методов)
        if self._current_text:
            self._current_text += " "
        self._current_text += text
        
        if not self.has_pending():
            return False
        
        # Сохраняем snapshot перед вызовом Gemini (для валидации Pro ответа)
        import time
        thesis_text = self.theses[self._index]
        self._pro_snapshots[thesis_text] = {
            "index": self._index,
            "context_size": len(self._dialogue_context),
            "timestamp": time.time()
        }
        
        # Проверяем только через Gemini (убрали проценты)
        if self._gemini_enabled and self._gemini_judge is not None:
            covered, conf = self._gemini_judge.judge(thesis_text, self._dialogue_context)
            logger.debug(
                f"GeminiJudge для тезиса '{thesis_text[:40]}...': covered={covered}, conf={conf:.3f}"
            )
            if covered and conf >= self.gemini_min_conf:
                self._advance()
                return True
        
        return False

    def _advance(self) -> None:
        self._index += 1
        self._announced = False
        self._current_tokens.clear()
        self._current_text = ""
        self._dialogue_context.clear()

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
        return "Остались тезисы: " + "; ".join(nums)
    
    def _on_pro_upgrade(
        self, 
        thesis: str, 
        covered: bool, 
        confidence: float,
        snapshot_timestamp: float,
        snapshot_context_size: int,
        response_timestamp: float
    ) -> bool:
        """
        Callback для обновлений от Pro модели.
        Проверяет можно ли применить решение Pro.
        
        Returns:
            True если можно применить решение Pro, False если контекст устарел
        """
        # Проверяем что у нас есть snapshot для этого тезиса
        if thesis not in self._pro_snapshots:
            logger.debug(f"⏭️  Pro: нет snapshot для тезиса {thesis[:50]}...")
            return False
        
        snapshot = self._pro_snapshots[thesis]
        
        # Проверка 1: Тезис не сменился (индекс тот же или тезис уже закрыт)
        current_thesis_text = self.theses[self._index] if self.has_pending() else None
        if current_thesis_text != thesis:
            # Тезис уже закрылся или сменился
            logger.debug(f"⏭️  Pro: тезис сменился (было: {thesis[:30]}..., сейчас: {current_thesis_text[:30] if current_thesis_text else 'None'}...)")
            return False
        
        # Проверка 2: Контекст не изменился (новых реплик не было)
        if len(self._dialogue_context) != snapshot_context_size:
            logger.debug(f"⏭️  Pro: контекст изменился (было {snapshot_context_size} реплик, сейчас {len(self._dialogue_context)})")
            return False
        
        # Всё в порядке - можно применять решение Pro
        logger.info(f"✅ Pro решение валидно для тезиса {thesis[:50]}...")
        
        # Применяем решение Pro
        if covered:
            logger.info(f"🔄 Pro закрывает тезис (conf={confidence:.2f})")
            self._advance()
        
        return True


__all__ = ["ThesisPrompter"]
