from __future__ import annotations

from typing import List

import numpy as np

from live_recognizer import LiveVoiceVerifier, ThesisPrompter


class DummyEmbedder:
    """Минимальный эмбеддер для тестов: возвращает нули нужной формы."""

    def __call__(self, tensor):  # type: ignore[override]
        batch = getattr(tensor, "shape", [1, 1, 0])[0]
        dim = 192
        return np.zeros((batch, dim), dtype=np.float32)


def _base_verifier(**kwargs) -> LiveVoiceVerifier:
    params = dict(
        asr_enable=False,
        llm_enable=False,
        theses_path=None,
        thesis_autogen_enable=False,
        thesis_gemini_enable=False,
        thesis_semantic_enable=False,
        embedder=DummyEmbedder(),
    )
    params.update(kwargs)
    return LiveVoiceVerifier(**params)


def test_simulate_dialogue_closes_thesis() -> None:
    verifier = _base_verifier()
    verifier.thesis_prompter = ThesisPrompter(
        theses=["столица франции париж"],
        match_threshold=0.8,
        enable_semantic=False,
        enable_gemini=False,
    )

    dialogue: List[tuple[str, str]] = [
        ("self", "Столица Франции — Париж."),
    ]

    verifier.simulate_dialogue(dialogue)

    assert verifier.thesis_prompter is not None
    assert not verifier.thesis_prompter.has_pending()


def test_simulate_dialogue_creates_ai_theses(monkeypatch) -> None:
    verifier = _base_verifier()
    verifier._ai_only_thesis = True

    def fake_extract(text: str) -> List[str]:
        assert "архитектуру" in text.lower()
        return ["укажи архитектуру сервиса"]

    monkeypatch.setattr(verifier, "_extract_theses_ai", fake_extract)

    dialogue: List[tuple[str, str]] = [
        ("other", "Расскажите, пожалуйста, архитектуру сервиса?"),
    ]

    verifier.simulate_dialogue(dialogue)

    assert verifier.thesis_prompter is not None
    assert verifier.thesis_prompter.current_text() == "укажи архитектуру сервиса"
