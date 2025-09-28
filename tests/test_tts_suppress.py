from __future__ import annotations

import time

from live_recognizer import LiveVoiceVerifier


def _verifier() -> LiveVoiceVerifier:
    return LiveVoiceVerifier(
        asr_enable=False,
        llm_enable=False,
        theses_path=None,
        thesis_autogen_enable=False,
    )


def test_tts_tail_suppression_skip_ai_extraction(monkeypatch):
    v = _verifier()
    v._ai_only_thesis = True

    def should_not_be_called(_text: str):  # pragma: no cover
        raise AssertionError("_extract_theses_ai must not be called under suppression")

    monkeypatch.setattr(v, "_extract_theses_ai", should_not_be_called)
    v._suppress_until = time.time() + 0.3  # будущее, значит подавление активно
    v._handle_foreign_text("Это хвост TTS, который должен быть проигнорирован")
    # если мы здесь, значит исключения не было и подавление сработало
    assert True
