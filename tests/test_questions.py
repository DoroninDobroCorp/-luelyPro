from __future__ import annotations

from live_recognizer import LiveVoiceVerifier


def _verifier() -> LiveVoiceVerifier:
    return LiveVoiceVerifier(
        asr_enable=False,
        llm_enable=False,
        theses_path=None,
        thesis_autogen_enable=False,
    )


def test_extract_questions_basic():
    text = (
        "Расскажите архитектуру сервиса? И как работает TCP/IP стек. Это не вопрос.\n"
        "Когда вышла первая версия Python?"
    )
    qs = LiveVoiceVerifier._extract_questions(text)
    assert any("архитектуру" in q.lower() for q in qs)
    assert any("python" in q.lower() for q in qs)


def test_filter_questions_heuristic():
    v = _verifier()
    # Принудительно отключим режим Gemini в фильтре, чтобы тест был оффлайн
    v._question_filter_mode = "heuristic"
    qs = [
        "что такое tcp/ip стек",
        "как дела",
        "в каком году вышел python?",
        "что такое бинарное дерево",
    ]
    kept = v._filter_questions_by_importance(qs)
    assert any("tcp" in q.lower() for q in kept)
    assert not any("как дела" == q.lower() for q in kept)
    assert any("python" in q.lower() for q in kept)
