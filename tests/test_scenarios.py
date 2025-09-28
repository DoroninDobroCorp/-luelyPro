from __future__ import annotations

from typing import List

from live_recognizer import LiveVoiceVerifier, ThesisPrompter


def test_ai_only_varied_questions_and_non_questions(monkeypatch):
    v = LiveVoiceVerifier(
        asr_enable=False,
        llm_enable=False,
        theses_path=None,
        thesis_autogen_enable=False,
    )
    v._ai_only_thesis = True

    def fake_ai_extract(text: str) -> List[str]:
        t = text.lower()
        if "два плюс два" in t or "2+2" in t:
            return ["скажи ответ четыре", "кратко поясни вычисление"]
        if "архитектур" in t:
            return [
                "назови основные компоненты",
                "опиши взаимодействие клиент-сервер",
                "упомяни REST и кеширование",
            ]
        return []

    monkeypatch.setattr(v, "_extract_theses_ai", fake_ai_extract)

    spoken: List[str] = []

    def capture_say(s: str) -> None:
        if s:
            spoken.append(s)

    # перехватываем озвучку, даже если _tts отсутствует
    monkeypatch.setattr(v, "_speak_text", capture_say)

    dialogue = [
        ("other", "Привет, как дела, давайте начнём"),  # не вопрос — ожидание: без тезисов
        ("other", "Сколько будет два плюс два?"),       # точный вопрос — ожидание: полезные тезисы
        ("self", "Ответ четыре, это базовая арифметика"),
        ("other", "Расскажите архитектуру вашего REST сервиса"),  # пространный вопрос — 3 тезиса
    ]

    v.simulate_dialogue(dialogue)

    # Проверяем, что были предложения тезисов и они содержат полезные ключи
    assert any("Предлагаю тезисы ответа" in s for s in spoken)
    assert any("четыре" in s.lower() for s in spoken)  # для вопроса 2+2
    # Для архитектуры — должен быть список с нумерацией 1) 2) ...
    assert any(
        ("1)" in s and "2)" in s) or ("опиши" in s.lower()) for s in spoken
    )


def test_paraphrase_marks_thesis_done_with_semantic(monkeypatch):
    # Включаем семантику заменой матчера на фиктивный, чтобы не требовать весов модели
    tp = ThesisPrompter(
        theses=["объясни разницу процесс и поток"],
        match_threshold=0.85,  # высокий порог по токенам
        enable_semantic=True,
        enable_gemini=False,
    )

    class DummyMatcher:
        def score(self, a: str, b: str) -> float:
            return 0.92  # достаточно, чтобы пройти semantic_threshold по умолчанию 0.55

    tp._semantic_enabled = True
    tp._semantic_matcher = DummyMatcher()  # type: ignore

    # Парафраз без явного токенного покрытия ('поток' ~ 'нит(ь)/тред')
    closed = tp.consume_transcript("Расскажите, в чём различия процессов и нитей (тредов)?")
    assert closed is True
    assert tp.has_pending() is False
