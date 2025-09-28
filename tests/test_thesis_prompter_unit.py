from __future__ import annotations

from thesis_prompter import ThesisPrompter


def test_thesis_prompter_progress_and_announce():
    tp = ThesisPrompter(
        theses=["столица франции париж", "приведи определение rest"],
        match_threshold=0.6,
        enable_semantic=False,
        enable_gemini=False,
    )
    assert tp.has_pending()
    assert tp.need_announce()
    assert tp.current_text() == "столица франции париж"

    tp.mark_announced()
    assert not tp.need_announce()

    # Покрываем первый тезис простой фразой
    closed = tp.consume_transcript("Столица Франции — Париж.")
    assert closed
    assert tp.has_pending()
    assert tp.current_text() == "приведи определение rest"
    assert tp.remaining_count() == 1
    rem = tp.remaining_announcement(limit=1)
    assert "1) приведи определение rest" in rem
