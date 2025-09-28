from __future__ import annotations

import time
from pathlib import Path

from live_recognizer import LiveVoiceVerifier, ThesisPrompter, setup_logging


def test_log_contains_thesis_progress(tmp_path: Path) -> None:
    # Configure file logging to tmp dir
    log_file = setup_logging(log_dir=tmp_path)

    v = LiveVoiceVerifier(
        asr_enable=False,
        llm_enable=False,
        theses_path=None,
        thesis_autogen_enable=False,
    )
    v.thesis_prompter = ThesisPrompter(
        theses=["столица франции париж"],
        match_threshold=0.8,
        enable_semantic=False,
        enable_gemini=False,
    )

    v.simulate_dialogue([("self", "Столица Франции — Париж.")])

    # небольшой retry: запись в файл может быть отложена файловой системой
    content = ""
    for _ in range(50):
        try:
            content = log_file.read_text(encoding="utf-8")
            if content:
                break
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    assert "Тезис закрыт" in content


def test_log_tts_tail_suppression_message(tmp_path: Path) -> None:
    log_file = setup_logging(log_dir=tmp_path)
    v = LiveVoiceVerifier(
        asr_enable=False,
        llm_enable=False,
        theses_path=None,
        thesis_autogen_enable=False,
    )
    v._ai_only_thesis = True
    v._suppress_until = time.time() + 0.3

    v._handle_foreign_text("Любой текст")

    content = ""
    for _ in range(50):
        try:
            content = log_file.read_text(encoding="utf-8")
            if content:
                break
        except FileNotFoundError:
            pass
        time.sleep(0.01)
    assert "Игнорирую распознанный TTS-хвост" in content
