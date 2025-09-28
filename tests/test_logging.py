from __future__ import annotations

from pathlib import Path

from live_recognizer import setup_logging


def test_setup_logging_creates_file(tmp_path: Path):
    log_path = setup_logging(log_dir=tmp_path)
    assert log_path.parent == tmp_path
    # файл должен существовать после первого лог-сообщения в setup_logging
    assert log_path.exists()
