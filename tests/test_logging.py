from __future__ import annotations

from pathlib import Path

import time
from live_recognizer import setup_logging


def test_setup_logging_creates_file(tmp_path: Path):
    log_path = setup_logging(log_dir=tmp_path)
    assert log_path.parent == tmp_path
    # подождём кратко, чтобы sink создал файл (на случай буферизации ОС)
    for _ in range(50):
        if log_path.exists():
            break
        time.sleep(0.01)
    assert log_path.exists()
