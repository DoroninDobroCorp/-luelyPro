from __future__ import annotations

from live_recognizer import LiveVoiceVerifier


def test_parse_theses_json_object():
    raw = '{"theses": ["первый", "второй", "третий"]}'
    out = LiveVoiceVerifier._parse_theses_from_raw(raw, n_max=3)
    assert out == ["первый", "второй", "третий"]


def test_parse_theses_code_fence():
    raw = """
    ```json
    {"theses": ["a", "b"]}
    ```
    """.strip()
    out = LiveVoiceVerifier._parse_theses_from_raw(raw, n_max=3)
    assert out == ["a", "b"]


def test_parse_theses_list_fallback():
    raw = "- один\n- два\n- три"
    out = LiveVoiceVerifier._parse_theses_from_raw(raw, n_max=2)
    assert out == ["один", "два"]
