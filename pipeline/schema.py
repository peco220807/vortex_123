"""Формат результата — строго по разделу 10 ТЗ. Все части пайплайна возвращают именно это."""


def make_violation(type_: str, timecode_sec: float, description: str, evidence_frames: list[dict]) -> dict:
    return {
        "type": type_,
        "timecode_sec": round(float(timecode_sec), 2),
        "description": description,
        "evidence_frames": evidence_frames[:3],  # 1–3 кадра
    }


def make_result(violations: list[dict]) -> dict:
    return {
        "verdict": "violation" if violations else "no_violation",
        "violations": violations,
    }


def validate(result: dict) -> None:
    """Падает с AssertionError, если формат не совпадает с ТЗ."""
    assert result["verdict"] in ("violation", "no_violation")
    assert isinstance(result["violations"], list)
    assert (result["verdict"] == "violation") == bool(result["violations"])
    for v in result["violations"]:
        assert isinstance(v["type"], str) and v["type"]
        assert isinstance(v["timecode_sec"], (int, float))
        assert isinstance(v["description"], str)
        assert 1 <= len(v["evidence_frames"]) <= 3
        for f in v["evidence_frames"]:
            assert isinstance(f["timestamp_sec"], (int, float))
            assert isinstance(f["image_base64"], str) and f["image_base64"]
