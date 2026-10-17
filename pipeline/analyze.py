"""Главная точка входа ML-части (Чел 1).

    from pipeline.analyze import analyze
    result = analyze("video.mov")   # dict: verdict + violations (раздел 10 ТЗ, без job_id/status)

Сервис (Чел 2) вызывает только эту функцию. Внутренности меняем свободно,
пока возвращается формат из pipeline/schema.py.

Сейчас это ЗАГЛУШКА: детекторы пустые, поэтому всегда no_violation.
Режим STUB_ALWAYS_VIOLATION=1 возвращает фейковое нарушение — для отладки сервиса и фронта.
"""
import json
import os
import tempfile
import time
from pathlib import Path

from .schema import make_result, make_violation, validate
from .video_io import extract_frames, frame_b64, probe

RULES = json.loads((Path(__file__).resolve().parent.parent / "rules" / "rules.json").read_text(encoding="utf-8"))
ANALYSIS_FPS = float(os.getenv("ANALYSIS_FPS", "2"))


def detect_events(frames: list[tuple[float, Path]], meta: dict) -> list[dict]:
    """Найти события-кандидаты.

    Вернуть список: {"type_id": <ключ из rules.json>, "t": <сек>, "score": 0..1, "evidence_t": [сек, ...]}
    TODO (Чел 1): детекция людей/объектов -> трекинг -> поза -> правила по типам.
    """
    if os.getenv("STUB_ALWAYS_VIOLATION") == "1":
        t = meta["duration"] / 2
        return [{"type_id": "example", "t": t, "score": 1.0, "evidence_t": [t]}]
    return []


def describe(event: dict) -> str:
    """Текст обоснования: что видно + почему нарушение + источник."""
    rule = RULES[event["type_id"]]
    src = rule["source"]
    return f'{rule["observation"]} {rule["why"]} Источник: {src["document"]}, {src["clause"]}.'


def analyze(video_path: str, threshold: float = 0.5) -> dict:
    t0 = time.time()
    meta = probe(video_path)
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(video_path, tmp, fps=ANALYSIS_FPS)
        events = [e for e in detect_events(frames, meta) if e["score"] >= threshold]

    violations = []
    for e in sorted(events, key=lambda e: e["t"]):
        ev = [{"timestamp_sec": round(t, 2), "image_base64": frame_b64(video_path, t)} for t in e["evidence_t"][:3]]
        violations.append(make_violation(RULES[e["type_id"]]["name"], e["t"], describe(e), ev))

    result = make_result(violations)
    validate(result)
    result["_debug"] = {"elapsed_sec": round(time.time() - t0, 1), "n_frames": len(frames), **meta}
    return result


if __name__ == "__main__":
    import sys
    r = analyze(sys.argv[1])
    for v in r["violations"]:
        for f in v["evidence_frames"]:
            f["image_base64"] = f["image_base64"][:20] + "..."
    print(json.dumps(r, ensure_ascii=False, indent=2))
