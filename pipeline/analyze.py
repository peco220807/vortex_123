"""Главная точка входа ML-части (Чел 1).

    from pipeline.analyze import analyze
    result = analyze("video.mov")   # dict: verdict + violations (раздел 10 ТЗ, без job_id/status)

Сервис (Чел 2) вызывает только эту функцию. Внутренности меняем свободно,
пока возвращается формат из pipeline/schema.py.

Поиск нарушений — pipeline/detector.py (окна по всему видео + локальная VLM).
Режим STUB_ALWAYS_VIOLATION=1 возвращает фейковое нарушение — для отладки сервиса и фронта без модели.
"""
import json
import os
import tempfile
import time
from pathlib import Path

from .schema import make_result, make_violation, validate
from .video_io import extract_frames, frame_b64, probe
from .violation_types import TYPES

RULES = json.loads((Path(__file__).resolve().parent.parent / "rules" / "rules.json").read_text(encoding="utf-8"))
ANALYSIS_FPS = float(os.getenv("ANALYSIS_FPS", "2"))


def detect_events(frames: list[tuple[float, Path]], meta: dict) -> list[dict]:
    """Список событий: {"type_id", "t", "score", "evidence_t", "description"}."""
    if os.getenv("STUB_ALWAYS_VIOLATION") == "1":
        t = meta["duration"] / 2
        return [{"type_id": "unattended_valuables", "t": t, "score": 1.0, "evidence_t": [t],
                 "description": "ТЕСТОВЫЙ РЕЖИМ: фейковое нарушение для отладки интерфейса."}]
    from .detector import detect_events as run
    return run(frames, meta)


def short_source(src: dict) -> str:
    """«Постановление ... № 231, пункты 20 и 22» — без длинного названия в кавычках «...»."""
    doc = (src.get("document") or "").strip()
    clause = (src.get("clause") or "").strip()
    if not doc or doc.startswith("<"):
        return ""
    doc = doc.split("«")[0].strip().rstrip(",")
    return doc + (f", {clause}" if clause and not clause.startswith("<") else "")


def describe(event: dict) -> str:
    """Что видно (от VLM) + признак нарушения + норматив из rules.json (если Чел 3 заполнил). Источники не выдумываем."""
    text = (event.get("description") or "").strip()
    if not text:
        text = f'На видео видны признаки: {TYPES[event["type_id"]]["visual"]}.'
    src = short_source((RULES.get(event["type_id"]) or {}).get("source") or {})
    if src:
        text = text.rstrip(".") + f". Источник: {src}."
    return text


def analyze(video_path: str, threshold: float = 0.5, deadline: float | None = None) -> dict:
    """deadline — unix-время (time.time()), к которому надо вернуть результат.
    Сервис передаёт его сам. Если time.time() близко к deadline — пропускать медленные шаги (VLM)
    и возвращать то, что уже найдено. Проверка: meta["deadline"] внутри detect_events.
    """
    t0 = time.time()
    meta = probe(video_path)
    meta["deadline"] = deadline
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(video_path, tmp, fps=ANALYSIS_FPS)
        events = [e for e in detect_events(frames, meta) if e["score"] >= threshold]

    violations = []
    for e in sorted(events, key=lambda e: e["t"]):
        ev = [{"timestamp_sec": round(t, 2), "image_base64": frame_b64(video_path, t)} for t in e["evidence_t"][:3]]
        violations.append(make_violation(TYPES[e["type_id"]]["name"], e["t"], describe(e), ev))

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
