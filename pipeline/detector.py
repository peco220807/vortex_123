"""Поиск нарушений по всему видео (Чел 1).

1. Видео делится на окна (по умолчанию 4 с, шаг 4 с; для длинных видео шаг растёт, чтобы уложиться в бюджет).
   + отдельное окно на каждое резкое потемнение (выключили свет).
2. В каждом окне VLM отвечает на ФАКТЫ (кто есть, деньги на столе, сейф открыт, ...) — pipeline/vlm.py.
3. Правила (pipeline/rules_engine.py) по последовательности фактов решают, есть ли нарушение.
4. Таймкод — пик движения в первом окне события; кадры-доказательства — оттуда же.

Переменные: WINDOW_SEC, WINDOW_STEP, WINDOW_FRAMES, TIME_BUDGET_SEC, SEC_PER_WINDOW, VLM_RESERVE_SEC.
"""
import os
import time
from pathlib import Path

from . import rules_engine, signals

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "4"))
WINDOW_STEP = float(os.getenv("WINDOW_STEP", "4"))
WINDOW_FRAMES = int(os.getenv("WINDOW_FRAMES", "4"))
TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "170"))   # на все окна VLM (лимит жюри 300 с)
SEC_PER_WINDOW = float(os.getenv("SEC_PER_WINDOW", "4.5"))     # замер на RTX 4050, 4 кадра по 448 px
VLM_RESERVE_SEC = float(os.getenv("VLM_RESERVE_SEC", "15"))


def windows(duration: float, extra_starts: list[float] = ()) -> list[tuple[float, float]]:
    max_windows = max(4, int(TIME_BUDGET_SEC / SEC_PER_WINDOW))
    step = max(WINDOW_STEP, duration / max_windows)
    starts, s = [], 0.0
    while s < duration - 0.5:
        starts.append(s)
        s += step
    starts += [max(0.0, x - 1.0) for x in extra_starts]
    return sorted({(round(a, 2), round(min(duration, a + WINDOW_SEC), 2)) for a in starts})


def pick(frames: list[tuple[float, Path]], lo: float, hi: float, n: int) -> list[tuple[float, bytes]]:
    inside = [(t, p) for t, p in frames if lo <= t <= hi]
    if len(inside) > n:
        step = (len(inside) - 1) / (n - 1)
        inside = [inside[round(i * step)] for i in range(n)]
    return [(t, Path(p).read_bytes()) for t, p in inside]


def build_timeline(frames: list[tuple[float, Path]], meta: dict, sig: dict | None = None) -> list[dict]:
    """Факты VLM по каждому окну, по порядку времени. Останавливается у deadline."""
    from .vlm import observe  # тяжёлый импорт — только когда нужен

    deadline = meta.get("deadline")
    sig = sig or signals.compute(frames)
    drops = signals.light_drops(sig)
    wins = windows(meta["duration"], drops)
    timeline, skipped = [], 0
    for lo, hi in wins:
        if deadline and time.time() > deadline - VLM_RESERVE_SEC:
            skipped += 1
            continue
        imgs = pick(frames, lo, hi, WINDOW_FRAMES)
        if len(imgs) < 2:
            continue
        f = observe(imgs)
        f["lights_off"] = f["lights_off"] or any(lo <= d <= hi for d in drops)
        timeline.append({"lo": lo, "hi": hi, "f": f})
    meta.update(windows_total=len(wins), windows_skipped=skipped, light_drops=drops)
    return timeline


def events_from_timeline(timeline: list[dict], sig: dict) -> list[dict]:
    events = rules_engine.apply(timeline)
    for e in events:
        # нарушение часто начинается на стыке окон (кассир ушёл между ними) — ищем пик и в предыдущем окне
        lo = timeline[e["first_i"] - 1]["lo"] if e.get("first_i") else e["lo"]
        t = signals.peak_motion_time(sig, lo, min(e["hi"], e["lo"] + WINDOW_SEC))
        t = (e["lo"] + e["hi"]) / 2 if t is None else t
        e["t"] = round(t, 2)
        e["evidence_t"] = sorted({round(t, 2), round(min(t + 1.0, e["hi"]), 2)})
    return events


def detect_events(frames: list[tuple[float, Path]], meta: dict) -> list[dict]:
    sig = signals.compute(frames)
    timeline = build_timeline(frames, meta, sig)
    meta["timeline_windows"] = len(timeline)
    return events_from_timeline(timeline, sig)
