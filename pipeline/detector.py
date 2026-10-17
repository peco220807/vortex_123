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

from . import rules_engine, signals, window_model
from .violation_types import TYPES

WINDOW_SEC = float(os.getenv("WINDOW_SEC", "4"))
WINDOW_STEP = float(os.getenv("WINDOW_STEP", "4"))
WINDOW_FRAMES = int(os.getenv("WINDOW_FRAMES", "4"))
TIME_BUDGET_SEC = float(os.getenv("TIME_BUDGET_SEC", "170"))   # на все окна VLM (лимит жюри 300 с)
SEC_PER_WINDOW = float(os.getenv("SEC_PER_WINDOW", "4.5"))     # стартовая оценка; дальше — по замеру
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
    from .vlm import load, observe  # тяжёлый импорт — только когда нужен

    load()  # загрузка модели не должна попадать в замер скорости окон
    deadline = meta.get("deadline")
    sig = sig or signals.compute(frames)
    drops = signals.light_drops(sig)
    wins = windows(meta["duration"], drops)
    budget = TIME_BUDGET_SEC
    if deadline:
        budget = min(budget, deadline - time.time() - VLM_RESERVE_SEC)
    timeline, done, t_start = [], 0, time.time()
    rest = list(wins)  # ещё не обработанные окна, по времени
    while rest:
        elapsed = time.time() - t_start
        if deadline and time.time() > deadline - VLM_RESERVE_SEC:
            break
        # сколько окон ещё успеем при текущей скорости; если меньше, чем осталось, — шагаем через окна,
        # но каждый раз пересчитываем (если модель ускорилась, снова идём подряд)
        if done >= 2:
            per = elapsed / done
            can = int((budget - elapsed) / per)
            if can <= 0:
                break
            stride = max(1, round(len(rest) / can)) if can < len(rest) else 1
        else:
            stride = 1
        lo, hi = rest[0]
        rest = rest[stride:]
        imgs = pick(frames, lo, hi, WINDOW_FRAMES)
        if len(imgs) < 2:
            continue
        f = observe(imgs)
        done += 1
        f["lights_off"] = f["lights_off"] or any(lo <= d <= hi for d in drops)
        timeline.append({"lo": lo, "hi": hi, "f": f})
    skipped = len(wins) - len(timeline)
    meta.update(windows_total=len(wins), windows_skipped=skipped, light_drops=drops)
    return timeline


MAX_EVENTS = int(os.getenv("MAX_EVENTS", "3"))


def _classifier_events(timeline: list[dict], sig: dict, m: dict) -> list[dict]:
    probs = window_model.predict(timeline, sig)
    thr = float(os.getenv("CLF_THRESHOLD", m["threshold"]))
    groups, cur = [], []
    for i, p in enumerate(probs):
        if p >= thr:
            cur.append(i)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    rule_events = rules_engine.apply(timeline)
    events = []
    for g in groups:
        best = max(g, key=lambda i: probs[i])
        tid = window_model.guess_type(timeline, best)
        why = next((e["why"] for e in rule_events if e["type_id"] == tid
                    and e["first_i"] - 1 <= best <= e["last_i"] + 1), TYPES[tid]["visual"])
        act = timeline[best]["f"].get("action", "")
        events.append({"type_id": tid, "first_i": g[0], "last_i": g[-1], "peak_i": best, "n": len(g), "why": why,
                       "lo": timeline[g[0]]["lo"], "hi": timeline[g[-1]]["hi"],
                       # analyze() отсекает score < 0.5, а порог классификатора свой (и у ctx может быть ~0.1):
                       # переводим так, чтобы «ровно на пороге» = 0.5
                       "score": float(min(1.0, max(0.0, 0.5 + (max(probs[g]) - thr)))),
                       "description": (f"{act.rstrip('.')}. " if act else "") + f"Признак нарушения: {why}."})
    events.sort(key=lambda e: -e["score"])
    return sorted(events[:MAX_EVENTS], key=lambda e: e["lo"])


def events_from_timeline(timeline: list[dict], sig: dict) -> list[dict]:
    m = window_model.load()
    events = _classifier_events(timeline, sig, m) if m and timeline else rules_engine.apply(timeline)
    for e in events:
        # нарушение часто начинается на стыке окон (кассир ушёл между ними) — ищем пик и в предыдущем окне
        if "peak_i" in e:  # классификатор: ищем момент в окне с максимальной вероятностью (и стыке с предыдущим)
            i = e["peak_i"]
            lo, hi = (timeline[i - 1]["lo"] if i else timeline[i]["lo"]), timeline[i]["hi"]
        else:
            lo = timeline[e["first_i"] - 1]["lo"] if e.get("first_i") else e["lo"]
            hi = min(e["hi"], e["lo"] + WINDOW_SEC)
        t = signals.peak_motion_time(sig, lo, hi)
        t = (e["lo"] + e["hi"]) / 2 if t is None else t
        e["t"] = round(t, 2)
        e["evidence_t"] = sorted({round(t, 2), round(min(t + 1.0, e["hi"]), 2)})
    return events


def detect_events(frames: list[tuple[float, Path]], meta: dict) -> list[dict]:
    from .vlm import describe

    sig = signals.compute(frames)
    timeline = build_timeline(frames, meta, sig)
    meta["timeline_windows"] = len(timeline)
    events = events_from_timeline(timeline, sig)
    deadline = meta.get("deadline")
    for e in events[:5]:  # фраза «что видно» — только для найденных событий
        if deadline and time.time() > deadline - 8:
            break
        try:
            act = describe(pick(frames, max(0.0, e["t"] - 1.5), e["t"] + 2.5, WINDOW_FRAMES))
        except Exception:  # noqa: BLE001 — без фразы ответ всё равно валиден
            act = ""
        if act:
            e["description"] = f"{act.rstrip('.')}. Признак нарушения: {e['why']}."
    return events
