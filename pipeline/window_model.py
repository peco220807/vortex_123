"""Обученный классификатор окон: факты VLM + движение -> вероятность «в этом окне нарушение».

Обучается scripts/train_window_clf.py на размеченных видео, сохраняется в models/window_clf.joblib.
Если файла нет или он обучен на старом наборе признаков — detector использует ручные правила (rules_engine).

Движение и яркость считаются ОТНОСИТЕЛЬНО САМОГО ВИДЕО (делим на медиану по видео): абсолютные значения
зависят от камеры, ракурса и освещения, и модель училась различать камеры, а не нарушения.

smooth() — как из вероятностей окон получить «силу события» в окне (агрегация выбирается при обучении):
  max   — как есть (одно окно с пиком = нарушение; много ложных);
  pair  — среднее окна и лучшего соседа (нужны 2 соседних окна выше порога);
  ctx   — окно минус медиана по видео (фон этой камеры вычитается).
"""
from pathlib import Path

import numpy as np

from . import rules_engine
from .violation_types import TYPES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "window_clf.joblib"

FACT_KEYS = [
    "people_count", "cashier_at_desk", "visitor_count", "person_leaving", "person_entering",
    "money_on_desk", "money_to_cashier", "money_to_visitor", "counting_machine_used",
    "id_document_visible", "bag_on_desk", "bag_being_opened", "safe_open", "keys_in_safe_lock",
    "laptop_open", "lights_off", "touching_equipment",
]
DELTA_KEYS = ["people_count", "cashier_at_desk", "visitor_count", "money_on_desk", "bag_on_desk",
              "safe_open", "laptop_open"]
TYPE_IDS = list(TYPES)


AGGS = ("max", "pair", "ctx")
FEATURE_NAMES = (FACT_KEYS + [f"d_{k}" for k in DELTA_KEYS]
                 + ["cashier_seen_before", "cashier_gone",
                    "motion_mean_rel", "motion_max_rel", "bright_rel", "bright_drop_rel"]
                 + [f"rule_{t}" for t in TYPE_IDS])


def _num(v) -> float:
    return float(min(int(v), 4)) if isinstance(v, (int, float)) and not isinstance(v, bool) else float(bool(v))


def _motion(sig, lo, hi):
    t, m = np.asarray(sig["t"]), np.asarray(sig["motion"])
    mask = (t >= lo) & (t <= hi)
    return (float(m[mask].mean()), float(m[mask].max())) if mask.any() else (0.0, 0.0)


def _bright(sig, lo, hi):
    t, b = np.asarray(sig["t"]), np.asarray(sig["bright"])
    mask = (t >= lo) & (t <= hi)
    return float(b[mask].mean()) if mask.any() else 0.0


def features(timeline: list[dict], sig: dict) -> tuple[np.ndarray, list[str]]:
    """Матрица признаков [окна x признаки] + имена признаков."""
    hits = {}  # (окно, тип) от ручных правил — как признаки
    for e in rules_engine.apply(timeline):
        for i in range(e["first_i"], e["last_i"] + 1):
            hits[(i, e["type_id"])] = 1.0
    names = FEATURE_NAMES
    # фон этого видео: типичное движение и яркость (медиана — устойчива к самому событию)
    m_all = np.asarray(sig["motion"], dtype=float)
    b_all = np.asarray(sig["bright"], dtype=float)
    m_base = float(np.median(m_all)) + 0.01 if len(m_all) else 0.01
    b_base = float(np.median(b_all)) + 1e-3 if len(b_all) else 1.0
    rows, seen = [], False
    for i, w in enumerate(timeline):
        f = w["f"]
        p = timeline[i - 1]["f"] if i else f
        cur = [_num(f.get(k, 0)) for k in FACT_KEYS]
        delta = [_num(f.get(k, 0)) - _num(p.get(k, 0)) for k in DELTA_KEYS]
        cashier = bool(f.get("cashier_at_desk"))
        mm, mx = _motion(sig, w["lo"], w["hi"])
        b = _bright(sig, w["lo"], w["hi"])
        b_prev = _bright(sig, timeline[i - 1]["lo"], timeline[i - 1]["hi"]) if i else b
        rel = [min(mm / m_base, 10.0), min(mx / m_base, 10.0), b / b_base - 1.0, max(0.0, b_prev - b) / b_base]
        rows.append(cur + delta + [float(seen), float(seen and not cashier)] + rel
                    + [hits.get((i, t), 0.0) for t in TYPE_IDS])
        seen = seen or cashier
    return np.array(rows, dtype=float).reshape(len(rows), len(names)), names


def guess_type(timeline: list[dict], i: int) -> str:
    """Тип для окна, где классификатор увидел нарушение: сначала ручные правила рядом, иначе — по фактам."""
    near = [e for e in rules_engine.apply(timeline) if e["first_i"] - 1 <= i <= e["last_i"] + 1]
    if near:
        return max(near, key=lambda e: e["n"])["type_id"]
    f = timeline[i]["f"]
    if f.get("keys_in_safe_lock") or f.get("safe_open"):
        return "safe_unsecured"
    if f.get("bag_being_opened") or f.get("bag_on_desk"):
        return "collection_procedure"
    if not f.get("cashier_at_desk"):
        if f.get("laptop_open") and f.get("people_count", 0) == 0:
            return "workplace_abandoned"
        if f.get("money_on_desk"):
            return "unattended_valuables"
        return "unauthorized_person"
    if f.get("visitor_count", 0) >= 2:
        return "mixed_operations"
    return "cash_intake_procedure"


def smooth(probs, agg: str = "max") -> np.ndarray:
    """Вероятности окон -> сила события в окне (см. описание модуля)."""
    p = np.asarray(probs, dtype=float)
    if len(p) == 0 or agg == "max":
        return p
    if agg == "pair":
        left = np.concatenate([[0.0], p[:-1]])
        right = np.concatenate([p[1:], [0.0]])
        return (p + np.maximum(left, right)) / 2
    if agg == "ctx":
        return p - float(np.median(p))
    raise ValueError(f"неизвестная агрегация: {agg}")


_cache = {}


def load():
    if "m" not in _cache:
        m = None
        if MODEL_PATH.exists():
            import joblib
            m = joblib.load(MODEL_PATH)
            if list(m.get("features", [])) != list(FEATURE_NAMES):
                print(f"[window_model] {MODEL_PATH.name} обучен на старых признаках — "
                      "переобучите: python scripts/train_window_clf.py. Пока работают ручные правила.")
                m = None
        _cache["m"] = m
    return _cache["m"]


def predict(timeline: list[dict], sig: dict) -> np.ndarray:
    """Сила события по окнам (уже с агрегацией, с которой модель обучалась)."""
    m = load()
    X, _ = features(timeline, sig)
    if not len(X):
        return np.zeros(0)
    return smooth(m["clf"].predict_proba(X)[:, 1], m.get("agg", "max"))
