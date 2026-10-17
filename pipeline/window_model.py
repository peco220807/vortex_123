"""Обученный классификатор окон: факты VLM + движение -> вероятность «в этом окне нарушение».

Обучается scripts/train_window_clf.py на размеченных видео, сохраняется в models/window_clf.joblib.
Если файла нет — detector использует ручные правила (rules_engine).
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
    names = (FACT_KEYS + [f"d_{k}" for k in DELTA_KEYS]
             + ["cashier_seen_before", "cashier_gone", "motion_mean", "motion_max", "bright", "bright_drop"]
             + [f"rule_{t}" for t in TYPE_IDS])
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
        rows.append(cur + delta + [float(seen), float(seen and not cashier), mm, mx, b, max(0.0, b_prev - b)]
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


_cache = {}


def load():
    if "m" not in _cache:
        import joblib
        _cache["m"] = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    return _cache["m"]


def predict(timeline: list[dict], sig: dict) -> np.ndarray:
    m = load()
    X, _ = features(timeline, sig)
    return m["clf"].predict_proba(X)[:, 1] if len(X) else np.zeros(0)
