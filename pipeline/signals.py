"""Дешёвые сигналы по кадрам: движение и яркость. Не зависят от ракурса камеры."""
from pathlib import Path

import cv2
import numpy as np


def compute(frames: list[tuple[float, Path]], size: int = 160) -> dict:
    """motion[i] — доля изменившихся пикселей между кадрами i-1 и i; bright[i] — средняя яркость 0..1."""
    ts, motion, bright = [], [], []
    prev = None
    for t, p in frames:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        k = size / max(h, w)
        small = cv2.GaussianBlur(cv2.resize(img, (max(1, int(w * k)), max(1, int(h * k)))), (5, 5), 0)
        ts.append(t)
        bright.append(float(small.mean()) / 255.0)
        if prev is None:
            motion.append(0.0)
        else:
            diff = cv2.absdiff(small, prev)
            motion.append(float((diff > 18).mean()))
        prev = small
    return {"t": np.array(ts), "motion": np.array(motion), "bright": np.array(bright)}


def light_drops(sig: dict, rel: float = 0.35) -> list[float]:
    """Моменты резкого падения яркости (выключили свет)."""
    b, t = sig["bright"], sig["t"]
    out = []
    for i in range(1, len(b)):
        before = b[max(0, i - 4):i].mean()
        if before > 0.08 and b[i] < before * (1 - rel):
            if not out or t[i] - out[-1] > 3:
                out.append(float(t[i]))
    return out


def peak_motion_time(sig: dict, lo: float, hi: float) -> float | None:
    """Момент максимального движения в интервале [lo, hi]."""
    t, m = sig["t"], sig["motion"]
    mask = (t >= lo) & (t <= hi)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    return float(t[idx[np.argmax(m[idx])]])


def window_activity(sig: dict, lo: float, hi: float) -> float:
    t, m = sig["t"], sig["motion"]
    mask = (t >= lo) & (t <= hi)
    return float(m[mask].mean()) if mask.any() else 0.0
