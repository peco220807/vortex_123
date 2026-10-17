"""Тест: видит ли VLM нарушения в РАЗМЕЧЕННЫЙ момент.

Для видео с нарушением берём кадры вокруг таймкода, для видео без нарушения — из середины
(и ещё из 1/4 и 3/4 ролика, чтобы проверить ложные срабатывания). Спрашиваем VLM и считаем точность.

  python scripts/vlm_probe.py --videos data/videos --split train
  python scripts/vlm_probe.py --videos data/videos --split train --limit 6     # быстрый прогон
  set VLM_MODEL=Qwen/Qwen3-VL-4B-Instruct   (другая модель)

Результат: печать в консоль + data/vlm_probe.csv (локально, в git не попадает).
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.video_io import frame_at  # noqa: E402
from pipeline.vlm import MODEL_ID, classify_window  # noqa: E402
from pipeline.ffbin import ffprobe  # noqa: E402
from scripts.common import find_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--split", default="train")
ap.add_argument("--window", type=float, default=3.0, help="± секунд вокруг момента")
ap.add_argument("--n", type=int, default=6, help="кадров на окно")
ap.add_argument("--width", type=int, default=640, help="макс. ширина кадра (маленькие не растягиваются)")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()


def duration(p):
    return float(subprocess.run([ffprobe(), "-v", "error", "-show_entries", "format=duration",
                                 "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout)


def window(vp, t, dur):
    lo, hi = max(t - a.window, 0), min(t + a.window, dur - 0.1)
    ts = [lo + i * (hi - lo) / (a.n - 1) for i in range(a.n)]
    return [(x, frame_at(str(vp), x, a.width)) for x in ts]


rows = [r for r in csv.DictReader(open(a.labels, encoding="utf-8")) if a.split == "all" or r["split"] == a.split]
if a.limit:
    v = [r for r in rows if r["violation"] == "1"][: a.limit // 2]
    c = [r for r in rows if r["violation"] == "0"][: a.limit - len(v)]
    rows = v + c

print(f"Модель: {MODEL_ID}, окно ±{a.window} с, {a.n} кадров, ширина {a.width}")
out, times = [], []
for r in rows:
    vp = find_video(a.videos, r["video_id"])
    if vp is None:
        print(f"[нет файла] {r['video_id']}")
        continue
    dur = duration(vp)
    y = int(r["violation"])
    probes = [float(r["timecode_sec"])] if y else [dur * 0.25, dur * 0.5, dur * 0.75]
    for t in probes:
        t0 = time.time()
        res = classify_window(window(vp, t, dur))
        dt = time.time() - t0
        times.append(dt)
        ok = res["violation"] == bool(y)
        out.append({"video_id": r["video_id"], "label": y, "t": round(t, 1), "pred": int(res["violation"]),
                    "type": res["type"], "conf": res["confidence"], "ok": int(ok), "sec": round(dt, 1),
                    "description": res["description"], "raw": res["raw"]})
        mark = "OK " if ok else "ERR"
        print(f"{mark} {r['video_id']:6} t={t:5.1f} метка={y} ответ={int(res['violation'])} "
              f"тип={res['type']} ({dt:.1f} с) | {res['description'][:90]}")

if out:
    tp = sum(o["label"] and o["pred"] for o in out)
    fn = sum(o["label"] and not o["pred"] for o in out)
    fp = sum((not o["label"]) and o["pred"] for o in out)
    tn = sum((not o["label"]) and not o["pred"] for o in out)
    print(f"\nНарушения найдены: {tp}/{tp + fn}   Ложные на чистых окнах: {fp}/{fp + tn}")
    print(f"Время на окно: среднее {sum(times) / len(times):.1f} с, макс {max(times):.1f} с")
    p = Path("data/vlm_probe.csv")
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"Подробно: {p}")
