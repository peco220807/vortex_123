"""Один раз прогнать VLM по всем окнам всех видео и сохранить факты локально.
Потом правила перебираются за секунды через scripts/rules_eval.py — без модели.

  python scripts\\facts_dump.py --videos data\\videos --split train
  python scripts\\facts_dump.py --videos data\\videos --split train --only Х-1,Х-3

Пишет data/facts/<видео>.json (в git не попадает). Уже готовые видео пропускает (--force — пересчитать).
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import signals  # noqa: E402
from pipeline.analyze import ANALYSIS_FPS  # noqa: E402
from pipeline import detector  # noqa: E402
from pipeline.detector import build_timeline  # noqa: E402

# для настройки нужны ПОЛНЫЕ факты по всем окнам — без прореживания под лимит жюри
detector.TIME_BUDGET_SEC = 1e9
from pipeline.video_io import extract_frames, probe  # noqa: E402
from scripts.common import find_video, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
ap.add_argument("--only", default="")
ap.add_argument("--out", default="data/facts")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

only = {norm(x) for x in a.only.split(",") if x.strip()}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for r in csv.DictReader(open(a.labels, encoding="utf-8")):
    vid = r["video_id"]
    if (a.split != "all" and r["split"] != a.split) or (only and norm(vid) not in only):
        continue
    dst = out / f"{vid}.json"
    if dst.exists() and not a.force:
        print(f"[есть] {vid}")
        continue
    vp = find_video(a.videos, vid)
    if vp is None:
        print(f"[нет файла] {vid}")
        continue
    t0 = time.time()
    print(f"... {vid}", end="", flush=True)
    meta = probe(str(vp))
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(str(vp), tmp, fps=ANALYSIS_FPS)
        sig = signals.compute(frames)
        timeline = build_timeline(frames, meta, sig)
    for w in timeline:
        if w["f"].get("parsed"):
            w["f"].pop("raw", None)  # сырой ответ храним только у неразобранных — для отладки
    dst.write_text(json.dumps({
        "video_id": vid, "meta": meta, "timeline": timeline,
        "sig": {k: [round(float(x), 4) for x in v] for k, v in sig.items()},
    }, ensure_ascii=False), encoding="utf-8")
    bad = sum(not w["f"]["parsed"] for w in timeline)
    print(f"\r[ok] {vid}: {len(timeline)} окон за {time.time() - t0:.0f} с "
          f"({(time.time() - t0) / max(1, len(timeline)):.1f} с/окно, пропущено {meta.get('windows_skipped', 0)})"
          + (f", НЕ РАЗОБРАНО ответов: {bad}" if bad else ""))
