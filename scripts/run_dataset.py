"""Прогнать analyze() по датасету локально и сохранить ответы для evaluate.py.

  python scripts/run_dataset.py --videos data/videos --split train --out preds/
  python scripts/evaluate.py --labels data/labels_clean.csv --preds preds/ --split train
"""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.analyze import analyze  # noqa: E402
from scripts.common import find_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
ap.add_argument("--out", default="preds")
a = ap.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
for r in csv.DictReader(open(a.labels, encoding="utf-8")):
    if a.split != "all" and r["split"] != a.split:
        continue
    vp = find_video(a.videos, r["video_id"])
    if vp is None:
        print(f"[нет файла] {r['video_id']}")
        continue
    t0 = time.time()
    try:
        res = {"status": "done", **analyze(str(vp))}
    except Exception:
        traceback.print_exc()
        res = {"status": "failed"}
    dt = time.time() - t0
    flag = "  <-- БОЛЬШЕ 5 МИН" if dt > 300 else ""
    (out / f"{r['video_id']}.json").write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
    print(f"{r['video_id']:6} {res.get('verdict', 'FAILED'):13} {dt:6.1f} с{flag}")
