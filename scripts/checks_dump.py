"""Один раз задать VLM прицельные вопросы по всем кандидатам всех видео и сохранить ответы локально.
Потом scripts/checks_eval.py перебирает правила решения за секунды, без модели.

Нужны рамки детектора: сначала scripts/people_dump.py (уже сделано, если есть data/people/).

  python scripts\\checks_dump.py --videos data\\videos --split train
  python scripts\\checks_dump.py --videos data\\videos --only Х-1,Х-28 --force

Пишет data/checks/<видео>.json (в git не попадает).
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import checks  # noqa: E402
from pipeline.analyze import ANALYSIS_FPS  # noqa: E402
from pipeline.video_io import extract_frames  # noqa: E402
from scripts.common import find_video, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
ap.add_argument("--only", default="")
ap.add_argument("--people", default="data/people")
ap.add_argument("--out", default="data/checks")
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

checks.NOVLM = False  # дамп — именно ответы VLM, даже если сервис по умолчанию работает без неё
only = {norm(x) for x in a.only.split(",") if x.strip()}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

from pipeline.vlm import MODEL_ID as checks_model, load  # noqa: E402
load()
for r in csv.DictReader(open(a.labels, encoding="utf-8")):
    vid = norm(r["video_id"])
    if only and vid not in only:
        continue
    if not only and a.split != "all" and r["split"] != a.split:
        continue
    dst = out / f"{vid}.json"
    if dst.exists() and not a.force:
        print(f"[есть] {vid}")
        continue
    pp = Path(a.people) / f"{vid}.json"
    path = find_video(a.videos, vid)
    if path is None or not pp.exists():
        print(f"[нет видео или рамок] {vid}")
        continue
    pj = json.loads(pp.read_text(encoding="utf-8"))
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(str(path), tmp, fps=ANALYSIS_FPS)
        _, cands = checks.run(frames, pj["dets"], None, None, ask_all=True)
    sec = round(time.time() - t0, 1)
    dst.write_text(json.dumps({"video_id": vid, "candidates": cands, "sec": sec, "model": checks_model,
                               "single": checks.SINGLE}, ensure_ascii=False, default=str),
                   encoding="utf-8")
    kinds = {}
    for c in cands:
        kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    print(f"[ok] {vid}: {len(cands)} кандидатов {kinds} за {sec:.0f} с")
