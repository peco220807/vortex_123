"""Прогнать детектор людей по видео и сохранить сырые рамки локально (data/people/<видео>.json).

Детектор быстрый (секунды на видео). Сторона кассира — по геометрии (где дольше всех был человек у ноутбука).
VLM задаёт ОДИН вопрос на видео как запасной вариант и для сравнения (--no-vlm — не спрашивать).

Только пересчитать стороны по уже сохранённым рамкам (секунды, без детектора и VLM):
  python scripts\\people_dump.py --sides
Факты VLM (data/facts) заново гонять НЕ нужно: rules_eval / train_window_clf с --people data/people
подставят факты о людях поверх готовых.

  (transformers уже стоит из requirements.txt; веса детектора скачаются при первом запуске, ~80 МБ)
  python scripts\\people_dump.py --videos data\\videos --split train
  python scripts\\people_dump.py --videos data\\videos --split train --no-vlm --only Х-1,Х-30
"""
import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import people  # noqa: E402
from pipeline.analyze import ANALYSIS_FPS  # noqa: E402
from pipeline.detector import pick  # noqa: E402
from pipeline.video_io import extract_frames, probe  # noqa: E402
from scripts.common import find_video, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
ap.add_argument("--only", default="")
ap.add_argument("--out", default="data/people")
ap.add_argument("--no-vlm", action="store_true", help="не спрашивать VLM про сторону кассира")
ap.add_argument("--force", action="store_true")
ap.add_argument("--sides", action="store_true", help="только показать стороны кассира по сохранённым рамкам")
a = ap.parse_args()

only = {norm(x) for x in a.only.split(",") if x.strip()}
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if a.sides:
    for p in sorted(out.glob("*.json"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0)):
        pj = json.loads(p.read_text(encoding="utf-8"))
        sc = people.layout(pj["dets"])
        print(f"{pj['video_id']:6} кассир: {people.side_name(sc['dir']):6} (геометрия, по «{sc['anchor_src']}»), "
              f"VLM: {pj.get('side_hint')}")
    sys.exit(0)

jobs = []
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
    path = find_video(a.videos, vid)
    if path is None:
        print(f"[нет видео] {vid}")
        continue
    jobs.append((vid, path, dst))

# 1) детектор по всем видео (VLM в это время не держим в памяти)
results = {}
for vid, path, dst in jobs:
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        frames = extract_frames(str(path), tmp, fps=ANALYSIS_FPS)
        dets = people.detect(frames)
        side_frames = pick(frames, 0.0, probe(str(path))["duration"], 3) if not a.no_vlm else None
    results[vid] = {"dets": dets, "side_frames": side_frames, "dst": dst, "sec": time.time() - t0}
    n_p = sum(1 for d in dets for b in d["boxes"] if b[5] == "person") / max(1, len(dets))
    print(f"[детектор] {vid}: {len(dets)} кадров за {time.time() - t0:.0f} с, людей в кадре в среднем {n_p:.1f}")
people.unload()

# 2) сторона кассира от VLM — один вопрос на видео
if not a.no_vlm and results:
    from pipeline.vlm import cashier_side, load
    load()
    for vid, r in results.items():
        try:
            r["side"] = cashier_side(r["side_frames"])
        except Exception as e:  # noqa: BLE001
            print(f"[vlm] {vid}: {e!r}")
            r["side"] = None

# 3) сохранить + коротко показать раскладку сцены
for vid, r in results.items():
    side = r.get("side")
    scene = people.layout(r["dets"], side_hint=side)
    r["dst"].write_text(json.dumps({"video_id": vid, "fps": ANALYSIS_FPS, "side_hint": side, "dets": r["dets"]},
                                   ensure_ascii=False), encoding="utf-8")
    geo = people.side_name(people.layout(r["dets"])["dir"])  # только геометрия, без VLM
    flag = "" if side is None or geo == side else "   <-- VLM не согласна, проверьте глазами"
    print(f"[ok] {vid}: рабочее место по «{scene['anchor_src']}», кассир: {geo} (геометрия), {side} (VLM){flag}")
