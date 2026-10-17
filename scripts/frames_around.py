"""Нарезать кадры вокруг размеченного таймкода, чтобы глазами разобрать, что за нарушение.

  python scripts/frames_around.py --videos data/videos --window 4 --fps 4

Для каждого видео с нарушением (split=train) кладёт в data/frames_around/<видео>/
кадры с t-window до t+window и один лист-коллаж contact.jpg для быстрого просмотра.
Для видео без нарушения берёт кадры из середины (для сравнения).
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.common import find_video  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--out", default="data/frames_around")
ap.add_argument("--window", type=float, default=4.0)
ap.add_argument("--fps", type=float, default=4.0)
ap.add_argument("--split", default="train")
a = ap.parse_args()

for r in csv.DictReader(open(a.labels, encoding="utf-8")):
    if r["split"] != a.split:
        continue
    vp = find_video(a.videos, r["video_id"])
    if vp is None:
        print(f"[нет файла] {r['video_id']}")
        continue
    t = float(r["timecode_sec"]) if r["timecode_sec"] else None
    if t is None:  # без нарушения — середина ролика
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                    "-of", "csv=p=0", str(vp)], capture_output=True, text=True).stdout)
        t = dur / 2
    start = max(t - a.window, 0)
    out = Path(a.out) / f"{r['video_id']}_{'V' if r['violation'] == '1' else 'OK'}_t{t:.0f}"
    out.mkdir(parents=True, exist_ok=True)
    vf = f"fps={a.fps},scale=640:-2,drawtext=text='%{{pts\\:hms}}':x=10:y=10:fontsize=28:fontcolor=yellow:box=1:boxcolor=black"
    # таймкод на кадре = время от начала ролика
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{2 * a.window}", "-i", str(vp),
                    "-copyts", "-vf", vf, "-q:v", "3", str(out / "f_%03d.jpg")], check=False)
    n = int(2 * a.window * a.fps)
    cols = 4
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(out / "f_%03d.jpg"),
                    "-vf", f"scale=480:-2,tile={cols}x{(n + cols - 1) // cols}", "-frames:v", "1",
                    str(out / "contact.jpg")], check=False)
    print(f"[ok] {out}")
