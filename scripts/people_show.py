"""Нарисовать рамки детектора на кадре — посмотреть глазами, кого он считает кассиром / посторонним.

Картинка сохраняется ЛОКАЛЬНО в data/debug/ (в git не попадает, никуда не отправляем).

  python scripts\\people_show.py Х-5             # 3 кадра: начало, середина, конец
  python scripts\\people_show.py Х-5 --t 10 25   # конкретные секунды

Цвета (подписи латиницей — шрифт по умолчанию без кириллицы): зелёный CASHIER — кассир, красный CASHIER SIDE — ещё кто-то на стороне кассира («за спиной» / «чужой на месте»),
синий VISITOR — посетитель, серый OTHER / SHORT — остальные или мелькнул < 2 с, жёлтый — ноутбук (рабочее место),
белая стрелка — от ноутбука в сторону кассира. Пунктир — рамки ниже порога PERSON_THR (не учитываются).
Плюс таблица треков в консоли: кто, сколько секунд, насколько «неподвижен» (неподвижный весь ролик
«человек» на стороне кассира — скорее отражение, плакат или вещь на стуле).
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import people  # noqa: E402
from pipeline.video_io import frame_at  # noqa: E402
from scripts.common import find_video, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--t", type=float, nargs="*", default=None)
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--people", default="data/people")
ap.add_argument("--out", default="data/debug")
a = ap.parse_args()

vid = norm(a.video)
pj = json.loads((Path(a.people) / f"{vid}.json").read_text(encoding="utf-8"))
dets = pj["dets"]
tracks = people.track(dets)
scene = people.layout(dets, tracks)
cids = people.cashier_tracks(dets, tracks, scene)
twins = people.twin_tracks(dets, tracks, cids)
bg = people.background_tracks(dets, tracks, cids, scene)
s = scene["s"] or 1.0
owner = {id(b): tr["id"] for tr in tracks for b in tr["boxes"]}
steady = {tr["id"] for tr in tracks
          if dets[tr["frames"][-1]]["t"] - dets[tr["frames"][0]]["t"] >= people.MIN_ROLE_SEC}

print(f"{vid}: сторона кассира {people.side_name(scene['dir'])}, размер человека {s:.0f} px, "
      f"кассир — треки {sorted(cids)}")
print(" трек |  с  -  по  | сек | роль (большинство кадров) | неподвижность* | кассир / фон")
for tr in tracks:
    t0, t1 = dets[tr["frames"][0]]["t"], dets[tr["frames"][-1]]["t"]
    roles = [people.role(b, scene) for b in tr["boxes"]]
    main = max(set(roles), key=roles.count)
    c = np.array([people._center(b) for b in tr["boxes"]])
    jitter = float(np.linalg.norm(c.std(axis=0)) / s)
    print(f" {tr['id']:4} | {t0:5.1f}-{t1:5.1f} | {t1 - t0:3.0f} | {main:25} | {jitter:6.3f}        | "
          f"{'да' if tr['id'] in cids else ('двойник (отражение?)' if tr['id'] in twins else ('фон с начала' if tr['id'] in bg else ''))}")
print(" * разброс центра рамки в размерах человека; < 0.02 весь ролик — подозрительно (не движется совсем)")

video = find_video(a.videos, vid)
if video is None:
    sys.exit(f"нет видео {vid} в {a.videos}")
times = a.t if a.t else [dets[0]["t"] + 1, dets[len(dets) // 2]["t"], dets[-1]["t"] - 1]
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
P = people.persons(dets)
for t in times:
    fi = min(range(len(dets)), key=lambda i: abs(dets[i]["t"] - t))
    d = dets[fi]
    im = Image.open(io.BytesIO(frame_at(str(video), d["t"]))).convert("RGB")
    k = im.width / d["w"]
    dr = ImageDraw.Draw(im)
    kept = {id(b) for b in P[fi]}
    for b in d["boxes"]:
        x1, y1, x2, y2 = (v * k for v in b[:4])
        if b[5] == "laptop":
            dr.rectangle([x1, y1, x2, y2], outline=(255, 210, 0), width=3)
            continue
        if b[5] != "person":
            continue
        if id(b) not in kept:  # ниже порога или дубль
            for x in range(int(x1), int(x2), 10):
                dr.line([x, y1, x + 5, y1], fill=(150, 150, 150))
                dr.line([x, y2, x + 5, y2], fill=(150, 150, 150))
            continue
        tid = owner.get(id(b))
        r = people.role(b, scene)
        if tid in cids:
            col, label = (0, 220, 0), "CASHIER"
        elif tid in twins:
            col, label = (0, 140, 0), "TWIN (mirror?)"
        elif tid is None or tid not in steady:
            col, label = (160, 160, 160), "SHORT"
        elif r == "cashier_side" and tid in bg:
            col, label = (255, 160, 0), "BACKGROUND"
        elif r == "cashier_side":
            col, label = (255, 40, 40), "CASHIER SIDE"
        elif r == "visitor":
            col, label = (40, 120, 255), "VISITOR"
        else:
            col, label = (160, 160, 160), "OTHER"
        dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
        txt = f"#{tid} {label} {b[4]:.2f}"
        tb = dr.textbbox((x1 + 4, y1 + 4), txt)
        dr.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(0, 0, 0))
        dr.text((x1 + 4, y1 + 4), txt, fill=col)
    if scene["anchor"] and scene["dir"]:
        ax, ay = (v * k for v in scene["anchor"])
        dx, dy = scene["dir"]
        dr.line([ax, ay, ax + dx * s * k * 1.5, ay + dy * s * k * 1.5], fill=(255, 255, 255), width=3)
        dr.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill=(255, 255, 255))
    dst = out / f"{vid}_{d['t']:05.1f}s.jpg"
    im.save(dst, quality=90)
    print(f"сохранено: {dst}")
