"""Сверить факты детектора с тем, что видно глазами — по каждому факту, на ВСЕХ видео (и чистых тоже).

1. Первый запуск создаёт заготовку data/people_truth.csv (локально, в git не попадает) со списком видео.
2. Заполните колонки глазами (пустую клетку скрипт пропускает — можно заполнять постепенно):
     clients         — сколько РАЗНЫХ клиентов подходило к кассе за ролик (0, 1, 2, ...)
     cashier_side    — где сидит кассир на картинке относительно ноутбука: top / bottom / left / right
     cashier_leaves  — кассир уходит с рабочего места (1) или нет (0)
     approached      — кто-то посторонний подходит к кассиру / садится на его место (1) или нет (0)
3. Запустите ещё раз — увидите, где детектор ошибается.

  python scripts\\people_check.py
  python scripts\\people_check.py --only-errors
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import people  # noqa: E402
from scripts.common import norm  # noqa: E402

COLS = ["clients", "cashier_side", "cashier_leaves", "approached"]

ap = argparse.ArgumentParser()
ap.add_argument("--truth", default="data/people_truth.csv")
ap.add_argument("--people", default="data/people")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--split", default="train")
ap.add_argument("--only-errors", action="store_true")
a = ap.parse_args()

truth_path = Path(a.truth)
if not truth_path.exists():
    vids = [norm(r["video_id"]) for r in csv.DictReader(open(a.labels, encoding="utf-8"))
            if a.split == "all" or r["split"] == a.split]
    with open(truth_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")  # ; — чтобы Excel на русской Windows открыл по колонкам
        w.writerow(["video_id"] + COLS + ["notes"])
        for v in vids:
            w.writerow([v] + [""] * (len(COLS) + 1))
    print(f"Создана заготовка {truth_path} ({len(vids)} видео). Заполните колонки и запустите ещё раз.")
    print("Колонки:", ", ".join(COLS), "— описание в начале этого скрипта.")
    sys.exit(0)

text = truth_path.read_text(encoding="utf-8-sig")
delim = ";" if text.count(";") >= text.count(",") else ","
truth = {norm(r["video_id"]): r for r in csv.DictReader(text.splitlines(), delimiter=delim)}

stats = {c: [0, 0] for c in COLS}  # [совпало, всего заполнено]
lines = []
for vid, t in truth.items():
    p = Path(a.people) / f"{vid}.json"
    if not p.exists():
        continue
    got = people.summary(json.loads(p.read_text(encoding="utf-8"))["dets"])
    cells, bad = [], False
    for c in COLS:
        want = (t.get(c) or "").strip().lower()
        have = str(got.get(c, "")).lower()
        if not want:
            cells.append(f"{c}={have}")
            continue
        ok = want == have
        stats[c][0] += ok
        stats[c][1] += 1
        bad |= not ok
        cells.append(f"{c}={have}" + ("" if ok else f" (надо {want})"))
    if bad or not a.only_errors:
        lines.append(f"  {'ERR' if bad else 'OK '} {vid:6} " + " | ".join(cells))

print("Детектор против глаз:")
print("\n".join(lines))
print("\nТочность по фактам (только заполненные клетки):")
for c, (ok, n) in stats.items():
    print(f"  {c:15} {ok:3}/{n:<3}" + (f" ({ok / n:.0%})" if n else ""))
