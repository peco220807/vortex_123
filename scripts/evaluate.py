"""Оценка ответов сервиса по разметке.

Ответы: папка с JSON в формате API (раздел 10 ТЗ), файл = <имя видео>.json,
например preds/Х-1.json или preds/X-1.json (латинская/кириллическая X не важна).

Запуск:
  python evaluate.py --labels labels_clean.csv --preds preds/ --split train
  python evaluate.py --labels labels_clean.csv --preds preds/ --split holdout
"""
import argparse, csv, json, pathlib, statistics

def norm(name):  # латинская X -> кириллица Х, без расширения
    return pathlib.Path(name).stem.strip().replace("X", "Х").replace("x", "Х")

ap = argparse.ArgumentParser()
ap.add_argument("--labels", default="labels_clean.csv")
ap.add_argument("--preds", required=True)
ap.add_argument("--split", default="all", choices=["train", "holdout", "all"])
ap.add_argument("--tol", type=float, default=2.0, help="допуск по таймкоду, сек")
a = ap.parse_args()

labels = {}
for r in csv.DictReader(open(a.labels, encoding="utf-8")):
    if a.split == "all" or r["split"] == a.split:
        labels[norm(r["video_id"])] = (int(r["violation"]), float(r["timecode_sec"]) if r["timecode_sec"] else None)

preds = {norm(p.name): json.load(open(p, encoding="utf-8")) for p in pathlib.Path(a.preds).glob("*.json")}

tp = fp = fn = tn = 0
missing, misses, false_alarms, errs, extra = [], [], [], [], []
for vid, (y, t) in sorted(labels.items(), key=lambda kv: int(kv[0].split("-")[1])):
    p = preds.get(vid)
    if p is None or p.get("status") != "done":
        missing.append(vid)  # как у жюри: не обработано = ошибка
        y and misses.append(vid); fn += y; fp += 0; continue
    pv = 1 if p.get("verdict") == "violation" else 0
    if y and pv:
        tp += 1
        tcs = [v["timecode_sec"] for v in p.get("violations", [])]
        if tcs:
            e = min(abs(x - t) for x in tcs); errs.append((vid, e))
            extra.append((vid, len(tcs) - 1)) if len(tcs) > 1 else None
    elif y and not pv: fn += 1; misses.append(vid)
    elif pv and not y: fp += 1; false_alarms.append(vid)
    else: tn += 1

n = len(labels)
print(f"Видео: {n} (split={a.split}), ответов найдено: {n - len(missing)}")
print(f"Точность вердикта: {(tp + tn) / n:.1%}   TP={tp} TN={tn} FP={fp} FN={fn}")
print(f"ПРОПУСКИ (недопустимы): {len(misses)}  {misses}")
print(f"Ложные срабатывания:    {len(false_alarms)}  {false_alarms}")
if missing: print(f"Нет ответа / failed:   {missing}")
if errs:
    es = [e for _, e in errs]
    print(f"Таймкод: медианная ошибка {statistics.median(es):.1f} с, "
          f"в пределах {a.tol:.0f} с: {sum(e <= a.tol for e in es)}/{len(es)}")
    bad = [(v, round(e, 1)) for v, e in errs if e > a.tol]
    if bad: print(f"  Промахи по времени: {bad}")
if extra: print(f"Лишние события в видео с нарушением: {extra}")
