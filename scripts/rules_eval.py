"""Проверить правила на сохранённых фактах (data/facts/*.json) — без модели, за секунды.

  python scripts\\rules_eval.py
  python scripts\\rules_eval.py --tol 3
  python scripts\\rules_eval.py --people data\\people   # факты о людях — от детектора (scripts/people_dump.py)
  set RULES_OFF=mixed_operations,cash_intake_procedure   (выключить правила и сравнить)

Печатает только сводку — её можно пересылать (там нет кадров).
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.detector import events_from_timeline  # noqa: E402
from pipeline.violation_types import TYPES  # noqa: E402
from scripts.common import apply_people, norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--facts", default="data/facts")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--tol", type=float, default=3.0, help="допуск по таймкоду, сек")
ap.add_argument("--people", default="", help="папка с рамками детектора (data/people)")
a = ap.parse_args()

labels = {norm(r["video_id"]): r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
rows, per_rule_v, per_rule_c = [], Counter(), Counter()
EXTRA_SEC = 5.0
extra_total = type_ok = type_n = 0
used_people, people_stats = 0, {}
for p in sorted(Path(a.facts).glob("*.json"), key=lambda p: int(''.join(c for c in p.stem if c.isdigit()) or 0)):
    d = json.loads(p.read_text(encoding="utf-8"))
    lab = labels.get(norm(d["video_id"]))
    if not lab:
        continue
    sig = {k: np.array(v) for k, v in d["sig"].items()}
    used_people += apply_people(d, a.people, people_stats)
    ev = events_from_timeline(d["timeline"], sig)
    y = lab["violation"] == "1"
    t_true = float(lab["timecode_sec"]) if lab["timecode_sec"] else None
    types = sorted({e["type_id"] for e in ev})
    for tid in types:
        (per_rule_v if y else per_rule_c)[tid] += 1
    err = min((abs(e["t"] - t_true) for e in ev), default=None) if (y and ev) else None
    # лишние срабатывания: всё на чистом видео; в видео с нарушением — события дальше EXTRA_SEC от метки
    extra = len(ev) if not y else sum(abs(e["t"] - t_true) > EXTRA_SEC for e in ev) if t_true is not None else 0
    extra_total += extra
    # тип: если в разметке есть type_hypothesis, совпадающий с id или названием типа
    want = (lab.get("type_hypothesis") or "").strip().lower()
    want_id = next((k for k, v in TYPES.items() if want and (want == k or want == v["name"].lower())), None)
    if y and want_id:
        type_n += 1
        near = [e["type_id"] for e in ev if abs(e["t"] - t_true) <= EXTRA_SEC]
        type_ok += want_id in near
    rows.append((d["video_id"], y, bool(ev), t_true, err, types, [e["t"] for e in ev]))

tp = sum(y and p for _, y, p, *_ in rows)
fn = [v for v, y, p, *_ in rows if y and not p]
fp = [v for v, y, p, *_ in rows if p and not y]
tn = sum((not y) and (not p) for _, y, p, *_ in rows)
errs = [r[4] for r in rows if r[4] is not None]
print(f"Видео: {len(rows)}  |  точность вердикта: {(tp + tn) / max(1, len(rows)):.0%}  "
      f"(TP={tp} TN={tn} FP={len(fp)} FN={len(fn)})")
print(f"ПРОПУСКИ: {fn}")
print(f"ЛОЖНЫЕ:   {fp}")
if errs:
    print(f"Таймкод: медиана ошибки {np.median(errs):.1f} с, в пределах {a.tol:.0f} с: "
          f"{sum(e <= a.tol for e in errs)}/{len(errs)}")
print(f"Лишние срабатывания: {extra_total} (на чистых видео + дальше {EXTRA_SEC:.0f} с от метки)")
print(f"Тип нарушения угадан: {type_ok}/{type_n}" if type_n else
      "Тип нарушения: в разметке нет type_hypothesis с id/названием типа — не считаю")
print("\nСрабатывания правил (видео с нарушением / без):")
for tid in sorted(set(per_rule_v) | set(per_rule_c)):
    print(f"  {tid:24} {per_rule_v[tid]:3} / {per_rule_c[tid]:3}")
print("\nПо видео:")
for v, y, p, t_true, err, types, ts in rows:
    mark = "OK " if y == p else "ERR"
    print(f"  {mark} {v:6} метка={'Н' if y else '-'}{'' if t_true is None else f'@{t_true:.0f}'} "
          f"найдено={','.join(types) or '-'} t={ts[:3]}" + (f" ошибка={err:.1f}" if err is not None else ""))

if a.people:
    n = people_stats.get("n", 0)
    print(f"\nДетектор людей: подставлен в {used_people} видео; «кассир на месте» совпадает с VLM "
          f"в {people_stats.get('same', 0)}/{n} окон" + (f" ({people_stats.get('same', 0) / n:.0%})" if n else ""))
