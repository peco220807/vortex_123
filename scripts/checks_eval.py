"""Оценка новой схемы (детектор + прицельная VLM) по сохранённым ответам data/checks/ — за секунды.

  python scripts\\checks_eval.py
  python scripts\\checks_eval.py --tol 3
  set CHECKS_OFF=mixed_operations          (выключить тип и сравнить)
  set TRUST_STRANGER=0                     («чужой на месте» — только с подтверждением VLM)

Печатает: вердикт, пропуски, ложные, таймкод, лишние срабатывания, тип (если заполнен type_hypothesis),
и по каждому вопросу VLM — как часто он отвечает «нарушение» на видео С нарушением и БЕЗ: вопрос, который
одинаково часто «да» и там и там, ничего не различает — его надо менять или выключать.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import checks  # noqa: E402
from pipeline.violation_types import TYPES  # noqa: E402
from scripts.common import norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--checks", default="data/checks")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--tol", type=float, default=3.0)
ap.add_argument("--extra", type=float, default=5.0, help="событие дальше стольких секунд от метки — лишнее")
ap.add_argument("--sweep", action="store_true", help="перебор порогов режима без VLM (checks.decide_novlm)")
ap.add_argument("--people", default="data/people", help="рамки детектора: для --sweep кандидаты пересчитываются")
a = ap.parse_args()

labels = {norm(r["video_id"]): r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}


def sweep():
    """Без VLM: перебор порогов, на каждый вариант — пропуски/ложные/таймкод. Видео — те, что есть в --checks."""
    import itertools
    vids = []
    for p in sorted(Path(a.checks).glob("*.json")):
        vid = norm(json.loads(p.read_text(encoding="utf-8"))["video_id"])
        pp = Path(a.people) / f"{vid}.json"
        if vid in labels and pp.exists():
            lab = labels[vid]
            cands = checks.candidates(json.loads(pp.read_text(encoding="utf-8"))["dets"], None)
            vids.append((vid, lab["violation"] == "1",
                         float(lab["timecode_sec"]) if lab["timecode_sec"] else None, cands))
    n_pos = sum(v[1] for v in vids)
    print(f"Видео: {len(vids)} (с нарушением {n_pos}, без {len(vids) - n_pos}); кандидаты пересчитаны из {a.people}")
    grid = {"left_min": [0, 2, 3, 4, 6, 8, 12], "left_end": [0, 1, 2, 3],
            "behind_min": [0, 1, 2, 3, 4, 6, 8], "stranger_min": [0, 1, 2, 3, 5]}
    res = []
    for combo in itertools.product(*grid.values()):
        prm = dict(zip(grid, combo))
        fp, fn, tc = [], [], 0
        for vid, y, tt, cands in vids:
            ev = checks.decide_novlm(cands, prm)
            if y and not ev:
                fn.append(vid)
            elif ev and not y:
                fp.append(vid)
            elif y and tt is not None and min(abs(e["t"] - tt) for e in ev) <= a.tol:
                tc += 1
        res.append((len(fp), len(fn), -tc, prm, fp, fn, tc))
    res.sort(key=lambda r: r[:3])
    front, best_fn = [], 10 ** 9
    for r in res:  # парето: меньше ложных -> не больше пропусков
        if r[1] < best_fn:
            front.append(r)
            best_fn = r[1]
    print("Парето-фронт (ложные, пропуски) — для каждого числа ложных лучший по пропускам и таймкоду:")
    for fpn, fnn, _, prm, fp, fn, tc in front:
        acc = (len(vids) - fpn - fnn) / max(1, len(vids))
        print(f"  FP={fpn:2} FN={fnn:2} точн={acc:.0%} таймкод±{a.tol:.0f}с={tc:2}  {prm}")
        print(f"      ложные={fp}  пропуски={fn}")


if a.sweep:
    sweep()
    sys.exit()
rows, secs = [], []
extra_total = type_ok = type_n = 0
qstat = defaultdict(lambda: [0, 0, 0, 0])  # (вид, вопрос) -> [да на видео с нарушением, всего, да на чистых, всего]
for p in sorted(Path(a.checks).glob("*.json"), key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0)):
    d = json.loads(p.read_text(encoding="utf-8"))
    lab = labels.get(norm(d["video_id"]))
    if not lab:
        continue
    y = lab["violation"] == "1"
    t_true = float(lab["timecode_sec"]) if lab["timecode_sec"] else None
    cands = d["candidates"]
    for c in cands:
        for k in checks.PLAN.get(c["kind"], ([], ""))[0]:
            ans = (c.get("answers") or {}).get(k)
            if ans is None:
                continue
            st = qstat[(c["kind"], k)]
            if y:
                st[0] += checks.positive(k, ans)
                st[1] += 1
            else:
                st[2] += checks.positive(k, ans)
                st[3] += 1
    ev = checks.decide(cands)
    err = min((abs(e["t"] - t_true) for e in ev), default=None) if (y and ev and t_true is not None) else None
    extra = len(ev) if not y else (sum(abs(e["t"] - t_true) > a.extra for e in ev) if t_true is not None else 0)
    extra_total += extra
    want = (lab.get("type_hypothesis") or "").strip().lower()
    want_id = next((k for k, v in TYPES.items() if want and (want == k or want == v["name"].lower())), None)
    if y and want_id:
        type_n += 1
        type_ok += want_id in [e["type_id"] for e in ev if abs(e["t"] - t_true) <= a.extra]
    rows.append((d["video_id"], y, bool(ev), t_true, err, ev, len(cands)))
    if "sec" in d:
        secs.append((d["sec"], d["video_id"], d.get("model", "?")))

tp = sum(y and pr for _, y, pr, *_ in rows)
tn = sum((not y) and (not pr) for _, y, pr, *_ in rows)
fn = [v for v, y, pr, *_ in rows if y and not pr]
fp = [v for v, y, pr, *_ in rows if pr and not y]
errs = [r[4] for r in rows if r[4] is not None]
n_pos = sum(r[1] for r in rows)
print(f"Видео: {len(rows)}  |  точность вердикта: {(tp + tn) / max(1, len(rows)):.0%}  "
      f"(TP={tp} TN={tn} FP={len(fp)} FN={len(fn)})  |  угадайка: {max(n_pos, len(rows) - n_pos) / max(1, len(rows)):.0%}")
print(f"ПРОПУСКИ: {fn}")
print(f"ЛОЖНЫЕ:   {fp}")
if errs:
    print(f"Таймкод: медиана ошибки {np.median(errs):.1f} с, в пределах {a.tol:.0f} с: "
          f"{sum(e <= a.tol for e in errs)}/{len(errs)}")
print(f"Лишние срабатывания: {extra_total} (на чистых видео + дальше {a.extra:.0f} с от метки)")
print(f"Тип нарушения угадан: {type_ok}/{type_n}" if type_n else
      "Тип: в labels_clean.csv не заполнен type_hypothesis — не считаю")

if secs:
    print(f"Время VLM на видео ({secs[0][2]}): медиана {np.median([x[0] for x in secs]):.0f} с, "
          f"макс {max(secs)[0]:.0f} с ({max(secs)[1]})")
print("\nВопросы VLM: доля ответов «нарушение» на видео С нарушением | БЕЗ (хороший вопрос: слева больше)")
for (kind, k), (pv, nv, pc, nc) in sorted(qstat.items()):
    l = f"{pv}/{nv} ({pv / nv:.0%})" if nv else "-"
    r = f"{pc}/{nc} ({pc / nc:.0%})" if nc else "-"
    print(f"  {kind:10} {k:13} {l:>14} | {r:>14}")

print("\nПо видео:")
for v, y, pr, t_true, err, ev, nc in rows:
    mark = "OK " if y == pr else "ERR"
    found = ", ".join(f"{e['type_id']}@{e['t']:.1f}" for e in ev) or "-"
    print(f"  {mark} {v:6} метка={'Н' if y else '-'}{'' if t_true is None else f'@{t_true:.0f}'} "
          f"кандидатов={nc:2} найдено: {found}" + (f"  ошибка={err:.1f}" if err is not None else ""))
