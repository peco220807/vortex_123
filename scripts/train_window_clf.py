"""Обучить классификатор окон на сохранённых фактах (data/facts) и честно проверить.

Проверка «отложи одно видео»: для каждого из N видео модель учится на остальных N-1 и предсказывает отложенное.
Так метрики показывают работу на НЕвиденных видео.

  pip install scikit-learn joblib
  python scripts\\train_window_clf.py              # сравнить модели, сохранить лучшую в models/
  python scripts\\train_window_clf.py --no-save
  python scripts\\train_window_clf.py --ids        # показать ID видео в ошибках (не пересылать наружу!)

Разметка окна: в видео с нарушением окно, пересекающее [t-2, t+2], — «нарушение»; окна в 2..8 с от t
не используются (переходная зона); окна дальше 8 с — «норма»; все окна чистых видео — «норма».

Перебираются модели (logreg, forest) и агрегации окон (max, pair, ctx — см. pipeline/window_model.py).
Выбор: не больше 1 пропуска, затем минимум ошибок.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import signals  # noqa: E402
from pipeline.window_model import AGGS, MODEL_PATH, features, smooth  # noqa: E402
from scripts.common import norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--facts", default="data/facts")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--no-save", action="store_true")
ap.add_argument("--ids", action="store_true", help="печатать ID видео в списках ошибок")
a = ap.parse_args()

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

labels = {norm(r["video_id"]): r for r in csv.DictReader(open(a.labels, encoding="utf-8"))}
videos = []
for p in sorted(Path(a.facts).glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    lab = labels.get(norm(d["video_id"]))
    if not lab or lab["split"] != "train" or not d["timeline"]:
        continue
    sig = {k: np.array(v) for k, v in d["sig"].items()}
    X, names = features(d["timeline"], sig)
    y_vid = lab["violation"] == "1"
    t = float(lab["timecode_sec"]) if lab["timecode_sec"] else None
    yw, use = [], []
    for w in d["timeline"]:
        if not y_vid:
            yw.append(0)
            use.append(True)
        else:
            hit = w["lo"] <= t + 2 and w["hi"] >= t - 2
            far = abs((w["lo"] + w["hi"]) / 2 - t) > 8
            yw.append(int(hit))
            use.append(hit or far)
    videos.append(dict(id=d["video_id"], X=X, yw=np.array(yw), use=np.array(use), y=y_vid, t=t,
                       timeline=d["timeline"], sig=sig))

print(f"Видео: {len(videos)} (с нарушением {sum(v['y'] for v in videos)}), "
      f"окон: {sum(len(v['X']) for v in videos)}, признаков: {len(names)}")

MODELS = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight="balanced",
                                                                        max_iter=2000)),
    "forest": lambda: RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=3,
                                             class_weight="balanced_subsample", random_state=0),
}


def fit(make, vids):
    X = np.vstack([v["X"][v["use"]] for v in vids])
    y = np.concatenate([v["yw"][v["use"]] for v in vids])
    return make().fit(X, y)


def loo(make):
    out = []
    for i, v in enumerate(videos):
        clf = fit(make, videos[:i] + videos[i + 1:])
        out.append(clf.predict_proba(v["X"])[:, 1])
    return out


THRESHOLDS = {
    "max": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "pair": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "ctx": [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6],
}


def timecode(v, probs):
    i = int(np.argmax(probs))
    lo = v["timeline"][i - 1]["lo"] if i else v["timeline"][i]["lo"]
    t = signals.peak_motion_time(v["sig"], lo, v["timeline"][i]["hi"])
    return (v["timeline"][i]["lo"] + v["timeline"][i]["hi"]) / 2 if t is None else t


def ids(lst):
    return lst if a.ids else len(lst)


def report(name, raw_probs, agg):
    probs = [smooth(p, agg) for p in raw_probs]
    scores = np.array([p.max() if len(p) else 0.0 for p in probs])
    ys = np.array([v["y"] for v in videos])
    print(f"\n=== {name} + {agg} (проверка на невиденных видео) ===")
    print(" порог | пропуски | ложные | точность")
    best = None
    for thr in THRESHOLDS[agg]:
        pred = scores >= thr
        fn, fp = int((ys & ~pred).sum()), int((~ys & pred).sum())
        acc = (pred == ys).mean()
        print(f"  {thr:<4} |   {fn:3}    |  {fp:3}   |  {acc:.0%}")
        key = (fn > 1, fp + fn, fn)  # не больше 1 пропуска, затем минимум ошибок
        if best is None or key < best[0]:
            best = (key, thr, fn, fp, acc)
    key, thr, fn, fp, acc = best
    errs = [abs(timecode(v, p) - v["t"]) for v, p, s in zip(videos, probs, scores) if v["y"] and s >= thr]
    miss = [v["id"] for v, s in zip(videos, scores) if v["y"] and s < thr]
    false = [v["id"] for v, s in zip(videos, scores) if not v["y"] and s >= thr]
    print(f" выбран порог {thr}: точность {acc:.0%}, пропуски {ids(miss)}, ложные {ids(false)}")
    if errs:
        print(f" таймкод: медиана ошибки {np.median(errs):.1f} с, в пределах 3 с: "
              f"{sum(e <= 3 for e in errs)}/{len(errs)}")
    return key, thr, acc


base_rate = max(np.mean([v["y"] for v in videos]), 1 - np.mean([v["y"] for v in videos]))
results = {}
for name, make in MODELS.items():
    raw = loo(make)  # LOO один раз на модель, агрегации — поверх
    for agg in AGGS:
        results[(name, agg)] = report(name, raw, agg)

(best_name, best_agg) = min(results, key=lambda k: results[k][0])
_, thr, acc = results[(best_name, best_agg)]
print(f"\nЛучшая: {best_name} + {best_agg}, порог {thr}, точность {acc:.0%} "
      f"(угадайка «всегда один ответ»: {base_rate:.0%})")

clf = fit(MODELS[best_name], videos)
if best_name == "forest":
    top = sorted(zip(clf.feature_importances_, names), reverse=True)[:10]
    print("Самые важные признаки:", ", ".join(f"{n} {w:.2f}" for w, n in top))

if not a.no_save:
    import joblib
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"clf": clf, "threshold": thr, "kind": best_name, "agg": best_agg,
                 "features": list(names)}, MODEL_PATH)
    print(f"Сохранено: {MODEL_PATH}")
