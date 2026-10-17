"""Обучить классификатор окон на сохранённых фактах (data/facts) и честно проверить.

Проверка «отложи одно видео»: для каждого из N видео модель учится на остальных N-1 и предсказывает отложенное.
Так метрики показывают работу на НЕвиденных видео.

  pip install scikit-learn joblib
  python scripts\\train_window_clf.py              # сравнить модели, сохранить лучшую в models/
  python scripts\\train_window_clf.py --no-save

Разметка окна: в видео с нарушением окно, пересекающее [t-2, t+2], — «нарушение»; окна дальше 8 с от t
не используются (там могут быть другие, неразмеченные нарушения); все окна чистых видео — «норма».
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import signals  # noqa: E402
from pipeline.window_model import MODEL_PATH, features  # noqa: E402
from scripts.common import norm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--facts", default="data/facts")
ap.add_argument("--labels", default="data/labels_clean.csv")
ap.add_argument("--no-save", action="store_true")
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


def timecode(v, probs):
    i = int(np.argmax(probs))
    lo = v["timeline"][i - 1]["lo"] if i else v["timeline"][i]["lo"]
    t = signals.peak_motion_time(v["sig"], lo, v["timeline"][i]["hi"])
    return (v["timeline"][i]["lo"] + v["timeline"][i]["hi"]) / 2 if t is None else t


def report(name, probs):
    scores = np.array([p.max() for p in probs])
    ys = np.array([v["y"] for v in videos])
    print(f"\n=== {name} (проверка на невиденных видео) ===")
    print(" порог | пропуски | ложные | точность")
    best = None
    for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = scores >= thr
        fn, fp = int((ys & ~pred).sum()), int((~ys & pred).sum())
        acc = (pred == ys).mean()
        print(f"  {thr:.1f}  |   {fn:3}    |  {fp:3}   |  {acc:.0%}")
        key = (fn > 1, fp + fn, fn)  # не больше 1 пропуска, затем минимум ошибок
        if best is None or key < best[0]:
            best = (key, thr, fn, fp, acc)
    _, thr, fn, fp, acc = best
    errs = [abs(timecode(v, p) - v["t"]) for v, p, s in zip(videos, probs, scores) if v["y"] and s >= thr]
    miss = [v["id"] for v, s in zip(videos, scores) if v["y"] and s < thr]
    false = [v["id"] for v, s in zip(videos, scores) if not v["y"] and s >= thr]
    print(f" выбран порог {thr}: точность {acc:.0%}, пропуски {miss}, ложные {false}")
    if errs:
        print(f" таймкод: медиана ошибки {np.median(errs):.1f} с, в пределах 3 с: "
              f"{sum(e <= 3 for e in errs)}/{len(errs)}")
    return (fn > 1, fp + fn, fn), thr


results = {name: report(name, loo(make)) for name, make in MODELS.items()}
best_name = min(results, key=lambda k: results[k][0])
thr = results[best_name][1]
print(f"\nЛучшая модель: {best_name}, порог {thr}")

if best_name == "forest":
    clf = fit(MODELS[best_name], videos)
    top = sorted(zip(clf.feature_importances_, names), reverse=True)[:10]
    print("Самые важные признаки:", ", ".join(f"{n} {w:.2f}" for w, n in top))

if not a.no_save:
    import joblib
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"clf": fit(MODELS[best_name], videos), "threshold": thr, "kind": best_name,
                 "features": names}, MODEL_PATH)
    print(f"Сохранено: {MODEL_PATH}")
