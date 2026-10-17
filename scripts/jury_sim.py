"""Имитация жюри: гоняет видео через ПУБЛИЧНЫЙ API так, как это будет делать жюри.

  python scripts/jury_sim.py --url https://<наш-адрес> --key <ключ> --videos data/videos --parallel 5
  python scripts/evaluate.py --labels data/labels_clean.csv --preds preds_api/ --split train

Что проверяет:
  * все видео отправляются ОДНОВРЕМЕННО (--parallel), как может сделать жюри;
  * время от POST до done по каждому видео; > 300 с = видео не засчитано;
  * ответ строго по схеме раздела 10 ТЗ (pipeline/schema.validate);
  * запрос без ключа отклоняется (401).
Ответы кладёт в preds_api/<видео>.json (картинки сохраняются отдельно в preds_api/frames/),
чтобы их можно было оценить тем же evaluate.py.
"""
import argparse
import base64
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.schema import validate  # noqa: E402
from scripts.common import find_video  # noqa: E402

LIMIT = 300.0

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--key", required=True)
ap.add_argument("--videos", default="data/videos")
ap.add_argument("--labels", default=None, help="labels_clean.csv — чтобы взять только нужный split")
ap.add_argument("--split", default="train", choices=["train", "holdout", "all"])
ap.add_argument("--limit", type=int, default=0, help="взять только первые N видео")
ap.add_argument("--parallel", type=int, default=5, help="сколько видео отправлять одновременно")
ap.add_argument("--poll", type=float, default=3.0)
ap.add_argument("--out", default="preds_api")
a = ap.parse_args()
base = a.url.rstrip("/")

# --- список видео
if a.labels:
    vids = []
    for r in csv.DictReader(open(a.labels, encoding="utf-8")):
        if a.split == "all" or r["split"] == a.split:
            p = find_video(a.videos, r["video_id"])
            if p:
                vids.append((r["video_id"], p))
            else:
                print(f"[нет файла] {r['video_id']}")
else:
    vids = [(p.stem, p) for p in sorted(Path(a.videos).iterdir()) if p.suffix.lower() in (".mov", ".mp4")]
if a.limit:
    vids = vids[: a.limit]
if not vids:
    sys.exit("Нет видео для прогона")

out = Path(a.out)
(out / "frames").mkdir(parents=True, exist_ok=True)

# --- проверка ключа
r = requests.get(f"{base}/api/v1/jobs/nonexistent", timeout=30)
print(f"Запрос без ключа -> {r.status_code} {'OK' if r.status_code == 401 else '!!! ДОЛЖНО БЫТЬ 401'}")
H = {"X-API-Key": a.key}


def run_one(name: str, path: Path) -> dict:
    t0 = time.time()
    try:
        with open(path, "rb") as f:
            r = requests.post(f"{base}/api/v1/analyze", headers=H, files={"file": (path.name, f)}, timeout=120)
        r.raise_for_status()
        job_id = r.json()["job_id"]
    except Exception as e:  # noqa: BLE001
        return {"name": name, "status": "upload_error", "error": str(e), "sec": time.time() - t0}
    upload_sec = time.time() - t0
    while True:
        time.sleep(a.poll)
        try:
            j = requests.get(f"{base}/api/v1/jobs/{job_id}", headers=H, timeout=30).json()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] ошибка опроса: {e}")
            continue
        if j.get("status") in ("done", "failed") or time.time() - t0 > LIMIT + 60:
            break
    sec = time.time() - t0
    res = {"name": name, "job_id": job_id, "status": j.get("status"), "sec": sec, "upload_sec": upload_sec}
    if j.get("status") == "done":
        try:
            validate(j)
            res["schema"] = "ok"
        except Exception as e:  # noqa: BLE001
            res["schema"] = f"НЕВЕРНО: {e!r}"
        res["verdict"] = j.get("verdict")
        res["n_viol"] = len(j.get("violations", []))
        res["kb"] = len(json.dumps(j)) / 1024
        # картинки -> файлы, в JSON оставляем только имя файла
        for vi, v in enumerate(j.get("violations", [])):
            for fi, fr in enumerate(v.get("evidence_frames", [])):
                fn = f"{name}_v{vi}_{fi}_{fr.get('timestamp_sec')}.jpg"
                try:
                    (out / "frames" / fn).write_bytes(base64.b64decode(fr["image_base64"]))
                except Exception:  # noqa: BLE001
                    res["schema"] = "НЕВЕРНО: image_base64 не декодируется"
                fr["image_base64"] = f"<frames/{fn}>"
    else:
        res["error"] = j.get("error")
    (out / f"{name}.json").write_text(json.dumps(j, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


print(f"Отправляем {len(vids)} видео, одновременно до {a.parallel}, на {base}\n")
T0 = time.time()
results = []
with ThreadPoolExecutor(max_workers=a.parallel) as ex:
    futs = [ex.submit(run_one, n, p) for n, p in vids]
    for fu in as_completed(futs):
        r = fu.result()
        results.append(r)
        flag = "  <-- БОЛЬШЕ 5 МИН, НЕ ЗАСЧИТАЕТСЯ" if r["sec"] > LIMIT else ""
        print(f"{r['name']:12} {r['status']:12} {r.get('verdict', ''):13} "
              f"нарушений={r.get('n_viol', '-'):<2} {r['sec']:6.1f} с  "
              f"ответ={r.get('kb', 0):6.0f} КБ  схема={r.get('schema', '-')}{flag}"
              + (f"  ошибка: {r['error']}" if r.get("error") else ""))

secs = [r["sec"] for r in results]
bad = [r["name"] for r in results if r["status"] != "done" or r["sec"] > LIMIT or r.get("schema", "ok") != "ok"]
print(f"\nВсего {len(results)} видео за {time.time() - T0:.0f} с. "
      f"Макс. время: {max(secs):.0f} с, среднее: {sum(secs) / len(secs):.0f} с.")
print(f"Не засчитаются у жюри: {len(bad)} {bad if bad else ''}")
print(f"Ответы: {out}/  ->  python scripts/evaluate.py --preds {out}/ ...")
