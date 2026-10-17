"""Люди относительно рабочего места кассира (Чел 1).

Готовый детектор (RT-DETR v2, Apache 2.0, через transformers — БЕЗ Ultralytics) рисует рамки «человек»,
«ноутбук», «стол» на каждом кадре. Конкретных людей НЕ узнаём, лица НЕ распознаём, на датасете НЕ обучаем:
роль человека определяется только тем, где он находится относительно рабочего места.

Схема:
  1. detect()     — рамки на кадрах (2 кадра/с, те же кадры, что идут в VLM).
  2. track()      — сопровождение рамок между кадрами (жадное по IoU), чтобы видеть «вошёл / ушёл».
  3. layout()     — раскладка сцены ОДИН раз на видео: где рабочее место (ноутбук / стол) и с какой
                    стороны от него кассир. Сторона берётся из подсказки VLM («куда смотрит экран»),
                    иначе — сторона, где дольше всех находился человек у рабочего места.
                    Все расстояния — в «размерах человека» (медиана sqrt(площади рамки)), а не в пикселях:
                    так правила не зависят от камеры и масштаба.
  4. cashier_tracks() — КТО кассир: тот, кто был на рабочем месте с начала видео (+ его продолжение, если
                    детектор на пару секунд потерял человека). Кто пришёл на место кассира позже — посторонний:
                    без распознавания личности иначе не отличить, а в датасете это и есть нарушение
                    («женщина входит в пустой кабинет и садится за стол»).
  5. window_facts() — факты по окну [lo, hi]: people_count, cashier_at_desk, visitor_count,
                    person_entering, person_leaving, behind_cashier, stranger_at_desk.
                    Они ЗАМЕНЯЮТ те же факты от VLM.

Сырые рамки сохраняются (scripts/people_dump.py -> data/people/<видео>.json), а шаги 2–4 — чистый Python:
параметры зон можно перебирать за секунды через scripts/rules_eval.py --people data/people.

Переменные: PEOPLE_MODEL, PEOPLE_DET=0 (выключить), PEOPLE_THR.
"""
import math
import os
from pathlib import Path

import numpy as np

MODEL_ID = os.getenv("PEOPLE_MODEL", "PekingU/rtdetr_v2_r18vd")
DET_THR = float(os.getenv("PEOPLE_THR", "0.35"))
# классы COCO, которые нам нужны (остальные выбрасываем сразу)
KEEP = {"person", "laptop", "dining table", "keyboard", "tv", "handbag", "backpack", "suitcase", "chair"}
DESK_CLASSES = ("laptop", "keyboard", "tv")  # что считаем «рабочим местом кассира», по приоритету

# параметры зон — в размерах человека (s)
CASHIER_R = 2.0      # кассир: на своей стороне и не дальше 2 s от рабочего места
VISITOR_R = 3.0      # посетитель: на другой стороне и не дальше 3 s
TRACK_IOU = 0.2
TRACK_MAX_GAP = 3    # кадров без рамки, после которых считаем, что человек ушёл (1.5 с при 2 к/с)
TRACK_MIN_LEN = 2    # короче — мерцание детектора, не учитываем
START_SEC = 2.0      # кассир — кто на рабочем месте в первые 2 с видео
REJOIN_SEC = 4.0     # трек кассира оборвался и через <= 4 с рядом начался новый -> это он же
REJOIN_R = 1.0       # ... не дальше 1 s от места, где оборвался

SIDES = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "top": (0.0, -1.0), "bottom": (0.0, 1.0)}

_model = None
_proc = None


def enabled() -> bool:
    return os.getenv("PEOPLE_DET", "1") != "0"


def load():
    global _model, _proc
    if _model is None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        _proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForObjectDetection.from_pretrained(MODEL_ID).eval()
        if torch.cuda.is_available():
            _model = _model.to("cuda").half()
    return _model, _proc


def unload():
    global _model, _proc
    _model = _proc = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def detect(frames: list[tuple[float, Path]], batch: int = 8) -> list[dict]:
    """[(t, path)] -> [{"t", "w", "h", "boxes": [[x1, y1, x2, y2, score, label], ...]}]."""
    import torch
    from PIL import Image

    model, proc = load()
    id2label = model.config.id2label
    out = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        imgs = [Image.open(p).convert("RGB") for _, p in chunk]
        inputs = proc(images=imgs, return_tensors="pt").to(model.device)
        if model.dtype == torch.float16:
            inputs["pixel_values"] = inputs["pixel_values"].half()
        with torch.inference_mode():
            res = model(**inputs)
        sizes = [(im.height, im.width) for im in imgs]
        post = proc.post_process_object_detection(res, target_sizes=sizes, threshold=DET_THR)
        for (t, _), im, r in zip(chunk, imgs, post):
            boxes = []
            for box, score, lab in zip(r["boxes"].tolist(), r["scores"].tolist(), r["labels"].tolist()):
                name = id2label.get(int(lab), str(lab))
                if name in KEEP:
                    boxes.append([round(v, 1) for v in box] + [round(float(score), 3), name])
            out.append({"t": round(float(t), 2), "w": im.width, "h": im.height, "boxes": boxes})
    return out


# ---------------------------------- чистый Python: всё ниже работает на сохранённых рамках ----------------------

def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def persons(dets: list[dict]) -> list[list]:
    return [[b for b in d["boxes"] if b[5] == "person"] for d in dets]


def track(dets: list[dict]) -> list[dict]:
    """Жадное сопровождение людей по IoU. -> [{"id", "frames": [индексы кадров], "boxes": [...]}]."""
    P = persons(dets)
    tracks, active = [], []
    for fi, boxes in enumerate(P):
        used = set()
        # продолжаем активные треки лучшими совпадениями
        pairs = sorted(((_iou(tr["boxes"][-1], b), ti, bi) for ti, tr in enumerate(active)
                        for bi, b in enumerate(boxes)), reverse=True)
        taken_t = set()
        for iou, ti, bi in pairs:
            if iou < TRACK_IOU or ti in taken_t or bi in used:
                continue
            active[ti]["frames"].append(fi)
            active[ti]["boxes"].append(boxes[bi])
            taken_t.add(ti)
            used.add(bi)
        for bi, b in enumerate(boxes):
            if bi not in used:
                tr = {"id": len(tracks), "frames": [fi], "boxes": [b]}
                tracks.append(tr)
                active.append(tr)
        active = [tr for tr in active if fi - tr["frames"][-1] <= TRACK_MAX_GAP]
    return [tr for tr in tracks if len(tr["frames"]) >= TRACK_MIN_LEN]


def _median_point(pts):
    a = np.asarray(pts, dtype=float)
    return (float(np.median(a[:, 0])), float(np.median(a[:, 1])))


def layout(dets: list[dict], tracks: list[dict] | None = None, side_hint: str | None = None) -> dict:
    """Раскладка сцены: anchor (рабочее место), dir (единичный вектор к стороне кассира), s (размер человека)."""
    tracks = track(dets) if tracks is None else tracks
    P = [b for boxes in persons(dets) for b in boxes]
    s = float(np.median([math.sqrt(max(1.0, (b[2] - b[0]) * (b[3] - b[1]))) for b in P])) if P else None
    anchor, anchor_src = None, None
    for cls in DESK_CLASSES:
        pts = [_center(b) for d in dets for b in d["boxes"] if b[5] == cls]
        if len(pts) >= max(2, len(dets) // 5):  # предмет виден хотя бы на 1/5 кадров
            anchor, anchor_src = _median_point(pts), cls
            break
    if anchor is None:
        pts = [_center(b) for d in dets for b in d["boxes"] if b[5] == "dining table"]
        if pts:
            anchor, anchor_src = _median_point(pts), "dining table"

    # «основной» человек у рабочего места: дольше всех рядом с ним; кто был там с самого начала — в приоритете
    main_pos = None
    if tracks:
        t_start = dets[0]["t"] if dets else 0.0

        def near_count(tr):
            if anchor is None or not s:
                n = len(tr["frames"])
            else:
                n = sum(math.dist(_center(b), anchor) < CASHIER_R * s for b in tr["boxes"])
            from_start = dets[tr["frames"][0]]["t"] <= t_start + START_SEC
            return n * (1.5 if from_start else 1.0)

        best = max(tracks, key=near_count)
        if near_count(best) > 0:
            main_pos = _median_point([_center(b) for b in best["boxes"]])

    # сторона кассира: ГЕОМЕТРИЯ — главная (любое направление, не только 4); VLM — только если людей у стола нет.
    # 2B-модель на датасете почти всегда отвечала left/right, хотя кассир сверху или снизу (проверено глазами).
    direction, dir_src = None, None
    if anchor is not None and main_pos is not None and math.dist(anchor, main_pos) > 0.05 * (s or 1.0):
        dx, dy = main_pos[0] - anchor[0], main_pos[1] - anchor[1]
        n = math.hypot(dx, dy)
        direction, dir_src = (dx / n, dy / n), "dwell"
    elif side_hint in SIDES and os.getenv("SIDE_FROM_VLM", "1") != "0":
        direction, dir_src = SIDES[side_hint], "vlm"
    return {"anchor": anchor, "anchor_src": anchor_src, "dir": direction, "dir_src": dir_src,
            "main_pos": main_pos, "s": s}


def role(box, scene: dict) -> str:
    """'cashier_side' | 'visitor' | 'other' для рамки человека."""
    s = scene.get("s")
    if not s:
        return "other"
    c = _center(box)
    a, d = scene.get("anchor"), scene.get("dir")
    if a is not None and d is not None:
        v = (c[0] - a[0], c[1] - a[1])
        dist = math.hypot(*v) / s
        along = (v[0] * d[0] + v[1] * d[1]) / s
        if along > -0.25 and dist <= CASHIER_R:
            return "cashier_side"
        if along <= -0.25 and dist <= VISITOR_R:
            return "visitor"
        return "other"
    # нет рабочего места — только по месту «основного» человека
    m = scene.get("main_pos")
    if m is not None and math.dist(c, m) / s <= 0.8:
        return "cashier_side"
    return "other"


def cashier_tracks(dets: list[dict], tracks: list[dict], scene: dict) -> set[int]:
    """id треков, которые считаем кассиром (см. описание модуля, шаг 4)."""
    if not dets:
        return set()
    t_start, s = dets[0]["t"], scene.get("s") or 1.0
    ids, ended = set(), []  # ended: (t_конца, последняя рамка) треков кассира
    for tr in sorted(tracks, key=lambda tr: tr["frames"][0]):
        t0 = dets[tr["frames"][0]]["t"]
        head = tr["boxes"][:3]
        in_zone = sum(role(b, scene) == "cashier_side" for b in head) * 2 > len(head)
        if not in_zone:
            continue
        if t0 <= t_start + START_SEC:
            ids.add(tr["id"])
        else:
            for t1, last in ended:
                if 0 < t0 - t1 <= REJOIN_SEC and math.dist(_center(last), _center(tr["boxes"][0])) <= REJOIN_R * s:
                    ids.add(tr["id"])
                    break
        if tr["id"] in ids:
            ended.append((dets[tr["frames"][-1]]["t"], tr["boxes"][-1]))
    return ids


def window_facts(dets: list[dict], tracks: list[dict], scene: dict, lo: float, hi: float,
                 cashier_ids: set[int] | None = None) -> dict | None:
    """Факты о людях в окне [lo, hi]. None, если в окне нет кадров."""
    idx = [i for i, d in enumerate(dets) if lo <= d["t"] <= hi]
    if not idx:
        return None
    cashier_ids = cashier_tracks(dets, tracks, scene) if cashier_ids is None else cashier_ids
    owner = {}  # id(рамки) -> id трека
    for tr in tracks:
        for b in tr["boxes"]:
            owner[id(b)] = tr["id"]
    P = persons(dets)
    counts, cashier, visitors, behind, stranger = [], [], [], [], []
    for i in idx:
        n_vis = n_cash = n_other_side = 0
        for b in P[i]:
            r = role(b, scene)
            if r == "visitor":
                n_vis += 1
            elif r == "cashier_side":
                if owner.get(id(b)) in cashier_ids:
                    n_cash += 1
                else:
                    n_other_side += 1
        counts.append(len(P[i]))
        cashier.append(n_cash >= 1)
        visitors.append(n_vis)
        behind.append(n_cash >= 1 and n_other_side >= 1)    # кассир на месте и рядом с ним ещё кто-то
        stranger.append(n_cash == 0 and n_other_side >= 1)  # кассира нет, а на его месте кто-то есть
    first_t, last_t = dets[0]["t"], dets[-1]["t"]
    entering = leaving = False
    for tr in tracks:
        t0, t1 = dets[tr["frames"][0]]["t"], dets[tr["frames"][-1]]["t"]
        if lo <= t0 <= hi and t0 > first_t + 1.0:
            entering = True
        if lo <= t1 <= hi and t1 < last_t - 1.0:
            leaving = True
    med = lambda xs: int(round(float(np.median(xs))))  # noqa: E731
    two = lambda xs: sum(xs) >= 2 or (len(xs) <= 2 and any(xs))  # noqa: E731 — хотя бы 2 кадра (1 с)
    return {
        "people_count": med(counts),
        "cashier_at_desk": sum(cashier) * 2 >= len(cashier),  # большинство кадров окна
        "visitor_count": med(visitors),
        "person_entering": entering,
        "person_leaving": leaving,
        "behind_cashier": two(behind),
        "stranger_at_desk": two(stranger),
    }


PEOPLE_KEYS = ("people_count", "cashier_at_desk", "visitor_count", "person_entering", "person_leaving")


def merge(timeline: list[dict], dets: list[dict], side_hint: str | None = None) -> dict:
    """Заменить в timeline факты о людях на посчитанные по детектору. Возвращает раскладку сцены."""
    tracks = track(dets)
    scene = layout(dets, tracks, side_hint)
    cids = cashier_tracks(dets, tracks, scene)
    scene["cashier_tracks"] = len(cids)
    for w in timeline:
        pf = window_facts(dets, tracks, scene, w["lo"], w["hi"], cids)
        if pf is None:
            continue
        f = w["f"]
        f["vlm_people"] = {k: f.get(k) for k in PEOPLE_KEYS}  # для сравнения
        f.update(pf)
    return scene


def side_name(direction) -> str:
    """Ближайшее из left/right/top/bottom — для печати и сравнения с ответом VLM."""
    if not direction:
        return "?"
    dx, dy = direction
    return ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) else ("bottom" if dy > 0 else "top")
