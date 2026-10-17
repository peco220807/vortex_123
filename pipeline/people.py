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
DET_THR = float(os.getenv("PEOPLE_THR", "0.35"))            # порог при детекции (сохраняем с запасом)
PERSON_THR = float(os.getenv("PERSON_THR", "0.5"))          # порог для людей при разборе (меняется без перезапуска)
DUP_IOU = 0.45       # две рамки людей перекрываются сильнее — это один человек
DUP_INSIDE = 0.75    # маленькая рамка на 75% внутри большой (рука, голова) — тот же человек
MIN_ROLE_SEC = 2.0   # «посторонний», «за спиной», «посетитель» — только тот, кого видно >= 2 с
TWIN_OVERLAP = 0.85  # «двойник кассира» (отражение в зеркале / стекле, отдельная рамка рук): виден >= 85%
TWIN_CORR = 0.3      # времени кассира и двигается синхронно с ним (корреляция скоростей >= 0.3)
# классы COCO, которые нам нужны (остальные выбрасываем сразу)
KEEP = {"person", "laptop", "dining table", "keyboard", "tv", "handbag", "backpack", "suitcase", "chair"}
DESK_CLASSES = ("laptop", "keyboard", "tv")  # что считаем «рабочим местом кассира», по приоритету

# параметры зон — в размерах человека (s)
CASHIER_R = 2.5      # сторона кассира: не дальше 2.5 s от рабочего места (стоящий «за спиной» дальше самого кассира)
VISITOR_R = 3.0      # посетитель: на другой стороне и не дальше 3 s
TRACK_GATE = 0.6     # за 0.5 с (кадр при 2 к/с) человек сдвигается не больше чем на 0.6 своего размера
TRACK_MAX_GAP = 3    # кадров без рамки, после которых считаем, что человек ушёл (1.5 с при 2 к/с)
TRACK_MIN_LEN = 2    # короче — мерцание детектора, не учитываем
START_SEC = 3.0      # кассир — кто на рабочем месте в первые 3 с видео
REJOIN_SEC = 6.0     # трек кассира оборвался и через <= 6 с рядом начался новый -> это он же
REJOIN_R = 1.5       # ... не дальше 1.5 s от места, где оборвался
BG_START_SEC = 1.0   # фон — то, что видно с первой секунды ролика
BG_REJOIN_SEC = 3.0  # обрывок «фона»: начинается не позже 3 с после того, как фон был виден,
BG_REJOIN_R = 0.8    # ... и не дальше 0.8 s от него (строже, чтобы не «съесть» подошедшего вплотную)
BG_COVER = 0.6       # фон виден >= 60% ролика (клиент, пришедший в начале и ушедший, — не фон)
BEHIND_DEEP = float(os.getenv("BEHIND_DEEP", "0.6"))           # «за спиной»: глубже кассира на >= 0.6 s — фон
BEHIND_VIS_SHARE = float(os.getenv("BEHIND_VIS_SHARE", "0.3"))  # ... или >= 30% трека на стороне клиента — клиент

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


def _dedupe(boxes: list[list]) -> list[list]:
    """Убрать дубли одного человека: сильное перекрытие или рамка почти целиком внутри другой."""
    keep = []
    for b in sorted(boxes, key=lambda b: -b[4]):
        area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
        dup = False
        for k in keep:
            ix = max(0.0, min(b[2], k[2]) - max(b[0], k[0]))
            iy = max(0.0, min(b[3], k[3]) - max(b[1], k[1]))
            area_k = max(1.0, (k[2] - k[0]) * (k[3] - k[1]))
            if _iou(b, k) > DUP_IOU or ix * iy / min(area_b, area_k) > DUP_INSIDE:
                dup = True
                break
        if not dup:
            keep.append(b)
    return keep


_pcache = {}


def persons(dets: list[dict]) -> list[list]:
    """Люди по кадрам: порог PERSON_THR + склейка дублей. Возвращает те же объекты рамок (важно для треков)."""
    key = id(dets)
    if key not in _pcache or _pcache[key][0] is not dets:
        _pcache.clear()
        _pcache[key] = (dets, [_dedupe([b for b in d["boxes"] if b[5] == "person" and b[4] >= PERSON_THR])
                               for d in dets])
    return _pcache[key][1]


def track(dets: list[dict]) -> list[dict]:
    """Сопровождение людей между кадрами. -> [{"id", "frames": [индексы кадров], "boxes": [...]}].
    Связываем по расстоянию между центрами (в размерах человека), жадно от ближайших пар: люди, стоящие
    вплотную (кассир и тот, кто рядом), не путаются, как при связывании по перекрытию рамок."""
    P = persons(dets)
    sizes = [math.sqrt(max(1.0, (b[2] - b[0]) * (b[3] - b[1]))) for boxes in P for b in boxes]
    s = float(np.median(sizes)) if sizes else 100.0
    gate = TRACK_GATE * s
    tracks, active = [], []
    for fi, boxes in enumerate(P):
        pairs = []
        for ti, tr in enumerate(active):
            last = tr["boxes"][-1]
            gap = fi - tr["frames"][-1]
            for bi, b in enumerate(boxes):
                dist = math.dist(_center(last), _center(b))
                if dist <= gate * (1 + 0.3 * (gap - 1)):  # после пропуска кадра допускаем чуть больший сдвиг
                    pairs.append((dist, ti, bi))
        pairs.sort()
        taken_t, used = set(), set()
        for dist, ti, bi in pairs:
            if ti in taken_t or bi in used:
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


def _near_anchor_frames(tr, scene) -> int:
    a, s = scene.get("anchor"), scene.get("s")
    if a is None or not s:
        return len(tr["frames"])
    return sum(math.dist(_center(b), a) < CASHIER_R * s for b in tr["boxes"])


def cashier_tracks(dets: list[dict], tracks: list[dict], scene: dict) -> set[int]:
    """id треков, которые считаем кассиром (см. описание модуля, шаг 4).
    В начале видео кассир ОДИН — кто из стоящих в зоне кассира дольше всех у рабочего места. Дальше — его
    продолжения: новый трек в зоне кассира, начавшийся в пределах REJOIN_SEC после конца трека кассира
    и рядом с местом, где тот оборвался (кассир не может быть в двух местах, поэтому только после конца)."""
    if not dets:
        return set()
    t_start, s = dets[0]["t"], scene.get("s") or 1.0

    def in_zone(tr):
        head = tr["boxes"][:3]
        return sum(role(b, scene) == "cashier_side" for b in head) * 2 > len(head)

    first = [tr for tr in tracks if dets[tr["frames"][0]]["t"] <= t_start + START_SEC and in_zone(tr)]
    if not first:
        return set()
    main = max(first, key=lambda tr: _near_anchor_frames(tr, scene))
    ids = {main["id"]}
    ended = [(dets[main["frames"][-1]]["t"], main["boxes"][-1])]
    for tr in sorted(tracks, key=lambda tr: tr["frames"][0]):
        if tr["id"] in ids or not in_zone(tr):
            continue
        t0 = dets[tr["frames"][0]]["t"]
        for t1, last in ended:
            if 0 < t0 - t1 <= REJOIN_SEC and math.dist(_center(last), _center(tr["boxes"][0])) <= REJOIN_R * s:
                ids.add(tr["id"])
                ended.append((dets[tr["frames"][-1]]["t"], tr["boxes"][-1]))
                break
    return ids


def background_tracks(dets: list[dict], tracks: list[dict], cashier_ids: set[int], scene: dict) -> set[int]:
    """«Фон» сцены: то, что видно С САМОГО НАЧАЛА и остаётся ПОЧТИ ВЕСЬ РОЛИК (отражение в зеркале, руки кассира,
    коллега рядом), плюс обрывки этого же объекта (новый трек начинается рядом с местом, где фон был в последние
    BG_REJOIN_SEC, когда самого фона в кадре уже нет). Клиент, который пришёл в начале и потом ушёл, — НЕ фон:
    цепочка должна покрывать >= BG_COVER ролика.
    Для «за спиной» / «чужой на месте» / подсчёта клиентов фон не считается: нарушение — когда кто-то ПОДОШЁЛ
    по ходу видео (Х-30: «сзади подошёл посторонний»)."""
    if not dets:
        return set()
    t_start, s = dets[0]["t"], scene.get("s") or 1.0
    duration = max(1e-6, dets[-1]["t"] - t_start)
    chains = []  # [[треки цепочки]]
    for tr in sorted(tracks, key=lambda tr: tr["frames"][0]):
        if tr["id"] in cashier_ids:
            continue
        t0 = dets[tr["frames"][0]]["t"]
        if t0 <= t_start + BG_START_SEC:
            chains.append([tr])
            continue
        c0, f0 = _center(tr["boxes"][0]), tr["frames"][0]
        for ch in chains:
            if any(f0 in m["frames"] for m in ch):  # объект цепочки виден в тот же момент — это другой объект
                continue
            hit = any(t0 - BG_REJOIN_SEC <= dets[f]["t"] < t0 and math.dist(_center(b), c0) <= BG_REJOIN_R * s
                      for m in ch for f, b in zip(m["frames"], m["boxes"]))
            if hit:
                ch.append(tr)
                break
    ids = set()
    for ch in chains:
        covered = sum(dets[m["frames"][-1]]["t"] - dets[m["frames"][0]]["t"] for m in ch)
        if covered >= BG_COVER * duration:
            ids |= {m["id"] for m in ch}
    return ids


def side_excluded(dets: list[dict], tracks: list[dict], cashier_ids: set[int], scene: dict) -> set[int]:
    """Треки, которые на стороне кассира НЕ считаем «подошедшим посторонним» (ложные Х-5, 12, 14, 25, 48):
      - клиент, зашедший на край стола: >= BEHIND_VIS_SHARE его рамок — на стороне клиента;
      - фон за спиной кассира (стена, проход, отражение): в среднем глубже кассира на >= BEHIND_DEEP s
        по оси «стол -> кассир». Подошедший к кассиру встаёт рядом или между ним и столом (Х-30: -0.8 s)."""
    a, d, s = scene.get("anchor"), scene.get("dir"), scene.get("s")
    if a is None or d is None or not s:
        return set()
    along = lambda b: ((_center(b)[0] - a[0]) * d[0] + (_center(b)[1] - a[1]) * d[1]) / s  # noqa: E731
    cash = [along(b) for tr in tracks if tr["id"] in cashier_ids for b in tr["boxes"]
            if role(b, scene) == "cashier_side"]
    c_along = float(np.median(cash)) if cash else None
    out = set()
    for tr in tracks:
        if tr["id"] in cashier_ids:
            continue
        roles = [role(b, scene) for b in tr["boxes"]]
        side = [b for b, r in zip(tr["boxes"], roles) if r == "cashier_side"]
        if not side:
            continue
        if BEHIND_VIS_SHARE > 0 and roles.count("visitor") >= BEHIND_VIS_SHARE * len(roles):
            out.add(tr["id"])
        elif BEHIND_DEEP > 0 and c_along is not None and float(np.median([along(b) for b in side])) - c_along >= BEHIND_DEEP:
            out.add(tr["id"])
    return out


def _speed(dets, tr) -> dict[int, float]:
    """Скорость центра рамки по кадрам трека: {индекс кадра: смещение с прошлого кадра}."""
    out = {}
    for (f0, b0), (f1, b1) in zip(zip(tr["frames"], tr["boxes"]), zip(tr["frames"][1:], tr["boxes"][1:])):
        out[f1] = math.dist(_center(b0), _center(b1)) / max(1, f1 - f0)
    return out


def twin_tracks(dets: list[dict], tracks: list[dict], cashier_ids: set[int]) -> set[int]:
    """Треки-«двойники» кассира: отражение в зеркале / стекле или отдельная рамка его рук.
    Признак: появился вместе с кассиром, виден почти всё его время и двигается синхронно с ним.
    Живой второй человек двигается по-своему — корреляция скоростей низкая."""
    cash = [tr for tr in tracks if tr["id"] in cashier_ids]
    if not cash or not dets:
        return set()
    t_start = dets[0]["t"]
    cash_frames = set().union(*(tr["frames"] for tr in cash))
    cash_speed = {}
    for tr in cash:
        cash_speed.update(_speed(dets, tr))
    twins = set()
    for tr in tracks:
        if tr["id"] in cashier_ids or dets[tr["frames"][0]]["t"] > t_start + START_SEC:
            continue
        overlap = len(set(tr["frames"]) & cash_frames) / max(1, len(cash_frames))
        if overlap < TWIN_OVERLAP:
            continue
        sp = _speed(dets, tr)
        common = sorted(set(sp) & set(cash_speed))
        if len(common) < 6:
            continue
        a = np.array([sp[i] for i in common])
        b = np.array([cash_speed[i] for i in common])
        if a.std() < 1e-6 or b.std() < 1e-6:
            continue
        if float(np.corrcoef(a, b)[0, 1]) >= TWIN_CORR:
            twins.add(tr["id"])
    return twins


def window_facts(dets: list[dict], tracks: list[dict], scene: dict, lo: float, hi: float,
                 cashier_ids: set[int] | None = None, twins: set[int] | None = None,
                 background: set[int] | None = None, excluded: set[int] | None = None) -> dict | None:
    """Факты о людях в окне [lo, hi]. None, если в окне нет кадров."""
    idx = [i for i, d in enumerate(dets) if lo <= d["t"] <= hi]
    if not idx:
        return None
    cashier_ids = cashier_tracks(dets, tracks, scene) if cashier_ids is None else cashier_ids
    twins = twin_tracks(dets, tracks, cashier_ids) if twins is None else twins
    background = background_tracks(dets, tracks, cashier_ids, scene) if background is None else background
    excluded = side_excluded(dets, tracks, cashier_ids, scene) if excluded is None else excluded
    owner, steady = {}, set()  # id(рамки) -> id трека; треки, которые видно >= MIN_ROLE_SEC
    for tr in tracks:
        for b in tr["boxes"]:
            owner[id(b)] = tr["id"]
        if dets[tr["frames"][-1]]["t"] - dets[tr["frames"][0]]["t"] >= MIN_ROLE_SEC:
            steady.add(tr["id"])
    P = persons(dets)
    counts, cashier, visitors, behind, stranger = [], [], [], [], []
    t_behind = t_stranger = None
    for i in idx:
        n_vis = n_cash = n_other_side = 0
        n_seen = 0
        for b in P[i]:
            tid = owner.get(id(b))
            if tid is None or tid in twins:  # мерцание детектора / двойник кассира (отражение)
                continue
            n_seen += 1
            r = role(b, scene)
            if tid in cashier_ids:
                n_cash += r == "cashier_side"
            elif tid not in steady:
                continue  # мелькнул меньше 2 с — для ролей не учитываем
            elif r == "visitor" and tid not in background:  # фон с начала видео (Х-1, Х-3) — не клиент
                n_vis += 1
            elif r == "cashier_side" and tid not in background and tid not in excluded:  # фон / клиент у края
                n_other_side += 1
        counts.append(n_seen)
        cashier.append(n_cash >= 1)
        visitors.append(n_vis)
        behind.append(n_cash >= 1 and n_other_side >= 1)    # кассир на месте и рядом с ним ещё кто-то
        stranger.append(n_cash == 0 and n_other_side >= 1)  # кассира нет, а на его месте кто-то есть
        if behind[-1] and t_behind is None:
            t_behind = dets[i]["t"]
        if stranger[-1] and t_stranger is None:
            t_stranger = dets[i]["t"]
    first_t, last_t = dets[0]["t"], dets[-1]["t"]
    entering = leaving = False
    for tr in tracks:
        if tr["id"] not in steady or tr["id"] in twins:
            continue
        t0, t1 = dets[tr["frames"][0]]["t"], dets[tr["frames"][-1]]["t"]
        if lo <= t0 <= hi and t0 > first_t + 1.0:
            entering = True
        if lo <= t1 <= hi and t1 < last_t - 1.0:
            leaving = True
    med = lambda xs: int(round(float(np.median(xs))))  # noqa: E731
    two = lambda xs: sum(xs) >= 2 or (len(xs) <= 2 and any(xs))  # noqa: E731 — хотя бы 2 кадра (1 с)
    # момент, когда кассира в последний раз видели на месте (до конца окна) — таймкод «ушёл»
    t_left = None
    for tr in tracks:
        if tr["id"] in cashier_ids:
            ts = [dets[f]["t"] for f in tr["frames"] if dets[f]["t"] <= hi]
            if ts:
                t_left = max(t_left or ts[-1], ts[-1])
    t_people = {}
    if t_stranger is not None or t_behind is not None:
        t_people["unauthorized_person"] = t_stranger if t_stranger is not None else t_behind
    if t_left is not None and sum(cashier) * 2 < len(cashier):
        t_people["workplace_abandoned"] = t_people["unattended_valuables"] = t_left
    return {
        "t_people": t_people,
        "people_count": med(counts),
        "cashier_at_desk": sum(cashier) * 2 >= len(cashier),  # большинство кадров окна
        "visitor_count": med(visitors),
        "person_entering": entering,
        "person_leaving": leaving,
        "behind_cashier": two(behind),
        "stranger_at_desk": two(stranger),
    }


PEOPLE_KEYS = ("people_count", "cashier_at_desk", "visitor_count", "person_entering", "person_leaving")
# t_people: {тип нарушения: точный момент по детектору} — detector.events_from_timeline берёт таймкод отсюда


def merge(timeline: list[dict], dets: list[dict], side_hint: str | None = None) -> dict:
    """Заменить в timeline факты о людях на посчитанные по детектору. Возвращает раскладку сцены."""
    tracks = track(dets)
    scene = layout(dets, tracks, side_hint)
    cids = cashier_tracks(dets, tracks, scene)
    tw = twin_tracks(dets, tracks, cids)
    bg = background_tracks(dets, tracks, cids, scene)
    scene["cashier_tracks"], scene["twins"], scene["background"] = len(cids), len(tw), len(bg)
    for w in timeline:
        pf = window_facts(dets, tracks, scene, w["lo"], w["hi"], cids, tw, bg)
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


def diag(dets: list[dict], side_hint: str | None = None) -> dict:
    """Короткая диагностика по видео: людей в кадре до/после склейки, треки, сколько окон с каждой ролью."""
    raw = [sum(1 for b in d["boxes"] if b[5] == "person") for d in dets]
    P = persons(dets)
    tracks = track(dets)
    scene = layout(dets, tracks, side_hint)
    cids = cashier_tracks(dets, tracks, scene)
    tw = twin_tracks(dets, tracks, cids)
    bg = background_tracks(dets, tracks, cids, scene)
    t_end = dets[-1]["t"] if dets else 0
    wins = [window_facts(dets, tracks, scene, lo, lo + 4, cids, tw, bg) for lo in np.arange(0, t_end + 0.01, 4)]
    wins = [w for w in wins if w]
    cnt = lambda k: sum(bool(w[k]) for w in wins)  # noqa: E731
    return {"raw": float(np.mean(raw)) if raw else 0.0, "kept": float(np.mean([len(x) for x in P])) if P else 0.0,
            "tracks": len(tracks), "cashier_tracks": len(cids), "twins": len(tw), "background": len(bg), "windows": len(wins),
            "no_cashier": sum(not w["cashier_at_desk"] for w in wins), "behind": cnt("behind_cashier"),
            "stranger": cnt("stranger_at_desk"), "vis2": sum(w["visitor_count"] >= 2 for w in wins),
            "side": side_name(scene["dir"])}


def summary(dets: list[dict]) -> dict:
    """Факты по видео целиком — для сверки с тем, что видно глазами (scripts/people_check.py)."""
    if not dets:
        return {}
    tracks = track(dets)
    scene = layout(dets, tracks)
    cids = cashier_tracks(dets, tracks, scene)
    tw = twin_tracks(dets, tracks, cids)
    bg = background_tracks(dets, tracks, cids, scene)
    sc = scene.get("s") or 1.0
    # клиенты: устойчивые треки на стороне посетителя (не фон), обрывки одного клиента склеиваем
    vis = []
    for tr in sorted(tracks, key=lambda tr: tr["frames"][0]):
        if tr["id"] in cids or tr["id"] in tw or tr["id"] in bg:
            continue
        if dets[tr["frames"][-1]]["t"] - dets[tr["frames"][0]]["t"] < MIN_ROLE_SEC:
            continue
        roles = [role(b, scene) for b in tr["boxes"]]
        if roles.count("visitor") * 2 < len(roles):
            continue
        t0 = dets[tr["frames"][0]]["t"]
        same = any(0 <= t0 - v["t1"] <= 2.0 and math.dist(v["last"], _center(tr["boxes"][0])) <= 0.8 * sc
                   for v in vis)
        if same:
            v = next(v for v in vis if 0 <= t0 - v["t1"] <= 2.0)
            v["t1"], v["last"] = dets[tr["frames"][-1]]["t"], _center(tr["boxes"][-1])
        else:
            vis.append({"t1": dets[tr["frames"][-1]]["t"], "last": _center(tr["boxes"][-1])})
    t_end = dets[-1]["t"]
    wins = [window_facts(dets, tracks, scene, lo, lo + 4, cids, tw, bg) for lo in np.arange(0, t_end + 0.01, 4)]
    wins = [w for w in wins if w]
    seen = False
    leaves = False
    for w in wins:
        if w["cashier_at_desk"]:
            seen = True
        elif seen:
            leaves = True
    return {"clients": len(vis), "cashier_side": side_name(scene["dir"]), "cashier_leaves": int(leaves),
            "approached": int(any(w["behind_cashier"] or w["stranger_at_desk"] for w in wins))}
