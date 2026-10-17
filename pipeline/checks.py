"""Кандидаты от детектора людей + прицельные вопросы VLM по вырезу кадра (Чел 1).

Главный принцип: о нарушении сообщаем, только если его подтвердили ДВА независимых источника.
  1. Детектор людей (pipeline/people.py) находит КАНДИДАТОВ — моменты, где что-то произошло:
       left      — кассир ушёл с рабочего места (точный момент: когда его последний раз видели на месте);
       stranger  — кассира нет, а на его месте кто-то есть (пришёл по ходу видео);
       behind    — кассир работает, и к нему на его сторону ПОДОШЁЛ ещё кто-то;
       visit_end — клиент уходит (проверяем: не ушёл ли он до конца пересчёта);
       visit     — визит клиента целиком (проверяем: не вернули ли часть денег в руки, предъявлялся ли документ);
       second    — пришёл новый клиент, пока прежний у кассы или только что ушёл (деньги прежнего на столе?).
  2. VLM отвечает на 1–5 вопросов ТОЛЬКО по этому моменту и по ВЫРЕЗУ стола (крупно), а не по всему кадру.
     Вопросы с вариантами ответа («open / closed»), а не «да/нет» — маленькая модель меньше поддакивает.
  3. decide() — чистый Python: ответы -> события (тип, точный таймкод, кадры, описание без домыслов).

Сохранённые ответы (scripts/checks_dump.py -> data/checks/) позволяют перебирать decide() за секунды
(scripts/checks_eval.py). Модель VLM — переменная VLM_MODEL (по умолчанию Qwen3-VL-2B, влезает в 6 ГБ).
"""
import io
import math
import os
import re
from pathlib import Path

import numpy as np

from . import people

# ---------------------------------------------------------------- вопросы к VLM
# ключ -> (вопрос, допустимые ответы, ответ «нарушение»)
QUESTIONS = {
    "laptop": ("Is the laptop on the desk open or closed?", ("open", "closed", "none"), "open"),
    "cash_desk": ("Are banknotes or cash bundles lying on the desk?", ("yes", "no"), "yes"),
    "bag_desk": ("Is a cash collection bag (sealed bag for money) lying on the desk?", ("yes", "no"), "yes"),
    "safe": ("Is the door of the safe open or closed?", ("open", "closed", "none"), "open"),
    "keys": ("Are keys left hanging in the lock of the safe?", ("yes", "no"), "yes"),
    "two_staff": ("How many people are on the cashier's side of the desk (behind the desk, where the cashier "
                  "sits)? Answer with a number.", ("0", "1", "2", "3"), "2+"),
    "counting": ("Is the money counting machine running or is someone still counting banknotes?", ("yes", "no"),
                 "yes"),
    "hand_back": ("Is the cashier handing banknotes back into the client's hands?", ("yes", "no"), "yes"),
    "id_doc": ("Is an ID card or passport being handed over, held or looked at?", ("yes", "no"), "yes"),
    "someone_side": ("Is there a person on the cashier's side of the desk (where the cashier sits)?", ("yes", "no"),
                     "yes"),
    "money_in": ("Is the client handing banknotes to the cashier, or is the cashier counting the client's "
                 "banknotes?", ("yes", "no"), "yes"),
}

# какие вопросы для какого кандидата и по какому вырезу
PLAN = {
    "left": (["laptop", "cash_desk", "bag_desk", "safe", "keys"], "room"),
    "behind": (["two_staff"], "cashier"),
    "stranger": (["someone_side"], "cashier"),
    "visit_end": (["counting"], "desk"),
    "visit": (["money_in", "hand_back", "id_doc"], "desk"),
    "second": (["cash_desk"], "desk"),
}

LEFT_MIN_SEC = 2.0     # кассира нет хотя бы 2 с подряд
LEFT_END_SEC = 1.0     # ... или 1 с, если он так и не вернулся до конца ролика
SEAT_R = 0.5           # «встал с места»: отошёл дальше 0.5 размера человека от места, где сидел
ENTER_R = float(os.getenv("ENTER_R", "0.5"))  # «кассир» в первые 3 с прошёл >= 0.5 s — это вошедший в пустой кабинет
SECOND_GAP_SEC = 8.0   # новый клиент пришёл, когда прежний ещё у кассы или ушёл меньше 8 с назад
CROP_R = 2.4           # вырез стола: ± 2.4 размера человека вокруг ноутбука
VISIT_REJOIN_SEC = 2.0  # обрывок того же клиента: не позже 2 с и не дальше 0.8 размера человека
VISIT_REJOIN_R = 0.8    # (иначе два клиента подряд на одном месте склеиваются в один визит)


def _runs(flags: list[bool], min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def state(dets: list[dict], side_hint: str | None = None) -> dict:
    """Всё, что детектор знает о видео: треки, роли, кто кассир, по кадрам."""
    tracks = people.track(dets)
    scene = people.layout(dets, tracks, side_hint)
    cids = people.cashier_tracks(dets, tracks, scene)
    tw = people.twin_tracks(dets, tracks, cids)
    bg = people.background_tracks(dets, tracks, cids, scene)
    ex = people.side_excluded(dets, tracks, cids, scene)  # для «за спиной»/«чужой на месте» не считаем
    owner = {id(b): tr["id"] for tr in tracks for b in tr["boxes"]}
    steady = {tr["id"] for tr in tracks
              if dets[tr["frames"][-1]]["t"] - dets[tr["frames"][0]]["t"] >= people.MIN_ROLE_SEC}
    P = people.persons(dets)
    frames = []
    for i, boxes in enumerate(P):
        st = {"cashier": [], "side": [], "visitor": []}
        for b in boxes:
            tid = owner.get(id(b))
            if tid is None or tid in tw:
                continue
            r = people.role(b, scene)
            if tid in cids:
                if r == "cashier_side":
                    st["cashier"].append(b)
            elif tid not in steady:
                continue
            elif r == "visitor" and tid not in bg:
                st["visitor"].append((tid, b))
            elif r == "cashier_side" and tid not in bg and tid not in ex:
                st["side"].append((tid, b))
        frames.append(st)
    return {"tracks": tracks, "scene": scene, "cids": cids, "bg": bg, "tw": tw, "steady": steady, "ex": ex,
            "frames": frames, "owner": owner}


def _seat(F) -> tuple[float, float] | None:
    """Где кассир сидел почти весь ролик: медиана центров его рамок."""
    pts = [people._center(b) for f in F for b in f["cashier"]]
    if not pts:
        return None
    a = np.asarray(pts)
    return (float(np.median(a[:, 0])), float(np.median(a[:, 1])))


def _visits(dets, S) -> list[dict]:
    """Визиты клиентов: обрывки одного клиента склеиваем (как в people.summary)."""
    sc = S["scene"].get("s") or 1.0
    vis = []
    for tr in sorted(S["tracks"], key=lambda tr: tr["frames"][0]):
        if tr["id"] in S["cids"] or tr["id"] in S["tw"] or tr["id"] in S["bg"] or tr["id"] not in S["steady"]:
            continue
        roles = [people.role(b, S["scene"]) for b in tr["boxes"]]
        if roles.count("visitor") * 2 < len(roles):
            continue
        t0, t1 = dets[tr["frames"][0]]["t"], dets[tr["frames"][-1]]["t"]
        prev = next((v for v in vis if 0 <= t0 - v["t1"] <= VISIT_REJOIN_SEC
                     and math.dist(v["last"], people._center(tr["boxes"][0])) <= VISIT_REJOIN_R * sc), None)
        if prev:
            prev["t1"], prev["last"] = t1, people._center(tr["boxes"][-1])
        else:
            vis.append({"t0": t0, "t1": t1, "last": people._center(tr["boxes"][-1])})
    return vis


def candidates(dets: list[dict], side_hint: str | None = None) -> list[dict]:
    """Кандидаты в нарушения: [{"kind", "t", "times": [моменты для кадров], "info": {...}}]."""
    if not dets:
        return []
    S = state(dets, side_hint)
    F = S["frames"]
    T = [d["t"] for d in dets]
    t_end = T[-1]
    fps = 1.0 / max(1e-6, (T[1] - T[0])) if len(T) > 1 else 2.0
    out = []

    # кассир ушёл: после того как он был на месте, его нет >= LEFT_MIN_SEC
    # (или >= LEFT_END_SEC, если он так и не вернулся до конца ролика: постановочные ролики кончаются сразу после ухода)
    present = [bool(f["cashier"]) for f in F]
    seen_idx = next((i for i, p in enumerate(present) if p), None)
    if seen_idx is not None:
        absent = [(not p) and i > seen_idx for i, p in enumerate(present)]
        seat = _seat(F)
        sc = S["scene"].get("s") or 100.0
        for a, b in _runs(absent, 2):
            at_end = b >= len(F) - 1
            if T[b] - T[a] + 1 / fps < (LEFT_END_SEC if at_end else LEFT_MIN_SEC):
                continue
            t_left = T[a - 1] if a > 0 else T[a]
            # момент, когда кассир ВСТАЛ со своего обычного места (разметчики отмечают начало ухода, а не выход
            # из кадра): последний кадр до ухода, где он был не дальше SEAT_R от места, где сидел почти весь ролик
            t_stand = t_left
            if seat is not None:
                for i in range(a - 1, -1, -1):
                    bx = F[i]["cashier"]
                    if bx and min(math.dist(people._center(x), seat) for x in bx) <= SEAT_R * sc:
                        t_stand = T[i]
                        break
            look = [min(t_end, t_left + d) for d in (0.5, 1.0, 1.5)]
            out.append({"kind": "left", "t": t_stand, "times": look,
                        "info": {"absent_sec": round(T[b] - T[a] + 1 / fps, 1), "t_left": t_left,
                                 "t_stand": t_stand, "at_end": at_end}})

    # пустой кабинет: тот, кого детектор считает кассиром, в начале ролика ВОШЁЛ (Х-32, 33, 40), а не сидел.
    # Настоящий кассир в первые 3 с почти неподвижен (сдвиг <= 0.33 s на всех видео); вошедший — 0.65..0.95 s
    first = min((tr for tr in S["tracks"] if tr["id"] in S["cids"]), key=lambda tr: tr["frames"][0], default=None)
    sc_s = S["scene"].get("s") or 100.0
    if first is not None and ENTER_R > 0:
        t0 = T[first["frames"][0]]
        pts = [(T[f], people._center(b)) for f, b in zip(first["frames"], first["boxes"]) if T[f] <= t0 + 3]
        if t0 <= T[0] + people.START_SEC and max(math.dist(pts[0][1], q) for _, q in pts) >= ENTER_R * sc_s:
            # таймкод — когда вошедший ближе всего к столу (трогает технику, Х-33, Х-40): первый кадр в 20% ближайших
            a = S["scene"].get("anchor") or _seat(F) or pts[-1][1]
            dd = [(T[f], math.dist(people._center(b), a)) for f, b in zip(first["frames"], first["boxes"])]
            q = float(np.percentile([x for _, x in dd], 20))
            t_in = next(t for t, x in dd if x <= q)
            out.append({"kind": "entered", "t": t_in, "times": [t0, t_in, min(t_end, t_in + 1.0)],
                        "info": {"t0": t0, "move": round(max(math.dist(pts[0][1], q) for _, q in pts) / sc_s, 2)}})

    # посторонний на месте кассира (кассира нет) / подошёл к работающему кассиру
    stranger = [bool(f["side"]) and not f["cashier"] for f in F]
    behind = [bool(f["side"]) and bool(f["cashier"]) for f in F]
    for a, b in _runs(stranger, 2):
        out.append({"kind": "stranger", "t": T[a], "times": [T[a], T[min(b, a + 2)]],
                    "info": {"dur": round(T[b] - T[a] + 1 / fps, 1)}})
    for a, b in _runs(behind, 2):
        out.append({"kind": "behind", "t": T[a], "times": [T[a], T[min(b, a + 2)]],
                    "info": {"dur": round(T[b] - T[a] + 1 / fps, 1)}})

    # визиты клиентов
    vis = _visits(dets, S)
    for k, v in enumerate(vis):
        dur = v["t1"] - v["t0"]
        if dur < 3:
            continue
        times = [v["t0"] + dur * q for q in (0.2, 0.4, 0.6, 0.8)]
        out.append({"kind": "visit", "t": v["t0"], "times": times, "info": {"t0": v["t0"], "t1": v["t1"]}})
        if v["t1"] < t_end - 1.0:  # клиент ушёл до конца ролика — не ушёл ли до конца пересчёта
            out.append({"kind": "visit_end", "t": v["t1"], "times": [v["t1"], min(t_end, v["t1"] + 1.5)],
                        "info": {"t0": v["t0"], "t1": v["t1"]}})
        if k > 0 and v["t0"] - vis[k - 1]["t1"] <= SECOND_GAP_SEC:
            out.append({"kind": "second", "t": v["t0"], "times": [v["t0"], min(t_end, v["t0"] + 1.5)],
                        "info": {"prev_t1": vis[k - 1]["t1"]}})
    return sorted(out, key=lambda c: c["t"])


# ---------------------------------------------------------------- вырезы кадров

def regions(dets: list[dict], side_hint: str | None = None) -> dict:
    """Прямоугольники вырезов в координатах кадров детектора: desk (стол), cashier (кассир и его сторона)."""
    S = state(dets, side_hint)
    sc = S["scene"]
    s = sc.get("s") or 100.0
    w, h = dets[0]["w"], dets[0]["h"]
    clip = lambda x1, y1, x2, y2: [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]  # noqa: E731
    a = sc.get("anchor") or (w / 2, h / 2)
    desk = clip(a[0] - CROP_R * s, a[1] - CROP_R * s, a[0] + CROP_R * s, a[1] + CROP_R * s)
    d = sc.get("dir") or (0.0, 0.0)
    c = (a[0] + d[0] * 1.5 * s, a[1] + d[1] * 1.5 * s)
    cashier = clip(c[0] - 2.2 * s, c[1] - 2.2 * s, c[0] + 2.2 * s, c[1] + 2.2 * s)
    return {"desk": desk, "cashier": cashier, "room": [0, 0, w, h]}


def crop_jpeg(path, box, det_w: int) -> bytes:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    k = im.width / det_w
    im = im.crop([int(v * k) for v in box])
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def nearest_frame(frames: list[tuple[float, Path]], t: float):
    return min(frames, key=lambda f: abs(f[0] - t))


# ---------------------------------------------------------------- VLM

def _prompt(keys: list[str]) -> str:
    lines = []
    for i, k in enumerate(keys):
        q, opts, _ = QUESTIONS[k]
        lines.append(f"{i + 1}. {q} ({' / '.join(opts)})")
    return ("These images come from a ceiling camera above a bank cashier desk, in time order"
            + (" (the last image may show the whole room)" if len(keys) > 1 else "") + ". "
            "Answer strictly about what is VISIBLE. If you cannot see it, answer no / none.\n"
            f"Answer ALL {len(keys)} questions on ONE line as number:answer, e.g. 1:yes 2:closed. No other words.\n"
            + "\n".join(lines))


def parse_answers(text: str, keys: list[str]) -> dict:
    out = {}
    for num, ans in re.findall(r"(?<!\d)(\d{1,2})\s*[:=.)]\s*([a-zA-Z]+|\d+)", text):
        i = int(num) - 1
        if 0 <= i < len(keys) and keys[i] not in out:
            out[keys[i]] = ans.lower()
    if len(keys) == 1 and not out:  # одиночный вопрос: модель могла ответить одним словом
        m = re.search(r"[a-zA-Z]+|\d+", text)
        if m:
            out[keys[0]] = m.group(0).lower()
    return out


def ask(cand: dict, frames: list[tuple[float, Path]], regs: dict, det_w: int) -> dict:
    """Задать VLM вопросы по кандидату. -> {ключ: ответ}, плюс "raw"."""
    from .vlm import _generate
    keys, reg = PLAN[cand["kind"]]
    imgs = []
    for t in cand["times"]:
        ft, p = nearest_frame(frames, t)
        box = regs["desk"] if reg == "room" else regs[reg]  # для «room»: крупно стол + в конце вся комната
        imgs.append((ft, crop_jpeg(p, box, det_w)))
    if reg == "room":
        ft, p = nearest_frame(frames, cand["times"][-1])
        imgs.append((ft, Path(p).read_bytes()))
    if SINGLE and len(keys) > 1:  # по одному вопросу за вызов: модель не путает номера и не копирует ответы
        ans, raws = {}, []
        for k in keys:
            text = _generate(imgs, _prompt([k]), max_new_tokens=16)
            ans.update(parse_answers(text, [k]))
            raws.append(text.strip()[:60])
        ans["raw"] = " | ".join(raws)
        return ans
    text = _generate(imgs, _prompt(keys), max_new_tokens=8 * len(keys) + 8)
    ans = parse_answers(text, keys)
    ans["raw"] = text.strip()[:200]
    return ans


# ---------------------------------------------------------------- решение (чистый Python)

def positive(key: str, ans: str | None) -> bool:
    if ans is None:
        return False
    want = QUESTIONS[key][2]
    if want == "2+":
        return ans.isdigit() and int(ans) >= 2
    return ans.startswith(want)


SINGLE = os.getenv("CHECKS_SINGLE", "0") == "1"  # каждый вопрос отдельным вызовом VLM
OFF = {x.strip() for x in os.getenv("CHECKS_OFF", "").split(",") if x.strip()}
TRUST_STRANGER = os.getenv("TRUST_STRANGER", "1") != "0"  # «чужой на месте кассира» — без подтверждения VLM


def decide(cands: list[dict]) -> list[dict]:
    """Кандидаты с ответами VLM -> события {type_id, t, evidence_t, why, description, score}."""
    ev = []

    def add(tid, t, why, ev_t):
        if tid in OFF:
            return
        ev.append({"type_id": tid, "t": round(float(t), 2), "why": why,
                   "evidence_t": sorted({round(float(x), 2) for x in ev_t})[:3], "score": 0.9})

    for c in cands:
        a = c.get("answers") or {}
        k, t = c["kind"], c["t"]
        if k == "left":
            if positive("cash_desk", a.get("cash_desk")) or positive("bag_desk", a.get("bag_desk")):
                what = "деньги" if positive("cash_desk", a.get("cash_desk")) else "инкассаторская сумка"
                add("unattended_valuables", t, f"кассир покинул рабочее место, на столе остались {what}",
                    [t] + c["times"][:1])
            if positive("safe", a.get("safe")) or positive("keys", a.get("keys")):
                what = "ключи остались в замке сейфа" if positive("keys", a.get("keys")) else "сейф остался открытым"
                add("safe_unsecured", t, f"кассир покинул рабочее место, {what}", [t] + c["times"][-1:])
            if positive("laptop", a.get("laptop")):
                add("workplace_abandoned", t, "кассир покинул рабочее место, ноутбук остался открытым",
                    [t, (c.get("info") or {}).get("t_left", t)] + c["times"][:1])
        elif k == "entered":
            add("unauthorized_person", t, "в пустой кабинет вошёл человек и занял рабочее место кассира", c["times"])
        elif k == "stranger":
            if TRUST_STRANGER or positive("someone_side", a.get("someone_side")):
                add("unauthorized_person", t, "на рабочем месте кассира находится посторонний, кассира нет",
                    c["times"])
        elif k == "behind":
            if positive("two_staff", a.get("two_staff")):
                add("unauthorized_person", t, "к кассиру на его сторону стола подошёл посторонний", c["times"])
        elif k == "visit_end":
            if positive("counting", a.get("counting")):
                add("cash_intake_procedure", t, "клиент ушёл, не дождавшись окончания пересчёта денег", c["times"])
        elif k == "visit":
            per = c.get("answers_by_time") or []
            money = [positive("money_in", x.get("money_in")) for x in per] or [positive("money_in", a.get("money_in"))]
            back = [positive("hand_back", x.get("hand_back")) for x in per] or [positive("hand_back", a.get("hand_back"))]
            if any(money) and any(back[i] for i in range(len(back)) if i > money.index(True)):
                i = next(i for i in range(len(back)) if i > money.index(True) and back[i])
                tt = c["times"][i] if i < len(c["times"]) else t
                add("cash_intake_procedure", tt, "после приёма денег часть возвращена клиенту в руки", [tt])
        elif k == "second":
            if positive("cash_desk", a.get("cash_desk")):
                add("mixed_operations", t,
                    "новый клиент начал обслуживаться, пока на столе остаются деньги предыдущего", c["times"])

    # одинаковый тип в пределах 5 с — одно событие
    ev.sort(key=lambda e: e["t"])
    out = []
    for e in ev:
        if any(o["type_id"] == e["type_id"] and abs(o["t"] - e["t"]) <= 5 for o in out):
            continue
        out.append(e)
    return out


# ---------------------------------------------------------------- запасной режим без VLM
# Только кандидаты детектора людей, с порогами длительности (подобраны scripts/checks_eval.py --sweep).
# Включается CHECKS_NOVLM=1 (VLM не грузится) — если VLM не влезла по времени/памяти.
NOVLM = os.getenv("CHECKS_NOVLM", "0") == "1"
NOVLM_P = {
    "left_min": float(os.getenv("NOVLM_LEFT_MIN", "12")),          # кассира нет >= столько секунд подряд
    "left_end": float(os.getenv("NOVLM_LEFT_END", "2")),          # ... или >= столько, если не вернулся до конца
    "behind_min": float(os.getenv("NOVLM_BEHIND_MIN", "5")),      # подошедший к кассиру виден >= с (0 — выкл.)
    "stranger_min": float(os.getenv("NOVLM_STRANGER_MIN", "0")),  # чужой на месте кассира >= с (0 — выкл.)
    "entered": float(os.getenv("NOVLM_ENTERED", "1")),            # вошедший в пустой кабинет (0 — выкл.)
}


def decide_novlm(cands: list[dict], p: dict | None = None) -> list[dict]:
    """Кандидаты -> события без VLM: доверяем детектору, отсекаем короткое (обычно мерцание/проход мимо)."""
    p = p or NOVLM_P
    ev = []
    for c in cands:
        k, t, info = c["kind"], c["t"], c.get("info") or {}
        if k == "left":
            lim = p["left_end"] if info.get("at_end") else p["left_min"]
            if lim > 0 and info.get("absent_sec", 0) >= lim:
                ev.append({"type_id": "workplace_abandoned", "t": t, "why": "кассир покинул рабочее место",
                           "evidence_t": [t, info.get("t_left", t)]})
        elif k == "stranger" and p["stranger_min"] > 0 and info.get("dur", 99) >= p["stranger_min"]:
            ev.append({"type_id": "unauthorized_person", "t": t, "evidence_t": c["times"],
                       "why": "на рабочем месте кассира находится посторонний, кассира нет"})
        elif k == "entered" and p.get("entered", 1) > 0:
            ev.append({"type_id": "unauthorized_person", "t": t, "evidence_t": c["times"],
                       "why": "в пустой кабинет вошёл человек и занял рабочее место кассира"})
        elif k == "behind" and p["behind_min"] > 0 and info.get("dur", 0) >= p["behind_min"]:
            ev.append({"type_id": "unauthorized_person", "t": t, "evidence_t": c["times"],
                       "why": "к кассиру на его сторону стола подошёл посторонний"})
    ev = [dict(e, t=round(float(e["t"]), 2), score=0.9,
               evidence_t=sorted({round(float(x), 2) for x in e["evidence_t"]})[:3])
          for e in ev if e["type_id"] not in OFF]
    ev.sort(key=lambda e: e["t"])
    out = []
    for e in ev:
        if not any(o["type_id"] == e["type_id"] and abs(o["t"] - e["t"]) <= 5 for o in out):
            out.append(e)
    return out


def run(frames: list[tuple[float, Path]], dets: list[dict], side_hint: str | None = None,
        deadline: float | None = None, ask_all: bool = False) -> tuple[list[dict], list[dict]]:
    """Полный проход для сервиса: кандидаты -> вопросы VLM -> события. -> (события, кандидаты с ответами)."""
    import time
    cands = candidates(dets, side_hint)
    if not cands:
        return [], []
    if NOVLM:
        return decide_novlm(cands), cands
    regs = regions(dets, side_hint)
    det_w = dets[0]["w"]
    for c in cands:
        if c["kind"] == "entered" or (c["kind"] == "stranger" and TRUST_STRANGER and not ask_all):
            continue  # VLM не нужна (ask_all — для офлайн-перебора в scripts/checks_eval.py)
        if deadline and time.time() > deadline - 10:
            break
        if c["kind"] == "visit":  # по моментам визита отдельно — чтобы понять порядок «деньги -> вернули»
            keys, _ = PLAN["visit"]
            per = []
            for t in c["times"]:
                sub = dict(c, times=[max(0.0, t - 0.5), t, t + 0.5])
                per.append(ask(sub, frames, regs, det_w))
            c["answers_by_time"] = per
            c["answers"] = {k: next((x[k] for x in per if positive(k, x.get(k))), per[0].get(k)) for k in keys}
        else:
            c["answers"] = ask(c, frames, regs, det_w)
    return decide(cands), cands
