from pathlib import Path


def norm(name: str) -> str:
    """Х-1 / X-1 / x-1.mov -> Х-1 (латинская X в названиях встречается)."""
    return Path(name).stem.strip().replace("X", "Х").replace("x", "Х")


def find_video(videos_dir: str, video_id: str):
    target = norm(video_id)
    for p in Path(videos_dir).iterdir():
        if p.suffix.lower() in (".mov", ".mp4") and norm(p.name) == target:
            return p
    return None


def apply_people(d: dict, people_dir: str | None, stats: dict | None = None) -> bool:
    """Подставить в d["timeline"] факты о людях из data/people/<видео>.json (если файл есть).
    stats — счётчик совпадений «кассир на месте» VLM vs детектор (для сравнения)."""
    if not people_dir:
        return False
    import json
    from pipeline import people
    p = Path(people_dir) / f"{norm(d['video_id'])}.json"
    if not p.exists():
        return False
    pj = json.loads(p.read_text(encoding="utf-8"))
    people.merge(d["timeline"], pj["dets"], pj.get("side_hint"))
    if stats is not None:
        for w in d["timeline"]:
            old = (w["f"].get("vlm_people") or {}).get("cashier_at_desk")
            if old is not None:
                stats["n"] = stats.get("n", 0) + 1
                stats["same"] = stats.get("same", 0) + (bool(old) == bool(w["f"]["cashier_at_desk"]))
    return True
