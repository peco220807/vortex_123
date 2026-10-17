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
