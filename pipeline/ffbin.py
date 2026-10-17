"""Где лежат ffmpeg/ffprobe. Порядок: переменные FFMPEG/FFPROBE -> PATH -> папка winget на Windows."""
import os
import shutil
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def find(name: str) -> str:
    env = os.getenv(name.upper())
    if env and Path(env).exists():
        return env
    p = shutil.which(name)
    if p:
        return p
    local = os.getenv("LOCALAPPDATA")
    if local:
        for root in (Path(local) / "Microsoft" / "WinGet" / "Links", Path(local) / "Microsoft" / "WinGet" / "Packages"):
            if root.exists():
                hit = next(root.rglob(f"{name}.exe"), None)
                if hit:
                    return str(hit)
    raise FileNotFoundError(
        f"{name} не найден. Установите ffmpeg (winget install Gyan.FFmpeg) "
        f"или укажите путь в переменной {name.upper()}")


def ffmpeg() -> str:
    return find("ffmpeg")


def ffprobe() -> str:
    return find("ffprobe")
