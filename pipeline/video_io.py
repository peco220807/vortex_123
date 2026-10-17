"""Чтение видео через ffmpeg/ffprobe (без OpenCV, работает на любой машине с ffmpeg)."""
import base64
import json
import subprocess
from pathlib import Path

from .ffbin import ffmpeg, ffprobe


def probe(video_path: str) -> dict:
    """Длительность, fps и размер кадра."""
    out = subprocess.run(
        [ffprobe(), "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout
    j = json.loads(out)
    s = j["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {
        "duration": float(j["format"]["duration"]),
        "fps": float(num) / float(den),
        "width": int(s["width"]),
        "height": int(s["height"]),
    }


def _scale(width: int) -> str:
    # уменьшаем до width, но маленькие видео не растягиваем
    return f"scale='min(iw,{width})':-2"


def extract_frames(video_path: str, out_dir: str, fps: float = 2.0, width: int = 960) -> list[tuple[float, Path]]:
    """Нарезать кадры с частотой fps. Возвращает [(timestamp_sec, path), ...]."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [ffmpeg(), "-v", "error", "-y", "-i", str(video_path),
         "-vf", f"fps={fps},{_scale(width)}", "-q:v", "3", str(out / "f_%05d.jpg")],
        check=True,
    )
    frames = sorted(out.glob("f_*.jpg"))
    # фильтр fps выдаёт i-й кадр в момент i/fps
    return [(i / fps, p) for i, p in enumerate(frames)]


def frame_at(video_path: str, t: float, width: int = 960) -> bytes:
    """Один JPEG-кадр в момент t (сек)."""
    return subprocess.run(
        [ffmpeg(), "-v", "error", "-ss", f"{max(t, 0):.3f}", "-i", str(video_path),
         "-frames:v", "1", "-vf", _scale(width), "-q:v", "3", "-f", "image2", "-"],
        capture_output=True, check=True,
    ).stdout


def frame_b64(video_path: str, t: float, width: int = 960) -> str:
    return base64.b64encode(frame_at(video_path, t, width)).decode("ascii")
