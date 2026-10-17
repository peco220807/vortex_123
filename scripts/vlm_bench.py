"""Замер VLM: сколько памяти и секунд уходит на одно окно при разном числе кадров и размере.

  python scripts\\vlm_bench.py data\\videos\\X-1.mov

Если «Пик памяти» близок к 6 ГБ или время > 20 с — память переполняется, уменьшаем кадры.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from pipeline import vlm  # noqa: E402
from pipeline.video_io import frame_at  # noqa: E402

video = sys.argv[1]
t0 = time.time()
model, _ = vlm.load()
gb = 1024 ** 3
print(f"Модель {vlm.MODEL_ID}: загрузка {time.time() - t0:.1f} с, устройство {model.device}, "
      f"занято {torch.cuda.memory_allocated() / gb:.2f} ГБ")

for side, n in [(320, 1), (448, 1), (448, 4), (448, 6), (640, 6)]:
    vlm.MAX_SIDE = side
    frames = [(5.0 + i, frame_at(video, 5.0 + i, 960)) for i in range(n)]
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t = time.time()
    r = vlm.classify_window(frames)
    torch.cuda.synchronize()
    print(f"сторона {side:4} px, кадров {n}: {time.time() - t:6.1f} с, "
          f"пик памяти {torch.cuda.max_memory_allocated() / gb:.2f} ГБ | ответ: {r['raw'][:70]!r}")
