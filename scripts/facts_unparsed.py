"""Показать сырые ответы модели в неразобранных окнах (для отладки формата).

  python scripts\\facts_unparsed.py            # первые 10
"""
import json
import sys
from pathlib import Path

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
shown = 0
for p in sorted(Path("data/facts").glob("*.json")):
    d = json.loads(p.read_text(encoding="utf-8"))
    for w in d["timeline"]:
        if not w["f"].get("parsed") and shown < n:
            print(f"{d['video_id']} {w['lo']:.0f}-{w['hi']:.0f}s: {w['f'].get('raw', '')[:200]!r}")
            shown += 1
print(f"показано {shown}")
