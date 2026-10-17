"""Локальная VLM (Qwen3-VL) — проверка момента и текст обоснования.

Модель грузится один раз и держится в памяти. Работает полностью офлайн после первой загрузки весов
(HF_HUB_OFFLINE=1 запрещает обращения в интернет).
"""
import io
import json
import os
import re

from PIL import Image

from .violation_types import TYPES, types_prompt_block

MODEL_ID = os.getenv("VLM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")

_model = None
_processor = None


def load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="cuda" if torch.cuda.is_available() else "cpu"
        ).eval()
    return _model, _processor


def unload():
    """Освободить видеопамять (если перед VLM гоняли детекторы и наоборот)."""
    global _model, _processor
    import torch
    _model = _processor = None
    torch.cuda.empty_cache()


PROMPT = """Ты помощник проверяющего в банке. Перед тобой кадры с камеры над кассой, снятые подряд \
за несколько секунд (время указано у каждого кадра). Определи, видно ли на этих кадрах одно из нарушений:

{types}

Правила ответа:
- Опирайся только на то, что видно на кадрах. Если не уверен — violation=false.
- Не распознавай и не описывай лица.
- Ответь ОДНОЙ строкой JSON без пояснений:
{{"violation": true/false, "type": "<id из списка или null>", "confidence": 0.0-1.0, \
"description": "<1-2 предложения по-русски: что видно и почему это нарушение>"}}"""


def _to_pil(img) -> Image.Image:
    if isinstance(img, Image.Image):
        return img
    return Image.open(io.BytesIO(img)).convert("RGB")


def _parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    t = d.get("type")
    return {
        "violation": bool(d.get("violation")) and t in TYPES,
        "type": t if t in TYPES else None,
        "confidence": float(d.get("confidence") or 0),
        "description": str(d.get("description") or ""),
        "raw": text.strip(),
    }


def classify_window(frames: list[tuple[float, bytes]], max_new_tokens: int = 160) -> dict:
    """frames: [(timestamp_sec, jpeg_bytes), ...] — 4–8 кадров одного момента."""
    import torch

    model, processor = load()
    content = []
    for t, img in frames:
        content.append({"type": "text", "text": f"Кадр {t:.1f} с:"})
        content.append({"type": "image", "image": _to_pil(img)})
    content.append({"type": "text", "text": PROMPT.format(types=types_prompt_block())})
    messages = [{"role": "user", "content": content}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    return _parse(text)
