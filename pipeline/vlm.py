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
# длинная сторона кадра для VLM: меньше пикселей -> меньше токенов -> меньше памяти и быстрее
MAX_SIDE = int(os.getenv("VLM_MAX_SIDE", "448"))

_model = None
_processor = None


def load():
    global _model, _processor
    if _model is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, dtype=torch.float16, attn_implementation="sdpa",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
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
    im = img if isinstance(img, Image.Image) else Image.open(io.BytesIO(img)).convert("RGB")
    k = MAX_SIDE / max(im.size)
    if k < 1:
        im = im.resize((round(im.width * k), round(im.height * k)), Image.BILINEAR)
    return im


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


def _generate(frames: list[tuple[float, bytes]], prompt: str, max_new_tokens: int) -> str:
    import torch

    model, processor = load()
    content = []
    for t, img in frames:
        content.append({"type": "text", "text": f"Frame at {t:.1f}s:"})
        content.append({"type": "image", "image": _to_pil(img)})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


def classify_window(frames: list[tuple[float, bytes]], max_new_tokens: int = 160) -> dict:
    """Старый режим «есть ли нарушение?» — маленькая модель почти всегда отвечает «да». Оставлен для сравнения."""
    return _parse(_generate(frames, PROMPT.format(types=types_prompt_block()), max_new_tokens))


# ---------- Основной режим: модель отвечает на ФАКТЫ, решение принимает код (pipeline/rules_engine.py) ----------

FACTS = {
    "people_count": "how many people are visible in the room (integer)",
    "cashier_at_desk": "is someone sitting/standing at the cashier workplace (behind the desk with the "
                       "money counter / laptop) and working there",
    "visitor_count": "how many people are on the visitor/client side of the desk (integer)",
    "person_leaving": "is someone walking out of the room or out of the frame",
    "person_entering": "is someone walking into the room or into the frame",
    "money_on_desk": "are banknotes / cash bundles lying on the desk",
    "money_handed_over": "is cash being passed from one person's hands to another person",
    "counting_machine_used": "is someone putting banknotes into or taking them from a money counting machine",
    "id_document_visible": "is an ID card / passport / document being handed over or looked at",
    "bag_on_desk": "is a cash-collection bag / sealed bag lying on the desk",
    "bag_being_opened": "is someone opening a bag or taking cash bundles out of a bag",
    "safe_open": "is the safe door open",
    "keys_in_safe_lock": "are keys left hanging in the safe lock",
    "laptop_open": "is a laptop open (screen up)",
    "lights_off": "is the room dark / lights switched off",
    "touching_equipment": "is someone touching the laptop, money counter or items on the cashier desk",
}

KEYS = list(FACTS)

OBSERVE_PROMPT = (
    "These frames come from a ceiling camera above a bank cashier desk, in time order. "
    "Answer strictly about what is VISIBLE. Do not guess.\n"
    f"Answer the {len(FACTS)} questions below IN THIS ORDER with numbers only, separated by spaces, "
    "on one line. Counts: integer. Yes/no: 1 or 0. No words, no explanation.\n"
    + "\n".join(f"{i + 1}. {v}" for i, v in enumerate(FACTS.values()))
)

DESCRIBE_PROMPT = (
    "These frames come from a ceiling camera above a bank cashier desk, in time order. "
    "In ONE short sentence IN RUSSIAN describe what the people are doing with money, bags, the safe or "
    "equipment. Only what is visible. Do not describe faces or appearance."
)


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    return str(v).strip().lower() in ("true", "yes", "да", "1")


def _as_int(v) -> int:
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return 1 if _as_bool(v) else 0


def parse_facts(text: str) -> dict:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if sum(bool(re.match(r"\s*\d+[.):]\s*\S", ln)) for ln in lines) >= len(KEYS) // 2:
        # ответ нумерованным списком «1. 2» — берём последнее число в каждой строке
        nums = [re.findall(r"-?\d+", ln)[-1] for ln in lines if re.findall(r"-?\d+", ln)]
    else:
        nums = re.findall(r"-?\d+", text)
    d = {}
    if len(nums) >= len(KEYS):
        d = dict(zip(KEYS, nums[:len(KEYS)]))
    else:
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
    out = {}
    for k in FACTS:
        out[k] = _as_int(d.get(k)) if k.endswith("_count") else _as_bool(d.get(k))
    out["action"] = str(d.get("action") or "").strip()
    out["parsed"] = bool(d)
    return out


def observe(frames: list[tuple[float, bytes]], max_new_tokens: int = 48) -> dict:
    """Факты об окне: {people_count, cashier_at_desk, money_on_desk, ..., parsed, raw}."""
    text = _generate(frames, OBSERVE_PROMPT, max_new_tokens)
    f = parse_facts(text)
    f["raw"] = text.strip()
    return f


def describe(frames: list[tuple[float, bytes]], max_new_tokens: int = 60) -> str:
    """Одна фраза по-русски о том, что происходит (только для окон с найденным нарушением)."""
    return _generate(frames, DESCRIBE_PROMPT, max_new_tokens).strip().split("\n")[0]
