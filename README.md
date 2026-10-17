# Vortex — AI-проверка кассовой дисциплины по видео

Datathon 26, кейс ForteBank. Сервис принимает видео кассы (.mov), находит визуально наблюдаемые признаки
возможных нарушений и возвращает вердикт, таймкод, кадры-доказательства, описание и ссылку на норматив РК.
Всё работает локально, без облачных AI-сервисов.

## Структура

```
pipeline/            ML-часть (Чел 1)
  analyze.py         analyze(video_path) -> dict  — единственная точка входа для сервиса
  schema.py          формат результата по разделу 10 ТЗ + validate()
  video_io.py        кадры через ffmpeg
rules/rules.json     типы нарушений -> описание -> документ -> пункт (Чел 3)
service/             FastAPI + веб-интерфейс (Чел 2)
scripts/
  frames_around.py   кадры и коллаж вокруг размеченных таймкодов — для разбора типов
  run_dataset.py     прогон analyze() по датасету -> preds/
  evaluate.py        метрики по разметке (вердикт, пропуски, ложные, таймкод)
data/
  labels_clean.csv   разметка: 42 видео, split train/holdout (локально, не в git)
  videos/            .mov (локально, не в git)
```

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # + ffmpeg в системе
# положить видео в data/videos/

# 1. Разобрать нарушения глазами
python scripts/frames_around.py --videos data/videos --window 4 --fps 4
#   -> data/frames_around/<видео>/contact.jpg

# 2. Прогнать пайплайн и посчитать метрики
python scripts/run_dataset.py --split train --out preds/
python scripts/evaluate.py --labels data/labels_clean.csv --preds preds/ --split train

# 3. Запустить сервис
API_KEY=<ключ> uvicorn service.app:app --host 0.0.0.0 --port 8000
#   UI:  http://localhost:8000/
#   STUB_ALWAYS_VIOLATION=1 — фейковое нарушение, чтобы отлаживать фронт без моделей
```

Docker: `docker build -t vortex . && docker run -p 8000:8000 -e API_KEY=<ключ> vortex`

## API (раздел 10 ТЗ)

```bash
curl -X POST https://<host>/api/v1/analyze -H "X-API-Key: <ключ>" -F "file=@video.mov"
# {"job_id": "3f2a9c1e"}
curl https://<host>/api/v1/jobs/3f2a9c1e -H "X-API-Key: <ключ>"
# {"job_id": "...", "status": "processing|done|failed", "verdict": "...", "violations": [...]}
```

## Правила команды

- **holdout** (9 видео в `labels_clean.csv`) не смотрим и не настраиваемся на них до финальной проверки.
- Метрики — только через `scripts/evaluate.py`, цифры пишем в PR.
- Модели — только open source с коммерческой лицензией (Apache/MIT/BSD). **Ultralytics YOLO — AGPL, нельзя.**
- Никакого распознавания лиц, никаких облачных API при обработке видео.
- **Датасет не покидает наши машины.** Видео, кадры, разметку и результаты по ним не кладём в git,
  Google Drive/Sheets, облачные мессенджеры и не загружаем в ChatGPT/Claude/Gemini и т.п.
  Передаём флешкой или по локальной сети. Подробно: `data/README.md`.
- Тестовые видео жюри не сохраняем: сервис удаляет файл сразу после обработки.
- Контракт `analyze()` -> формат из `pipeline/schema.py` не ломаем: от него зависит API жюри.
