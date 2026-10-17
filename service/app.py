"""Сервис (Чел 2): API по разделу 10 ТЗ + веб-интерфейс.

  API_KEY=секрет uvicorn service.app:app --host 0.0.0.0 --port 8000

POST /api/v1/analyze        (X-API-Key, multipart поле file=.mov) -> {"job_id": ...}
GET  /api/v1/jobs/{job_id}  (X-API-Key) -> {"job_id", "status": processing|done|failed, ...результат}
GET  /                       веб-интерфейс (использует внутренние /ui/* без ключа)
GET  /health                 состояние очереди и анализатора

Задания хранятся в SQLite (DATA_DIR/jobs.db) и переживают перезапуск.
Анализ идёт в отдельном процессе с таймаутом — см. service/worker.py.
"""
import logging
import os
import secrets
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse

from pipeline.schema import validate

from .store import JobStore
from .worker import Supervisor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("vortex.app")

API_KEY = os.getenv("API_KEY") or secrets.token_urlsafe(24)
if not os.getenv("API_KEY"):
    log.warning("API_KEY не задан, сгенерирован временный: %s", API_KEY)
DATA_DIR = Path(os.getenv("DATA_DIR", "var"))
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
ALLOWED_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}

store = JobStore(DATA_DIR / "jobs.db")


def _save_result(job_id: str, res: dict) -> None:
    """Проверить формат по ТЗ и сохранить. Неверный формат = failed (жюри всё равно не примет)."""
    res = dict(res)
    debug = res.pop("_debug", None)
    try:
        validate(res)
    except (AssertionError, KeyError, TypeError) as e:
        log.error("job %s: analyze() вернула неверный формат: %r", job_id, e)
        store.mark_failed(job_id, "analyze() вернула результат не по схеме")
        return
    store.mark_done(job_id, {"verdict": res["verdict"], "violations": res["violations"]})
    if debug:
        log.info("job %s debug: %s", job_id, debug)


supervisor = Supervisor(store, on_result=_save_result)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    supervisor.start()
    # задания, которые не успели обработаться до перезапуска, ставим обратно в очередь
    for j in store.unfinished():
        if j["path"] and Path(j["path"]).exists():
            log.info("восстановлено задание %s", j["job_id"])
            supervisor.submit(j["job_id"])
        else:
            store.mark_failed(j["job_id"], "сервис перезапущен, файл видео утерян")
    yield


app = FastAPI(title="Vortex — кассовая дисциплина", lifespan=lifespan)


def _check_key(key: str | None) -> None:
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


async def _submit(file: UploadFile) -> dict:
    if not file.filename:
        raise HTTPException(400, "file is required")
    ext = Path(file.filename).suffix.lower() or ".mov"
    if ext not in ALLOWED_EXT:
        raise HTTPException(415, f"unsupported file type {ext}, expected .mov")
    job_id = uuid.uuid4().hex[:8]
    path = UPLOAD_DIR / f"{job_id}{ext}"

    def _copy() -> int:
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f, length=1024 * 1024)
        return path.stat().st_size

    size = await run_in_threadpool(_copy)  # копирование не блокирует ответы на другие запросы
    if size == 0:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "empty file")
    if size > MAX_MB * 1024 * 1024:
        path.unlink(missing_ok=True)
        raise HTTPException(413, "file too large")

    store.create(job_id, str(path), file.filename)
    supervisor.submit(job_id)
    log.info("job %s: принят %s (%.1f МБ)", job_id, file.filename, size / 1e6)
    return {"job_id": job_id, "status": "processing"}


def _public(job_id: str) -> dict:
    """Ответ строго в формате раздела 10 ТЗ."""
    j = store.get(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    if j["status"] == "done":
        return {"job_id": job_id, "status": "done", **j["result"]}
    if j["status"] == "failed":
        return {"job_id": job_id, "status": "failed", "error": j["error"]}
    return {"job_id": job_id, "status": "processing"}


# --- публичный API для жюри ---
@app.post("/api/v1/analyze")
async def api_analyze(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return await _submit(file)


@app.get("/api/v1/jobs/{job_id}")
def api_job(job_id: str, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return _public(job_id)


# --- для веб-интерфейса (без ключа, чтобы ключ жюри не светился в браузере) ---
@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _submit(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    out = _public(job_id)
    j = store.get(job_id)
    out["stage"] = j["status"]  # queued | running | done | failed — для подсказки в интерфейсе
    out["elapsed_sec"] = round((j["finished_at"] or time.time()) - j["created_at"], 1)
    if j["status"] == "queued":
        out["queue_position"] = store.queue_position(job_id)
    return out


@app.get("/health")
def health():
    return {"ok": True, **supervisor.info(), "jobs": store.counts()}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
