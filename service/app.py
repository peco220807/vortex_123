"""Сервис (Чел 2): API по разделу 10 ТЗ + веб-интерфейс.

  API_KEY=секрет uvicorn service.app:app --host 0.0.0.0 --port 8000

POST /api/v1/analyze        (X-API-Key, multipart поле file=.mov) -> {"job_id": ...}
GET  /api/v1/jobs/{job_id}  (X-API-Key) -> {"job_id", "status": processing|done|failed, ...результат}
GET  /                       веб-интерфейс (использует внутренние /ui/* без ключа)
"""
import os
import secrets
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

from pipeline.analyze import analyze

API_KEY = os.getenv("API_KEY") or secrets.token_urlsafe(24)
if not os.getenv("API_KEY"):
    print(f"[!] API_KEY не задан, сгенерирован временный: {API_KEY}")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/vortex_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))

app = FastAPI(title="Vortex — кассовая дисциплина")
executor = ThreadPoolExecutor(max_workers=int(os.getenv("WORKERS", "1")))  # 1 = одна GPU
jobs: dict[str, dict] = {}
lock = threading.Lock()


def _check_key(key: str | None):
    if not key or not secrets.compare_digest(key, API_KEY):
        raise HTTPException(status_code=401, detail="invalid API key")


def _run(job_id: str, path: Path):
    try:
        res = analyze(str(path))
        res.pop("_debug", None)
        with lock:
            jobs[job_id] = {"job_id": job_id, "status": "done", **res}
    except Exception as e:  # noqa: BLE001
        with lock:
            jobs[job_id] = {"job_id": job_id, "status": "failed", "error": str(e)}
    finally:
        path.unlink(missing_ok=True)


async def _submit(file: UploadFile) -> dict:
    if not file.filename:
        raise HTTPException(400, "file is required")
    job_id = uuid.uuid4().hex[:8]
    path = UPLOAD_DIR / f"{job_id}{Path(file.filename).suffix.lower() or '.mov'}"
    with path.open("wb") as f:
        shutil.copyfileobj(file.file, f, length=1024 * 1024)
    if path.stat().st_size > MAX_MB * 1024 * 1024:
        path.unlink()
        raise HTTPException(413, "file too large")
    with lock:
        jobs[job_id] = {"job_id": job_id, "status": "processing"}
    executor.submit(_run, job_id, path)
    return {"job_id": job_id}


def _get(job_id: str) -> dict:
    with lock:
        j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return j


# --- публичный API для жюри ---
@app.post("/api/v1/analyze")
async def api_analyze(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return await _submit(file)


@app.get("/api/v1/jobs/{job_id}")
def api_job(job_id: str, x_api_key: str | None = Header(None)):
    _check_key(x_api_key)
    return _get(job_id)


# --- для веб-интерфейса (без ключа) ---
@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...)):
    return await _submit(file)


@app.get("/ui/jobs/{job_id}")
def ui_job(job_id: str):
    return _get(job_id)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")
