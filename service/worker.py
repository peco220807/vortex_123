"""Очередь и воркер.

Как это устроено:
  * Supervisor — поток внутри сервиса. Берёт job_id из очереди по одному (GPU одна).
  * Сам анализ идёт в ОТДЕЛЬНОМ ПРОЦЕССЕ (child_main). Модели грузятся в нём один раз
    и переиспользуются для всех видео.
  * Если analyze() зависла дольше RUN_TIMEOUT_SEC или процесс упал (например, нехватка
    памяти GPU), процесс убивается и перезапускается, задание помечается failed,
    а очередь идёт дальше. Одно плохое видео не блокирует остальные.
  * В analyze() передаётся deadline (unix-время): к этому моменту надо вернуть то,
    что уже найдено (например, пропустить VLM). Это мягкий таймер, его соблюдает Чел 1.
"""
import inspect
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from pathlib import Path

from .store import JobStore

log = logging.getLogger("vortex.worker")

# Лимит жюри — 300 с от POST до done. Берём с запасом.
SOFT_DEADLINE_SEC = float(os.getenv("SOFT_DEADLINE_SEC", "200"))  # от загрузки: analyze() должна закругляться
MIN_RUN_SEC = float(os.getenv("MIN_RUN_SEC", "20"))  # минимум времени на анализ, даже если задание долго ждало
RUN_TIMEOUT_SEC = float(os.getenv("RUN_TIMEOUT_SEC", "270"))  # жёсткий лимит одного запуска
MODEL_LOAD_TIMEOUT_SEC = float(os.getenv("MODEL_LOAD_TIMEOUT_SEC", "900"))


# ---------------------------------------------------------------- дочерний процесс
def child_main(conn) -> None:
    """Работает в отдельном процессе: грузит модели один раз и анализирует видео по запросу."""
    try:
        from pipeline.analyze import analyze  # тут Чел 1 грузит веса

        takes_deadline = "deadline" in inspect.signature(analyze).parameters
        extra_sleep = float(os.getenv("DEBUG_EXTRA_SLEEP_SEC", "0"))  # только для тестов зависания
        conn.send(("ready", None))
    except Exception:  # noqa: BLE001
        conn.send(("load_error", traceback.format_exc()))
        return

    while True:
        msg = conn.recv()
        if msg is None:
            return
        job_id, path, deadline = msg
        try:
            if extra_sleep:
                time.sleep(extra_sleep)
            kw = {"deadline": deadline} if takes_deadline else {}
            res = analyze(path, **kw)
            conn.send(("ok", res))
        except Exception as e:  # noqa: BLE001
            conn.send(("err", f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"))


# ---------------------------------------------------------------- супервизор
class Supervisor:
    def __init__(self, store: JobStore, on_result):
        self.store = store
        self.on_result = on_result  # (job_id, result_dict) -> None: проверка формата и сохранение
        self.q: "queue.Queue[str]" = queue.Queue()
        self._ctx = mp.get_context("spawn")  # spawn нужен для CUDA и одинаково работает в Windows/Linux
        self._proc = None
        self._conn = None
        self.ready = threading.Event()
        self.current: str | None = None
        self.last_error: str | None = None

    # --- управление процессом
    def _spawn(self) -> None:
        self._kill()
        self.ready.clear()
        parent, child = self._ctx.Pipe()
        p = self._ctx.Process(target=child_main, args=(child,), daemon=True, name="vortex-analyzer")
        p.start()
        child.close()
        self._proc, self._conn = p, parent
        t0 = time.time()
        if not parent.poll(MODEL_LOAD_TIMEOUT_SEC):
            raise RuntimeError("модели не загрузились за отведённое время")
        kind, payload = parent.recv()
        if kind != "ready":
            raise RuntimeError(f"ошибка загрузки моделей:\n{payload}")
        log.info("анализатор готов за %.1f с (pid %s)", time.time() - t0, p.pid)
        self.ready.set()

    def _kill(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.kill()
            self._proc.join(5)
        self._proc = self._conn = None

    # --- основной цикл
    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True, name="vortex-supervisor").start()

    def submit(self, job_id: str) -> None:
        self.q.put(job_id)

    def _loop(self) -> None:
        while True:
            try:
                self._spawn()
                break
            except Exception as e:  # noqa: BLE001
                self.last_error = str(e)
                log.error("не удалось запустить анализатор: %s — повтор через 10 с", e)
                time.sleep(10)

        while True:
            job_id = self.q.get()
            try:
                self._process(job_id)
            except Exception as e:  # noqa: BLE001  — супервизор не должен умирать никогда
                log.exception("сбой супервизора на %s", job_id)
                self.store.mark_failed(job_id, f"internal: {e}")
            finally:
                self.current = None

    def _process(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None or job["status"] not in ("queued", "running"):
            return
        path = job["path"]
        if not path or not Path(path).exists():
            self.store.mark_failed(job_id, "файл видео не найден")
            return

        if self._proc is None or not self._proc.is_alive():
            self._spawn()

        self.current = job_id
        self.store.mark_running(job_id)
        started = time.time()
        deadline = max(job["created_at"] + SOFT_DEADLINE_SEC, started + MIN_RUN_SEC)
        log.info("job %s: старт (ждал в очереди %.1f с)", job_id, started - job["created_at"])

        try:
            self._conn.send((job_id, path, deadline))
            if not self._conn.poll(RUN_TIMEOUT_SEC):
                log.error("job %s: превышен лимит %.0f с — перезапуск анализатора", job_id, RUN_TIMEOUT_SEC)
                self.store.mark_failed(job_id, f"timeout: анализ дольше {RUN_TIMEOUT_SEC:.0f} с")
                self._spawn()
                return
            kind, payload = self._conn.recv()
        except (EOFError, BrokenPipeError, ConnectionResetError, OSError):
            log.error("job %s: процесс анализа упал — перезапуск", job_id)
            self.store.mark_failed(job_id, "процесс анализа аварийно завершился")
            self._spawn()
            return
        finally:
            Path(path).unlink(missing_ok=True)  # тестовые видео жюри не храним

        if kind == "ok":
            self.on_result(job_id, payload)
            log.info("job %s: готово за %.1f с (всего с загрузки %.1f с)",
                     job_id, time.time() - started, time.time() - job["created_at"])
        else:
            log.error("job %s: ошибка анализа\n%s", job_id, payload)
            self.store.mark_failed(job_id, payload.splitlines()[0] if payload else "ошибка анализа")

    def info(self) -> dict:
        return {
            "analyzer_ready": self.ready.is_set(),
            "analyzer_alive": bool(self._proc and self._proc.is_alive()),
            "current_job": self.current,
            "queued": self.q.qsize(),
            "last_error": self.last_error,
        }
