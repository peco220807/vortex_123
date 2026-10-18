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
SOFT_DEADLINE_SEC = float(os.getenv("SOFT_DEADLINE_SEC", "230"))  # от загрузки: к этому моменту видео должно быть готово
MIN_RUN_SEC = float(os.getenv("MIN_RUN_SEC", "20"))  # минимум времени на анализ, даже если задание долго ждало
MAX_RUN_SEC = float(os.getenv("MAX_RUN_SEC", "75"))  # максимум на одно видео, даже если очередь пуста
RUN_TIMEOUT_SEC = float(os.getenv("RUN_TIMEOUT_SEC", "270"))  # жёсткий лимит одного запуска
MODEL_LOAD_TIMEOUT_SEC = float(os.getenv("MODEL_LOAD_TIMEOUT_SEC", "900"))


# ---------------------------------------------------------------- дочерний процесс
def _warmup() -> None:
    """Загрузить модели заранее, чтобы первое видео жюри не тратило на это свой лимит.
    Грузим только то, что реально используется в текущем режиме:
      * PIPELINE=checks (по умолчанию): детектор людей RT-DETR; VLM — только если CHECKS_NOVLM=0;
      * PIPELINE=windows: VLM + классификатор окон.
    Ошибка прогрева не мешает старту: модели тогда догрузятся на первом видео."""
    if os.getenv("STUB_ALWAYS_VIOLATION") == "1" or os.getenv("SKIP_WARMUP") == "1":
        return
    t0 = time.time()
    loaded = []
    try:
        mode = os.getenv("PIPELINE", "checks")
        need_vlm = mode != "checks" or os.getenv("CHECKS_NOVLM", "1") == "0"
        if mode == "checks":
            from pipeline import people

            if people.enabled():
                people.load()  # заодно импортирует torch/transformers — это тоже секунды
                loaded.append("RT-DETR")
        if need_vlm:
            from pipeline import vlm

            vlm.load()
            loaded.append("VLM")
        if mode != "checks":
            from pipeline import window_model

            window_model.load()
            loaded.append("классификатор окон")
        log.info("модели прогреты за %.1f с: %s", time.time() - t0, ", ".join(loaded) or "нечего грузить")
    except Exception:  # noqa: BLE001
        log.error("прогрев моделей не удался, загрузятся на первом видео:\n%s", traceback.format_exc())


def child_main(conn) -> None:
    """Работает в отдельном процессе: грузит модели один раз и анализирует видео по запросу."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        from pipeline.analyze import analyze  # тут Чел 1 грузит веса

        takes_deadline = "deadline" in inspect.signature(analyze).parameters
        extra_sleep = float(os.getenv("DEBUG_EXTRA_SLEEP_SEC", "0"))  # только для тестов зависания
        _warmup()
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
        log.info("job %s: старт (ждал в очереди %.1f с)", job_id, started - job["created_at"])
        deadline = self._fair_deadline(job, started)

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

    def _fair_deadline(self, job: dict, now: float) -> float:
        """Срок для analyze(): время до SOFT_DEADLINE делится поровну с теми, кто ждёт в очереди.
        Задание на позиции p должно успеть после текущего и p-1 предыдущих, поэтому текущему —
        не больше 1/(p+1) от остатка времени этого задания."""
        slot = job["created_at"] + SOFT_DEADLINE_SEC - now
        waiting = [j for j in self.store.unfinished() if j["job_id"] != job["job_id"]]
        for p, j in enumerate(waiting, start=1):
            slot = min(slot, (j["created_at"] + SOFT_DEADLINE_SEC - now) / (p + 1))
        slot = max(min(slot, MAX_RUN_SEC), MIN_RUN_SEC)
        log.info("job %s: срок на анализ %.0f с (в очереди ещё %d)", job["job_id"], slot, len(waiting))
        return now + slot

    def info(self) -> dict:
        return {
            "analyzer_ready": self.ready.is_set(),
            "analyzer_alive": bool(self._proc and self._proc.is_alive()),
            "current_job": self.current,
            "queued": self.q.qsize(),
            "last_error": self.last_error,
        }
