"""Хранилище заданий в SQLite: переживает перезапуск сервиса.

Статусы внутри сервиса: queued -> running -> done | failed.
Наружу (API жюри) queued и running отдаются как "processing".
"""
import json
import sqlite3
import threading
import time
from pathlib import Path


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    job_id      TEXT PRIMARY KEY,
                    status      TEXT NOT NULL,
                    path        TEXT,
                    filename    TEXT,
                    created_at  REAL NOT NULL,
                    started_at  REAL,
                    finished_at REAL,
                    result      TEXT,
                    error       TEXT
                )"""
            )
            self._db.commit()

    def _exec(self, sql: str, args: tuple = ()):
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            return cur

    def create(self, job_id: str, path: str, filename: str) -> None:
        self._exec(
            "INSERT INTO jobs (job_id, status, path, filename, created_at) VALUES (?, 'queued', ?, ?, ?)",
            (job_id, path, filename, time.time()),
        )

    def mark_running(self, job_id: str) -> None:
        self._exec("UPDATE jobs SET status='running', started_at=? WHERE job_id=?", (time.time(), job_id))

    def mark_done(self, job_id: str, result: dict) -> None:
        self._exec(
            "UPDATE jobs SET status='done', finished_at=?, result=?, path=NULL WHERE job_id=?",
            (time.time(), json.dumps(result, ensure_ascii=False), job_id),
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        self._exec(
            "UPDATE jobs SET status='failed', finished_at=?, error=?, path=NULL WHERE job_id=?",
            (time.time(), error[:2000], job_id),
        )

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        j = dict(row)
        j["result"] = json.loads(j["result"]) if j["result"] else None
        return j

    def unfinished(self) -> list[dict]:
        """Задания, которые не успели завершиться (нужно после перезапуска)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT job_id, path, created_at FROM jobs WHERE status IN ('queued', 'running') ORDER BY created_at"
            ).fetchall()
        return [dict(r) for r in rows]

    def queue_position(self, job_id: str) -> int:
        """Сколько заданий будет обработано раньше этого (0 = следующее или уже идёт)."""
        with self._lock:
            row = self._db.execute(
                """SELECT COUNT(*) FROM jobs
                   WHERE status='queued' AND created_at < (SELECT created_at FROM jobs WHERE job_id=?)""",
                (job_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def counts(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}
