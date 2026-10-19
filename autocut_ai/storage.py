from __future__ import annotations

import sqlite3
from pathlib import Path

from autocut_ai.schemas import JobEvent, JobRecord
from autocut_ai.timers import refresh_timers


class JobNotFoundError(LookupError):
    pass


class JobRepository:
    """Repository SQLite privo di stato condiviso tra richieste."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def save(self, job: JobRecord) -> JobRecord:
        # Ogni operazione apre una connessione breve: SQLite gestisce il commit
        # atomico e i worker non condividono oggetti connection tra thread.
        refreshed = refresh_timers(job)
        payload = refreshed.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO jobs(job_id, payload)
                VALUES (?, ?)
                ON CONFLICT(job_id) DO UPDATE SET payload = excluded.payload
                """,
                (refreshed.job_id, payload),
            )
            connection.commit()
        return refreshed

    def get(self, job_id: str) -> JobRecord:
        with self._connect() as connection:
            row = connection.execute("SELECT payload FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFoundError(job_id)
        return refresh_timers(JobRecord.model_validate_json(row["payload"]))

    def append_event(self, event: JobEvent) -> JobEvent:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO job_events(job_id, payload) VALUES (?, ?)",
                (event.job_id, event.model_dump_json()),
            )
            sequence = int(cursor.lastrowid)
            connection.commit()
        event.sequence = sequence
        return event

    def list_events_after(self, job_id: str, sequence: int) -> list[JobEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, payload
                FROM job_events
                WHERE job_id = ? AND sequence > ?
                ORDER BY sequence ASC
                """,
                (job_id, sequence),
            ).fetchall()
        events: list[JobEvent] = []
        for row in rows:
            event = JobEvent.model_validate_json(row["payload"])
            event.sequence = int(row["sequence"])
            events.append(event)
        return events
