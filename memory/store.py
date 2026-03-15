from __future__ import annotations

import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec

from memory.embedder import OllamaEmbedder

_log = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    row_id: int
    user_input: str
    assistant_output: str
    task_summary: str
    distance: float
    created_at: str
    metadata: dict


def _row_to_result(r: sqlite3.Row, distance: float | None = None) -> MemoryResult:
    """Convert a DB row to a MemoryResult.

    Args:
        r: Row from the conversations (+ optional distance) query.
        distance: Override distance value (used when the column is absent, e.g. get_recent).

    Returns:
        Populated MemoryResult dataclass.
    """
    return MemoryResult(
        row_id=r["id"],
        user_input=r["user_input"],
        assistant_output=r["assistant_output"],
        task_summary=r["task_summary"],
        distance=distance if distance is not None else r["distance"],
        created_at=r["created_at"],
        metadata=json.loads(r["metadata"]),
    )


class MemoryStore:
    def __init__(self, db_path: str, embedder: OllamaEmbedder):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lock = threading.Lock()
        self._conn = self._connect()
        self.init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def init_schema(self) -> None:
        schema_path = Path(__file__).parent / "schema.sql"
        sql = schema_path.read_text(encoding="utf-8")
        with self._lock:
            self._conn.executescript(sql)

    def store(
        self,
        user_input: str,
        assistant_output: str,
        task_summary: list[dict] | None = None,
        metadata: dict | None = None,
    ) -> int:
        task_summary_json = json.dumps(task_summary or [])
        metadata_json = json.dumps(metadata or {})

        combined = f"{user_input} {assistant_output}"
        try:
            embedding = self._embedder.embed(combined)  # slow HTTP — outside lock
        except Exception as e:
            _log.warning("Memory store skipped (embedding failed): %s", e)
            return -1
        embedding_bytes = json.dumps(embedding)

        with self._lock:
            with self._conn:  # BEGIN … COMMIT / ROLLBACK on exception
                cur = self._conn.execute(
                    "INSERT INTO conversations (user_input, assistant_output, task_summary, metadata)"
                    " VALUES (?, ?, ?, ?)",
                    (user_input, assistant_output, task_summary_json, metadata_json),
                )
                row_id = cur.lastrowid
                self._conn.execute(
                    "INSERT INTO memory_vec(rowid, embedding) VALUES (?, vec_f32(?))",
                    (row_id, embedding_bytes),
                )
        return row_id

    def search(self, query: str, top_k: int = 5, max_distance: float | None = None) -> list[MemoryResult]:
        try:
            embedding = self._embedder.embed(query)  # slow HTTP — outside lock
        except Exception as e:
            _log.warning("Memory search skipped (embedding failed): %s", e)
            return []
        embedding_bytes = json.dumps(embedding)

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT c.id, c.user_input, c.assistant_output, c.task_summary,
                       c.metadata, c.created_at, m.distance
                FROM memory_vec m
                JOIN conversations c ON c.id = m.rowid
                WHERE m.embedding MATCH vec_f32(?)
                  AND k = ?
                ORDER BY m.distance
                """,
                (embedding_bytes, top_k),
            ).fetchall()

        results = [_row_to_result(r) for r in rows]
        if max_distance is not None:
            results = [r for r in results if r.distance <= max_distance]
        return results

    def get_recent(self, n: int = 10) -> list[MemoryResult]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, user_input, assistant_output, task_summary, metadata,"
                " created_at, 0.0 as distance FROM conversations"
                " ORDER BY created_at DESC LIMIT ?",
                (n,),
            ).fetchall()

        return [_row_to_result(r) for r in rows]

    def delete(self, row_id: int) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM conversations WHERE id = ?", (row_id,))
                self._conn.execute("DELETE FROM memory_vec WHERE rowid = ?", (row_id,))

    def clear_all(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM conversations")
                self._conn.execute("DELETE FROM memory_vec")

    def close(self) -> None:
        self._conn.close()
