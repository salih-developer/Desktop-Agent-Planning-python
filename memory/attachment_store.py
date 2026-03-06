from __future__ import annotations

import re
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memory.embedder import OllamaEmbedder

from logger import get_logger

_log = get_logger(__name__)

_CHUNK_SIZE = 2000   # target chars per chunk
_OVERLAP = 200       # overlap chars between consecutive chunks
_SUMMARY_RE = re.compile(
    r"\b(özetle|özetini|özetler|summarize|summary|overview|tümünü|hepsini|tamamen|full|all)\b",
    re.IGNORECASE,
)


class AttachmentStore:
    """Session-scoped in-memory vector store for large attachment chunks."""

    def __init__(self, embedder: "OllamaEmbedder", dim: int = 768) -> None:
        self._embedder = embedder
        self._dim = dim
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._setup()
        self._has_data = False

    # ── Schema ──────────────────────────────────────────────────────────

    def _setup(self) -> None:
        self._conn.enable_load_extension(True)
        import sqlite_vec
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        dim = self._dim
        self._conn.executescript(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                chunk_idx INTEGER NOT NULL,
                content   TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                embedding float[{dim}]
            );
        """)
        self._conn.commit()

    # ── Indexing ─────────────────────────────────────────────────────────

    def index(self, file_path: str, file_name: str, content: str) -> int:
        """Chunk `content`, embed each chunk, store in the in-memory DB. Returns chunk count."""
        chunks = self._split(content)
        _log.info("AttachmentStore: indexing %r → %d chunk(s)", file_name, len(chunks))
        for idx, chunk in enumerate(chunks):
            try:
                vec = self._embedder.embed(chunk)
            except Exception as exc:
                _log.warning("Embedding chunk %d of %r failed: %s", idx, file_name, exc)
                continue
            cur = self._conn.execute(
                "INSERT INTO chunks (file_path, file_name, chunk_idx, content) VALUES (?, ?, ?, ?)",
                (file_path, file_name, idx, chunk),
            )
            rowid = cur.lastrowid
            import struct
            blob = struct.pack(f"{len(vec)}f", *vec)
            self._conn.execute(
                "INSERT INTO chunks_vec (rowid, embedding) VALUES (?, ?)",
                (rowid, blob),
            )
        self._conn.commit()
        self._has_data = True
        return len(chunks)

    # ── Retrieval ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-K relevant chunks for `query`."""
        if not self._has_data:
            return []
        try:
            vec = self._embedder.embed(query)
        except Exception as exc:
            _log.warning("AttachmentStore: query embedding failed: %s", exc)
            return []
        import struct
        blob = struct.pack(f"{len(vec)}f", *vec)
        rows = self._conn.execute(
            """
            SELECT c.file_name, c.chunk_idx, c.content, v.distance
            FROM chunks_vec v
            JOIN chunks c ON c.id = v.rowid
            WHERE v.embedding MATCH ?
              AND k = ?
            ORDER BY v.distance
            """,
            (blob, top_k),
        ).fetchall()
        return [
            {"file_name": r[0], "chunk_idx": r[1], "content": r[2], "distance": r[3]}
            for r in rows
        ]

    def get_first_chunks(self, file_name: str, n: int = 3) -> list[dict]:
        """Return the first N chunks of a specific file (for summary queries)."""
        rows = self._conn.execute(
            "SELECT file_name, chunk_idx, content FROM chunks WHERE file_name = ? ORDER BY chunk_idx LIMIT ?",
            (file_name, n),
        ).fetchall()
        return [{"file_name": r[0], "chunk_idx": r[1], "content": r[2]} for r in rows]

    def get_all_file_names(self) -> list[str]:
        rows = self._conn.execute(
            "SELECT DISTINCT file_name FROM chunks ORDER BY file_name"
        ).fetchall()
        return [r[0] for r in rows]

    def has_data(self) -> bool:
        return self._has_data

    def clear(self) -> None:
        self._conn.executescript("""
            DELETE FROM chunks;
            DELETE FROM chunks_vec;
        """)
        self._conn.commit()
        self._has_data = False
        _log.debug("AttachmentStore cleared")

    # ── Splitting ────────────────────────────────────────────────────────

    @staticmethod
    def _split(text: str) -> list[str]:
        """Split text into ~CHUNK_SIZE char chunks with OVERLAP, preferring paragraph breaks."""
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if not para.strip():
                continue
            if len(current) + len(para) + 2 <= _CHUNK_SIZE:
                current = (current + "\n\n" + para).lstrip()
            else:
                if current:
                    chunks.append(current)
                    # overlap: carry last OVERLAP chars into next chunk
                    overlap_text = current[-_OVERLAP:] if len(current) > _OVERLAP else current
                    current = overlap_text + "\n\n" + para
                else:
                    # Single paragraph exceeds chunk size — split by sentence
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sub = ""
                    for sent in sentences:
                        if len(sub) + len(sent) + 1 <= _CHUNK_SIZE:
                            sub = (sub + " " + sent).lstrip()
                        else:
                            if sub:
                                chunks.append(sub)
                                sub = sub[-_OVERLAP:] + " " + sent
                            else:
                                # Single sentence > CHUNK_SIZE: hard split
                                for i in range(0, len(sent), _CHUNK_SIZE - _OVERLAP):
                                    chunks.append(sent[i:i + _CHUNK_SIZE])
                                sub = ""
                    if sub:
                        current = sub

        if current.strip():
            chunks.append(current.strip())

        return chunks or [text[:_CHUNK_SIZE]]

    # ── Context builder ──────────────────────────────────────────────────

    @staticmethod
    def is_summary_query(query: str) -> bool:
        return bool(_SUMMARY_RE.search(query))

    def build_context(self, query: str, top_k: int = 5, first_n: int = 3) -> str:
        """
        Build a context block to inject into the planner prompt.

        - Summary queries: always prepend the first `first_n` chunks of each indexed file.
        - All queries: append top-K semantically relevant chunks (deduped).
        """
        if not self._has_data:
            return ""

        seen_keys: set[tuple[str, int]] = set()
        parts: list[dict] = []

        if self.is_summary_query(query):
            for fname in self.get_all_file_names():
                for ch in self.get_first_chunks(fname, first_n):
                    key = (ch["file_name"], ch["chunk_idx"])
                    if key not in seen_keys:
                        seen_keys.add(key)
                        parts.append(ch)

        for ch in self.search(query, top_k=top_k):
            key = (ch["file_name"], ch["chunk_idx"])
            if key not in seen_keys:
                seen_keys.add(key)
                parts.append(ch)

        if not parts:
            return ""

        lines = ["### Retrieved Document Chunks (RAG)"]
        for ch in parts:
            lines.append(f"\n[{ch['file_name']} — chunk {ch['chunk_idx']}]\n{ch['content']}")
        return "\n".join(lines)
