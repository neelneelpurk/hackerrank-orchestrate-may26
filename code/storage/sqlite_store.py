"""SQLite store for raw chunk text + embedding bookkeeping.

Two columns make resumable indexing possible:
- `content_hash` (sha256 of chunk text): detects when re-chunking produced different text
- `embedded_at` (ISO timestamp or NULL): NULL until Chroma upsert succeeds, then set

On a re-run the indexer can skip a chunk iff
  (source_path, chunk_index) row exists  AND  content_hash matches  AND  embedded_at IS NOT NULL.
This makes Jina rate-limit failures recoverable — interrupted runs resume where they stopped.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    company      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    product_area TEXT NOT NULL,
    doc_type     TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    content      TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    content_hash TEXT,
    embedded_at  TEXT,
    UNIQUE(source_path, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_company      ON chunks(company);
CREATE INDEX IF NOT EXISTS idx_product_area ON chunks(product_area);
CREATE INDEX IF NOT EXISTS idx_doc_type     ON chunks(doc_type);
CREATE INDEX IF NOT EXISTS idx_embedded_at  ON chunks(embedded_at);
"""


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


class SqliteStore:
    def __init__(self, db_path: str = "knowledge_base.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._ensure_columns()
        self.conn.commit()

    def _ensure_columns(self) -> None:
        """Add content_hash / embedded_at to a pre-existing DB that lacked them."""
        cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(chunks)").fetchall()}
        if "content_hash" not in cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN content_hash TEXT")
        if "embedded_at" not in cols:
            self.conn.execute("ALTER TABLE chunks ADD COLUMN embedded_at TEXT")

    def clear_company(self, company: str) -> None:
        self.conn.execute("DELETE FROM chunks WHERE company = ?", (company,))
        self.conn.commit()

    def lookup_chunk(self, company: str, source_path: str, chunk_index: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, content_hash, embedded_at FROM chunks "
            "WHERE company = ? AND source_path = ? AND chunk_index = ?",
            (company, source_path, chunk_index),
        ).fetchone()
        return dict(row) if row else None

    def upsert_chunk(
        self,
        company: str,
        source_path: str,
        product_area: str,
        doc_type: str,
        heading_path: str,
        content: str,
        chunk_index: int,
        content_hash: str,
    ) -> int:
        """Upsert a chunk row keyed by (source_path, chunk_index). Resets embedded_at to NULL
        if the content changed; preserves it otherwise.

        Returns the SQLite id of the row.
        """
        existing = self.lookup_chunk(company, source_path, chunk_index)
        if existing and existing.get("content_hash") == content_hash:
            # Idempotent: same content, just refresh metadata fields (heading_path / product_area).
            self.conn.execute(
                """UPDATE chunks
                      SET product_area = ?, doc_type = ?, heading_path = ?
                    WHERE id = ?""",
                (product_area, doc_type, heading_path, existing["id"]),
            )
            return int(existing["id"])

        # Content changed (or new): replace and clear embedded_at so we re-embed.
        cur = self.conn.execute(
            """INSERT INTO chunks
                 (company, source_path, product_area, doc_type, heading_path,
                  content, chunk_index, content_hash, embedded_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
               ON CONFLICT(source_path, chunk_index) DO UPDATE SET
                 company       = excluded.company,
                 product_area  = excluded.product_area,
                 doc_type      = excluded.doc_type,
                 heading_path  = excluded.heading_path,
                 content       = excluded.content,
                 content_hash  = excluded.content_hash,
                 embedded_at   = NULL""",
            (company, source_path, product_area, doc_type, heading_path,
             content, chunk_index, content_hash),
        )
        if cur.lastrowid:
            return int(cur.lastrowid)
        # ON CONFLICT path returns rowid of 0 / unchanged: re-fetch.
        again = self.lookup_chunk(company, source_path, chunk_index)
        return int(again["id"])

    def mark_embedded(self, sqlite_ids: list[int]) -> None:
        if not sqlite_ids:
            return
        ts = _utcnow()
        self.conn.executemany(
            "UPDATE chunks SET embedded_at = ? WHERE id = ?",
            [(ts, sid) for sid in sqlite_ids],
        )
        self.conn.commit()

    def commit(self) -> None:
        self.conn.commit()

    def get_chunk(self, sqlite_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, company, source_path, product_area, doc_type, "
            "       heading_path, content, chunk_index, content_hash, embedded_at "
            "  FROM chunks WHERE id = ?",
            (sqlite_id,),
        ).fetchone()
        return dict(row) if row else None

    def list_company(self, company: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, company, source_path, product_area, doc_type, "
            "       heading_path, content, chunk_index, content_hash, embedded_at "
            "  FROM chunks WHERE company = ? ORDER BY id",
            (company,),
        ).fetchall()
        return [dict(r) for r in rows]

    def counts_by_company(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT company, COUNT(*) FROM chunks GROUP BY company"
        ).fetchall()
        return {r[0]: r[1] for r in rows}

    def counts_pending(self, company: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE company = ? AND embedded_at IS NULL",
            (company,),
        ).fetchone()
        return int(row[0]) if row else 0
