from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class SQLiteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _utcnow(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds")

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags TEXT NOT NULL DEFAULT '',
                    importance INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personality_profiles (
                    profile_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    json_blob TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    doc_type TEXT NOT NULL DEFAULT '',
                    source_path TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_documents_title ON knowledge_documents(title)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER,
                    content TEXT NOT NULL,
                    embedding_ref TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_id ON knowledge_chunks(document_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_page_number ON knowledge_chunks(page_number)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    page_number INTEGER,
                    image_path TEXT NOT NULL DEFAULT '',
                    caption TEXT NOT NULL DEFAULT '',
                    ocr_text TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_images_document_id ON knowledge_images(document_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_knowledge_images_page_number ON knowledge_images(page_number)"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scope TEXT NOT NULL DEFAULT 'user',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,

                    memory_type TEXT NOT NULL DEFAULT 'note',
                    status TEXT NOT NULL DEFAULT 'candidate',

                    source_type TEXT NOT NULL DEFAULT 'chat',
                    source_ref TEXT NOT NULL DEFAULT '',

                    confidence REAL NOT NULL DEFAULT 0.0,
                    importance INTEGER NOT NULL DEFAULT 0,

                    tags TEXT NOT NULL DEFAULT '',

                    created_from_user_text TEXT NOT NULL DEFAULT '',
                    created_from_assistant_text TEXT NOT NULL DEFAULT '',

                    verified_by TEXT NOT NULL DEFAULT '',
                    verified_at TEXT NOT NULL DEFAULT '',

                    rejected_by TEXT NOT NULL DEFAULT '',
                    rejected_at TEXT NOT NULL DEFAULT '',

                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_status ON memory_items(status)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_type ON memory_items(memory_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_scope ON memory_items(scope)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_items_source_type ON memory_items(source_type)"
            )

            conn.commit()

    # -------------------------------------------------------------------------
    # Sessions / messages
    # -------------------------------------------------------------------------

    def upsert_session(
        self,
        session_id: str,
        channel: str,
        user_id: str,
        chat_id: str,
    ) -> None:
        now = self._utcnow()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE sessions
                    SET channel = ?, user_id = ?, chat_id = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (channel, user_id, chat_id, now, session_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, channel, user_id, chat_id, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, channel, user_id, chat_id, now, now),
                )
            conn.commit()

    def add_message(self, session_id: str, role: str, content: str) -> int:
        now = self._utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            conn.commit()
            return int(cur.lastrowid)

    def get_recent_messages(self, session_id: str, max_items: int = 8) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, int(max_items)),
            ).fetchall()

        rows = list(reversed(rows))
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def reset_session_messages(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()

    # -------------------------------------------------------------------------
    # Legacy memories
    # -------------------------------------------------------------------------

    def add_memory(
        self,
        scope: str,
        title: str,
        content: str,
        tags: str = "",
        importance: int = 0,
    ) -> int:
        now = self._utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO memories (
                    scope, title, content, tags, importance, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (scope, title, content, tags, int(importance), now, now),
            )
            conn.commit()
            return int(cur.lastrowid)

    def add_memory_if_new(
        self,
        scope: str,
        title: str,
        content: str,
        tags: str = "",
        importance: int = 0,
    ) -> int | None:
        title_norm = (title or "").strip().lower()
        content_norm = (content or "").strip().lower()

        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM memories
                WHERE scope = ?
                  AND lower(trim(title)) = ?
                  AND lower(trim(content)) = ?
                LIMIT 1
                """,
                (scope, title_norm, content_norm),
            ).fetchone()

            if row:
                return None

        return self.add_memory(
            scope=scope,
            title=title,
            content=content,
            tags=tags,
            importance=importance,
        )

    def list_memories(self, scope: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        query = "SELECT * FROM memories"
        params: list[Any] = []

        if scope:
            query += " WHERE scope = ?"
            params.append(scope)

        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_memory_prompt_lines(self, scope: str = "user", limit: int = 5) -> list[str]:
        items = self.list_memories(scope=scope, limit=limit)
        lines: list[str] = []

        for item in items:
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            if title and content:
                lines.append(f"- {title}: {content}")
            elif content:
                lines.append(f"- {content}")

        return lines

    # -------------------------------------------------------------------------
    # New memory items: candidate / verified / rejected
    # -------------------------------------------------------------------------

    def create_memory_item(
        self,
        scope: str,
        title: str,
        content: str,
        memory_type: str = "note",
        status: str = "candidate",
        source_type: str = "chat",
        source_ref: str = "",
        confidence: float = 0.0,
        importance: int = 0,
        tags: str = "",
        created_from_user_text: str = "",
        created_from_assistant_text: str = "",
    ) -> int:
        now = self._utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO memory_items (
                    scope, title, content,
                    memory_type, status,
                    source_type, source_ref,
                    confidence, importance, tags,
                    created_from_user_text, created_from_assistant_text,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scope,
                    title,
                    content,
                    memory_type,
                    status,
                    source_type,
                    source_ref,
                    float(confidence),
                    int(importance),
                    tags,
                    created_from_user_text,
                    created_from_assistant_text,
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def list_memory_items(
        self,
        status: str | None = None,
        memory_type: str | None = None,
        scope: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM memory_items WHERE 1=1"
        params: list[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        if scope:
            query += " AND scope = ?"
            params.append(scope)

        query += " ORDER BY updated_at DESC, id DESC LIMIT ?"
        params.append(int(limit))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]

    def get_memory_item(self, memory_id: int) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM memory_items WHERE id = ?",
                (int(memory_id),),
            ).fetchone()
            return dict(row) if row else None

    def verify_memory_item(self, memory_id: int, verified_by: str) -> bool:
        now = self._utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE memory_items
                SET status = 'verified',
                    verified_by = ?,
                    verified_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (verified_by, now, now, int(memory_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def reject_memory_item(self, memory_id: int, rejected_by: str) -> bool:
        now = self._utcnow()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE memory_items
                SET status = 'rejected',
                    rejected_by = ?,
                    rejected_at = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (rejected_by, now, now, int(memory_id)),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_memory_item(self, memory_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM memory_items WHERE id = ?",
                (int(memory_id),),
            )
            conn.commit()
            return cur.rowcount > 0

    def get_verified_memory_prompt_lines(
        self,
        scope: str = "user",
        limit: int = 8,
    ) -> list[str]:
        items = self.list_memory_items(
            status="verified",
            scope=scope,
            limit=limit,
        )

        lines: list[str] = []
        for item in items:
            title = (item.get("title") or "").strip()
            content = (item.get("content") or "").strip()
            memory_type = (item.get("memory_type") or "").strip()

            if title and content:
                lines.append(f"- [{memory_type}] {title}: {content}")
            elif content:
                lines.append(f"- [{memory_type}] {content}")

        return lines

    # -------------------------------------------------------------------------
    # Personality profiles
    # -------------------------------------------------------------------------

    def upsert_personality_profile(
        self,
        profile_id: str,
        name: str,
        json_blob: str,
        active: int = 0,
    ) -> None:
        now = self._utcnow()

        with self._connect() as conn:
            if active:
                conn.execute("UPDATE personality_profiles SET active = 0")

            existing = conn.execute(
                "SELECT profile_id FROM personality_profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE personality_profiles
                    SET name = ?, json_blob = ?, active = ?, updated_at = ?
                    WHERE profile_id = ?
                    """,
                    (name, json_blob, int(active), now, profile_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO personality_profiles (
                        profile_id, name, json_blob, active, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (profile_id, name, json_blob, int(active), now, now),
                )

            conn.commit()

    def list_personality_profiles(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT profile_id, name, json_blob, active, created_at, updated_at
                FROM personality_profiles
                ORDER BY updated_at DESC, profile_id DESC
                """
            ).fetchall()
            return [dict(row) for row in rows]

    # -------------------------------------------------------------------------
    # Knowledge documents / chunks / images
    # -------------------------------------------------------------------------

    def find_knowledge_documents_by_title(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        query = (query or "").strip()

        with self._connect() as conn:
            if not query:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM knowledge_documents
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()
            else:
                like = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT *
                    FROM knowledge_documents
                    WHERE title LIKE ?
                    ORDER BY updated_at DESC, id DESC
                    LIMIT ?
                    """,
                    (like, int(limit)),
                ).fetchall()

        return [dict(row) for row in rows]

    def get_knowledge_chunks_for_document_page(
        self,
        document_id: int,
        page_number: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM knowledge_chunks
                WHERE document_id = ? AND page_number = ?
                ORDER BY id ASC
                """,
                (int(document_id), int(page_number)),
            ).fetchall()
            return [dict(row) for row in rows]

    def get_knowledge_images_for_document_page(
        self,
        document_id: int,
        page_number: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM knowledge_images
                WHERE document_id = ? AND page_number = ?
                ORDER BY id ASC
                """,
                (int(document_id), int(page_number)),
            ).fetchall()
            return [dict(row) for row in rows]