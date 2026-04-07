from pathlib import Path
import sqlite3
from datetime import datetime


class SQLiteStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        conn = self.connect()
        try:
            cur = conn.cursor()

            cur.execute(
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

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT NOT NULL,
                  role TEXT NOT NULL,
                  content TEXT NOT NULL,
                  timestamp TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  scope TEXT NOT NULL,
                  title TEXT NOT NULL,
                  content TEXT NOT NULL,
                  tags TEXT,
                  importance INTEGER DEFAULT 0,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS personality_profiles (
                  profile_id TEXT PRIMARY KEY,
                  name TEXT NOT NULL,
                  json_blob TEXT NOT NULL,
                  active INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS policies (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  domain TEXT NOT NULL,
                  action TEXT NOT NULL,
                  mode TEXT NOT NULL,
                  rule_json TEXT,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  scope_type TEXT NOT NULL DEFAULT 'global',
                  scope_value TEXT NOT NULL DEFAULT '*'
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source_path TEXT NOT NULL UNIQUE,
                  doc_type TEXT NOT NULL,
                  title TEXT NOT NULL,
                  checksum TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  document_id INTEGER NOT NULL,
                  chunk_index INTEGER NOT NULL,
                  content TEXT NOT NULL,
                  page_number INTEGER,
                  created_at TEXT NOT NULL
                )
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_images (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  document_id INTEGER,
                  source_path TEXT NOT NULL,
                  image_path TEXT NOT NULL,
                  page_number INTEGER,
                  caption TEXT,
                  ocr_text TEXT,
                  checksum TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )

            conn.commit()
        finally:
            conn.close()

    def upsert_session(self, session_id: str, channel: str, user_id: str, chat_id: str) -> None:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO sessions (session_id, channel, user_id, chat_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                  updated_at = excluded.updated_at
                """,
                (session_id, channel, user_id, chat_id, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_recent_messages(self, session_id: str, max_items: int) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, max_items),
            )
            rows = cur.fetchall()
            rows.reverse()
            return [{"role": row["role"], "content": row["content"]} for row in rows]
        finally:
            conn.close()

    def reset_session_messages(self, session_id: str) -> None:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.commit()
        finally:
            conn.close()

    def add_memory(self, scope: str, title: str, content: str, tags: str = "", importance: int = 0) -> int:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO memories (scope, title, content, tags, importance, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (scope, title, content, tags, importance, now, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()

    def memory_exists(self, scope: str, content: str) -> bool:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT 1
                FROM memories
                WHERE scope = ? AND content = ?
                LIMIT 1
                """,
                (scope, content),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    def add_memory_if_new(self, scope: str, title: str, content: str, tags: str = "", importance: int = 0) -> int | None:
        if self.memory_exists(scope=scope, content=content):
            return None
        return self.add_memory(scope=scope, title=title, content=content, tags=tags, importance=importance)

    def list_memories(self, scope: str | None = None, limit: int = 20) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            if scope:
                cur.execute(
                    """
                    SELECT id, scope, title, content, tags, importance, created_at, updated_at
                    FROM memories
                    WHERE scope = ?
                    ORDER BY importance DESC, id DESC
                    LIMIT ?
                    """,
                    (scope, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id, scope, title, content, tags, importance, created_at, updated_at
                    FROM memories
                    ORDER BY importance DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_memory_prompt_lines(self, scope: str, limit: int = 5) -> list[str]:
        memories = self.list_memories(scope=scope, limit=limit)
        lines: list[str] = []
        for memory in memories:
            title = str(memory.get("title", "")).strip()
            content = str(memory.get("content", "")).strip()
            if title and content:
                lines.append(f"- {title}: {content}")
            elif content:
                lines.append(f"- {content}")
        return lines

    def upsert_personality_profile(self, profile_id: str, name: str, json_blob: str, active: int = 1) -> None:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            if active:
                cur.execute("UPDATE personality_profiles SET active = 0")
            cur.execute(
                """
                INSERT INTO personality_profiles (profile_id, name, json_blob, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(profile_id) DO UPDATE SET
                  name = excluded.name,
                  json_blob = excluded.json_blob,
                  active = excluded.active,
                  updated_at = excluded.updated_at
                """,
                (profile_id, name, json_blob, active, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_active_personality_profile(self) -> dict | None:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT profile_id, name, json_blob, active, created_at, updated_at
                FROM personality_profiles
                WHERE active = 1
                ORDER BY updated_at DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def list_personality_profiles(self) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT profile_id, name, json_blob, active, created_at, updated_at
                FROM personality_profiles
                ORDER BY updated_at DESC
                """
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def upsert_knowledge_document(self, source_path: str, doc_type: str, title: str, checksum: str, status: str) -> int:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO knowledge_documents (source_path, doc_type, title, checksum, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_path) DO UPDATE SET
                  title = excluded.title,
                  checksum = excluded.checksum,
                  status = excluded.status,
                  updated_at = excluded.updated_at
                """,
                (source_path, doc_type, title, checksum, status, now, now),
            )
            conn.commit()
            cur.execute("SELECT id FROM knowledge_documents WHERE source_path = ?", (source_path,))
            row = cur.fetchone()
            return int(row["id"])
        finally:
            conn.close()

    def replace_knowledge_chunks(self, document_id: int, chunks: list[dict]) -> None:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (document_id,))
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks (document_id, chunk_index, content, page_number, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (document_id, idx, chunk["content"], chunk.get("page_number"), now),
                )
            conn.commit()
        finally:
            conn.close()

    def list_knowledge_chunks(self) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  kc.id,
                  kc.document_id,
                  kc.chunk_index,
                  kc.content,
                  kc.page_number,
                  kd.title,
                  kd.source_path
                FROM knowledge_chunks kc
                JOIN knowledge_documents kd ON kd.id = kc.document_id
                ORDER BY kc.id
                """
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_knowledge_chunk_by_id(self, chunk_id: int) -> dict | None:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  kc.id,
                  kc.document_id,
                  kc.chunk_index,
                  kc.content,
                  kc.page_number,
                  kd.title,
                  kd.source_path
                FROM knowledge_chunks kc
                JOIN knowledge_documents kd ON kd.id = kc.document_id
                WHERE kc.id = ?
                """,
                (chunk_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def upsert_knowledge_image(
        self,
        document_id: int | None,
        source_path: str,
        image_path: str,
        page_number: int | None,
        caption: str,
        ocr_text: str,
        checksum: str,
    ) -> int:
        now = datetime.utcnow().isoformat()
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id
                FROM knowledge_images
                WHERE image_path = ?
                LIMIT 1
                """,
                (image_path,),
            )
            existing = cur.fetchone()

            if existing:
                image_id = int(existing["id"])
                cur.execute(
                    """
                    UPDATE knowledge_images
                    SET document_id = ?, source_path = ?, page_number = ?, caption = ?, ocr_text = ?, checksum = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (document_id, source_path, page_number, caption, ocr_text, checksum, now, image_id),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO knowledge_images
                    (document_id, source_path, image_path, page_number, caption, ocr_text, checksum, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (document_id, source_path, image_path, page_number, caption, ocr_text, checksum, now, now),
                )
                image_id = int(cur.lastrowid)

            conn.commit()
            return image_id
        finally:
            conn.close()

    def find_knowledge_documents_by_title(self, query: str, limit: int = 10) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, source_path, doc_type, title, checksum, status, created_at, updated_at
                FROM knowledge_documents
                WHERE lower(title) LIKE lower(?)
                    OR lower(source_path) LIKE lower(?)
                ORDER BY title
                LIMIT ?
                """,
                (f"%{query}%", f"%{query}%", limit),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_knowledge_chunks_for_document_page(self, document_id: int, page_number: int) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  kc.id,
                  kc.document_id,
                  kc.chunk_index,
                  kc.content,
                  kc.page_number
                FROM knowledge_chunks kc
                WHERE kc.document_id = ? AND kc.page_number = ?
                ORDER BY kc.chunk_index
                """,
                (document_id, page_number),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_knowledge_images_for_document_page(self, document_id: int, page_number: int) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  id,
                  document_id,
                  source_path,
                  image_path,
                  page_number,
                  caption,
                  ocr_text,
                  checksum,
                  created_at,
                  updated_at
                FROM knowledge_images
                WHERE document_id = ? AND page_number = ?
                ORDER BY id
                """,
                (document_id, page_number),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def find_knowledge_documents_by_title(self, query: str, limit: int = 10) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT id, source_path, doc_type, title, checksum, status, created_at, updated_at
                FROM knowledge_documents
                WHERE lower(title) LIKE lower(?)
                ORDER BY title
                LIMIT ?
                """,
                (f"%{query}%", limit),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_knowledge_chunks_for_document_page(self, document_id: int, page_number: int) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  kc.id,
                  kc.document_id,
                  kc.chunk_index,
                  kc.content,
                  kc.page_number
                FROM knowledge_chunks kc
                WHERE kc.document_id = ? AND kc.page_number = ?
                ORDER BY kc.chunk_index
                """,
                (document_id, page_number),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_knowledge_images_for_document_page(self, document_id: int, page_number: int) -> list[dict]:
        conn = self.connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT
                  id,
                  document_id,
                  source_path,
                  image_path,
                  page_number,
                  caption,
                  ocr_text,
                  checksum,
                  created_at,
                  updated_at
                FROM knowledge_images
                WHERE document_id = ? AND page_number = ?
                ORDER BY id
                """,
                (document_id, page_number),
            )
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()