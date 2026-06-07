from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class Note:
    id: int
    title: str
    content: str
    status: str
    tags: str
    source: str
    created_at: str
    updated_at: str
    synced_at: str | None = None
    google_doc_ref: str | None = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect(db_path: Path | str) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_notes_db(db_path: Path | str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                tags TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL DEFAULT 'manual',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                synced_at TEXT,
                google_doc_ref TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_notes_status ON notes(status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_notes_updated_at ON notes(updated_at)"
        )
        conn.commit()


def _row_to_note(row: sqlite3.Row) -> Note:
    return Note(
        id=int(row["id"]),
        title=row["title"],
        content=row["content"],
        status=row["status"],
        tags=row["tags"],
        source=row["source"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        synced_at=row["synced_at"],
        google_doc_ref=row["google_doc_ref"],
    )


def _make_title(content: str, max_len: int = 80) -> str:
    text = " ".join((content or "").strip().split())
    if not text:
        return "Untitled note"
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def create_note(
    db_path: Path | str,
    content: str,
    title: str | None = None,
    tags: str = "",
    source: str = "telegram",
) -> Note:
    init_notes_db(db_path)

    content = (content or "").strip()
    if not content:
        raise ValueError("note content is empty")

    now = _utcnow()
    note_title = (title or _make_title(content)).strip()

    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO notes
                (title, content, status, tags, source, created_at, updated_at)
            VALUES
                (?, ?, 'open', ?, ?, ?, ?)
            """,
            (note_title, content, tags.strip(), source.strip() or "telegram", now, now),
        )
        conn.commit()
        note_id = int(cur.lastrowid)

    note = get_note(db_path, note_id)
    if note is None:
        raise RuntimeError("created note could not be loaded")
    return note


def get_note(db_path: Path | str, note_id: int) -> Note | None:
    init_notes_db(db_path)

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?",
            (int(note_id),),
        ).fetchone()

    return _row_to_note(row) if row else None


def list_notes(
    db_path: Path | str,
    status: str = "open",
    limit: int = 10,
) -> list[Note]:
    init_notes_db(db_path)

    limit = max(1, min(int(limit), 50))

    with _connect(db_path) as conn:
        if status == "all":
            rows = conn.execute(
                """
                SELECT * FROM notes
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM notes
                WHERE status = ?
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (status, limit),
            ).fetchall()

    return [_row_to_note(row) for row in rows]


def search_notes(
    db_path: Path | str,
    query: str,
    limit: int = 10,
) -> list[Note]:
    init_notes_db(db_path)

    query = (query or "").strip()
    if not query:
        return []

    like = f"%{query}%"
    limit = max(1, min(int(limit), 50))

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM notes
            WHERE title LIKE ?
               OR content LIKE ?
               OR tags LIKE ?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (like, like, like, limit),
        ).fetchall()

    return [_row_to_note(row) for row in rows]


def append_to_note(
    db_path: Path | str,
    note_id: int,
    extra_content: str,
) -> Note:
    init_notes_db(db_path)

    extra_content = (extra_content or "").strip()
    if not extra_content:
        raise ValueError("append content is empty")

    note = get_note(db_path, note_id)
    if note is None:
        raise ValueError(f"note not found: {note_id}")

    now = _utcnow()
    new_content = note.content.rstrip() + "\n\n" + extra_content

    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE notes
            SET content = ?, updated_at = ?
            WHERE id = ?
            """,
            (new_content, now, int(note_id)),
        )
        conn.commit()

    updated = get_note(db_path, note_id)
    if updated is None:
        raise RuntimeError("updated note could not be loaded")
    return updated


def update_note_status(
    db_path: Path | str,
    note_id: int,
    status: str,
) -> Note:
    init_notes_db(db_path)

    status = status.strip().lower()
    if status not in {"open", "done", "archived"}:
        raise ValueError("invalid note status")

    note = get_note(db_path, note_id)
    if note is None:
        raise ValueError(f"note not found: {note_id}")

    now = _utcnow()

    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE notes
            SET status = ?, updated_at = ?
            WHERE id = ?
            """,
            (status, now, int(note_id)),
        )
        conn.commit()

    updated = get_note(db_path, note_id)
    if updated is None:
        raise RuntimeError("updated note could not be loaded")
    return updated


def format_note(note: Note, include_content: bool = True) -> str:
    status_icon = {
        "open": "🟡",
        "done": "✅",
        "archived": "📦",
    }.get(note.status, "•")

    lines = [
        f"{status_icon} #{note.id} — {note.title}",
        f"Status: {note.status}",
    ]

    if note.tags:
        lines.append(f"Tags: {note.tags}")

    if include_content:
        lines.append("")
        lines.append(note.content)

    return "\n".join(lines)


def format_note_list(notes: list[Note]) -> str:
    if not notes:
        return "Keine Notizen gefunden."

    lines: list[str] = []

    for note in notes:
        first_line = " ".join(note.content.split())
        if len(first_line) > 120:
            first_line = first_line[:117] + "..."

        status_icon = {
            "open": "🟡",
            "done": "✅",
            "archived": "📦",
        }.get(note.status, "•")

        lines.append(f"{status_icon} #{note.id} — {first_line}")

    return "\n".join(lines)