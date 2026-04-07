import json
from datetime import datetime

from avacore.config.settings import settings
from avacore.memory.sqlite_store import SQLiteStore
from avacore.policy.engine import DEFAULT_RULES


def migrate_policies_scope_columns(store: SQLiteStore) -> None:
    conn = store.connect()
    try:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(policies)")
        cols = {row["name"] for row in cur.fetchall()}

        if "scope_type" not in cols:
            cur.execute("ALTER TABLE policies ADD COLUMN scope_type TEXT NOT NULL DEFAULT 'global'")
            print("Added policies.scope_type")

        if "scope_value" not in cols:
            cur.execute("ALTER TABLE policies ADD COLUMN scope_value TEXT NOT NULL DEFAULT '*'")
            print("Added policies.scope_value")

        conn.commit()
    finally:
        conn.close()


def seed_default_policies(store: SQLiteStore) -> None:
    conn = store.connect()
    try:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) AS cnt FROM policies")
        count = int(cur.fetchone()["cnt"])

        if count > 0:
            print("Policies already present, skipping seed.")
            return

        now = datetime.utcnow().isoformat()

        for rule in DEFAULT_RULES:
            cur.execute(
                """
                INSERT INTO policies
                (domain, action, mode, scope_type, scope_value, rule_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rule.domain,
                    rule.action,
                    rule.mode,
                    rule.scope_type,
                    rule.scope_value,
                    json.dumps({"reason": rule.reason}, ensure_ascii=False),
                    now,
                    now,
                ),
            )

        conn.commit()
        print(f"Inserted {len(DEFAULT_RULES)} default policies.")
    finally:
        conn.close()


def backfill_existing_policies(store: SQLiteStore) -> None:
    conn = store.connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE policies
            SET scope_type = COALESCE(scope_type, 'global'),
                scope_value = COALESCE(scope_value, '*')
            """
        )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    store = SQLiteStore(settings.db_path)
    store.init_db()
    migrate_policies_scope_columns(store)
    backfill_existing_policies(store)
    seed_default_policies(store)
    print(f"Initialized DB at {settings.db_path}")


if __name__ == "__main__":
    main()
