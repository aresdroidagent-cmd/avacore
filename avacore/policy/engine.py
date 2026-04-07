from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from avacore.policy.schema import PolicyRule


DEFAULT_RULES: list[PolicyRule] = [
    PolicyRule(
        domain="coding",
        action="generate_code",
        mode="ask",
        scope_type="global",
        scope_value="*",
        reason="Vor Erzeugung von Code soll Ava bei unklarer Absicht erst rückfragen.",
    ),
    PolicyRule(
        domain="external",
        action="send_mail",
        mode="ask",
        scope_type="global",
        scope_value="*",
        reason="Externe Kommunikation soll nicht ungefragt ausgelöst werden.",
    ),
    PolicyRule(
        domain="web",
        action="web_fetch",
        mode="allow",
        scope_type="global",
        scope_value="*",
        reason="Webzugriff ist grundsätzlich erlaubt.",
    ),
    PolicyRule(
        domain="vision",
        action="camera_access",
        mode="deny",
        scope_type="global",
        scope_value="*",
        reason="Kamerazugriff ist standardmäßig deaktiviert.",
    ),
]


class PolicyEngine:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def list_rules(self) -> list[PolicyRule]:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT domain, action, mode, scope_type, scope_value, rule_json
                FROM policies
                ORDER BY
                  CASE scope_type
                    WHEN 'user' THEN 1
                    WHEN 'channel' THEN 2
                    ELSE 3
                  END,
                  domain,
                  action
                """
            )
            rows = cur.fetchall()
        finally:
            conn.close()

        if not rows:
            return DEFAULT_RULES

        rules: list[PolicyRule] = []
        for row in rows:
            rule_json = row["rule_json"]
            reason = ""
            if rule_json:
                try:
                    parsed = json.loads(rule_json)
                    reason = str(parsed.get("reason", ""))
                except Exception:
                    reason = ""

            rules.append(
                PolicyRule(
                    domain=row["domain"],
                    action=row["action"],
                    mode=row["mode"],
                    scope_type=row["scope_type"],
                    scope_value=row["scope_value"],
                    reason=reason,
                )
            )
        return rules

    def resolve(
        self,
        domain: str,
        action: str,
        *,
        channel: str | None = None,
        user_id: str | None = None,
    ) -> PolicyRule | None:
        rules = self.list_rules()

        # 1) user
        if user_id:
            for rule in rules:
                if (
                    rule.domain == domain
                    and rule.action == action
                    and rule.scope_type == "user"
                    and rule.scope_value == user_id
                ):
                    return rule

        # 2) channel
        if channel:
            for rule in rules:
                if (
                    rule.domain == domain
                    and rule.action == action
                    and rule.scope_type == "channel"
                    and rule.scope_value == channel
                ):
                    return rule

        # 3) global
        for rule in rules:
            if (
                rule.domain == domain
                and rule.action == action
                and rule.scope_type == "global"
                and rule.scope_value == "*"
            ):
                return rule

        return None
