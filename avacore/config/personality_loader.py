import json

from avacore.config.settings import settings
from avacore.memory.sqlite_store import SQLiteStore
from avacore.personality.manager import PersonalityManager


def load_personality_manager() -> PersonalityManager:
    return PersonalityManager(settings.personality_path)


def load_personality_json_text() -> str:
    store = SQLiteStore(settings.db_path)
    active = store.get_active_personality_profile()
    if active and active.get("json_blob"):
        return str(active["json_blob"])

    path = settings.personality_path
    if path.exists():
        return path.read_text(encoding="utf-8")

    return json.dumps(
        {
            "name": "Ava",
            "language_default": "de",
            "tone": {"style": "klar", "verbosity": "kurz", "humor": "leicht"},
            "behavior": {
                "honest_when_uncertain": True,
                "ask_before_code_generation": True,
                "ask_before_external_send": True,
            },
            "user_preferences": {
                "prefers_directness": True,
                "technical_depth": "adaptive",
            },
        },
        ensure_ascii=False,
    )
