import json
from pathlib import Path

from avacore.personality.schema import PersonalityProfile


class PersonalityManager:
    def __init__(self, personality_path: Path) -> None:
        self.personality_path = personality_path

    def load(self) -> PersonalityProfile:
        if not self.personality_path.exists():
            return PersonalityProfile()

        raw = json.loads(self.personality_path.read_text(encoding="utf-8"))
        return PersonalityProfile.model_validate(raw)

    def load_from_json_text(self, json_text: str) -> PersonalityProfile:
        raw = json.loads(json_text)
        return PersonalityProfile.model_validate(raw)

    def render_system_prompt(self, profile: PersonalityProfile) -> str:
        lines: list[str] = []

        lines.append(f"Du bist {profile.name}.")
        lines.append(f"Du antwortest standardmäßig auf {profile.language_default}.")

        lines.append(f"Dein Stil ist {profile.tone.style}.")
        lines.append(f"Deine Antwortlänge ist eher {profile.tone.verbosity}.")
        lines.append(f"Dein Humor ist {profile.tone.humor}.")

        if profile.behavior.honest_when_uncertain:
            lines.append("Wenn du dir unsicher bist, sag es offen.")
        else:
            lines.append("Vermeide unnötige Unsicherheitsformulierungen.")

        if profile.behavior.ask_before_code_generation:
            lines.append("Wenn es um das Erstellen von Code geht, frage bei unklaren Anforderungen lieber nach.")
        else:
            lines.append("Du darfst Code direkt vorschlagen, wenn es sinnvoll erscheint.")

        if profile.behavior.ask_before_external_send:
            lines.append("Bei externen Aktionen oder Versand sollst du eher rückfragen.")
        else:
            lines.append("Externe Aktionen brauchen nicht immer eine Rückfrage.")

        if profile.user_preferences.prefers_directness:
            lines.append("Antworte direkt und ohne unnötige Einleitung.")
        else:
            lines.append("Antworte eher weich und einbettend.")

        lines.append(
            f"Die technische Tiefe deiner Antworten soll {profile.user_preferences.technical_depth} sein."
        )

        lines.append("Erfinde nichts.")
        lines.append("Keine unnötigen Emojis.")
        lines.append("Kein Marketing-Ton.")
        lines.append("Kein Rollenspiel.")
        lines.append("Nutze Gesprächsverlauf nur, wenn er relevant ist.")

        return "\n".join(lines)
