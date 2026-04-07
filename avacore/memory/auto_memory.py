import re
from dataclasses import dataclass


@dataclass
class MemoryCandidate:
    title: str
    content: str
    tags: str = "auto"
    importance: int = 4


class AutoMemoryExtractor:
    def extract(self, text: str) -> list[MemoryCandidate]:
        text = text.strip()
        if not text:
            return []

        candidates: list[MemoryCandidate] = []

        rules: list[tuple[re.Pattern[str], str]] = [
            (
                re.compile(r"^meine?\s+(.+?)\s+ist\s+(.+)$", re.IGNORECASE),
                "Persönliche Angabe",
            ),
            (
                re.compile(r"^ich\s+nutze\s+(.+)$", re.IGNORECASE),
                "Nutzt",
            ),
            (
                re.compile(r"^ich\s+verwende\s+(.+)$", re.IGNORECASE),
                "Verwendet",
            ),
            (
                re.compile(r"^ich\s+bevorzuge\s+(.+)$", re.IGNORECASE),
                "Präferenz",
            ),
            (
                re.compile(r"^wir\s+nehmen\s+(.+)$", re.IGNORECASE),
                "Entscheidung",
            ),
            (
                re.compile(r"^(.+?)\s+soll\s+(.+)\s+sein$", re.IGNORECASE),
                "Festlegung",
            ),
        ]

        for pattern, label in rules:
            match = pattern.match(text)
            if not match:
                continue

            content = text.rstrip(". ")
            if len(content) < 8:
                continue

            candidates.append(
                MemoryCandidate(
                    title=label,
                    content=content,
                    tags="auto,conversation",
                    importance=4,
                )
            )
            break

        return candidates
