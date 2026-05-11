from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


BRAIN_FILE_ORDER = ("SOUL.md", "USER.md", "OPERATING.md", "MEMORY.md")


@dataclass
class BrainContext:
    brain_dir: Path
    runtime_context: str
    files_context: str
    today_note: str
    yesterday_note: str

    def as_prompt(self, max_chars: int = 18000) -> str:
        parts = [
            "# AVA SHARED BRAIN",
            self.runtime_context,
            self.files_context,
        ]

        if self.yesterday_note:
            parts.append("# YESTERDAY'S DAILY NOTE")
            parts.append(self.yesterday_note)

        if self.today_note:
            parts.append("# TODAY'S DAILY NOTE")
            parts.append(self.today_note)

        prompt = "\n\n".join(part.strip() for part in parts if part and part.strip())
        if len(prompt) > max_chars:
            return prompt[:max_chars] + "\n\n[Shared brain context truncated]"
        return prompt


def _safe_read_text(path: Path, max_chars: int = 6000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[File truncated]"
    return text


def build_runtime_context(
    timezone: str = "Europe/Zurich",
    default_location: str = "Zurich, Switzerland",
    assistant_name: str = "Ava",
    system_name: str = "AvaCore",
    model_name: str = "unknown",
) -> str:
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)

    return "\n".join(
        [
            "# RUNTIME CONTEXT",
            f"Assistant name: {assistant_name}",
            f"System name: {system_name}",
            f"Current date: {now.strftime('%A, %Y-%m-%d')}",
            f"Current local time: {now.strftime('%H:%M:%S')}",
            f"Timezone: {timezone}",
            f"Location context: {default_location}",
            f"Backend model: {model_name}",
        ]
    )


def load_brain_files(brain_dir: Path, max_chars_per_file: int = 6000) -> str:
    brain_dir = Path(brain_dir)
    parts: list[str] = []

    for filename in BRAIN_FILE_ORDER:
        path = brain_dir / filename
        text = _safe_read_text(path, max_chars=max_chars_per_file)
        if text:
            parts.append(f"# {filename}\n{text}")

    return "\n\n".join(parts)


def daily_note_path(brain_dir: Path, day: datetime) -> Path:
    return Path(brain_dir) / "daily" / f"{day.date().isoformat()}.md"


def load_daily_note(brain_dir: Path, day: datetime, max_chars: int = 5000) -> str:
    return _safe_read_text(daily_note_path(brain_dir, day), max_chars=max_chars)


def load_brain_context(
    brain_dir: Path,
    timezone: str = "Europe/Zurich",
    default_location: str = "Zurich, Switzerland",
    assistant_name: str = "Ava",
    system_name: str = "AvaCore",
    model_name: str = "unknown",
) -> BrainContext:
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    yesterday = now - timedelta(days=1)

    brain_dir = Path(brain_dir)

    return BrainContext(
        brain_dir=brain_dir,
        runtime_context=build_runtime_context(
            timezone=timezone,
            default_location=default_location,
            assistant_name=assistant_name,
            system_name=system_name,
            model_name=model_name,
        ),
        files_context=load_brain_files(brain_dir),
        today_note=load_daily_note(brain_dir, now),
        yesterday_note=load_daily_note(brain_dir, yesterday),
    )


def append_daily_note(
    brain_dir: Path,
    text: str,
    section: str = "Notes",
    timezone: str = "Europe/Zurich",
) -> Path:
    tz = ZoneInfo(timezone)
    now = datetime.now(tz)
    path = daily_note_path(Path(brain_dir), now)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.write_text(f"# {now.date().isoformat()}\n\n", encoding="utf-8")

    entry = (
        f"\n## {section}\n"
        f"- {now.strftime('%H:%M')}: {text.strip()}\n"
    )

    with path.open("a", encoding="utf-8") as handle:
        handle.write(entry)

    return path
