from datetime import datetime
from zoneinfo import ZoneInfo

from avacore.core.brain import (
    BRAIN_FILE_ORDER,
    append_daily_note,
    daily_note_path,
    load_brain_context,
    load_brain_files,
)


def test_brain_files_are_loaded_in_defined_order(tmp_path) -> None:
    for filename in reversed(BRAIN_FILE_ORDER):
        (tmp_path / filename).write_text(f"content of {filename}", encoding="utf-8")

    context = load_brain_files(tmp_path)

    positions = [context.index(f"# {filename}") for filename in BRAIN_FILE_ORDER]
    assert positions == sorted(positions)


def test_missing_brain_files_are_ignored(tmp_path) -> None:
    (tmp_path / "SOUL.md").write_text("Ava identity", encoding="utf-8")

    context = load_brain_files(tmp_path)

    assert "Ava identity" in context
    assert "USER.md" not in context


def test_load_brain_context_includes_daily_notes(tmp_path) -> None:
    timezone = "Europe/Zurich"
    now = datetime.now(ZoneInfo(timezone))
    note = daily_note_path(tmp_path, now)
    note.parent.mkdir(parents=True)
    note.write_text("Today's test note", encoding="utf-8")

    context = load_brain_context(tmp_path, timezone=timezone, model_name="test-model")

    assert context.today_note == "Today's test note"
    assert "Backend model: test-model" in context.runtime_context
    assert "Today's test note" in context.as_prompt()


def test_append_daily_note_creates_expected_section(tmp_path) -> None:
    path = append_daily_note(
        tmp_path,
        "Test interaction",
        section="Tests",
        timezone="Europe/Zurich",
    )

    content = path.read_text(encoding="utf-8")
    assert path.parent == tmp_path / "daily"
    assert "## Tests" in content
    assert "Test interaction" in content


def test_prompt_is_truncated_to_requested_size(tmp_path) -> None:
    (tmp_path / "SOUL.md").write_text("x" * 500, encoding="utf-8")
    context = load_brain_context(tmp_path)

    prompt = context.as_prompt(max_chars=100)

    assert prompt.startswith("# AVA SHARED BRAIN")
    assert prompt.endswith("[Shared brain context truncated]")
