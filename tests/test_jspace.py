import json

import pytest

from avacore.core.jspace import (
    JSpaceState,
    clamp,
    infer_jspace_tags,
    stable_item_id,
    update_jspace_from_user_message,
)


def test_clamp_keeps_values_in_unit_interval() -> None:
    assert clamp(-1) == 0.0
    assert clamp(0.4) == 0.4
    assert clamp(2) == 1.0


def test_stable_item_id_normalizes_whitespace_and_case() -> None:
    first = stable_item_id("conversation", "user", "  Hello   Ava ")
    second = stable_item_id("conversation", "user", "hello ava")

    assert first == second


def test_tag_inference_detects_known_topics() -> None:
    tags = infer_jspace_tags("Debug AvaCore Python and camera support")

    assert {"avacore", "python", "programming", "debugging", "vision"} <= set(tags)


def test_new_state_is_seeded_with_protected_context(tmp_path) -> None:
    state = JSpaceState.load(tmp_path / "missing.json")

    assert len(state.items) == 2
    assert {item.source for item in state.items.values()} == {
        "identity",
        "operating_rule",
    }


def test_injecting_same_item_reinforces_instead_of_duplicating() -> None:
    state = JSpaceState()
    first = state.inject("test", "fact", "A useful fact", activation_boost=0.2)
    second = state.inject("test", "fact", "A useful fact", activation_boost=0.3)

    assert first.id == second.id
    assert len(state.items) == 1
    assert second.activation == pytest.approx(0.5)


def test_empty_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="content is empty"):
        JSpaceState().inject("test", "fact", "  ")


def test_state_round_trip_preserves_items(tmp_path) -> None:
    path = tmp_path / "state" / "jspace.json"
    state = JSpaceState(focus_mode="wide")
    item = state.inject("test", "goal", "Ship reliable tests", tags=["testing"])
    state.save(path)

    restored = JSpaceState.load(path, focus_mode="wide")

    assert restored.focus_mode == "wide"
    assert restored.items[item.id].content == "Ship reliable tests"
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_corrupt_state_falls_back_to_seed_items(tmp_path) -> None:
    path = tmp_path / "jspace.json"
    path.write_text("{not valid json", encoding="utf-8")

    state = JSpaceState.load(path)

    assert len(state.items) == 2


def test_user_update_persists_message_and_returns_prompt(tmp_path) -> None:
    path = tmp_path / "jspace.json"

    prompt = update_jspace_from_user_message(path, "Debug the AvaCore Python tests")
    state = JSpaceState.load(path)

    assert path.exists()
    assert "Current Dynamic Conscious Workspace" in prompt
    assert any(
        item.kind == "user_message" and "AvaCore Python tests" in item.content
        for item in state.items.values()
    )
