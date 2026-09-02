import pytest

from avacore.config.settings import Settings, split_csv


def test_split_csv_strips_and_ignores_empty_values() -> None:
    assert split_csv("one, two, ,three") == ["one", "two", "three"]


def test_unknown_profile_has_actionable_error(monkeypatch) -> None:
    monkeypatch.setenv("AVACORE_PROFILE", "does-not-exist")

    with pytest.raises(ValueError, match="Unknown AVACORE_PROFILE") as error:
        Settings()

    assert "low_vram" in str(error.value)
    assert "mid_vram" in str(error.value)


def test_autonomous_research_settings_are_bounded(monkeypatch) -> None:
    monkeypatch.setenv("AVACORE_RESEARCH_MAX_RUNS_PER_DAY", "999")
    monkeypatch.setenv("AVACORE_RESEARCH_MIN_SCORE", "-2")
    monkeypatch.setenv("AVACORE_RESEARCH_CURIOSITY_WEIGHT", "5")

    settings = Settings()

    assert settings.research_max_runs_per_day == 24
    assert settings.research_min_score == 0.0
    assert settings.research_curiosity_weight == 0.5


def test_invalid_auto_research_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("AVACORE_AUTO_RESEARCH", "unlimited")

    with pytest.raises(ValueError, match="off, ask, bounded"):
        Settings()


def test_person_recognition_follows_existing_identity_switch_by_default(monkeypatch) -> None:
    monkeypatch.setenv("AVACORE_IDENTITY_ENABLED", "1")
    monkeypatch.delenv("AVA_PERSON_RECOGNITION_ENABLED", raising=False)
    configured = Settings()
    assert configured.person_recognition_enabled is True
    assert configured.known_persons["roger"] == "Roger"


def test_person_recognition_can_still_be_explicitly_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AVACORE_IDENTITY_ENABLED", "1")
    monkeypatch.setenv("AVA_PERSON_RECOGNITION_ENABLED", "0")
    assert Settings().person_recognition_enabled is False


def test_vision_preempts_reasoning_by_default_and_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("AVACORE_VISION_PREEMPT_REASONING", raising=False)
    assert Settings().vision_preempt_reasoning is True
    monkeypatch.setenv("AVACORE_VISION_PREEMPT_REASONING", "0")
    assert Settings().vision_preempt_reasoning is False


def test_model_router_defaults_are_enabled_and_bounded(monkeypatch) -> None:
    monkeypatch.delenv("AVACORE_MODEL_ROUTER_ENABLED", raising=False)
    monkeypatch.setenv("AVACORE_MODEL_ROUTER_HISTORY_LIMIT", "9999")
    configured = Settings()
    assert configured.model_router_enabled is True
    assert configured.model_router_history_limit == 500


def test_resource_coordinator_defaults_and_bounds(monkeypatch) -> None:
    monkeypatch.delenv("AVACORE_RESOURCE_COORDINATOR_ENABLED", raising=False)
    monkeypatch.setenv("AVACORE_RESOURCE_HISTORY_LIMIT", "9999")
    monkeypatch.setenv("AVACORE_GPU_QUERY_TIMEOUT_SECONDS", "0")
    configured = Settings()
    assert configured.resource_coordinator_enabled is True
    assert configured.resource_history_limit == 500
    assert configured.gpu_query_timeout_seconds == .1
