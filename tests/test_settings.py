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
