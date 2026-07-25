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
