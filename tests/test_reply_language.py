from pathlib import Path
from types import SimpleNamespace


def test_system_prompt_uses_requested_response_language() -> None:
    from avacore.core.language import response_language_rule

    english_prompt = response_language_rule("en")
    german_prompt = response_language_rule("de")

    assert "Answer in English unless" in english_prompt
    assert "Antworte standardmässig auf Deutsch, sofern" in german_prompt


def test_camera_vlm_prompt_remains_english() -> None:
    describe_source = (
        Path(__file__).parents[1] / "avacore" / "vision" / "describe.py"
    ).read_text(encoding="utf-8")

    assert "You are looking at a live indoor camera image." in describe_source
    assert "Answer in one short factual sentence." in describe_source


def test_telegram_language_switch_defaults_to_german() -> None:
    from avacore.channels.telegram.bot import telegram_reply_language

    assert telegram_reply_language(SimpleNamespace(chat_data={})) == "de"
    assert (
        telegram_reply_language(SimpleNamespace(chat_data={"reply_language": "en"}))
        == "en"
    )


def test_telegram_help_mentions_language_commands() -> None:
    from avacore.channels.telegram.bot import command_help_text

    help_text = command_help_text()
    assert "/de - Antworten auf Deutsch" in help_text
    assert "/en - replies in English" in help_text
