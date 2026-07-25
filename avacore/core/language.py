from typing import Literal


ReplyLanguage = Literal["de", "en"]


def response_language_rule(language: ReplyLanguage) -> str:
    if language == "en":
        return "- Answer in English unless the user explicitly requests another language."
    return (
        "- Antworte standardmässig auf Deutsch, sofern der Nutzer nicht ausdrücklich "
        "eine andere Sprache verlangt."
    )
