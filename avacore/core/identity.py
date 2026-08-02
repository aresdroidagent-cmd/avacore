from __future__ import annotations

import re

from avacore.core.language import ReplyLanguage


def _normalize_question(text: str) -> str:
    normalized = " ".join((text or "").strip().casefold().split())
    normalized = normalized.replace("heisst", "heißt")
    return re.sub(r"[\s.!?…,:;]+$", "", normalized)


_AGENT_IDENTITY_DE = {
    "wer bist du",
    "was bist du",
    "wie heißt du",
    "was ist dein name",
    "wie ist dein name",
    "bist du ava",
    "du bist ava, oder",
}
_AGENT_IDENTITY_EN = {
    "who are you",
    "what are you",
    "what is your name",
    "what's your name",
    "are you ava",
    "you are ava, right",
}
_MODEL_DE = {
    "welches modell bist du",
    "welches modell verwendest du",
    "was ist dein modell",
    "was ist dein hintergrundmodell",
    "welches sprachmodell verwendest du",
    "auf welchem modell basierst du",
}
_MODEL_EN = {
    "which model are you",
    "which model do you use",
    "what model do you use",
    "what is your underlying model",
    "what is your background model",
    "which language model do you use",
}
_GEMMA_DE = {"bist du gemma", "bist du gemma4"}
_GEMMA_EN = {"are you gemma", "are you gemma4"}
_OLLAMA_DE = {"bist du ollama"}
_OLLAMA_EN = {"are you ollama"}
_LANGUAGE_MODEL_DE = {"bist du ein sprachmodell"}
_LANGUAGE_MODEL_EN = {"are you a language model"}


def answer_identity_question(
    text: str,
    *,
    language: ReplyLanguage,
    assistant_name: str,
    system_name: str,
    model_name: str,
) -> str | None:
    """Return a deterministic identity answer for a narrowly recognized question."""
    question = _normalize_question(text)

    if question in _GEMMA_DE or question in _GEMMA_EN:
        if language == "en":
            return (
                f"No. I am {assistant_name}. {model_name} is my current underlying "
                "language model."
            )
        return (
            f"Nein. Ich bin {assistant_name}. {model_name} ist das aktuell verwendete "
            "Hintergrundmodell."
        )

    if question in _OLLAMA_DE or question in _OLLAMA_EN:
        if language == "en":
            return (
                f"No. I am {assistant_name}. Ollama is only the local model runtime; "
                f"my current underlying language model is {model_name}."
            )
        return (
            f"Nein. Ich bin {assistant_name}. Ollama ist nur die lokale Modell-Laufzeit; "
            f"mein aktuelles Hintergrundmodell ist {model_name}."
        )

    if question in _LANGUAGE_MODEL_DE or question in _LANGUAGE_MODEL_EN:
        if language == "en":
            return (
                f"I am {assistant_name}, a local assistant in the {system_name} system. "
                f"For language processing, I use {model_name} as my underlying model."
            )
        return (
            f"Ich bin {assistant_name}, ein lokaler Assistent im {system_name}-System. "
            f"Für Sprachverarbeitung verwende ich {model_name} als Hintergrundmodell."
        )

    if question in _MODEL_DE or question in _MODEL_EN:
        if language == "en":
            return (
                f"I am {assistant_name}. My current underlying language model is {model_name}."
            )
        return (
            f"Ich bin {assistant_name}. Als Hintergrundmodell verwende ich derzeit "
            f"{model_name}."
        )

    if question in _AGENT_IDENTITY_DE or question in _AGENT_IDENTITY_EN:
        if language == "en":
            return (
                f"I am {assistant_name}, your local assistant in the {system_name} system."
            )
        return (
            f"Ich bin {assistant_name}, dein lokaler Assistent im {system_name}-System."
        )

    return None
