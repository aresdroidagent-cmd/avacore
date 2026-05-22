from __future__ import annotations

from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel


_model: WhisperModel | None = None


def get_whisper_model(
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
) -> WhisperModel:
    global _model

    if _model is None:
        _model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

    return _model


def transcribe_audio_file(
    audio_path: Path,
    model_name: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = "de",
) -> dict:
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    model = get_whisper_model(
        model_name=model_name,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=language or None,
        vad_filter=True,
    )

    text_parts: list[str] = []

    for segment in segments:
        text = segment.text.strip()
        if text:
            text_parts.append(text)

    text = " ".join(text_parts).strip()

    return {
        "ok": True,
        "text": text,
        "language": getattr(info, "language", language),
        "duration": getattr(info, "duration", None),
    }