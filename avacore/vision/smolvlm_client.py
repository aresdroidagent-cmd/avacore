from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


class SmolVLMClient:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 64,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)

        self.model.eval()

    def _build_conversation(self, image_path: Path, prompt: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        moved: dict[str, Any] = {}

        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved[key] = value
                continue

            if torch.is_floating_point(value):
                moved[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                moved[key] = value.to(device=self.device)

        return moved

    def describe_image(
        self,
        image_path: Path,
        prompt: str,
    ) -> str:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"image not found: {image_path}")

        # Früher Fehler, falls Datei kein gültiges Bild ist
        Image.open(image_path).convert("RGB")

        conversation = self._build_conversation(image_path, prompt)

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        inputs = self._move_inputs(inputs)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_length:]

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        return text