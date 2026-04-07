from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText


class SmolVLMClient:
    def __init__(self, model_name: str, max_new_tokens: int = 64) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=self.model_dtype,
        ).to(self.device)

        self.model.eval()

    def _clean_output(self, text: str) -> str:
        text = text.strip()

        if "Assistant:" in text:
            text = text.split("Assistant:", 1)[-1].strip()

        if "User:" in text:
            text = text.split("User:", 1)[0].strip()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = " ".join(lines)

        return text[:1200].strip()

    def describe_image(self, image_path: Path, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "path": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        model_device = next(self.model.parameters()).device

        moved_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved_inputs[key] = value
                continue

            if torch.is_tensor(value) and value.is_floating_point():
                moved_inputs[key] = value.to(device=model_device, dtype=self.model_dtype)
            else:
                moved_inputs[key] = value.to(device=model_device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **moved_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )

        output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return self._clean_output(output)
