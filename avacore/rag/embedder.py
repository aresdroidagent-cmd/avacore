from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")
