from pathlib import Path
import json

import faiss

from avacore.memory.sqlite_store import SQLiteStore
from avacore.rag.embedder import Embedder


class Retriever:
    def __init__(
        self,
        store: SQLiteStore,
        embedder: Embedder,
        index_dir: Path,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "knowledge.faiss"
        self.meta_path = self.index_dir / "knowledge_meta.json"

    def rebuild(self) -> int:
        chunks = self.store.list_knowledge_chunks()
        if not chunks:
            if self.index_path.exists():
                self.index_path.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()
            return 0

        texts = [chunk["content"] for chunk in chunks]
        vectors = self.embedder.encode(texts)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(self.index_path))

        meta = [
            {
                "chunk_id": chunk["id"],
                "document_id": chunk["document_id"],
                "title": chunk["title"],
                "source_path": chunk["source_path"],
                "page_number": chunk["page_number"],
            }
            for chunk in chunks
        ]
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(chunks)

    def search(self, query: str, top_k: int = 6) -> list[dict]:
        if not self.index_path.exists() or not self.meta_path.exists():
            return []

        index = faiss.read_index(str(self.index_path))
        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

        qvec = self.embedder.encode([query])
        scores, ids = index.search(qvec, top_k)

        results: list[dict] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            item = meta[idx]
            chunk = self.store.get_knowledge_chunk_by_id(int(item["chunk_id"]))
            if not chunk:
                continue
            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk["id"],
                    "document_id": chunk["document_id"],
                    "title": chunk["title"],
                    "source_path": chunk["source_path"],
                    "page_number": chunk["page_number"],
                    "content": chunk["content"],
                }
            )
        return results
