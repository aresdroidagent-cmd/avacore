from pathlib import Path

from avacore.config.settings import settings
from avacore.memory.sqlite_store import SQLiteStore
from avacore.rag.pdf_ingest import ingest_pdf, extract_pdf_images
from avacore.rag.image_ingest import ingest_image, SUPPORTED_IMAGE_EXTS
from avacore.rag.embedder import Embedder
from avacore.rag.retriever import Retriever
from avacore.vision.describe import describe_image_with_smolvlm, detect_image_mode


def build_image_chunk_text(
    title: str,
    caption: str,
    ocr_text: str,
    page_number: int | None = None,
    page_text: str = "",
    image_mode: str = "",
) -> str:
    parts = [f"Bildtitel: {title}"]

    if image_mode:
        parts.append(f"Bildmodus: {image_mode}")

    if caption:
        parts.append(f"Visuelle Beschreibung: {caption}")

    if page_number:
        parts.append(f"PDF-Seite: {page_number}")

    if page_text:
        parts.append(f"Seitenkontext: {' '.join(page_text.split())[:1200]}")

    if ocr_text:
        parts.append(f"OCR-Text: {ocr_text}")

    return "\n".join(parts).strip()


def build_page_context_map(chunks: list[dict]) -> dict[int, str]:
    page_map: dict[int, list[str]] = {}
    for chunk in chunks:
        page_number = chunk.get("page_number")
        content = (chunk.get("content") or "").strip()
        if not page_number or not content:
            continue
        page_map.setdefault(page_number, []).append(content)

    merged: dict[int, str] = {}
    for page_number, texts in page_map.items():
        merged[page_number] = " ".join(texts)[:2000]
    return merged


def main() -> None:
    store = SQLiteStore(settings.db_path)
    store.init_db()

    settings.knowledge_inbox_pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_inbox_images_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_pdf_images_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_index_dir.mkdir(parents=True, exist_ok=True)
    settings.knowledge_image_text_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(settings.knowledge_inbox_pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {settings.knowledge_inbox_pdf_dir}")

    for pdf_path in pdf_files:
        title, chunks, checksum = ingest_pdf(
            pdf_path,
            chunk_size=settings.rag_chunk_size,
            overlap=settings.rag_chunk_overlap,
        )

        document_id = store.upsert_knowledge_document(
            source_path=str(pdf_path.resolve()),
            doc_type="pdf",
            title=title,
            checksum=checksum,
            status="indexed",
        )

        combined_chunks = list(chunks)
        page_context_map = build_page_context_map(chunks)

        pdf_image_dir = settings.knowledge_pdf_images_dir / pdf_path.stem
        extracted_images = extract_pdf_images(pdf_path, pdf_image_dir)

        for item in extracted_images:
            image_path = Path(item["image_path"])
            page_number = item["page_number"]
            page_text = page_context_map.get(page_number, "")

            image_meta = ingest_image(image_path, ocr_enabled=settings.ocr_enabled)

            vision_caption = ""
            image_mode = ""

            if settings.vision_on_pdf_images:
                try:
                    image_mode = detect_image_mode(
                        image_path,
                        ocr_text=image_meta["ocr_text"],
                        page_text=page_text,
                    )
                    vision_caption = describe_image_with_smolvlm(
                        image_path,
                        ocr_text=image_meta["ocr_text"],
                        page_text=page_text,
                        mode=image_mode,
                    )
                except Exception as exc:
                    print(f"SmolVLM failed for {image_path.name}: {exc}")

            final_caption = vision_caption.strip() or image_meta["caption"]

            store.upsert_knowledge_image(
                document_id=document_id,
                source_path=str(pdf_path.resolve()),
                image_path=str(image_path.resolve()),
                page_number=page_number,
                caption=final_caption,
                ocr_text=image_meta["ocr_text"],
                checksum=image_meta["checksum"],
            )

            chunk_text = build_image_chunk_text(
                title=image_meta["title"],
                caption=final_caption,
                ocr_text=image_meta["ocr_text"],
                page_number=page_number,
                page_text=page_text,
                image_mode=image_mode,
            )

            if len(chunk_text.strip()) >= settings.ocr_min_text_length:
                combined_chunks.append(
                    {
                        "content": chunk_text,
                        "page_number": page_number,
                    }
                )

        store.replace_knowledge_chunks(document_id, combined_chunks)
        print(
            f"Indexed PDF: {pdf_path.name} -> {len(chunks)} text chunks, "
            f"{len(extracted_images)} extracted images, {len(combined_chunks)} total chunks"
        )

    image_files = sorted(
        [
            p
            for p in settings.knowledge_inbox_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
    )
    print(f"Found {len(image_files)} image files in {settings.knowledge_inbox_images_dir}")

    for image_path in image_files:
        image_meta = ingest_image(image_path, ocr_enabled=settings.ocr_enabled)

        vision_caption = ""
        image_mode = ""

        if settings.vision_on_loose_images:
            try:
                image_mode = detect_image_mode(
                    image_path,
                    ocr_text=image_meta["ocr_text"],
                    page_text="",
                )
                vision_caption = describe_image_with_smolvlm(
                    image_path,
                    ocr_text=image_meta["ocr_text"],
                    page_text="",
                    mode=image_mode,
                )
            except Exception as exc:
                print(f"SmolVLM failed for {image_path.name}: {exc}")

        final_caption = vision_caption.strip() or image_meta["caption"]

        document_id = store.upsert_knowledge_document(
            source_path=str(image_path.resolve()),
            doc_type="image",
            title=image_meta["title"],
            checksum=image_meta["checksum"],
            status="indexed",
        )

        store.upsert_knowledge_image(
            document_id=document_id,
            source_path=str(image_path.resolve()),
            image_path=str(image_path.resolve()),
            page_number=None,
            caption=final_caption,
            ocr_text=image_meta["ocr_text"],
            checksum=image_meta["checksum"],
        )

        chunk_text = build_image_chunk_text(
            title=image_meta["title"],
            caption=final_caption,
            ocr_text=image_meta["ocr_text"],
            page_number=None,
            page_text="",
            image_mode=image_mode,
        )

        chunks = [{"content": chunk_text, "page_number": None}]
        store.replace_knowledge_chunks(document_id, chunks)
        print(f"Indexed image: {image_path.name} -> {len(chunks)} chunk(s)")

    embedder = Embedder(settings.embedding_model)
    retriever = Retriever(
        store=store,
        embedder=embedder,
        index_dir=settings.knowledge_index_dir,
    )
    count = retriever.rebuild()
    print(f"FAISS rebuilt with {count} chunks")


if __name__ == "__main__":
    main()