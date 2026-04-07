from pathlib import Path
import hashlib
import fitz


from avacore.rag.chunker import chunk_text


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def ingest_pdf(path: Path, chunk_size: int, overlap: int) -> tuple[str, list[dict], str]:
    doc = fitz.open(path)
    try:
        all_chunks: list[dict] = []
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text")
            page_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            for chunk in page_chunks:
                all_chunks.append(
                    {
                        "content": chunk,
                        "page_number": page_index + 1,
                    }
                )
    finally:
        doc.close()

    checksum = file_checksum(path)
    title = path.stem
    return title, all_chunks, checksum


def extract_pdf_images(pdf_path: Path, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(pdf_path)
    extracted: list[dict] = []
    try:
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            images = page.get_images(full=True)

            for img_idx, img in enumerate(images):
                xref = img[0]
                base = doc.extract_image(xref)
                image_bytes = base["image"]
                ext = base.get("ext", "png")

                out_path = out_dir / f"page_{page_index+1:03d}_img_{img_idx+1:03d}.{ext}"
                out_path.write_bytes(image_bytes)

                extracted.append(
                    {
                        "image_path": out_path,
                        "page_number": page_index + 1,
                    }
                )
    finally:
        doc.close()

    return extracted
