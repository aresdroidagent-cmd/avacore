from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import faiss
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


_ALLOWED_LABELS = {"roger", "unknown", "empty"}

_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None
_clip_model_name: str | None = None
_clip_device: str | None = None


@dataclass
class IdentityDecision:
    identity: str
    confidence: float
    roger_votes: int
    top_label: str
    margin: float
    reason: str
    neighbors: list[dict[str, Any]]
    face_path: str | None = None


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_identity_dirs(identity_dir: Path | str) -> None:
    identity_dir = Path(identity_dir)

    for sub in [
        "raw/roger",
        "raw/unknown",
        "raw/empty",
        "faces/roger",
        "faces/unknown",
        "faces/empty",
        "index",
    ]:
        (identity_dir / sub).mkdir(parents=True, exist_ok=True)


def validate_label(label: str) -> str:
    normalized = (label or "").strip().lower()

    aliases = {
        "roger": "roger",
        "me": "roger",
        "ich": "roger",
        "unknown": "unknown",
        "unbekannt": "unknown",
        "not_roger": "unknown",
        "other": "unknown",
        "empty": "empty",
        "leer": "empty",
        "nobody": "empty",
    }

    normalized = aliases.get(normalized, normalized)

    if normalized not in _ALLOWED_LABELS:
        raise ValueError("label must be one of: roger, unknown, empty")

    return normalized


def copy_capture_to_identity_dataset(
    image_path: Path | str,
    identity_dir: Path | str,
    label: str,
) -> Path:
    identity_dir = Path(identity_dir)
    image_path = Path(image_path)
    label = validate_label(label)

    ensure_identity_dirs(identity_dir)

    timestamp = utc_timestamp()
    dst = identity_dir / "raw" / label / f"{timestamp}-{label}.jpg"
    shutil.copy2(image_path, dst)

    return dst


def _load_haar_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade_path))

    if detector.empty():
        raise RuntimeError(f"failed to load Haar cascade: {cascade_path}")

    return detector


def detect_largest_face(image_path: Path | str) -> tuple[int, int, int, int] | None:
    image_path = Path(image_path)

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"could not read image: {image_path}")

    original_height, original_width = img.shape[:2]

    # Upscale smaller images / distant faces for better Haar detection.
    target_width = 1600
    scale = 1.0

    if original_width < target_width:
        scale = target_width / float(original_width)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        img_work = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    else:
        img_work = img

    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY)

    # Improve contrast for indoor / low light scenes.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    cascade_dir = Path(cv2.data.haarcascades)

    cascades = [
        cascade_dir / "haarcascade_frontalface_default.xml",
        cascade_dir / "haarcascade_frontalface_alt2.xml",
        cascade_dir / "haarcascade_profileface.xml",
    ]

    all_faces: list[tuple[int, int, int, int]] = []

    for cascade_path in cascades:
        detector = cv2.CascadeClassifier(str(cascade_path))

        if detector.empty():
            continue

        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=4,
            minSize=(28, 28),
        )

        for face in faces:
            x, y, w, h = [int(v) for v in face]
            all_faces.append((x, y, w, h))

    if not all_faces:
        return None

    # Choose largest detected face.
    x, y, w, h = sorted(
        all_faces,
        key=lambda box: int(box[2]) * int(box[3]),
        reverse=True,
    )[0]

    # Convert back to original image coordinates.
    if scale != 1.0:
        x = int(x / scale)
        y = int(y / scale)
        w = int(w / scale)
        h = int(h / scale)

    x = max(0, min(x, original_width - 1))
    y = max(0, min(y, original_height - 1))
    w = max(1, min(w, original_width - x))
    h = max(1, min(h, original_height - y))

    return x, y, w, h


def crop_face(
    image_path: Path | str,
    output_dir: Path | str,
    label: str,
    padding_ratio: float = 0.35,
) -> Path | None:
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    label = validate_label(label)

    box = detect_largest_face(image_path)
    if box is None:
        return None

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    x, y, w, h = box

    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(width, x + w + pad_x)
    bottom = min(height, y + h + pad_y)

    face = img.crop((left, top, right, bottom))

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{image_path.stem}-face.jpg"
    face.save(out_path, quality=95)

    return out_path


def get_clip(model_name: str, device: str) -> tuple[CLIPModel, CLIPProcessor, str]:
    global _clip_model, _clip_processor, _clip_model_name, _clip_device

    device = device.strip() or "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if (
        _clip_model is not None
        and _clip_processor is not None
        and _clip_model_name == model_name
        and _clip_device == device
    ):
        return _clip_model, _clip_processor, device

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    _clip_model = model
    _clip_processor = processor
    _clip_model_name = model_name
    _clip_device = device

    return model, processor, device


def embed_image(
    image_path: Path | str,
    model_name: str,
    device: str,
) -> np.ndarray:
    model, processor, effective_device = get_clip(model_name, device)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(effective_device) for key, value in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)

    vector = features.detach().cpu().numpy().astype("float32")[0]
    norm = np.linalg.norm(vector)

    if norm <= 0:
        raise RuntimeError("zero vector embedding")

    vector = vector / norm
    return vector.astype("float32")


def build_identity_index(
    identity_dir: Path | str,
    model_name: str,
    device: str,
) -> dict[str, Any]:
    identity_dir = Path(identity_dir)
    ensure_identity_dirs(identity_dir)

    vectors: list[np.ndarray] = []
    meta: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for label in ["roger", "unknown"]:
        raw_dir = identity_dir / "raw" / label
        faces_dir = identity_dir / "faces" / label
        faces_dir.mkdir(parents=True, exist_ok=True)

        for image_path in sorted(raw_dir.glob("*.jpg")):
            face_path = crop_face(
                image_path=image_path,
                output_dir=faces_dir,
                label=label,
            )

            if face_path is None:
                skipped.append(
                    {
                        "image_path": str(image_path),
                        "reason": "no face detected",
                    }
                )
                continue

            vector = embed_image(
                image_path=face_path,
                model_name=model_name,
                device=device,
            )

            vectors.append(vector)
            meta.append(
                {
                    "label": label,
                    "raw_path": str(image_path),
                    "face_path": str(face_path),
                }
            )

    if not vectors:
        raise RuntimeError("no face embeddings created; capture images with visible faces first")

    matrix = np.vstack(vectors).astype("float32")
    faiss.normalize_L2(matrix)

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    index_dir = identity_dir / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_dir / "face_index.faiss"))

    payload = {
        "model_name": model_name,
        "embedding_dim": int(matrix.shape[1]),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "items": meta,
        "skipped": skipped,
    }

    (index_dir / "face_meta.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    counts = {
        "roger": sum(1 for item in meta if item["label"] == "roger"),
        "unknown": sum(1 for item in meta if item["label"] == "unknown"),
        "skipped": len(skipped),
        "total": len(meta),
    }

    return {
        "index_path": str(index_dir / "face_index.faiss"),
        "meta_path": str(index_dir / "face_meta.json"),
        "counts": counts,
    }


def load_identity_index(identity_dir: Path | str) -> tuple[faiss.Index, dict[str, Any]]:
    identity_dir = Path(identity_dir)
    index_path = identity_dir / "index" / "face_index.faiss"
    meta_path = identity_dir / "index" / "face_meta.json"

    if not index_path.exists() or not meta_path.exists():
        raise FileNotFoundError("identity index not found; run /idtrain first")

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    return index, meta


def recognize_face_image(
    image_path: Path | str,
    identity_dir: Path | str,
    model_name: str,
    device: str,
    threshold: float = 0.78,
    margin_threshold: float = 0.06,
    top_k: int = 5,
    min_roger_votes: int = 2,
) -> IdentityDecision:
    identity_dir = Path(identity_dir)
    image_path = Path(image_path)
    ensure_identity_dirs(identity_dir)

    check_dir = identity_dir / "faces" / "check"
    check_dir.mkdir(parents=True, exist_ok=True)

    face_path = crop_face(
        image_path=image_path,
        output_dir=check_dir,
        label="unknown",
    )

    if face_path is None:
        return IdentityDecision(
            identity="unknown",
            confidence=0.0,
            roger_votes=0,
            top_label="unknown",
            margin=0.0,
            reason="no face detected",
            neighbors=[],
            face_path=None,
        )

    query = embed_image(
        image_path=face_path,
        model_name=model_name,
        device=device,
    ).reshape(1, -1).astype("float32")

    faiss.normalize_L2(query)

    index, meta = load_identity_index(identity_dir)
    items = meta.get("items", [])

    if not items:
        raise RuntimeError("identity index contains no items")

    k = max(1, min(int(top_k), len(items)))
    distances, indices = index.search(query, k)

    neighbors: list[dict[str, Any]] = []

    for score, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue

        item = items[int(idx)]
        neighbors.append(
            {
                "label": item["label"],
                "score": float(score),
                "face_path": item.get("face_path"),
                "raw_path": item.get("raw_path"),
            }
        )

    if not neighbors:
        return IdentityDecision(
            identity="unknown",
            confidence=0.0,
            roger_votes=0,
            top_label="unknown",
            margin=0.0,
            reason="no nearest neighbors",
            neighbors=[],
            face_path=str(face_path),
        )

    top = neighbors[0]
    top_label = str(top["label"])
    top_score = float(top["score"])

    roger_scores = [
        float(item["score"]) for item in neighbors if item["label"] == "roger"
    ]
    non_roger_scores = [
        float(item["score"]) for item in neighbors if item["label"] != "roger"
    ]

    best_roger = max(roger_scores) if roger_scores else 0.0
    best_non_roger = max(non_roger_scores) if non_roger_scores else 0.0

    roger_votes = sum(1 for item in neighbors if item["label"] == "roger")
    margin = best_roger - best_non_roger

    if top_label != "roger":
        identity = "unknown"
        reason = "top neighbor is not Roger"
    elif best_roger < threshold:
        identity = "unknown"
        reason = f"Roger score below threshold: {best_roger:.3f} < {threshold:.3f}"
    elif roger_votes < min_roger_votes:
        identity = "unknown"
        reason = f"not enough Roger votes in top-{k}: {roger_votes} < {min_roger_votes}"
    elif non_roger_scores and margin < margin_threshold:
        identity = "unknown"
        reason = f"margin too small: {margin:.3f} < {margin_threshold:.3f}"
    else:
        identity = "roger"
        reason = "Roger accepted by threshold, votes and margin"

    return IdentityDecision(
        identity=identity,
        confidence=best_roger if identity == "roger" else top_score,
        roger_votes=roger_votes,
        top_label=top_label,
        margin=margin,
        reason=reason,
        neighbors=neighbors,
        face_path=str(face_path),
    )


def format_identity_decision(decision: IdentityDecision) -> str:
    lines = [
        f"Identität: {decision.identity}",
        f"Confidence: {decision.confidence:.3f}",
        f"Top Label: {decision.top_label}",
        f"Roger Votes: {decision.roger_votes}",
        f"Margin: {decision.margin:.3f}",
        f"Grund: {decision.reason}",
    ]

    if decision.face_path:
        lines.append(f"Face Crop: {decision.face_path}")

    if decision.neighbors:
        lines.append("")
        lines.append("Top Neighbors:")
        for i, item in enumerate(decision.neighbors, start=1):
            lines.append(
                f"{i}. {item['label']} — {float(item['score']):.3f}"
            )

    return "\n".join(lines)