from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from PIL import Image

import cv2
import os


def build_rtsp_url(
    user: str,
    password: str,
    ip: str,
    rtsp_path: str = "/play1.sdp",
) -> str:
    user_enc = quote(user or "")
    password_enc = quote(password or "")

    path = rtsp_path if rtsp_path.startswith("/") else f"/{rtsp_path}"

    if password:
        return f"rtsp://{user_enc}:{password_enc}@{ip}:554{path}"

    return f"rtsp://{user_enc}:@{ip}:554{path}"


def capture_rtsp_snapshot(
    url: str,
    output_dir: Path,
    camera_name: str = "camera",
) -> Path:
    import os
    import time

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    output_dir.mkdir(parents=True, exist_ok=True)

    last_error = None

    for attempt in range(1, 4):
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        try:
            if not cap.isOpened():
                last_error = RuntimeError(f"camera stream could not be opened, attempt {attempt}")
                time.sleep(1.0)
                continue

            frame = None

            # Warm-up: old RTSP cameras often need a few frames.
            for _ in range(20):
                ret, candidate = cap.read()
                if ret and candidate is not None:
                    frame = candidate
                    break
                time.sleep(0.1)

            if frame is None:
                last_error = RuntimeError(f"no frame received from camera stream, attempt {attempt}")
                time.sleep(1.0)
                continue

            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            out_path = output_dir / f"{camera_name}-{timestamp}.jpg"

            ok = cv2.imwrite(str(out_path), frame)
            if not ok:
                raise RuntimeError(f"failed to write snapshot: {out_path}")

            return out_path

        finally:
            cap.release()

    raise RuntimeError(str(last_error or "camera snapshot failed"))


def crop_camera_overlay(image_path: Path) -> Path:
    image_path = Path(image_path)

    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # D-Link OSD/time overlay is usually in the upper area.
    crop_top = min(100, int(height * 0.28))

    cropped = img.crop((0, crop_top, width, height))

    out_path = image_path.with_name(image_path.stem + "-scene.jpg")
    cropped.save(out_path, quality=95)

    return out_path