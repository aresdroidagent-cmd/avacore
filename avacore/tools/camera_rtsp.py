from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote

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
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        raise RuntimeError("camera stream could not be opened")

    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("no frame received from camera stream")

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_path = output_dir / f"{camera_name}-{timestamp}.jpg"

        ok = cv2.imwrite(str(out_path), frame)
        if not ok:
            raise RuntimeError(f"failed to write snapshot: {out_path}")

        return out_path
    finally:
        cap.release()