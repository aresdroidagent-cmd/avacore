from __future__ import annotations

import os
import logging
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434
_logger = logging.getLogger(__name__)


def _ollama_api_base(ollama_url: str) -> str:
    url = ollama_url.rstrip("/")
    for suffix in ("/api/chat", "/api/generate"):
        if url.endswith(suffix):
            return url[:-len(suffix)]
    return url


def loaded_ollama_models(
    ollama_url: str,
    timeout: float = 2.0,
    session: requests.Session | None = None,
) -> tuple[str, ...]:
    """Return Ollama's own resident-model state without loading a model."""
    client = session or requests.Session()
    response = client.get(f"{_ollama_api_base(ollama_url)}/api/ps", timeout=timeout)
    response.raise_for_status()
    return tuple(sorted({
        str(item.get("name") or item.get("model") or "").strip()
        for item in (response.json().get("models") or [])
        if item.get("name") or item.get("model")
    }))


def unload_ollama_model(
    model_name: str,
    ollama_url: str,
    timeout: float = 10.0,
    session: requests.Session | None = None,
) -> bool:
    """Unload one resident Ollama model; return False for no-op or failure."""
    client = session or requests.Session()
    base_url = _ollama_api_base(ollama_url)
    try:
        loaded_names = set(loaded_ollama_models(
            ollama_url, timeout=timeout, session=client
        ))
        if model_name not in loaded_names:
            return False

        response = client.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=timeout,
        )
        response.raise_for_status()
        return True
    except Exception:
        _logger.warning("Could not release Ollama model before vision", exc_info=True)
        return False


def is_port_open(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_ollama_binary() -> str:
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise RuntimeError("Ollama binary not found in PATH.")
    return ollama_bin


def ensure_runtime_dirs(log_file: Optional[str]) -> None:
    if not log_file:
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)


def start_ollama_server(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    startup_timeout: float = 30.0,
    log_file: Optional[str] = None,
) -> Optional[subprocess.Popen]:
    """
    Startet 'ollama serve' nur dann, wenn noch kein Server läuft.
    Gibt Popen zurück, wenn dieser Prozess hier gestartet wurde.
    Gibt None zurück, wenn Ollama bereits läuft.
    """
    if is_port_open(host, port):
        return None

    ollama_bin = find_ollama_binary()
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"{host}:{port}"

    stdout_target = subprocess.DEVNULL
    stderr_target = subprocess.DEVNULL
    log_handle = None

    if log_file:
        ensure_runtime_dirs(log_file)
        log_handle = open(log_file, "ab")
        stdout_target = log_handle
        stderr_target = log_handle

    process = subprocess.Popen(
        [ollama_bin, "serve"],
        env=env,
        stdout=stdout_target,
        stderr=stderr_target,
        start_new_session=True,
    )

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if is_port_open(host, port):
            return process

        if process.poll() is not None:
            raise RuntimeError(
                f"Ollama exited immediately with code {process.returncode}"
            )

        time.sleep(0.4)

    raise RuntimeError("Ollama server did not become ready in time.")
