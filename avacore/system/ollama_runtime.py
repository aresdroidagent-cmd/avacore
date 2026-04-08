from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 11434


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