from __future__ import annotations

import os
import requests

MYSTROM_IP = os.environ.get("AVACORE_MYSTROM_IP", "192.168.8.186").strip()
MYSTROM_TIMEOUT = float(os.environ.get("AVACORE_MYSTROM_TIMEOUT", "2"))


def _base_url() -> str:
    if not MYSTROM_IP:
        raise RuntimeError("AVACORE_MYSTROM_IP is not configured")
    return f"http://{MYSTROM_IP}"


def light_on() -> str:
    response = requests.get(f"{_base_url()}/relay?state=1", timeout=MYSTROM_TIMEOUT)
    response.raise_for_status()
    return "Licht eingeschaltet."


def light_off() -> str:
    response = requests.get(f"{_base_url()}/relay?state=0", timeout=MYSTROM_TIMEOUT)
    response.raise_for_status()
    return "Licht ausgeschaltet."


def light_status() -> dict:
    response = requests.get(f"{_base_url()}/report", timeout=MYSTROM_TIMEOUT)
    response.raise_for_status()
    return response.json()