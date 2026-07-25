from __future__ import annotations

import asyncio
from typing import Any, Literal

import requests
from requests import Response

HttpMethod = Literal["GET", "POST", "DELETE"]


async def request(method: HttpMethod, url: str, **kwargs: Any) -> Response:
    """Run a blocking requests call without blocking Telegram's event loop."""
    return await asyncio.to_thread(requests.request, method, url, **kwargs)


async def get(url: str, **kwargs: Any) -> Response:
    return await request("GET", url, **kwargs)


async def post(url: str, **kwargs: Any) -> Response:
    return await request("POST", url, **kwargs)


async def delete(url: str, **kwargs: Any) -> Response:
    return await request("DELETE", url, **kwargs)
