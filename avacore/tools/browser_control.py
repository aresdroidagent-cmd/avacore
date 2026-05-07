from __future__ import annotations

from datetime import datetime
from pathlib import Path
from urllib.parse import quote_plus

from playwright.sync_api import BrowserContext, Page, Playwright, sync_playwright


class BrowserController:
    """
    Read-only browser controller for AvaCore.

    Allowed v1 actions:
    - open URL
    - web search
    - read page text
    - take screenshot
    - close browser

    No clicking, typing or form submission in this first version.
    """

    def __init__(
        self,
        user_data_dir: Path,
        screenshot_dir: Path,
        headless: bool = True,
        timeout_ms: int = 30000,
        default_search: str = "https://duckduckgo.com/?q=",
    ) -> None:
        self.user_data_dir = Path(user_data_dir)
        self.screenshot_dir = Path(screenshot_dir)
        self.headless = headless
        self.timeout_ms = timeout_ms
        self.default_search = default_search

        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    def _ensure_started(self) -> Page:
        if self._page is not None:
            return self._page

        self.user_data_dir.mkdir(parents=True, exist_ok=True)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

        self._playwright = sync_playwright().start()

        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.user_data_dir),
            headless=self.headless,
            viewport={"width": 1400, "height": 950},
            args=[
                "--disable-dev-shm-usage",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        )

        self._context.set_default_timeout(self.timeout_ms)

        if self._context.pages:
            self._page = self._context.pages[0]
        else:
            self._page = self._context.new_page()

        return self._page

    def open_url(self, url: str) -> dict:
        if not url.startswith(("http://", "https://")):
            raise ValueError("Only http:// and https:// URLs are allowed.")

        page = self._ensure_started()
        page.goto(url, wait_until="domcontentloaded", timeout=self.timeout_ms)

        return self.status()

    def search(self, query: str) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("Search query is empty.")

        url = f"{self.default_search}{quote_plus(query)}"
        return self.open_url(url)

    def get_text(self, max_chars: int = 8000) -> dict:
        page = self._ensure_started()

        title = page.title()
        url = page.url

        try:
            text = page.locator("body").inner_text(timeout=self.timeout_ms)
        except Exception:
            text = page.content()

        text = text.strip()
        shortened = text[:max_chars]

        return {
            "ok": True,
            "title": title,
            "url": url,
            "text": shortened,
            "truncated": len(text) > len(shortened),
            "chars": len(text),
        }

    def screenshot(self, full_page: bool = True) -> dict:
        page = self._ensure_started()

        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        out_path = self.screenshot_dir / f"browser-{timestamp}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        page.screenshot(path=str(out_path), full_page=full_page)

        return {
            "ok": True,
            "path": str(out_path),
            "url": page.url,
            "title": page.title(),
        }

    def status(self) -> dict:
        page = self._ensure_started()
        return {
            "ok": True,
            "url": page.url,
            "title": page.title(),
            "headless": self.headless,
            "user_data_dir": str(self.user_data_dir),
        }

    def close(self) -> dict:
        if self._context is not None:
            self._context.close()
            self._context = None

        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

        self._page = None

        return {"ok": True, "closed": True}