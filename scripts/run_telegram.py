import logging

import httpx
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegram.ext import ContextTypes
from telegram.request import HTTPXRequest

from avacore.channels.telegram.bot import build_application
from avacore.config.settings import settings
from avacore.system.ollama_runtime import start_ollama_server


logger = logging.getLogger(__name__)


def build_telegram_requests() -> tuple[HTTPXRequest, HTTPXRequest]:
    """Build independent IPv4-only request clients for API calls and polling."""

    def build_request(*, read_timeout: float) -> HTTPXRequest:
        return HTTPXRequest(
            connect_timeout=15.0,
            read_timeout=read_timeout,
            write_timeout=30.0,
            pool_timeout=10.0,
            httpx_kwargs={
                "transport": httpx.AsyncHTTPTransport(
                    local_address="0.0.0.0",
                    retries=2,
                ),
            },
        )

    return build_request(read_timeout=30.0), build_request(read_timeout=45.0)


async def telegram_error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Log expected network failures tersely and retain tracebacks for other errors."""
    del update
    error = context.error

    if isinstance(error, (NetworkError, TimedOut, RetryAfter)):
        logger.warning("Temporary Telegram network error: %s", error)
        return

    if isinstance(error, BaseException):
        logger.error(
            "Unexpected error while processing a Telegram update",
            exc_info=(type(error), error, error.__traceback__),
        )
        return

    logger.error("Unexpected Telegram error without exception details: %r", error)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if settings.ollama_autostart:
        start_ollama_server(
            host=settings.ollama_host,
            port=settings.ollama_port,
            startup_timeout=settings.ollama_startup_timeout,
            log_file=settings.ollama_runtime_log,
        )

    request, get_updates_request = build_telegram_requests()
    logger.info("Telegram transport configured for IPv4")

    app = build_application(
        request=request,
        get_updates_request=get_updates_request,
    )
    app.add_error_handler(telegram_error_handler)
    logger.info("Starting Telegram polling")
    app.run_polling()


if __name__ == "__main__":
    main()
