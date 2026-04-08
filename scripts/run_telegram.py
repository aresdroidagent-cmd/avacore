from avacore.channels.telegram.bot import build_application
from avacore.config.settings import settings
from avacore.system.ollama_runtime import start_ollama_server


def main() -> None:
    if settings.ollama_autostart:
        start_ollama_server(
            host=settings.ollama_host,
            port=settings.ollama_port,
            startup_timeout=settings.ollama_startup_timeout,
            log_file=settings.ollama_runtime_log,
        )

    app = build_application()
    app.run_polling()


if __name__ == "__main__":
    main()