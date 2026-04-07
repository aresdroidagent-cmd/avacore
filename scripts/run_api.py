import uvicorn

from avacore.config.settings import settings


def main() -> None:
    uvicorn.run(
        "avacore.api.http_app:app",
        host=settings.http_host,
        port=settings.http_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
