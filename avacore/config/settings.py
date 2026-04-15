from pathlib import Path
from dotenv import load_dotenv
import os

from avacore.config.profiles import PROFILES

load_dotenv()


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


class Settings:
    def __init__(self) -> None:
        self.profile_name = os.environ.get("AVACORE_PROFILE", "low_vram")
        self.profile = PROFILES[self.profile_name]

        self.log_level = os.environ.get("AVACORE_LOG_LEVEL", "info")
        self.debug = os.environ.get("AVACORE_DEBUG", "0").strip() in {
            "1", "true", "True", "yes", "on"
        }

        self.db_path = Path(os.environ.get("AVACORE_DB_PATH", "./data/sqlite/avacore.db"))
        self.history_dir = Path(os.environ.get("AVACORE_HISTORY_DIR", "./data/history"))
        self.personality_path = Path(
            os.environ.get(
                "AVACORE_PERSONALITY_PATH",
                "./data/personality/default_personality.json",
            )
        )

        self.ollama_host = os.environ.get("OLLAMA_HOST", "127.0.0.1").strip()
        self.ollama_port = int(os.environ.get("OLLAMA_PORT", "11434"))
        self.ollama_url = os.environ.get(
            "OLLAMA_URL",
            f"http://{self.ollama_host}:{self.ollama_port}/api/chat",
        ).strip()
        self.ollama_model = os.environ.get("OLLAMA_MODEL", self.profile["default_model"]).strip()
        self.ollama_timeout_ms = int(
            os.environ.get("OLLAMA_TIMEOUT_MS", str(self.profile["request_timeout_ms"]))
        )
        self.ollama_autostart = os.environ.get("OLLAMA_AUTOSTART", "1").strip() in {
            "1", "true", "True", "yes", "on"
        }
        self.ollama_startup_timeout = float(
            os.environ.get("OLLAMA_STARTUP_TIMEOUT", "30")
        )
        self.ollama_runtime_log = os.environ.get(
            "OLLAMA_RUNTIME_LOG",
            "./data/logs/ollama_runtime.log",
        ).strip()

        self.http_host = os.environ.get("AVACORE_HTTP_HOST", "127.0.0.1")
        self.http_port = int(os.environ.get("AVACORE_HTTP_PORT", "8787"))
        self.web_admin_password = os.environ.get("AVACORE_WEB_ADMIN_PASSWORD", "").strip()
        self.web_avatar_path = Path(
            os.environ.get(
                "AVACORE_WEB_AVATAR_PATH",
                "./data/knowledge/inbox/images/synthese-bots-15.jpg",
            )
        ).expanduser()

        self.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.telegram_allowed_chat_id = os.environ.get("TELEGRAM_ALLOWED_CHAT_ID", "").strip()

        self.max_history_turns = int(self.profile["max_history_turns"])
        self.tool_mode = str(self.profile["tool_mode"])

        self.knowledge_inbox_pdf_dir = Path(
            os.environ.get("AVACORE_KNOWLEDGE_INBOX_PDF", "./data/knowledge/inbox/pdf")
        )
        self.knowledge_inbox_images_dir = Path(
            os.environ.get("AVACORE_KNOWLEDGE_INBOX_IMAGES", "./data/knowledge/inbox/images")
        )
        self.knowledge_processed_dir = Path(
            os.environ.get("AVACORE_KNOWLEDGE_PROCESSED_DIR", "./data/knowledge/processed")
        )
        self.knowledge_pdf_images_dir = Path(
            os.environ.get("AVACORE_KNOWLEDGE_PDF_IMAGES_DIR", "./data/knowledge/processed/pdf_images")
        )
        self.knowledge_image_text_dir = Path(
            os.environ.get("AVACORE_KNOWLEDGE_IMAGE_TEXT_DIR", "./data/knowledge/processed/image_text")
        )
        self.knowledge_index_dir = Path(
            os.environ.get("AVACORE_KNOWLEDGE_INDEX_DIR", "./data/knowledge/index")
        )

        self.embedding_model = os.environ.get(
            "AVACORE_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.rag_top_k = int(os.environ.get("AVACORE_RAG_TOP_K", "6"))
        self.rag_chunk_size = int(os.environ.get("AVACORE_RAG_CHUNK_SIZE", "800"))
        self.rag_chunk_overlap = int(os.environ.get("AVACORE_RAG_CHUNK_OVERLAP", "120"))
        self.rag_score_threshold = float(os.environ.get("AVACORE_RAG_SCORE_THRESHOLD", "0.35"))
        self.rag_max_context_hits = int(os.environ.get("AVACORE_RAG_MAX_CONTEXT_HITS", "4"))
        self.rag_max_sources = int(os.environ.get("AVACORE_RAG_MAX_SOURCES", "4"))
        self.rag_max_hits_per_doc = int(os.environ.get("AVACORE_RAG_MAX_HITS_PER_DOC", "2"))

        self.default_location = os.environ.get("AVACORE_DEFAULT_LOCATION", "Zurich").strip()
        self.medium_feeds = split_csv(os.environ.get("AVACORE_MEDIUM_FEEDS", ""))
        self.news_feeds = split_csv(os.environ.get("AVACORE_NEWS_FEEDS", ""))

        self.ocr_enabled = os.environ.get("AVACORE_OCR_ENABLED", "1").strip() not in {
            "0", "false", "False"
        }
        self.ocr_min_text_length = int(os.environ.get("AVACORE_OCR_MIN_TEXT_LENGTH", "10"))

        self.mail_imap_host = os.environ.get("AVACORE_MAIL_IMAP_HOST", "").strip()
        self.mail_imap_port = int(os.environ.get("AVACORE_MAIL_IMAP_PORT", "993"))
        self.mail_smtp_host = os.environ.get("AVACORE_MAIL_SMTP_HOST", "").strip()
        self.mail_smtp_port = int(os.environ.get("AVACORE_MAIL_SMTP_PORT", "587"))
        self.mail_username = os.environ.get("AVACORE_MAIL_USERNAME", "").strip()
        self.mail_password = os.environ.get("AVACORE_MAIL_PASSWORD", "").strip()
        self.mail_from = os.environ.get("AVACORE_MAIL_FROM", "").strip()
        self.mail_allowed_to = split_csv(os.environ.get("AVACORE_MAIL_ALLOWED_TO", ""))

        self.vision_enabled = os.environ.get("AVACORE_VISION_ENABLED", "1").strip() not in {
            "0", "false", "False"
        }
        self.vision_model = os.environ.get(
            "AVACORE_VISION_MODEL",
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        ).strip()
        self.vision_prompt = os.environ.get(
            "AVACORE_VISION_PROMPT",
            "",
        ).strip()
        self.vision_max_new_tokens = int(os.environ.get("AVACORE_VISION_MAX_NEW_TOKENS", "64"))
        self.vision_on_pdf_images = os.environ.get("AVACORE_VISION_ON_PDF_IMAGES", "0").strip() not in {
            "0", "false", "False"
        }
        self.vision_on_loose_images = os.environ.get("AVACORE_VISION_ON_LOOSE_IMAGES", "1").strip() not in {
            "0", "false", "False"
        }
        self.vision_min_image_pixels = int(os.environ.get("AVACORE_VISION_MIN_IMAGE_PIXELS", "90000"))


settings = Settings()