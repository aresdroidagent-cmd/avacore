# avacore

AvaCore is a local AI assistant core with:

- an Ollama backend for chat LLMs
- a Telegram bot
- an HTTP API via FastAPI
- RAG for PDFs and images
- SQLite-based memory
- a personality and policy layer
- a local vision module based on Hugging Face / SmolVLM2

## Status

Work in progress, but already usable in the environment described below.

## Validated setup

This setup has been tested in the following environment:

- Ubuntu 20.04.x
- Python 3.10
- NVIDIA Quadro RTX 4000 with 8 GB VRAM
- CUDA 13.0 / PyTorch
- Ollama 0.20.3

Models already used in this setup:

- `gemma4:e2b`
- `qwen2.5:3b-instruct-32k`
- `qwen2.5:3b-instruct`
- `llama3.2:3b-32k`
- `llama3.2:3b`
- `lfm2.5-thinking:latest`

# Create Models with bigger Standard-Ollama-Context
```bash
cat > ~/Modelfile.llama3.2-3b-32k <<'EOF'
FROM llama3.2:3b
PARAMETER num_ctx 32768
EOF

ollama create llama3.2:3b-32k -f ~/Modelfile.llama3.2-3b-32k
```
## Project overview

Important entry points and modules:

- `scripts/run_api.py` starts the HTTP API
- `scripts/run_telegram.py` starts the Telegram bot
- `avacore/api/http_app.py` is the central FastAPI worker
- `avacore/model/ollama_backend.py` talks to Ollama
- `avacore/vision/describe.py` and `avacore/vision/smolvlm_client.py` provide local image analysis
- `avacore/memory/` contains SQLite-based session and memory logic
- `avacore/rag/` contains embedding, chunking, and retrieval components

## System requirements

### Ubuntu packages

At minimum:

```bash
sudo apt update
sudo apt install -y \
  python3.10 python3.10-venv python3-pip \
  git curl build-essential rsync \
  libgl1 libglib2.0-0
```

Depending on your GPU and Torch setup, additional NVIDIA or CUDA packages may be required.

## Python setup

Inside the repository:

```bash
cd ~/avacore
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If `requirements.txt` is incomplete, install any missing packages manually as needed.

## Install Ollama

Ollama must work locally before starting AvaCore.

Check:

```bash
ollama --version
```

Example:

```bash
ollama version is 0.20.3
```

### Pull models

```bash
ollama pull gemma4:e2b
ollama pull qwen2.5:3b-instruct
ollama pull llama3.2:3b
```

Then verify:

```bash
ollama list
```
## open a Terminal and
```bash
ollama serve
```

## Create `.env` from `.env.example`

```bash
cd ~/avacore
cp .env.example .env
```

Then edit `.env`.

## Example `.env`

```env
AVACORE_PROFILE=low_vram
AVACORE_LOG_LEVEL=info
AVACORE_DEBUG=0

AVACORE_DB_PATH=./data/sqlite/avacore.db
AVACORE_HISTORY_DIR=./data/history
AVACORE_PERSONALITY_PATH=./data/personality/default_personality.json

OLLAMA_AUTOSTART=1
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_STARTUP_TIMEOUT=30
OLLAMA_RUNTIME_LOG=./data/logs/ollama_runtime.log
OLLAMA_MODEL=gemma4:e2b
OLLAMA_TIMEOUT_MS=180000
OLLAMA_URL=http://127.0.0.1:11434/api/chat

AVACORE_HTTP_HOST=127.0.0.1
AVACORE_HTTP_PORT=8787

AVACORE_RAG_TOP_K=6
AVACORE_RAG_SCORE_THRESHOLD=0.35
AVACORE_RAG_MAX_CONTEXT_HITS=4
AVACORE_RAG_MAX_SOURCES=4
AVACORE_RAG_MAX_HITS_PER_DOC=2

TELEGRAM_BOT_TOKEN=YOUR_TOKEN_XXXXX
TELEGRAM_ALLOWED_CHAT_ID=YOUR_CHAT_ID

AVACORE_DEFAULT_LOCATION=Schaffhausen
AVACORE_MEDIUM_FEEDS=https://medium.com/feed/tag/artificial-intelligence,https://medium.com/feed/tag/robotics
AVACORE_NEWS_FEEDS=https://feeds.reuters.com/reuters/technologyNews,https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml

AVACORE_MAIL_IMAP_HOST=imap.gmail.com
AVACORE_MAIL_IMAP_PORT=993
AVACORE_MAIL_SMTP_HOST=smtp.gmail.com
AVACORE_MAIL_SMTP_PORT=587
AVACORE_MAIL_USERNAME=ROBOT_MAIL@gmail.com
AVACORE_MAIL_PASSWORD=APP_PASSWORD_OR_PROVIDER_PASSWORD
AVACORE_MAIL_FROM=ROBOT_MAIL@gmail.com
AVACORE_MAIL_ALLOWED_TO=YOUR_MAIL@gmail.com

AVACORE_VISION_ENABLED=1
AVACORE_VISION_MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
AVACORE_VISION_PROMPT=
AVACORE_VISION_ON_PDF_IMAGES=0
AVACORE_VISION_ON_LOOSE_IMAGES=1
AVACORE_VISION_MIN_IMAGE_PIXELS=90000
AVACORE_VISION_MAX_NEW_TOKENS=64

AVACORE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AVACORE_KNOWLEDGE_INBOX_PDF=./data/knowledge/inbox/pdf
AVACORE_KNOWLEDGE_INBOX_IMAGES=./data/knowledge/inbox/images
AVACORE_KNOWLEDGE_PROCESSED_DIR=./data/knowledge/processed
AVACORE_KNOWLEDGE_PDF_IMAGES_DIR=./data/knowledge/processed/pdf_images
AVACORE_KNOWLEDGE_IMAGE_TEXT_DIR=./data/knowledge/processed/image_text
AVACORE_KNOWLEDGE_INDEX_DIR=./data/knowledge/index
AVACORE_RAG_CHUNK_SIZE=800
AVACORE_RAG_CHUNK_OVERLAP=120
AVACORE_OCR_ENABLED=1
AVACORE_OCR_MIN_TEXT_LENGTH=10
```

## Give documents and images to Ava and vectorize them

AvaCore is meant to accept PDFs and images in dedicated inbox folders, process them, and make them searchable through semantic retrieval.

### Input folders

Typically:

- PDFs: `./data/knowledge/inbox/pdf`
- loose images: `./data/knowledge/inbox/images`

Relevant image types in this setup are mainly:

- `jpg`
- `jpeg`
- `png`
- images extracted from PDFs

### What happens during processing

1. **PDF ingestion**
   - PDF files are loaded.
   - page text is extracted.
   - embedded PDF images can be exported separately.
2. **Chunking**
   - long text is split into searchable chunks.
3. **Embedding / vectorization**
   - text chunks are embedded with the configured embedding model.
4. **Image analysis**
   - loose images or PDF images can also be described.
   - OCR text and image captions become part of the searchable knowledge.
5. **Index build**
   - the retrieval index is created from those processed artifacts.

### Relevant variables and directories

- `AVACORE_KNOWLEDGE_INBOX_PDF`
- `AVACORE_KNOWLEDGE_INBOX_IMAGES`
- `AVACORE_KNOWLEDGE_PROCESSED_DIR`
- `AVACORE_KNOWLEDGE_PDF_IMAGES_DIR`
- `AVACORE_KNOWLEDGE_IMAGE_TEXT_DIR`
- `AVACORE_KNOWLEDGE_INDEX_DIR`
- `AVACORE_EMBEDDING_MODEL`
- `AVACORE_RAG_CHUNK_SIZE`
- `AVACORE_RAG_CHUNK_OVERLAP`
- `AVACORE_OCR_ENABLED`
- `AVACORE_OCR_MIN_TEXT_LENGTH`

### Typical user workflow

1. Put PDFs into `data/knowledge/inbox/pdf/`.
2. Put images into `data/knowledge/inbox/images/`.
3. Run the project's ingestion / indexing scripts.
4. Ask questions about the material through the API or Telegram.

### Practical notes

- scanned PDFs benefit from OCR if the PDF does not contain extractable native text
- text-heavy images should not blindly be treated as `photo`
- technical graphics are often better handled as `diagram`
- illustrations, posters, or paintings are better treated as `artwork`

## Vision settings

- leave `AVACORE_VISION_PROMPT=` empty if you want to use the built-in mode prompts
- auto mode distinguishes between `screen`, `diagram`, `assembly`, `artwork`, and `photo`
- for text-heavy images, slides, and screenshots, `screen` is usually better than `photo`
- for diagrams or technical graphics, `diagram` is usually better
- `SmolVLM2-500M` is lightweight and fast, but not fully reliable semantically

## Telegram

At minimum:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_CHAT_ID=...
```
# Start Project
## 1) Put PDFs into data/knowledge/inbox/pdf
## 2) Put images into data/knowledge/inbox/images
## 3) Build / refresh the knowledge index
```bash
cd ~/avacore
source .venv/bin/activate
python scripts/index_knowledge.py
```
## 4) Start the API
```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_api.py
```
## 5) Start Telegram in a second terminal
```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_telegram.py
```

# Quick tests

## Health

```bash
curl http://127.0.0.1:8787/health
```

## Model status

```bash
curl http://127.0.0.1:8787/model
```

## Ollama tags

```bash
curl http://127.0.0.1:11434/api/tags
```

## Detect vision mode

```bash
curl -X POST http://127.0.0.1:8787/vision/detect_mode \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/full/path/to/image.png",
    "ocr_text": ""
  }'
```

## Describe image

```bash
curl -X POST http://127.0.0.1:8787/vision/describe_image \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/full/path/to/image.png",
    "ocr_text": ""
  }'
```

## Optional with an explicit mode:

```bash
curl -X POST http://127.0.0.1:8787/vision/describe_image \
  -H "Content-Type: application/json" \
  -d '{
    "image_path": "/full/path/to/image.png",
    "mode": "diagram",
    "ocr_text": ""
  }'
```

## Gmail / IMAP / SMTP notes

AvaCore uses classic IMAP/SMTP settings from `.env`, not the Gmail API.

Relevant variables:

```env
AVACORE_MAIL_IMAP_HOST=imap.gmail.com
AVACORE_MAIL_IMAP_PORT=993
AVACORE_MAIL_SMTP_HOST=smtp.gmail.com
AVACORE_MAIL_SMTP_PORT=587
AVACORE_MAIL_USERNAME=your.name@gmail.com
AVACORE_MAIL_PASSWORD=APP_PASSWORD_OR_PROVIDER_PASSWORD
AVACORE_MAIL_FROM=your.name@gmail.com
AVACORE_MAIL_ALLOWED_TO=target1@example.com,target2@example.com
```

Important in practice:

- for personal Gmail accounts, app passwords are typically required for this kind of client, and 2-Step Verification must be enabled
- app passwords may be unavailable if the account uses security keys only, is a work/school account, or has Advanced Protection enabled
- app passwords are revoked after a password change and need to be recreated
- for personal Google Accounts, IMAP has been always-on since January 2025
- pure username/password sign-ins without modern authentication are no longer supported for Google Workspace accounts as of January 2025

## Common failures

### `404 ... http:127.0.0.1:11434/api/chat`

The URL is malformed. It must be:

```env
OLLAMA_URL=http://127.0.0.1:11434/api/chat
```

### `{"models":[]}`

Ollama is running but cannot see any models. Check `ollama list` and inspect the model store path.

### Vision returns useless short answers

Then the image type and the chosen mode usually do not match:

- screenshot / slide / text-heavy image -> `screen`
- technical graphic -> `diagram`
- assembly step -> `assembly`
- illustration / cover / painting -> `artwork`
- real photo -> `photo`

## Reproducible startup sequence

1. `source .venv/bin/activate`
2. check `ollama list`
3. `python scripts/run_api.py`
4. in a second terminal run `python scripts/run_telegram.py`
5. `curl http://127.0.0.1:8787/health`
6. optionally `curl http://127.0.0.1:11434/api/tags`

## Document status

This README includes:

- Ollama autostart from AvaCore
- Telegram bot usage
- SmolVLM2-based vision path
- notes about PDF and image vectorization


# Telegram integration

AvaCore can be connected to Telegram through a private bot interface.

Requirements

## You need:

a Telegram bot token from BotFather
the Telegram chat ID that is allowed to talk to AvaCore
the HTTP API running locally, because the Telegram bot forwards requests to the API worker
Environment variables

## Set these values in .env:

TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_ALLOWED_CHAT_ID=your_private_chat_id
How it works
scripts/run_telegram.py starts the Telegram bot
the bot only accepts messages from the configured private chat
incoming Telegram messages are forwarded to the AvaCore HTTP API
the API handles chat memory, retrieval, document lookup, vision endpoints, policies and model access
Start Telegram

## Run the API first:

```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_api.py
```

## Then start Telegram in a second terminal:

```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_telegram.py
```

## Security note

The Telegram bot is intentionally restricted to one configured private chat via TELEGRAM_ALLOWED_CHAT_ID.
If the chat ID does not match, AvaCore rejects the conversation.

## Telegram commands and Ava skills

AvaCore exposes a set of Telegram commands for system status, memory, documents, weather, web tools and mail actions.

### Basic commands
/start
Start Ava and show the command overview.

/help
Show the available commands.

/health
Show AvaCore runtime status.

/model
Show the currently active Ollama model and profile.

/personality
Show the active personality configuration.

/personalitybackup
Save the current personality into SQLite.

/personalityrestore <profile_id>
Restore a stored personality profile.

### Memory and policy commands

/memories
List stored memories.

/remember <text>
Store a manual memory entry.

/policies
Show active policies.

/reset
Reset the current Telegram chat history in AvaCore.

### Document and knowledge commands

/docs [keyword]
List known documents from the knowledge base.

/page <document name> | <page>
Explain a specific page from a document.
These commands rely on the indexed knowledge base.

If new PDFs or images were added, run:
python scripts/index_knowledge.py
before expecting retrieval to use the new content.

### Weather and feed commands

/weather [location]
Show a short weather summary.

/medium
Show current Medium feed entries.

/news
Show current news feed entries.

/mediumdigest
Summarize Medium feed entries.

/newsdigest
Summarize news feed entries.

### Web commands

/webfetch <url>
Fetch raw readable page text from a URL.

/webask <url> <question>
Ask a question about a specific webpage.

### Mail commands

/mail
Show recent inbox entries.

/maildigest
Summarize recent emails.

/sendmail <subject> | <text>
Send a mail to the configured default recipient.

/mailscript <filename.py> | <content>
Send Python script content by mail.

/mailnote <title> | <content>
Send an important note by mail.

### The default recipient is taken from:

AVACORE_MAIL_ALLOWED_TO=someone@example.com
The Telegram bot uses the first configured address as its default mail target.

### Free-text chat

In addition to commands, Ava also accepts normal Telegram text messages.
These are forwarded to the /reply API endpoint and can use:

chat history
stored memories
retrieved document context
the active Ollama model
policies and personality settings

## Web UI

AvaCore includes a minimal browser-based UI on top of the existing FastAPI server.

### Features

- chat page
- document list and page explain view
- status page
- admin page with read-only runtime/config view
- password-protected admin access for LAN use
- Ava avatar image shown in the UI

### Environment

Add these values to `.env`:

```env
AVACORE_HTTP_HOST=0.0.0.0
AVACORE_HTTP_PORT=8787
AVACORE_WEB_ADMIN_PASSWORD=change_me_in_lan
```

Then open in a browser:

http://127.0.0.1:8787/ui/chat
http://127.0.0.1:8787/ui/status
http://127.0.0.1:8787/ui/admin

For LAN access, replace 127.0.0.1 with the machine IP, for example:

http://192.168.x.x:8787/ui/chat

### Admin access

The admin page is read-only and protected by the password defined in:

```env
AVACORE_WEB_ADMIN_PASSWORD=...

```
### Current UI pages

- /ui/chat
Chat interface with document list and page explain panel.

- /ui/status
Runtime status view.

- /ui/admin
Read-only runtime/configuration view protected by password.

- /ui/avatar

Ava image served by the backend.
Security note

The current admin protection is intentionally simple and should only be used inside a trusted local network.

- `/ui/review`

This page is intended for reviewing memory candidates and managing the `candidate / verified / rejected` workflow.

Like the admin page, it is intended for trusted LAN use and protected by the configured admin password.

### Avatar image

The web UI can display an Ava avatar image served by the backend.

Set the image path in `.env`:

```env
AVACORE_WEB_AVATAR_PATH=./data/knowledge/inbox/images/synthese-bots-15.jpg
```

### Do not expose this setup directly to the internet without:

a reverse proxy
HTTPS
stronger authentication

## Memory review workflow

AvaCore now supports a reviewable memory workflow with explicit status handling.

### Memory states

Memory items can exist in three states:

- `candidate`
- `verified`
- `rejected`

Only **verified** memory should be treated as trusted long-term memory in the main hybrid chat prompt.

### Purpose

This allows AvaCore to:

- collect potentially useful findings
- keep unverified facts separate from trusted memory
- let the user verify or reject memory candidates
- prepare the system for future agent-style research and controlled learning

### Memory item model

Each reviewable memory item can store:

- `scope`
- `title`
- `content`
- `memory_type`
- `status`
- `source_type`
- `source_ref`
- `confidence`
- `importance`
- `tags`

Typical examples:

- `environment`
- `project`
- `preference`
- `workflow_rule`
- `document_fact`
- `web_fact`
- `research_lead`

### Review UI

AvaCore includes a dedicated browser page for memory review:

- `/ui/review`

The review page allows you to:

- inspect candidate memories
- verify candidate memories
- reject candidate memories
- delete memory items

### Review API

The following endpoints are available for reviewable memory items:

- `GET /memories/items`
- `GET /memories/candidates`
- `GET /memories/verified`
- `GET /memories/rejected`
- `GET /memories/items/{id}`
- `POST /memories/items`
- `POST /memories/items/{id}/verify`
- `POST /memories/items/{id}/reject`
- `DELETE /memories/items/{id}`

These endpoints are protected with the admin password and require the header:

```http
X-Admin-Password: <your admin password>