# AvaCore

AvaCore is a local AI assistant core for Roger Seeberger's workstation and LAN environment. It combines a local Ollama language model, Telegram, FastAPI, document/image RAG, SQLite memory, web research, camera snapshots, calendar briefings and a shared long-term brain.

The goal is not a toy chatbot. AvaCore is intended as a practical local assistant that can help with robotics, computer vision, AI engineering, research, documentation, automation and project continuity over longer periods of time.

<p>
  <strong>Author / Autor:</strong> Roger Seeberger (Swissbot) seeberger.robotics@gmail.com<br>
  <img src="docs/author_icon.png" alt="Author icon" width="64" />
</p>

## Current status

AvaCore is work in progress, but usable in the validated local setup described below.

Main capabilities:

- local chat backend through Ollama
- FastAPI HTTP API
- Telegram bot interface
- web UI for chat, status, admin and memory review
- RAG over PDFs, extracted text and images
- local image description through SmolVLM2
- SQLite session and memory store
- reviewed long-term memory workflow: `candidate`, `verified`, `rejected`
- shared brain files for identity, user context, operating rules and curated memory
- web research with source collection and memory candidates
- read-only browser control through Chromium / Playwright
- RTSP camera snapshot support
- private iCal calendar briefing
- optional IMAP/SMTP mail tools

## Validated environment

The current low-VRAM profile has been tested with:

- Ubuntu 20.04.x
- Python 3.10
- NVIDIA Quadro RTX 4000 with 8 GB VRAM
- Ollama 0.20.3
- local model: `gemma4:e2b`

Other systems may work, but this README focuses on a reproducible local installation.

## Repository layout

Important paths:

```text
avacore/
├── avacore/
│   ├── api/              # FastAPI app
│   ├── channels/telegram # Telegram bot
│   ├── core/             # shared brain and decision router
│   ├── memory/           # SQLite store and memory logic
│   ├── model/            # Ollama backend
│   ├── rag/              # document/image retrieval
│   ├── tools/            # camera, calendar, web, browser, mail helpers
│   ├── vision/           # SmolVLM / image description
│   └── web/static/       # web UI assets
├── data/
│   ├── brain/            # SOUL.md, USER.md, OPERATING.md, MEMORY.md, daily notes
│   ├── knowledge/        # PDF/image inbox, processed files and vector index
│   ├── sqlite/           # local SQLite database
│   ├── cache/            # camera/browser cache
│   └── logs/             # runtime logs
├── scripts/
└── tests/
```

## System packages

Install the basic Ubuntu packages:

```bash
sudo apt update
sudo apt install -y \
  git curl build-essential rsync \
  python3.10 python3.10-venv python3-pip \
  libgl1 libglib2.0-0 ffmpeg netcat-openbsd
```

For Playwright browser automation, system dependencies may be installed later with:

```bash
python -m playwright install-deps chromium
```

## Python environment

From the repository root:

```bash
cd ~/avacore
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If newer modules are not yet part of `requirements.txt`, install the currently used optional dependencies:

```bash
pip install \
  beautifulsoup4 \
  icalendar recurring-ical-events python-dateutil \
  python-dotenv \
  playwright

python -m playwright install chromium
```

For RTSP camera snapshots, keep OpenCV compatible with `numpy<2`:

```bash
pip uninstall -y opencv-python-headless opencv-python numpy
pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
```

Do not install an unpinned latest `opencv-python-headless` unless the project has been updated for NumPy 2.x.

## Ollama setup

Ollama must work locally before AvaCore starts.

Check:

```bash
ollama --version
```

Pull the default low-VRAM model:

```bash
ollama pull gemma4:e2b
ollama list
```

AvaCore can autostart Ollama if configured with `OLLAMA_AUTOSTART=1`. Manual start is also possible:

```bash
ollama serve
```

## Configuration

Create your local `.env`:

```bash
cd ~/avacore
cp .env.example .env
nano .env
```

Do not commit `.env`. It contains secrets such as Telegram tokens, iCal links, mail passwords and admin passwords.

A compact example:

```env
# AvaCore profile / logging
AVACORE_PROFILE=low_vram
AVACORE_LOG_LEVEL=info
AVACORE_DEBUG=0

# Core paths
AVACORE_DB_PATH=./data/sqlite/avacore.db
AVACORE_HISTORY_DIR=./data/history
AVACORE_PERSONALITY_PATH=./data/personality/default_personality.json

# Shared Brain
AVACORE_BRAIN_DIR=./data/brain
AVACORE_ASSISTANT_NAME=Ava
AVACORE_SYSTEM_NAME=AvaCore
AVACORE_DEFAULT_LOCATION=Zurich, Switzerland
AVACORE_AUTO_RESEARCH=ask

# HTTP / Web UI
AVACORE_HTTP_HOST=0.0.0.0
AVACORE_HTTP_PORT=8787
AVACORE_WEB_ADMIN_PASSWORD=change_me_in_lan
AVACORE_WEB_AVATAR_PATH=./data/knowledge/inbox/images/synthese-bots-15.jpg

# Ollama
OLLAMA_AUTOSTART=1
OLLAMA_HOST=127.0.0.1
OLLAMA_PORT=11434
OLLAMA_STARTUP_TIMEOUT=30
OLLAMA_RUNTIME_LOG=./data/logs/ollama_runtime.log
OLLAMA_MODEL=gemma4:e2b
OLLAMA_TIMEOUT_MS=180000
OLLAMA_URL=http://127.0.0.1:11434/api/chat

# Telegram
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_CHAT_ID=

# Knowledge ingestion / RAG
AVACORE_KNOWLEDGE_INBOX_PDF=./data/knowledge/inbox/pdf
AVACORE_KNOWLEDGE_INBOX_IMAGES=./data/knowledge/inbox/images
AVACORE_KNOWLEDGE_PROCESSED_DIR=./data/knowledge/processed
AVACORE_KNOWLEDGE_PDF_IMAGES_DIR=./data/knowledge/processed/pdf_images
AVACORE_KNOWLEDGE_IMAGE_TEXT_DIR=./data/knowledge/processed/image_text
AVACORE_KNOWLEDGE_INDEX_DIR=./data/knowledge/index
AVACORE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
AVACORE_RAG_TOP_K=6
AVACORE_RAG_CHUNK_SIZE=800
AVACORE_RAG_CHUNK_OVERLAP=120
AVACORE_RAG_SCORE_THRESHOLD=0.35
AVACORE_RAG_MAX_CONTEXT_HITS=4
AVACORE_RAG_MAX_SOURCES=4
AVACORE_RAG_MAX_HITS_PER_DOC=2

# OCR / Vision
AVACORE_OCR_ENABLED=1
AVACORE_OCR_MIN_TEXT_LENGTH=10
AVACORE_VISION_ENABLED=1
AVACORE_VISION_MODEL=HuggingFaceTB/SmolVLM2-500M-Video-Instruct
AVACORE_VISION_PROMPT=
AVACORE_VISION_ON_PDF_IMAGES=0
AVACORE_VISION_ON_LOOSE_IMAGES=1
AVACORE_VISION_MIN_IMAGE_PIXELS=90000
AVACORE_VISION_MAX_NEW_TOKENS=40

# Web research
AVACORE_RESEARCH_ENABLED=1
AVACORE_RESEARCH_MAX_RESULTS=4
AVACORE_RESEARCH_SAVE_MEMORY_CANDIDATE=1

# Browser control / Chromium read-only automation
AVACORE_BROWSER_ENABLED=0
AVACORE_BROWSER_HEADLESS=0
AVACORE_BROWSER_USER_DATA_DIR=./data/browser/chromium-profile
AVACORE_BROWSER_SCREENSHOT_DIR=./data/cache/browser
AVACORE_BROWSER_TIMEOUT_MS=30000
AVACORE_BROWSER_DEFAULT_SEARCH=https://duckduckgo.com/?q=

# RTSP camera
AVACORE_CAMERA_ENABLED=0
AVACORE_CAMERA_USER=admin
AVACORE_CAMERA_PASSWORD=
AVACORE_CAMERA_IP=
AVACORE_CAMERA_RTSP_PATH=/play1.sdp
AVACORE_CAMERA_CACHE_DIR=./data/cache/camera

# Calendar / daily briefing
AVACORE_CALENDAR_ICS_URL=
AVACORE_DAILY_BRIEFING_TIME=08:30
AVACORE_DAILY_BRIEFING_TIMEZONE=Europe/Zurich
AVACORE_API_URL=http://127.0.0.1:8787

# Mail / IMAP / SMTP
AVACORE_MAIL_IMAP_HOST=imap.gmail.com
AVACORE_MAIL_IMAP_PORT=993
AVACORE_MAIL_SMTP_HOST=smtp.gmail.com
AVACORE_MAIL_SMTP_PORT=587
AVACORE_MAIL_USERNAME=
AVACORE_MAIL_PASSWORD=
AVACORE_MAIL_FROM=
AVACORE_MAIL_ALLOWED_TO=

# Optional feeds
AVACORE_MEDIUM_FEEDS=
AVACORE_NEWS_FEEDS=
```

## Initialize local folders

```bash
mkdir -p \
  data/brain/daily \
  data/knowledge/inbox/pdf \
  data/knowledge/inbox/images \
  data/knowledge/processed \
  data/knowledge/index \
  data/cache/camera \
  data/cache/browser \
  data/logs \
  data/sqlite
```

## Shared Brain and long-term context

AvaCore uses a transparent shared-brain structure inspired by agent memory systems, but adapted for the local AvaCore architecture.

Files:

```text
data/brain/SOUL.md       # Ava identity, purpose, boundaries
data/brain/USER.md       # Roger context, working style, preferences
data/brain/OPERATING.md  # operating rules and tool policy
data/brain/MEMORY.md     # curated long-term memory summary
data/brain/daily/        # daily episodic notes
```

At reply time, Ava can combine:

- current date, time, timezone and location context
- Ava identity and Roger context
- active personality profile
- verified SQLite memories
- local RAG hits
- daily notes
- decision-router hints for memory/RAG/research needs

Core identity belongs in `SOUL.md`. Stable user/project context belongs in `USER.md` or `MEMORY.md`. Raw daily work notes belong in `data/brain/daily/YYYY-MM-DD.md`.

## Memory model

AvaCore separates unverified findings from trusted memory.

Memory states:

- `candidate` — collected but not trusted yet
- `verified` — approved and usable as trusted long-term context
- `rejected` — explicitly not used

Typical memory types:

- `user_profile`
- `project`
- `environment`
- `preference`
- `workflow_rule`
- `document_fact`
- `web_fact`
- `research_lead`
- `note`

Only verified memories should be injected into Ava's trusted prompt context.

Review UI:

```text
http://127.0.0.1:8787/ui/review
```

Review API endpoints are protected by `X-Admin-Password`:

```text
GET    /memories/items
GET    /memories/candidates
GET    /memories/verified
GET    /memories/rejected
GET    /memories/items/{id}
POST   /memories/items
POST   /memories/items/{id}/verify
POST   /memories/items/{id}/reject
DELETE /memories/items/{id}
```

## Knowledge ingestion and RAG

Put source files into the inbox folders:

```text
data/knowledge/inbox/pdf/
data/knowledge/inbox/images/
```

Build or refresh the knowledge index:

```bash
cd ~/avacore
source .venv/bin/activate
python scripts/index_knowledge.py
```

The pipeline extracts PDF text, chunks documents, creates embeddings, processes images/OCR when enabled and builds a local retrieval index.

Practical notes:

- scanned PDFs benefit from OCR
- screenshots and slides are usually better described as `screen`
- diagrams and technical graphics are better described as `diagram`
- assembly/manual images are better described as `assembly`
- real photos are better described as `photo`

## Start AvaCore

Terminal 1 — API:

```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_api.py
```

Terminal 2 — Telegram bot:

```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_telegram.py
```

Health checks:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/model
curl http://127.0.0.1:11434/api/tags
```

## Web UI

Available pages:

```text
/ui/chat     chat interface
/ui/status   runtime status
/ui/admin    read-only admin/config view
/ui/review   memory candidate review
/ui/avatar   Ava avatar image
```

Local URLs:

```text
http://127.0.0.1:8787/ui/chat
http://127.0.0.1:8787/ui/status
http://127.0.0.1:8787/ui/admin
http://127.0.0.1:8787/ui/review
```

For LAN usage, replace `127.0.0.1` with the machine IP.

Security note: the built-in admin password is intended for a trusted local network. Do not expose AvaCore directly to the internet without reverse proxy, HTTPS and stronger authentication.

## Telegram

Set in `.env`:

```env
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ALLOWED_CHAT_ID=...
```

The bot only accepts messages from the configured private chat.

Common commands:

```text
/start                         show command overview
/help                          show available commands
/health                        runtime status
/model                         active model and profile
/reset                         reset Telegram chat history
/memories                      list stored memories
/remember <text>               store a manual memory entry
/docs [keyword]                list indexed documents
/page <doc> | <page>           explain a specific document page
/camera or /snapshot           request current camera image
/briefing                      today's calendar briefing
/research <question>           web research with sources and memory candidate
/webfetch <url>                fetch readable page text
/webask <url> <question>       ask a question about a specific webpage
/mail                          recent inbox entries
/maildigest                    summarize recent emails
/sendmail <subject> | <body>   send mail to configured default recipient
```

Free text messages are forwarded to `/reply` and can use chat history, verified memories, RAG context, policies, personality and the active Ollama model.

## Web research

AvaCore has two web information paths.

Direct webpage question:

```bash
curl -X POST http://127.0.0.1:8787/tools/web_ask \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{"url":"https://example.com","question":"What is on this page?"}'
```

Research workflow:

```bash
curl -X POST http://127.0.0.1:8787/research \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{
    "query": "D-Link DCS-5222L RTSP play1.sdp",
    "max_results": 4,
    "save_memory": true
  }'
```

Telegram:

```text
/research D-Link DCS-5222L RTSP play1.sdp
```

Research results are stored as memory candidates when enabled:

```text
memory_type = research_lead
status      = candidate
source_type = web
```

Review and verify before trusting long-term.

## Decision router

AvaCore includes a lightweight decision router that classifies whether a request likely needs memory, RAG, web research, calendar, camera or a memory candidate.

Debug endpoint:

```bash
curl -X POST http://127.0.0.1:8787/debug/decision \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{"text":"Welche OpenCV Version ist aktuell stabil?"}'
```

The router should prefer local knowledge first and use web research only when current or external information is needed.

## Read-only browser control

AvaCore includes experimental Chromium control through Playwright.

Allowed v1 actions:

- open URL
- search web
- read visible page text
- take screenshot
- close browser

Not allowed in v1:

- form submission
- arbitrary typing
- purchases
- sending emails
- creating calendar events
- uncontrolled clicking

Enable only when needed:

```env
AVACORE_BROWSER_ENABLED=1
AVACORE_BROWSER_HEADLESS=0
```

Examples:

```bash
curl -X POST http://127.0.0.1:8787/browser/search \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{"query":"D-Link DCS-5222L RTSP play1.sdp"}'

curl -X POST http://127.0.0.1:8787/browser/text \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{"max_chars":4000}'

curl -X POST http://127.0.0.1:8787/browser/screenshot \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{"full_page":true}'
```

Browser actions are executed through one worker thread because Playwright contexts are thread-bound.

## RTSP camera snapshots

Enable in `.env`:

```env
AVACORE_CAMERA_ENABLED=1
AVACORE_CAMERA_USER=admin
AVACORE_CAMERA_PASSWORD=
AVACORE_CAMERA_IP=192.168.X.XXX
AVACORE_CAMERA_RTSP_PATH=/play1.sdp
AVACORE_CAMERA_CACHE_DIR=./data/cache/camera
```

For the D-Link DCS-5222L with empty password, the working URL format is:

```text
rtsp://admin:@192.168.8.184:554/play1.sdp
```

Snapshot API:

```bash
curl -X POST http://127.0.0.1:8787/camera/snapshot
```

Telegram:

```text
/camera
/snapshot
```

Cleanup old camera snapshots manually:

```bash
python scripts/cleanup_camera_cache.py
```

Systemd user timer example:

```ini
# ~/.config/systemd/user/avacore-camera-cleanup.service
[Unit]
Description=AvaCore camera cache cleanup

[Service]
Type=oneshot
WorkingDirectory=/home/ares/avacore
ExecStart=/home/ares/avacore/.venv/bin/python /home/ares/avacore/scripts/cleanup_camera_cache.py
```

```ini
# ~/.config/systemd/user/avacore-camera-cleanup.timer
[Unit]
Description=Run AvaCore camera cache cleanup weekly

[Timer]
OnCalendar=weekly
Persistent=true

[Install]
WantedBy=timers.target
```

Activate:

```bash
systemctl --user daemon-reload
systemctl --user enable --now avacore-camera-cleanup.timer
```

## Calendar briefing

AvaCore uses a private iCal URL for read-only calendar access. This avoids Google OAuth and browser-login problems.

Set in `.env`:

```env
AVACORE_CALENDAR_ICS_URL=https://calendar.google.com/calendar/ical/.../basic.ics
AVACORE_DAILY_BRIEFING_TIME=08:30
AVACORE_DAILY_BRIEFING_TIMEZONE=Europe/Zurich
```

Keep the iCal URL secret. Anyone with this URL can read the calendar feed.

API:

```bash
curl -X POST http://127.0.0.1:8787/briefing/calendar \
  -H "Content-Type: application/json" \
  -H "X-Admin-Password: YOUR_PASSWORD" \
  -d '{}'
```

Telegram:

```text
/briefing
```

Automatic daily Telegram briefing script:

```bash
python scripts/send_daily_briefing.py
```

Systemd user timer example:

```ini
# ~/.config/systemd/user/avacore-daily-briefing.service
[Unit]
Description=AvaCore daily Telegram briefing

[Service]
Type=oneshot
WorkingDirectory=/home/ares/avacore
EnvironmentFile=/home/ares/avacore/.env
ExecStart=/home/ares/avacore/.venv/bin/python /home/ares/avacore/scripts/send_daily_briefing.py
```

```ini
# ~/.config/systemd/user/avacore-daily-briefing.timer
[Unit]
Description=Run AvaCore daily briefing every morning

[Timer]
OnCalendar=*-*-* 08:30:00
Persistent=true

[Install]
WantedBy=timers.target
```

Activate:

```bash
systemctl --user daemon-reload
systemctl --user enable --now avacore-daily-briefing.timer
```

The timer expects the AvaCore API to be running.

## Vision endpoints

For SmolVLM2 vision support, the tested setup used Transformers 4.50.0.dev0.
If the stable package fails with the selected vision model, install the matching development version manually.

Detect image mode:

```bash
curl -X POST http://127.0.0.1:8787/vision/detect_mode \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/full/path/to/image.png","ocr_text":""}'
```

Describe image:

```bash
curl -X POST http://127.0.0.1:8787/vision/describe_image \
  -H "Content-Type: application/json" \
  -d '{"image_path":"/full/path/to/image.png","mode":"diagram","ocr_text":""}'
```

## Mail / IMAP / SMTP

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
AVACORE_MAIL_ALLOWED_TO=target@example.com
```

For personal Gmail accounts, app passwords are typically required and 2-Step Verification must be enabled.

## Git hygiene

Never commit:

```text
.env
.venv/
__pycache__/
*.pyc
data/cache/
data/browser/
data/logs/
data/sqlite/
data/knowledge/processed/
data/knowledge/index/
*.jpg
*.png
```

Before committing:

```bash
git status
git diff --cached --stat
```

## Common problems

### `Ollama URL malformed`

Use:

```env
OLLAMA_URL=http://127.0.0.1:11434/api/chat
```

Not:

```text
http:127.0.0.1:11434/api/chat
```

### `ollama list` shows no models

Check whether Ollama is running and which model store it uses. Start with:

```bash
ollama serve
ollama list
```

### OpenCV upgraded NumPy to 2.x

Repair the venv:

```bash
pip uninstall -y opencv-python-headless opencv-python numpy
pip install "numpy==1.26.4" "opencv-python-headless==4.10.0.84"
```

### Playwright error: `cannot switch to a different thread`

Browser actions must run in the dedicated single browser worker. Restart the API after changing browser-control code:

```bash
pkill -f "python scripts/run_api.py"
pkill -f "uvicorn"
python scripts/run_api.py
```

### Google Calendar login blocked in Playwright

Use the private iCal URL instead of browser login. This is the supported path for daily briefing.

## Recommended first validation sequence

```bash
cd ~/avacore
source .venv/bin/activate

python -m py_compile avacore/api/http_app.py
python -m py_compile avacore/core/brain.py
python -m py_compile avacore/core/decision.py
python -m py_compile avacore/tools/web_research.py
python -m py_compile avacore/tools/calendar_ics.py
python -m py_compile avacore/tools/camera_rtsp.py
python -m py_compile avacore/channels/telegram/bot.py

python scripts/index_knowledge.py
python scripts/run_api.py
```

In a second terminal:

```bash
cd ~/avacore
source .venv/bin/activate
python scripts/run_telegram.py
```

Then test:

```bash
curl http://127.0.0.1:8787/health
curl http://127.0.0.1:8787/model
```

In Telegram:

```text
/health
/model
/briefing
/research D-Link DCS-5222L RTSP play1.sdp
```

## Current design principle

AvaCore should prefer local verified knowledge before using the web:

```text
Shared Brain
→ verified SQLite memories
→ local RAG / documents
→ web research only when current or external information is needed
→ research results become candidates, not trusted facts
```

This is the basis for Ava becoming a useful long-term project assistant instead of a stateless chatbot.

## myStrom Switch integration

AvaCore can control a local myStrom Switch directly through the LAN API.

This is useful for simple local smart-home actions such as switching a lamp on or off without using a cloud service.

### Configuration

In this case we used an plug from myStrom, because it hase a REST API
Add the switch IP to `.env`:

```env
# myStrom Switch / local smart plug
AVACORE_MYSTROM_IP=192.168.X.XXX
AVACORE_MYSTROM_TIMEOUT=2
```

### Telegram commands

The Telegram bot supports direct switch commands:

/switchon
/switchoff
/switchstate

### Safety note

The current myStrom integration is intentionally limited to explicit local switch actions.
More autonomous device control should be routed through the AvaCore decision/tool router and should require confirmation for risky actions.

## Telegram voice input

AvaCore can process Telegram voice messages.

Flow:

```text
Telegram voice message
→ audio file download
→ local Whisper transcription
→ transcribed text enters the normal AvaCore reply flow
```
This allows Roger to talk to Ava through Telegram.

Dependencies

System dependency:
```bash
sudo apt install ffmpeg
```

Python dependency:

```bash
pip install faster-whisper
```
### Configuration

AVACORE_VOICE_ENABLED=1
AVACORE_VOICE_MODEL=base
AVACORE_VOICE_DEVICE=cpu
AVACORE_VOICE_COMPUTE_TYPE=int8
AVACORE_VOICE_CACHE_DIR=./data/cache/voice
AVACORE_VOICE_LANGUAGE=de

For first tests, CPU mode is recommended:

AVACORE_VOICE_DEVICE=cpu
AVACORE_VOICE_COMPUTE_TYPE=int8

Later, CUDA can be tested:

AVACORE_VOICE_DEVICE=cuda
AVACORE_VOICE_COMPUTE_TYPE=float16

Voice messages are also routed through the local Telegram intent layer.
This means spoken commands such as:

Mach das Licht an
Schalte die Lampe aus
Ist das Licht an?

can control the myStrom Switch without sending the command through the LLM first.


### myStrom Switch

```md
## myStrom Switch control
```
AvaCore can control a local myStrom Switch through the LAN API.

### Configuration

```env
AVACORE_MYSTROM_IP=192.168.8.186
AVACORE_MYSTROM_TIMEOUT=2
```
### Telegram commands
/switchon
/switchoff
/switchstate

### Natural language

Telegram text and voice messages support simple local switch intents:

Mach das Licht an
Schalte die Lampe aus
Ist das Licht an?
Wie ist der Status der Steckdose?

These commands are handled locally by the Telegram bot before the normal /reply flow.
This keeps device control deterministic and avoids letting the LLM freely decide whether a device should be switched.

## Ava Notes

AvaCore includes a local notes system for capturing and managing short notes through Telegram.

Notes are stored locally in the AvaCore SQLite database and can later be synced/exported to a shared Google Doc.

This avoids giving Ava full access to Roger's private Google account. The recommended workflow is:

```text
Telegram text or voice
→ Ava Notes local SQLite storage
→ optional later sync to a shared Google Doc
```
### Telegram note commands
/note <text>                  Create a new note
/notes                        Show recent open notes
/notes open                   Show open notes
/notes done                   Show completed notes
/notes archived               Show archived notes
/notes all                    Show all notes
/notesearch <query>           Search notes
/noteadd <id> <text>          Append text to an existing note
/notedone <id>                Mark a note as done
/notearchive <id>             Archive a note
/notesync                     lacal Ava notes upload to Google Drive shared with Ava

### Natural language notes

Telegram text messages can also create notes directly:

Notiere: D405 Halterung nochmals prüfen
Ava, notiere: myStrom Switch funktioniert über Telegram
Mach eine Notiz: Kamera-Overlay oben für VLM wegschneiden

### Voice notes

Telegram voice messages can also create notes after speech-to-text transcription.

Example spoken command:

Ava, notiere: Morgen die D405 Halterung prüfen

## Ava Notes Google Drive Sync

AvaCore can export local Ava Notes to a Markdown file and sync it to Google Drive using `rclone`.

This avoids Google Cloud, OAuth app setup, API keys and credit-card requirements.

The local SQLite notes remain the source of truth. Google Drive is only used as an export/sync target.

### Sync architecture

```text
Ava Notes SQLite
→ Markdown export
→ local file: data/exports/ava_notes/Ava Notes.md
→ rclone
→ shared Google Drive folder: AvaCore
```

### Google Drive setup

Recommended setup:

Roger's Google Drive
└── AvaCore

The folder AvaCore is shared with:

email_of_your_bot@gmail.com

Because Google Drive shortcuts are not always handled like real folders by rclone, the preferred setup is to configure an rclone remote directly with the real Google Drive folder ID of the shared AvaCore folder.

Example remote name:

avacorenotes

Example .env configuration:

AVACORE_NOTES_EXPORT_ENABLED=1
AVACORE_NOTES_EXPORT_PATH=./data/exports/ava_notes/Ava Notes.md
AVACORE_NOTES_RCLONE_ENABLED=1
AVACORE_NOTES_RCLONE_REMOTE=avacorenotes:

### Telegram command

/notesync

This command:

1. Reads local Ava Notes from SQLite
2. Creates/updates the Markdown export file
3. Uploads the file to the configured Google Drive folder via rclone

Example result:

Notes Export abgeschlossen.

Lokale Datei:
data/exports/ava_notes/Ava Notes.md

Google Drive Ziel:
avacorenotes:

Sync: abgeschlossen.

### rclone test

Before using /notesync, test the remote manually:

echo "AvaCore shared folder test" > /tmp/ava-shared-test.txt
rclone copyto /tmp/ava-shared-test.txt "avacorenotes:Ava shared test.txt" --progress
rclone lsf avacorenotes:
rclone deletefile "avacorenotes:Ava shared test.txt"

The test file should appear in the real shared AvaCore folder in Roger's Google Drive.

### Security model

No Google Cloud project
No credit card
No Google Docs API
No full access to Roger's private Google account
No credentials committed to GitHub

Only the local rclone configuration has access to the shared folder.



## Visual Identity RAG / Local Person Recognition PoC

AvaCore includes an experimental local visual identity workflow for testing whether a known person can be recognized from camera snapshots.

This is not a general-purpose face recognition system and should not be treated as biometric certainty. The system is designed to be conservative:

```text
If Roger is not recognized with enough confidence:
→ return unknown
```

### The first supported identity is:

Roger

### Telegram commands

/idcapture roger     Capture the current camera image as a Roger example
/idcapture unknown   Capture the current camera image as a non-Roger / unknown example
/idcapture empty     Capture the current camera image as an empty-room example
/idtrain             Build the local identity vector index
/idcheck             Check the current camera image against the local identity 

### Pipeline

Camera snapshot
→ crop camera overlay
→ detect largest face
→ create face crop
→ create CLIP image embedding
→ store/search embeddings with FAISS
→ conservative threshold / margin / vote check
→ Roger or unknown


### Conservative recognition logic

Ava only returns roger when several checks pass:

Top match must be Roger
Roger similarity must be above threshold
Enough Roger votes must appear in the top-k neighbors
Margin to non-Roger examples must be large enough

Otherwise the result is:

unknown

This is intentional. It is safer to return unknown too often than to falsely identify another person as Roger.

### Training data

Recommended first test dataset:

Roger examples:        20–50 images
Unknown examples:      20–50 images
Empty-room examples:    5–20 images

Useful Roger variations:

near camera
farther away
sitting on the sofa
standing
frontal face
slightly side-facing
different lighting
different clothes

### Useful negative examples:

empty room
synthetic non-Roger faces
other persons, only with consent
hard negatives such as posters, screens, shadows or face-like objects

Synthetic faces can be useful for initial unknown/hard-negative testing. Real third-party face examples should only be captured with consent.

### Limitations

The current PoC uses OpenCV face detection plus CLIP image embeddings. It works best when the face is clearly visible. Small, heavily cropped, blurred or partially hidden faces may result in:

unknown
reason: no face detected

This is expected behavior for the first version.

### Configuration

Example .env settings:

AVACORE_IDENTITY_ENABLED=1
AVACORE_IDENTITY_DIR=./data/vision_identity
AVACORE_IDENTITY_MODEL=openai/clip-vit-base-patch32
AVACORE_IDENTITY_DEVICE=cpu
AVACORE_IDENTITY_THRESHOLD=0.90
AVACORE_IDENTITY_MARGIN=0.08
AVACORE_IDENTITY_TOP_K=5
AVACORE_IDENTITY_MIN_ROGER_VOTES=3

For more conservative behavior:

AVACORE_IDENTITY_THRESHOLD=0.93
AVACORE_IDENTITY_MARGIN=0.10
AVACORE_IDENTITY_MIN_ROGER_VOTES=4

For more permissive testing:

AVACORE_IDENTITY_THRESHOLD=0.85
AVACORE_IDENTITY_MARGIN=0.05
AVACORE_IDENTITY_MIN_ROGER_VOTES=2

### Privacy

Identity images, face crops and indexes are local runtime data and must not be committed to GitHub.

The identity dataset is ignored by Git:

data/vision_identity/

## Daily Mail Digest

AvaCore can send a daily Telegram summary of recent emails.

The scheduler script follows the same pattern as the daily calendar briefing:

```text
systemd user timer
→ scripts/send_daily_mail_digest.py
→ AvaCore API: GET /mail/digest
→ Telegram message to TELEGRAM_ALLOWED_CHAT_ID
```
The script does not read Gmail directly. Mail access is handled by the AvaCore API and the existing mail service configuration.

### Configuration

AVACORE_API_URL=http://127.0.0.1:8787
TELEGRAM_BOT_TOKEN=
TELEGRAM_ALLOWED_CHAT_ID=

AVACORE_MAIL_DIGEST_LIMIT=8

The mail account itself is configured through the existing AvaCore mail settings:

AVACORE_MAIL_IMAP_HOST=imap.gmail.com
AVACORE_MAIL_IMAP_PORT=993
AVACORE_MAIL_USERNAME=
AVACORE_MAIL_PASSWORD=

### Manual test

Start the AvaCore API first:
```bash
python scripts/run_api.py
```
Then run:
```bash
python scripts/send_daily_mail_digest.py
```

### systemd user timer

The mail digest can be scheduled at 18:00 with:
```bash
nano ~/.config/systemd/user/avacore-mail-digest.service
```

[Unit]
Description=Run AvaCore daily mail digest at 18:00

[Timer]
OnCalendar=*-*-* 18:00:00
Persistent=true
Unit=avacore-mail-digest.service

[Install]
WantedBy=timers.target

### activate
```bash
systemctl --user daemon-reload
### systemd user service and timer

The daily mail digest requires both the AvaCore API service and the
mail-digest timer.

Service file:

```ini
# ~/.config/systemd/user/avacore-mail-digest.service

[Unit]
Description=AvaCore daily mail digest
Wants=avacore-api.service
After=avacore-api.service

[Service]
Type=oneshot
WorkingDirectory=/home/ares/avacore
ExecStart=/home/ares/avacore/.venv/bin/python -u /home/ares/avacore/scripts/send_daily_mail_digest.py
TimeoutStartSec=5min
```
Timer file:

# ~/.config/systemd/user/avacore-mail-digest.timer

[Unit]
Description=Run AvaCore daily mail digest at 18:00 Europe/Zurich

[Timer]
OnCalendar=*-*-* 18:00:00 Europe/Zurich
Persistent=true
AccuracySec=1min
Unit=avacore-mail-digest.service

[Install]
WantedBy=timers.target

Enable the API and timer:
```bash
systemctl --user daemon-reload
systemctl --user enable --now avacore-api.service
systemctl --user enable --now avacore-mail-digest.timer
systemctl --user restart avacore-mail-digest.timer
```
For execution without an active login session:
```ash
sudo loginctl enable-linger ares
```
Manual systemd test:
```bash
systemctl --user start avacore-mail-digest.service
journalctl --user -u avacore-mail-digest.service -n 100 --no-pager
```
Check the next run:
```bash
systemctl --user list-timers --all | grep avacore-mail-digest
```

## Dynamic Conscious Workspace / JSpace

AvaCore includes a first minimal implementation of the Dynamic Conscious Workspace, called **JSpace**.

JSpace is not a separate memory and not a separate intelligence. It is a dynamic activation field that represents Ava's current cognitive focus across the larger AvaCore context.

The current Phase 1 implementation is intentionally conservative:

- JSON-based state storage
- no autonomous background process
- no direct modification of verified long-term memory
- no permanent identity or goal changes without Roger's review
- injection of the current top activated JSpace items into the `/reply` 

system prompt
- debug visibility through `/debug/jspace`

This file is private runtime state and must not be committed.

Configuration:

AVACORE_JSPACE_ENABLED=1
AVACORE_JSPACE_PATH=./data/state/jspace.json
AVACORE_JSPACE_TOP_K=8
AVACORE_JSPACE_DECAY=0.92
AVACORE_JSPACE_MIN_ACTIVATION=0.05
AVACORE_JSPACE_FOCUS_MODE=balanced

Available focus modes for Phase 1:

balanced
narrow
wide
watchful

The JSpace is updated during normal /reply calls:

user message
→ JSpace tick / decay
→ user signal activation
→ top active JSpace items
→ system prompt injection
→ model response
→ assistant response activation

Debug endpoint:

curl -s http://127.0.0.1:8787/debug/jspace \\
  -H "X-Admin-Password: YOUR_ADMIN_PASSWORD" | jq

The architecture document is stored in:

docs/JSPACE.md

Safety boundary:

JSpace may influence focus.
JSpace may suggest what appears relevant.
JSpace must not silently rewrite durable identity, goals or verified long-term memory.

Durable memory still follows AvaCore's reviewed memory workflow:

candidate → verified → usable as trusted long-term context


