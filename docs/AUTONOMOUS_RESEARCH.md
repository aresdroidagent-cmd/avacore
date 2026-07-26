# Autonomous Research (controlled phase 1)

This phase adds a bounded scheduler around AvaCore's existing JSpace, web
research, Ollama and candidate-memory components. It does not introduce a
parallel search stack and it never verifies memories automatically.

## Architecture

```text
JSpace top items
  -> deterministic filtering and scoring
  -> data/state/research_queue.json
  -> at most one selected topic
  -> existing collect_research_sources()
  -> local Ollama summary
  -> memory_items candidate
  -> JSpace research/finding
  -> optional Telegram notification
```

`research_queue.py` owns atomic persistence, stable IDs, statuses, cooldowns and
run history. `research_curiosity.py` owns deterministic topic derivation and
scoring. `autonomous_research.py` enforces policy and budgets and coordinates
existing components.

## Freedom modes

`AVACORE_RESEARCH_ENABLED` is the master switch for all web research.
`AVACORE_AUTO_RESEARCH` controls only autonomous behavior:

- `off`: no derivation and no execution.
- `ask`: derive queue candidates, then return `approval_required`; do not search.
- `bounded`: derive and execute at most one eligible topic per scheduler call.

`ask` is the safe default. Return from `bounded` at any time by setting
`AVACORE_AUTO_RESEARCH=ask` or disable derivation with
`AVACORE_AUTO_RESEARCH=off`, then restart the API.

## Queue and scoring

The queue defaults to `data/state/research_queue.json` and is written through a
temporary file followed by atomic replacement. Runtime state remains ignored by
Git. Supported topic statuses are `candidate`, `pending`, `running`,
`completed`, `failed` and `dismissed`.

Each component is clamped to 0..1:

```text
base =
    activation     * 0.30
  + priority       * 0.25
  + knowledge_gap  * 0.20
  + freshness_need * 0.15
  + urgency        * 0.10
```

Activation and priority come from JSpace. Knowledge gap uses explicit question
and project signals. Freshness and urgency use documented lexical signals.
Curiosity uses question/tag diversity and is added only into the remaining
headroom. Its configured value is additionally capped to 25 percent of its
nominal weight so it cannot overtake project relevance. All inputs remain
visible in `score_components`.

Identity anchors, operating rules, greetings, short content, existing findings,
normalized duplicates and topics inside the cooldown are not executable
research topics.

## Budgets and safety boundaries

Defaults:

```env
AVACORE_RESEARCH_MAX_RUNS_PER_DAY=3
AVACORE_RESEARCH_MAX_TOPICS_PER_RUN=1
AVACORE_RESEARCH_MAX_SOURCES_PER_TOPIC=5
AVACORE_RESEARCH_MIN_SCORE=0.65
AVACORE_RESEARCH_COOLDOWN_HOURS=24
```

The phase-one scheduler always executes at most one topic, even if
`AVACORE_RESEARCH_MAX_TOPICS_PER_RUN` is configured higher. Run history counts
attempted runs, including failed attempts, against the daily UTC budget.

The loop only searches and reads public HTML sources. It does not install or
execute software, submit forms, send email, call paid services, recursively
research, change identity/goals, or verify memory candidates.

## Memory review and JSpace feedback

Successful results create exactly one `memory_items` row per queue topic:

```text
scope=user
memory_type=research_lead
status=candidate
source_type=autonomous_research
source_ref=<research-topic-id>
```

Review it through the existing memory review API/UI. Only Roger's existing
verify action can make it durable. A successful result also injects a bounded
`source=research`, `kind=finding` item into the existing JSpace.

## Telegram

Telegram is attempted only when the topic score meets
`AVACORE_RESEARCH_NOTIFY_SCORE` and both `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_ALLOWED_CHAT_ID` are set. A Telegram failure is reported as
`notification_error` but does not turn a completed research run into a failure.

## Administrative API tests

All endpoints require the existing `X-Admin-Password`:

```bash
export AVA_ADMIN_PASSWORD='your local admin password'

curl -sS \
  -H "X-Admin-Password: ${AVA_ADMIN_PASSWORD}" \
  http://127.0.0.1:8787/debug/research_queue

curl -sS -X POST \
  -H "X-Admin-Password: ${AVA_ADMIN_PASSWORD}" \
  http://127.0.0.1:8787/research/autonomous/derive

curl -sS -X POST \
  -H "X-Admin-Password: ${AVA_ADMIN_PASSWORD}" \
  http://127.0.0.1:8787/research/autonomous/run-next

curl -sS -X POST \
  -H "X-Admin-Password: ${AVA_ADMIN_PASSWORD}" \
  http://127.0.0.1:8787/research/autonomous/topics/RESEARCH_TOPIC_ID/dismiss
```

Expected scheduler statuses are `disabled`, `idle`, `approval_required`,
`budget_exhausted`, `completed` and `failed`.

## systemd user timer

The repository contains examples only. Review paths before installing:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/user/avacore-autonomous-research.service ~/.config/systemd/user/
cp deploy/systemd/user/avacore-autonomous-research.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now avacore-autonomous-research.timer
systemctl --user start avacore-autonomous-research.service
systemctl --user status avacore-autonomous-research.timer
journalctl --user -u avacore-autonomous-research.service -n 100 --no-pager
```

Disable it without deleting queue or memories:

```bash
systemctl --user disable --now avacore-autonomous-research.timer
```

These commands are documentation only; repository setup does not run them.
