# JSpace Phase 2 — Cognitive Continuity

Phase 2 is an event-driven engineering architecture for continuity and conscious access. It does not claim subjective consciousness and does not run a continuous thought loop.

## Architecture

Each `/reply` request receives one `cycle_id` and follows this path:

```text
user stimulus → persistent Working Memory → Cognitive Field/JSpace
→ deterministic Pre-LLM Spotlight → Conscious Workspace → LLM or resolver
→ deterministic Post-LLM Gate → response + assimilation → new state
```

The existing JSpace switch remains the single feature switch. Existing JSpace JSON remains loadable: absent Phase-2 fields receive bounded defaults, and writes use atomic temporary-file replacement.

## Self Model

The persistent Self Model records Ava's configured name, AvaCore system name, role, underlying model, Ollama runtime, confidence, persistence, and system authority. Configuration is authoritative for deployment-specific identity fields. A compact self state is always available; technical model/runtime detail is expanded when deterministic self affinity is high.

**The LLM is a reasoning component of Ava. It is not the persistent identity, memory or cognitive state of Ava.**

**AvaCore maintains continuity. The language model processes the currently active conscious workspace.**

## Working Memory

Working Memory is a separate bounded, atomic JSON store keyed by the existing `/reply` `session_id` (`channel:chat_id`). Defaults are 24 persistent items and 10 active items per session. It stores the current user input, recent user/assistant turns, decisions, topic, current task, unresolved questions, cycle ownership, importance, activation, and timestamps. One session cannot activate another session's short-term conversation. The Self Model remains global and identical across sessions.

Version-1 Working Memory files without a `session_id` are migrated without deletion into the documented `legacy/default` scope. New writes use the version-2 `sessions` map.

Active selection is transparent:

```text
working_score = 0.45 recency + 0.35 lexical/topic relevance + 0.20 importance
```

Important decisions decay more slowly. Capacity eviction ranks activation and importance instead of blindly retaining the last N messages. No LLM or external NLP library is used.

`current_task` is only set for explicit signals such as “Wir müssen”, “Als nächstes”, `TODO`, or “We need to”. `unresolved_questions` accepts explicit open-question markers and questions emitted by Ava. Ambiguous prose produces no state.

## Cognitive Field and attention

JSpace items expose normalized activation, relevance, recency, self affinity, continuity, goal affinity, confidence, novelty, urgency, persistence, and authority. Competition uses:

```text
attention_score = 0.25 relevance + 0.20 recency + 0.15 continuity
                + 0.15 self_affinity + 0.10 priority + 0.05 persistence
                + 0.05 confidence + 0.05 urgency
```

The Phase-1 `activation_score` API remains compatible for callers, while the Spotlight uses the new attention score. Debug items expose every score input and the complete `cognitive_state` vector.

Decay is source-sensitive: `AVACORE_WORKSPACE_DECAY_CONVERSATION` controls conversation and Working Memory, while `AVACORE_WORKSPACE_DECAY_GENERAL` controls ordinary non-identity field items unless a more specific memory/research/goal policy applies. Goals are slow, and the identity anchor has effectively no decay. Decay changes activation and never deletes an item merely for becoming inactive. Memory, knowledge, and research continue to enter as lower-base-activation candidates.

## Pre-LLM Spotlight and context

`build_conscious_workspace` / `run_pre_llm_spotlight` loads state, applies decay, injects the stimulus and candidates, updates lexical relevance, runs bounded competition and diversity limits, and selects top-K active representations. Current input is mandatory. The system prompt contains a single operational block with `CURRENT SELF`, `CURRENT FOCUS`, `WORKING MEMORY`, and `ACTIVE CONTEXT`. Previous assistant responses and user assertions are explicitly unverified context.

Self affinity uses bilingual token combinations for second-person, identity, agent, and model concepts. A bare model-installation question remains weak; combinations such as “who are you?” or “if Gemma is your model, who are you?” strongly activate Self.

Safety, source authority, identity, memory state, and attention belong to AvaCore. The model retains freedom over interpretation, reasoning, association, and wording inside those boundaries. Normal answers make at most one backend chat call; attention, topic detection, Working Memory, gate, and assimilation make none.

## Post-LLM gate and assimilation

`run_post_llm_gate` detects only clear first-person claims such as “Ich bin Gemma” or “I am Ollama.” Benign descriptions such as “Gemma ist mein Hintergrundmodell” pass. A conflict is locally repaired while preserving useful remaining text; if repair cannot be made safe, a short Self-Model-consistent answer is used. Debug records `identity_conflict` without storing the rejected claim as authoritative identity.

The accepted assistant response is then added to Working Memory and JSpace with low authority/confidence. Conservative regex signals may mark clear decisions; topic detection uses known deterministic tags and otherwise keeps `null` or a supported previous topic. Pre/post focus and whether it changed are retained.

## Performance, debug, and UI

Local `perf_counter` instrumentation records `workspace_pre_ms`, `llm_ms`, `workspace_post_ms`, and `total_ms`; no telemetry leaves AvaCore and tests impose no unstable wall-clock threshold.

- `GET /debug/workspace` returns Self Model, Working Memory, pre/post snapshots, focus, gate result, and timing.
- `GET /debug/workspace/history` returns a bounded default history of 20 completed cycle summaries.
- `/ui/workspace` polls every two seconds and shows headline status, Self Model, Working Memory, the non-semantic 2D Spotlight, Cognitive Cycle, history, and detailed score components.

The 2D layout encodes activation and active/latent state only. It is not semantic geometry.

## Current limits

Phase 2 deliberately does not rebuild long-term memory, RAG, research, emotion, vision, robotics, or retrieval embeddings. Topic extraction is conservative and lexical; pronoun resolution works through recently active Working Memory rather than general linguistic parsing. A later phase can deepen retrieval-to-field calibration and goal modeling without moving persistent identity or continuity into the LLM.

## Legacy Phase-1 API classification

- `GET /debug/jspace`, `GET /debug/workspace/history`, `activation_score`, and JSpace serialization helpers are compatibility/read-only/debug surfaces and remain supported.
- `POST /debug/workspace/cycle` is an explicit admin debug override, not a `/reply` prompt path.
- Research-completion workspace cycles are event ingestion into the same field core; they do not build a competing reply prompt or call an attention LLM.
- `update_jspace_from_assistant_response` and the legacy JSpace prompt helper remain compatibility APIs. The `/reply` handler does not use them when a Phase-2 cycle exists; its only authoritative flow is session Working Memory → JSpace → Spotlight → workspace → one LLM/resolver → gate → assimilation.
