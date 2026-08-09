# Ava Conscious Workspace — Phase 1

This is an engineering architecture for conscious access and cognitive continuity. It does not claim subjective or phenomenological consciousness.

## Goal and terms

JSpace is the persistent **Cognitive Field**: all currently available representations. Local fields (`identity`, `conversation`, `memory`, `knowledge`, `research`, `goal`, `system`, and `reasoning`) identify candidate origins. The **Conscious Workspace** is the small globally available subset selected by deterministic activation and competition. Attention answers: “What matters now?”

Activation is relevance, not truth or authority. Identity anchors and verified memories can have high confidence; retrieval hits and research findings carry source-dependent confidence; user statements and prior assistant responses remain unverified context.

## Cycle, activation, and competition

Each event-driven cycle loads JSpace, applies source-sensitive decay, injects the user stimulus and retrieved candidates, scores them, applies capacity/diversity limits, persists a snapshot atomically, and renders the selected subset as `CONSCIOUS WORKSPACE` for reasoning.

Focused-mode score:

```text
0.30 relevance + 0.20 priority + 0.15 recency + 0.10 persistence
+ 0.10 confidence + 0.10 novelty + 0.05 urgency
```

All inputs and the result are clamped to `0..1`. `associative` increases novelty and capacity latitude; `urgent` emphasizes priority and urgency. Conversation decays fastest, research/memory more slowly, goals slowly, and the identity anchor effectively does not decay. Current user input and the identity anchor are mandatory; per-source/per-kind limits prevent retrieval flooding.

## `/reply` flow

Previously, Memory, RAG, JSpace, conversation history, and Identity supplied independent context paths. Now verified Memory and selected RAG hits become `CognitiveEntity` candidates alongside Conversation and Identity. Competition selects one workspace block. The hard Identity Guard, personality, Shared Brain, safety rules, and language rules remain system authority outside the workspace. Dynamic Memory/RAG blocks are not duplicated.

After answering, `assistant_response` returns to JSpace with low confidence. It is context, never a verified fact. Deterministic identity questions still bypass the LLM, but their workspace cycle remains observable.

## Research feedback

Manual and autonomous results become `research/research_finding` entities while retaining the existing candidate-memory workflow. Autonomous topic derivation prefers current workspace items, while `off`, `ask`, `bounded`, daily budget, maximum topics, and cooldown safeguards remain unchanged.

## Snapshot and inspection

`data/state/conscious_workspace.json` stores the current snapshot plus a bounded 20-cycle summary history using atomic replacement. It records scores, components, selection explanations, active/latent state, focus changes, and a non-semantic `operational_layout_v1` projection.

- `GET /debug/workspace` — protected current snapshot
- `GET /debug/workspace/history` — protected bounded history
- `GET /debug/jspace` — compatible raw field view
- `/ui/workspace` — polling focus field, legend, history, details, filters, and technical table

The 2D geometry encodes operational selected/latent state and activation only. `semantic=false`; it does not imply embedding proximity.

## Phase 1 limits and Phase 2

Phase 1 is event-driven and deterministic. It has no emotion, vision/robotics field implementation, continuous thought loop, extra selector LLM call, semantic projection, or automatic memory verification. Phase 2 can add embedding-based clustering/projection, richer goal and uncertainty models, event ingestion beyond replies/research, and calibrated source authority without changing the field/workspace boundary.
