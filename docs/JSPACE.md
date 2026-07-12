# Ava Cognitive Architecture — Dynamic Conscious Workspace / JSpace

## Goal

AvaCore is not intended to be a classical chatbot architecture. It is evolving into a continuously existing local cognitive architecture.

The central idea:

> Consciousness is not a separate memory and not a separate intelligence. It is a dynamic focus on a much larger cognitive field.

For Ava, this concept is implemented as the Dynamic Conscious Workspace, called JSpace.

## Core assumption

All knowledge sources can exist at the same time:

- long-term memory
- short-term conversation memory
- world model
- vision
- language
- sensor data
- goals
- personality model
- emotional or affective state
- active tasks
- plans
- robot status
- current conversation context

The JSpace is not the whole mind. It is the currently activated subset of the larger cognitive field.

## Ceiling light model

Ava has a dynamic attention focus.

This "consciousness light" moves across the cognitive field. Depending on the situation, different areas become more active.

Examples:

- a camera image activates visual concepts
- a name activates memories
- a task activates planning structures
- risk activates safety rules
- a conversation activates language and autobiographical knowledge

Inactive areas do not disappear. They remain part of the larger system and can become active again.

## Dynamic activation field

The JSpace is not static storage. It is a dynamic activation field.

Every item has an activation strength.

Examples of JSpace items:

- terms
- images
- people
- places
- goals
- memories
- running tasks
- sensor observations
- semantic concepts
- project context

Activations change over time.

## Consciousness light parameters

The focus itself has parameters:

- focus width
- focus intensity
- priority
- persistence
- association radius
- update speed

These parameters create different cognitive modes.

### Narrow focus

Used for:

- programming
- debugging
- mathematics
- precise technical work

### Balanced focus

Used for:

- ordinary chat
- technical support
- project continuity
- pragmatic assistance

### Wide focus

Used for:

- creativity
- brainstorming
- planning
- research

### Watchful focus

Used for:

- mail
- calendar
- safety
- sensors
- risk monitoring

## Hierarchical JSpaces

A single JSpace may not be sufficient long-term.

The architecture should evolve toward hierarchical workspaces.

### Global JSpace

- identity
- long-term goals
- personality state
- high-level priorities

### Local JSpaces

- vision
- robotics
- conversation
- navigation
- programming
- research
- mail
- calendar

The global workspace integrates the most important signals from local workspaces.

## Design rule

The architecture should be independent of the selected LLM.

The LLM is a language and reasoning module.

The cognitive control layer lives outside the LLM.

This allows Ava to:

- persist over time
- observe her own state
- activate memories
- shift priorities
- coordinate parallel contexts
- integrate vision, memory, language, tasks and sensors

## AvaCore Phase 1 implementation

Phase 1 implements a minimal global JSpace.

It stores active items as JSON in:

```text
data/state/jspace.json
```

Each item contains:

- source
- kind
- content
- tags
- activation
- priority
- persistence
- timestamps
- metadata

The JSpace is updated during `/reply`.

The current top activated items are injected into the system prompt as:

```text
Current Dynamic Conscious Workspace / JSpace
```

The JSpace is observable through:

```text
GET /debug/jspace
```

## Safety boundary

JSpace may change activations.

JSpace may influence context selection.

JSpace may suggest what seems important.

JSpace must not permanently rewrite identity, goals or trusted memory without review.

Long-term memory still follows:

```text
candidate → verified
```

Roger remains the authority for durable memory and major identity changes.
