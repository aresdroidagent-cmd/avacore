import json

from avacore.core.cognitive_workspace import (
    activation_score,
    read_workspace_history,
    run_workspace_cycle,
    workspace_prompt,
)
from avacore.core.jspace import JSpaceItem, JSpaceState


def cycle(tmp_path, text, candidates=(), **kwargs):
    return run_workspace_cycle(
        jspace_path=tmp_path / "jspace.json",
        workspace_path=tmp_path / "workspace.json",
        stimulus=text,
        candidates=candidates,
        **kwargs,
    )


def test_user_and_identity_are_active_and_identity_focus_is_contextual(tmp_path):
    snapshot = cycle(tmp_path, "Wie funktioniert der Scheduler?")
    assert any(item["kind"] == "user_input" for item in snapshot.active_items)
    anchor = next(item for item in snapshot.active_items if item["kind"] == "identity_anchor")
    assert anchor["persistence"] == 1.0
    assert snapshot.active_items[0]["kind"] != "identity_anchor"

    identity = cycle(tmp_path, "Wer bist du?")
    assert identity.active_items[0]["kind"] == "identity_anchor"


def test_memory_knowledge_research_compete_and_irrelevant_memory_is_latent(tmp_path):
    candidates = [
        {"source":"memory","kind":"memory","content":"Scheduler uses a daily budget", "relevance":.9,"confidence":.9},
        {"source":"memory","kind":"memory","content":"Bananas are yellow", "relevance":0,"confidence":.9},
        {"source":"knowledge","kind":"knowledge_hit","content":"Autonomous scheduler source code", "relevance":.95,"source_ref":"chunk:1"},
        {"source":"research","kind":"research_finding","content":"Scheduler cooldown finding", "relevance":.85,"source_ref":"topic:1"},
    ]
    snapshot = cycle(tmp_path, "Explain the scheduler", candidates, min_activation=.35)
    active = {(item["source"], item["content"]) for item in snapshot.active_items}
    assert any(source == "knowledge" for source, _ in active)
    assert any(source == "research" for source, _ in active)
    assert any("daily budget" in content for source, content in active if source == "memory")
    assert any("Bananas" in item["content"] for item in snapshot.latent_items)


def test_decay_limits_diversity_scores_reasons_history_and_atomic_write(tmp_path):
    many = [{"source":"knowledge","kind":"knowledge_hit","content":f"scheduler hit {n}","relevance":.95,"source_ref":f"chunk:{n}"} for n in range(10)]
    first = cycle(tmp_path, "scheduler", many, max_active_items=5, max_per_source=2)
    assert len(first.active_items) <= 5
    assert sum(item["source"] == "knowledge" for item in first.active_items) <= 2
    old_user = next(item for item in first.active_items if item["kind"] == "user_input")
    second = cycle(tmp_path, "different topic", decay_factor=.5)
    prior = next(item for item in second.active_items + second.latent_items if item["id"] == old_user["id"])
    assert prior["activation"] < old_user["activation"]
    assert all(0 <= item["activation_score"] <= 1 and item["score_components"] and item["selection_reason"] for item in second.active_items + second.latent_items)
    assert second.focus_changed
    assert len(read_workspace_history(tmp_path / "workspace.json")) == 2
    assert not (tmp_path / "workspace.json.tmp").exists()


def test_history_is_bounded_and_legacy_jspace_migrates(tmp_path):
    path = tmp_path / "jspace.json"
    path.write_text(json.dumps({"version":1,"items":[{"source":"identity","kind":"self_anchor","content":"Ava identity"}]}))
    state = JSpaceState.load(path)
    assert next(iter(state.items.values())).kind == "identity_anchor"
    for n in range(4):
        cycle(tmp_path, f"topic {n}", history_limit=2)
    assert len(read_workspace_history(tmp_path / "workspace.json")) == 2


def test_assistant_response_is_unverified_and_prompt_has_single_workspace(tmp_path):
    state = JSpaceState()
    response = state.inject_assistant_response("I am Gemma4")
    assert response.confidence < .5
    snapshot = cycle(tmp_path, "What are you?")
    prompt = workspace_prompt(snapshot)
    assert prompt.count("CONSCIOUS WORKSPACE") == 1
    assert "not verified facts" in prompt


def test_score_is_bounded():
    item = JSpaceItem(id="x", source="system", kind="system_state", content="x", activation=99, priority=99, persistence=99, confidence=99, relevance=99, novelty=99, recency=99, urgency=99)
    score, components = activation_score(item)
    assert score == 1.0
    assert set(components) == {"relevance", "priority", "recency", "persistence", "confidence", "novelty", "urgency"}
