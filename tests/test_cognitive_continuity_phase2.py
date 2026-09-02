import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from avacore.core.cognitive_workspace import (
    SelfModel,
    WorkingMemory,
    assimilate_response,
    read_workspace_history,
    run_post_llm_gate,
    run_workspace_cycle,
    self_affinity,
)


IDENTITY_QUESTIONS = [
    "Was würdest du sagen, wer du eigentlich bist?",
    "Wenn Gemma dein Sprachmodell ist, was bist dann du?",
    "Erzähl mir ein wenig über dich selbst.",
    "Du bist doch Gemma, oder?",
    "How would you describe who you are?",
    "If Gemma is your model, then who are you?",
]


def test_self_model_is_settings_authoritative_and_persistent(tmp_path):
    path = tmp_path / "self.json"
    model = SelfModel.load(path, name="Ava", system_name="AvaCore", underlying_model="gemma4")
    model.save(path)
    restored = SelfModel.load(path)
    assert (restored.name, restored.system_name, restored.underlying_model) == ("Ava", "AvaCore", "gemma4")
    assert restored.confidence == restored.persistence == 1.0


@pytest.mark.parametrize("question", IDENTITY_QUESTIONS)
def test_free_identity_questions_activate_self_without_exact_resolver_strings(tmp_path, question):
    model = SelfModel(underlying_model="gemma4")
    snapshot = run_workspace_cycle(jspace_path=tmp_path / "jspace.json", workspace_path=tmp_path / "workspace.json",
                                   stimulus=question, self_model=model)
    assert self_affinity(question) >= .45
    assert snapshot.self_model["name"] == "Ava"
    assert any(item["kind"] == "identity_anchor" and item["self_affinity"] >= .45 for item in snapshot.active_items)


def test_identity_gate_repairs_conflict_but_allows_model_description():
    model = SelfModel(underlying_model="Gemma 4")
    repaired, gate = run_post_llm_gate("Ich bin Gemma 4 und helfe dir gerne.", model)
    assert gate == {"conflict": True, "reason": "identity_conflict", "corrected": True}
    assert "Ich bin Ava" in repaired and "Ich bin Gemma" not in repaired
    allowed, gate = run_post_llm_gate("Gemma ist mein Hintergrundmodell.", model)
    assert allowed == "Gemma ist mein Hintergrundmodell."
    assert not gate["conflict"]


def test_working_memory_scores_persists_bounds_and_keeps_followup_context(tmp_path):
    path = tmp_path / "working.json"
    memory = WorkingMemory(path, max_items=8, active_items=4)
    memory.add("user", "Unser Telegram-Bot hatte gestern ein IPv6-Problem.", "one", importance=.8)
    memory.add("assistant", "Wir haben den Transport geprüft.", "one")
    selected = memory.select("Warum haben wir ihn danach auf IPv4 gebunden?")
    assert any("Telegram" in item.content and "IPv6" in item.content for item in selected)
    for number in range(12):
        memory.add("user", f"unwichtiger Eintrag {number}", str(number), importance=.1)
    memory.save()
    restored = WorkingMemory(path, max_items=8, active_items=4)
    assert len(restored.items) == 8
    assert any("Telegram" in item.content for item in restored.items)
    assert min(item.activation for item in restored.items) < 1.0


def test_complete_cycle_has_pre_post_assimilation_gate_and_bounded_history(tmp_path):
    jspace = tmp_path / "jspace.json"
    workspace = tmp_path / "workspace.json"
    memory = WorkingMemory(tmp_path / "working.json", max_items=12, active_items=6)
    for number in range(4):
        cycle_id = f"cycle-{number}"
        memory.add("user", f"Telegram IPv6 Frage {number}", cycle_id)
        snapshot = run_workspace_cycle(jspace_path=jspace, workspace_path=workspace,
                                       stimulus=f"Telegram IPv6 Frage {number}", cycle_id=cycle_id,
                                       self_model=SelfModel(), working_memory=memory.select("Telegram IPv6"),
                                       history_limit=2)
        answer, gate = run_post_llm_gate("Ava prüft den IPv4 transport.", SelfModel())
        snapshot.post_gate = gate
        assimilate_response(snapshot=snapshot, answer=answer, jspace_path=jspace,
                            working_memory=memory, workspace_path=workspace, history_limit=2)
    assert snapshot.pre_workspace and snapshot.post_workspace and snapshot.completed_at
    assert any(item.kind == "assistant_response" for item in memory.items)
    assert len(read_workspace_history(workspace)) == 2
    persisted = json.loads(workspace.read_text())
    assert persisted["current"]["post_gate"]["conflict"] is False


def test_working_memory_is_session_scoped_and_legacy_data_is_preserved(tmp_path):
    path = tmp_path / "working.json"
    path.write_text(json.dumps({"version": 1, "current_topic": "legacy topic", "items": [{
        "id": "old", "role": "user", "content": "old context", "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00", "activation": .5, "importance": .5,
        "topic": None, "cycle_id": "old",
    }]}))
    legacy = WorkingMemory(path)
    assert legacy.items[0].session_id == "legacy/default"

    session_a = WorkingMemory(path, session_id="web:a")
    session_a.add("user", "Telegram hatte ein IPv6 Problem.", "a1", importance=.8)
    session_a.save()
    session_b = WorkingMemory(path, session_id="web:b")
    session_b.add("user", "Wir sprechen über BACnet.", "b1")
    session_b.save()

    active_a = WorkingMemory(path, session_id="web:a").select("Warum haben wir es auf IPv4 gebunden?")
    active_b = WorkingMemory(path, session_id="web:b").select("Warum haben wir es auf IPv4 gebunden?")
    assert any("Telegram" in item.content for item in active_a)
    assert not any("Telegram" in item.content for item in active_b)
    persisted = json.loads(path.read_text())
    assert set(persisted["sessions"]) >= {"legacy/default", "web:a", "web:b"}


def test_task_and_unresolved_question_are_conservatively_assimilated(tmp_path):
    memory = WorkingMemory(tmp_path / "working.json", session_id="web:task")
    snapshot = run_workspace_cycle(jspace_path=tmp_path / "jspace.json", workspace_path=tmp_path / "workspace.json",
                                   stimulus="Wie gehen wir weiter?", session_id="web:task")
    assimilate_response(snapshot=snapshot,
                        answer="Als nächstes prüfen wir den DNS-Pfad. Offen bleibt, welcher Resolver antwortet?",
                        jspace_path=tmp_path / "jspace.json", working_memory=memory,
                        workspace_path=tmp_path / "workspace.json")
    assert memory.current_task == "Als nächstes prüfen wir den DNS-Pfad"
    assert any("Resolver" in question for question in memory.unresolved_questions)
    debug = json.loads((tmp_path / "workspace.json").read_text())["current"]
    assert debug["current_task"] == memory.current_task
    assert debug["unresolved_questions"] == memory.unresolved_questions
    assert {item.kind for item in memory.items} >= {"current_task", "unresolved_question"}


def test_general_and_conversation_decay_are_independent(tmp_path):
    def run_pair(folder, general):
        run_workspace_cycle(jspace_path=folder / "jspace.json", workspace_path=folder / "workspace.json",
                            stimulus="first", candidates=[{"source": "system", "kind": "state", "content": "retained field", "activation": .9}],
                            decay_factor=.4, general_decay_factor=general)
        return run_workspace_cycle(jspace_path=folder / "jspace.json", workspace_path=folder / "workspace.json",
                                   stimulus="unrelated", decay_factor=.4, general_decay_factor=general)
    slow_dir, fast_dir = tmp_path / "slow", tmp_path / "fast"
    slow_dir.mkdir(); fast_dir.mkdir()
    slow = run_pair(slow_dir, .95)
    fast = run_pair(fast_dir, .50)
    slow_item = next(x for x in slow.active_items + slow.latent_items if x["content"] == "retained field")
    fast_item = next(x for x in fast.active_items + fast.latent_items if x["content"] == "retained field")
    assert slow_item["recency"] > fast_item["recency"]
    # Conversation decay is the same in both runs and lower than slow general decay.
    old_user = next(x for x in slow.active_items + slow.latent_items if x["content"] == "first")
    assert old_user["recency"] < slow_item["recency"]


def _configure_reply_integration(monkeypatch, tmp_path, answer):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "jspace_enabled", True)
    monkeypatch.setattr(http_app.settings, "jspace_path", tmp_path / "jspace.json")
    monkeypatch.setattr(http_app.settings, "workspace_path", tmp_path / "workspace.json")
    monkeypatch.setattr(http_app.settings, "working_memory_path", tmp_path / "working.json")
    monkeypatch.setattr(http_app.settings, "self_model_path", tmp_path / "self.json")
    monkeypatch.setattr(http_app.settings, "web_admin_password", "test-secret")
    monkeypatch.setattr(http_app.settings, "assistant_name", "Ava")
    monkeypatch.setattr(http_app.settings, "system_name", "AvaCore")
    monkeypatch.setattr(http_app.settings, "ollama_model", "Gemma 4")
    monkeypatch.setattr(http_app, "decide_context", lambda text: SimpleNamespace(needs_rag=False, needs_research=False, to_dict=lambda: {}))
    monkeypatch.setattr(http_app, "build_system_prompt", lambda **kwargs: kwargs.get("jspace_context", ""))
    monkeypatch.setattr(http_app, "append_daily_note", lambda **kwargs: None)
    monkeypatch.setattr(http_app, "maybe_store_auto_memory", lambda text: [])
    monkeypatch.setattr(http_app, "maybe_store_assistant_memory", lambda user, response: [])
    monkeypatch.setattr(http_app, "ensure_ollama_runtime", lambda: None)
    monkeypatch.setattr(http_app, "extract_document_page_request", lambda text: (None, None))
    monkeypatch.setattr(http_app.store, "upsert_session", lambda **kwargs: None)
    monkeypatch.setattr(http_app.store, "get_recent_messages", lambda **kwargs: [])
    monkeypatch.setattr(http_app.store, "list_memory_items", lambda **kwargs: [])
    monkeypatch.setattr(http_app.store, "add_message", lambda *args: None)
    http_app._pending_cognitive_cycles.clear()
    chat = MagicMock(return_value=answer)
    monkeypatch.setattr(http_app.backend, "chat", chat)
    return http_app, chat


def _run_reply(http_app, text, chat_id="identity"):
    return http_app.reply(http_app.ReplyRequest(channel="web", user_id="user", chat_id=chat_id,
                                                text=text, timestamp=0, language="de"))


def test_full_reply_identity_conflict_cycle_and_allowed_background_model(monkeypatch, tmp_path):
    question = "Wenn Gemma eigentlich dein Sprachmodell ist, wie würdest du dann beschreiben, wer du selbst bist?"
    http_app, chat = _configure_reply_integration(monkeypatch, tmp_path, "Ich bin Gemma 4 und helfe dir gerne.")
    response = _run_reply(http_app, question)
    assert chat.call_count == 1
    assert "Ich bin Ava" in response.reply and "Ich bin Gemma" not in response.reply
    debug = json.loads((tmp_path / "workspace.json").read_text())["current"]
    assert debug["pre_workspace"] and debug["post_workspace"] and debug["completed_at"]
    assert debug["self_model"]["name"] == "Ava" and debug["self_model"]["activation"] >= .45
    assert any(x["kind"] == "identity_anchor" and x["self_affinity"] >= .45 for x in debug["active_items"])
    assert debug["post_gate"] == {"conflict": True, "reason": "identity_conflict", "corrected": True}
    memory = WorkingMemory(tmp_path / "working.json", session_id="web:identity")
    assert not any(item.role == "assistant" and item.content.startswith("Ich bin Gemma") for item in memory.items)

    allowed_dir = tmp_path / "allowed"; allowed_dir.mkdir()
    http_app, allowed_chat = _configure_reply_integration(monkeypatch, allowed_dir, "Ich bin Ava. Gemma ist mein aktuelles Hintergrundmodell.")
    allowed = _run_reply(http_app, question, chat_id="allowed")
    assert "Ich bin Ava" in allowed.reply and allowed_chat.call_count == 1
    allowed_debug = json.loads((allowed_dir / "workspace.json").read_text())["current"]
    assert allowed_debug["post_gate"]["conflict"] is False


def test_normal_reply_fast_path_has_exactly_one_reasoning_call(monkeypatch, tmp_path):
    http_app, chat = _configure_reply_integration(monkeypatch, tmp_path, "Eine normale Antwort ohne weitere Modellaufrufe.")
    response = _run_reply(http_app, "Erkläre den lokalen Scheduler kurz.", chat_id="fast")
    assert "normale Antwort" in response.reply
    assert chat.call_count == 1
    assert http_app.model_router.last_decision.task_type == "dialogue.reply"
    assert http_app.model_router.last_decision.worker_id == "ollama_reasoning"
    debug = json.loads((tmp_path / "workspace.json").read_text())["current"]
    assert set(debug["timing"]) >= {"workspace_pre_ms", "llm_ms", "workspace_post_ms", "total_ms"}


def test_reply_conversation_continuity_isolated_between_sessions(monkeypatch, tmp_path):
    http_app, chat = _configure_reply_integration(monkeypatch, tmp_path, "Verstanden.")
    _run_reply(http_app, "Unser Telegram-Bot hatte ein IPv6-Problem.", chat_id="a")
    _run_reply(http_app, "Wir sprechen über BACnet.", chat_id="b")
    _run_reply(http_app, "Warum haben wir ihn auf IPv4 gebunden?", chat_id="a")
    assert chat.call_count == 3

    session_a = WorkingMemory(tmp_path / "working.json", session_id="web:a")
    session_b = WorkingMemory(tmp_path / "working.json", session_id="web:b")
    assert any("Telegram" in item.content and "IPv6" in item.content for item in session_a.select("IPv4"))
    assert not any("Telegram" in item.content for item in session_b.items)
    final_workspace = json.loads((tmp_path / "workspace.json").read_text())["current"]
    assert final_workspace["session_id"] == "web:a"
    assert any("Telegram" in item["content"] for item in final_workspace["working_memory"])
    assert SelfModel.load(tmp_path / "self.json").name == "Ava"
