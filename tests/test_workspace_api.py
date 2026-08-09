from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "avacore" / "api" / "http_app.py").read_text(encoding="utf-8")


def test_workspace_debug_routes_are_admin_protected():
    assert '@app.get("/debug/workspace")' in SOURCE
    assert '@app.get("/debug/workspace/history")' in SOURCE
    assert SOURCE.count("Depends(verify_admin_password)") >= 2


def test_reply_builds_one_workspace_context_without_legacy_dynamic_blocks():
    assert "run_workspace_cycle(" in SOURCE
    assert "workspace_prompt(snapshot)" in SOURCE
    assert 'if not getattr(settings, "jspace_enabled", False):' in SOURCE
    assert SOURCE.index('if not getattr(settings, "jspace_enabled", False):') < SOURCE.index('parts.append("VERIFIED LONG-TERM MEMORY:')


def test_memory_rag_and_research_are_cognitive_candidates():
    assert '"source": "memory", "kind": "memory"' in SOURCE
    assert '"source": "knowledge", "kind": "knowledge_hit"' in SOURCE
    assert '"source": "research", "kind": "research_finding"' in SOURCE
