from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

def test_workspace_ui_route_and_content():
    root = Path(__file__).parents[1]
    path = root / "avacore" / "web" / "static" / "workspace.html"
    content = path.read_text(encoding="utf-8")
    script = (path.parent / "workspace.js").read_text(encoding="utf-8")
    api_source = (root / "avacore" / "api" / "http_app.py").read_text(encoding="utf-8")
    assert '@app.get("/ui/workspace"' in api_source
    assert "Ava Conscious Workspace" in content
    assert "focus-field" in content
    assert "Item table" in content
    assert "Source legend" in content
    assert "/debug/workspace" in script
    assert "setInterval(refresh,2000)" in script


@pytest.mark.anyio
async def test_workspace_ui_and_protected_json_are_served_over_http(monkeypatch, tmp_path):
    from avacore.api import http_app

    monkeypatch.setattr(http_app.settings, "workspace_path", tmp_path / "workspace.json")
    monkeypatch.setattr(http_app.settings, "web_admin_password", "workspace-secret")
    async with AsyncClient(transport=ASGITransport(app=http_app.app), base_url="http://test") as client:
        response = await client.get("/ui/workspace")
        assert response.status_code == 200
        for heading in ("Ava Conscious Workspace", "Self Model", "Working Memory", "Spotlight", "Cognitive Cycle"):
            assert heading in response.text

        assert (await client.get("/debug/workspace")).status_code == 401
        debug = await client.get("/debug/workspace", headers={"X-Admin-Password": "workspace-secret"})
        assert debug.status_code == 200
        assert set(debug.json()) >= {"self_model", "working_memory", "pre_workspace", "post_workspace", "post_gate", "timing"}
