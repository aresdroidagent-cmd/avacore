from pathlib import Path

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
