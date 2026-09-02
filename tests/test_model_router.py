from dataclasses import replace
from types import SimpleNamespace

from avacore.core.orbits import CognitiveTask
from avacore.model.router import (
    ModelCapability,
    ModelRouter,
    ResourceSnapshot,
    TaskProfile,
    default_workers,
    profile_for_cognitive_task,
    profile_for_operation,
)


def settings(**overrides):
    values = {"ollama_model":"configured-dialogue", "vision_model":"configured-vision",
              "vision_enabled":True}
    values.update(overrides)
    return SimpleNamespace(**values)


def router(**kwargs):
    return ModelRouter(default_workers(settings()), **kwargs)


def test_who_routes_to_no_model_without_resource_action():
    decision = router().route(profile_for_operation("telegram:/who"),
        ResourceSnapshot(gpu_available=True, active_workers=("ollama_reasoning",)))
    assert decision.requires_model is False
    assert decision.worker_id is None
    assert decision.resource_actions == ()


def test_see_selects_configured_vision_worker():
    # CommandSpec.requires_llm means reasoning LLM; /see still explicitly
    # requires its semantic vision worker.
    decision = router().route(profile_for_operation("telegram:/see", command_requires_llm=False))
    assert decision.required_capability == "vision"
    assert decision.worker_id == "smolvlm_vision"
    assert decision.model_name == "configured-vision"


def test_reply_selects_configured_ollama_reasoning_worker():
    decision = router().route(profile_for_operation("reply"))
    assert decision.required_capability == "reasoning"
    assert decision.worker_id == "ollama_reasoning"
    assert decision.model_name == "configured-dialogue"
    assert decision.runtime == "ollama"


def test_disabled_vision_never_falls_back_to_reasoning():
    workers = tuple(replace(worker, enabled=False) if worker.worker_id == "smolvlm_vision" else worker
                    for worker in default_workers(settings()))
    decision = ModelRouter(workers).route(profile_for_operation("telegram:/see"))
    assert decision.worker_id is None
    assert decision.required_capability == "vision"
    assert "no cross-capability fallback" in decision.reason


def test_identity_is_not_a_generative_model_capability():
    assert "identity" not in {capability.value for capability in ModelCapability}
    assert all(worker.metadata.get("identity_source") is not True
               for worker in default_workers(settings()))


def test_same_input_produces_same_decision():
    model_router = router()
    profile = profile_for_operation("reply")
    resources = ResourceSnapshot(gpu_available=True, gpu_total_mb=8192, gpu_free_mb=500)
    assert model_router.route(profile, resources) == model_router.route(profile, resources)


def test_route_has_no_worker_or_network_side_effect():
    calls = []
    model_router = ModelRouter(default_workers(settings()),
        resource_provider=lambda: (calls.append("snapshot") or ResourceSnapshot()))
    decision = model_router.route(profile_for_operation("reply"), ResourceSnapshot())
    assert decision.worker_id == "ollama_reasoning"
    assert calls == []


def test_router_leaves_resource_policy_to_coordinator():
    decision = router().route(profile_for_operation("vision.camera"),
        ResourceSnapshot(active_workers=("ollama_reasoning",)))
    assert decision.worker_id == "smolvlm_vision"
    assert decision.resource_actions == ()


def test_coding_and_review_placeholders_report_unavailable():
    model_router = router()
    for capability in (ModelCapability.CODING, ModelCapability.REVIEW):
        decision = model_router.route(TaskProfile(f"test.{capability.value}",
            (capability,), capability, True))
        assert decision.worker_id is None
        assert decision.required_capability == capability.value


def test_command_requires_llm_false_has_real_no_model_semantics():
    decision = router().route(profile_for_operation(
        "telegram:/deterministic", command_requires_llm=False))
    assert decision.requires_model is False and decision.worker_id is None


def test_cognitive_task_profile_is_recommendation_only():
    task = CognitiveTask("task-1", "orbit-1", "Inspect", "Inspect vision", "vision")
    profile = profile_for_cognitive_task(task)
    assert profile.preferred_capability == ModelCapability.VISION
    assert profile.metadata == {"task_id":"task-1", "recommendation_only":True}


def test_history_is_bounded():
    model_router = router(history_limit=3)
    for index in range(7):
        model_router.route(TaskProfile(f"no-model-{index}", requires_model=False))
    assert len(model_router.history()) == 3
    assert model_router.history()[0].task_type == "no-model-4"


def test_worker_names_come_from_settings():
    workers = default_workers(settings(ollama_model="a", vision_model="b"))
    assert {worker.worker_id:worker.model_name for worker in workers} == {
        "ollama_reasoning":"a", "smolvlm_vision":"b",
        "coding_placeholder":None, "review_placeholder":None,
    }


def test_debug_api_routes_are_registered_without_execution_endpoint():
    from avacore.api.http_app import app
    routes = {(route.path, method) for route in app.routes
              for method in getattr(route, "methods", set())}
    assert ("/debug/model-router", "GET") in routes
    assert ("/debug/model-router/route", "POST") in routes
    assert ("/debug/resources", "GET") in routes
    assert ("/debug/resources/plan", "POST") in routes
    assert not any(path.startswith("/debug/model-router/execute") for path, _ in routes)


def test_camera_api_records_vision_and_structured_no_model_routes(monkeypatch):
    from avacore.api import http_app
    from avacore.vision.perception import PerceptionResult

    class Service:
        def request(self, **kwargs):
            return PerceptionResult("now", "now", "frame", "scene", [], [], [],
                                    reason=kwargs["reason"])

    monkeypatch.setattr(http_app, "camera_perception_service", lambda _decision=None: Service())
    monkeypatch.setattr(http_app.settings, "vision_preempt_reasoning", True)
    http_app.request_camera_perception(http_app.CameraPerceptionRequest(
        reason="see_command", include_scene=True), None)
    assert http_app.model_router.last_decision.worker_id == "smolvlm_vision"

    http_app.request_camera_perception(http_app.CameraPerceptionRequest(
        reason="who_command", include_scene=False), None)
    assert http_app.model_router.last_decision.requires_model is False
    assert http_app.model_router.last_decision.worker_id is None
    assert http_app.model_router.last_decision.resource_actions == ()
