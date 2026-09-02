from threading import Event, Thread
from types import SimpleNamespace
import subprocess

import pytest

from avacore.model.resources import (
    NvidiaSmiProvider,
    ResourceActionKind,
    ResourceCoordinator,
    RuntimeResourceProvider,
    default_resource_profiles,
)
from avacore.model.router import ResourceSnapshot, RouteDecision


def result(stdout="", returncode=0):
    return SimpleNamespace(stdout=stdout, returncode=returncode)


def decision(worker_id="smolvlm_vision", requires_model=True):
    return RouteDecision("test", requires_model, "vision" if worker_id else None,
                         worker_id, "model", "runtime", "test")


def coordinator(snapshot=ResourceSnapshot(), *, release=None):
    adapters = {"ollama_reasoning":release} if release else {}
    return ResourceCoordinator(default_resource_profiles(),
        snapshot_provider=lambda: snapshot, release_adapters=adapters)


def test_nvidia_provider_parses_real_memory_shape_without_gpu():
    calls = []
    provider = NvidiaSmiProvider(timeout=1.5,
        runner=lambda args, **kwargs: (calls.append((args, kwargs)) or result("8192, 5000\n")))
    snapshot = provider.snapshot()
    assert snapshot.gpu_available is True
    assert snapshot.gpu_total_mb == 8192 and snapshot.gpu_free_mb == 5000
    assert calls[0][1] == {"capture_output":True, "text":True, "timeout":1.5, "check":False}
    assert calls[0][0][0] == "nvidia-smi"


@pytest.mark.parametrize("failure", [FileNotFoundError(),
    subprocess.TimeoutExpired("nvidia-smi", 1), ValueError("malformed")])
def test_nvidia_provider_failures_are_safe(failure):
    def run(*_args, **_kwargs):
        raise failure
    assert NvidiaSmiProvider(runner=run).snapshot() == ResourceSnapshot()


def test_nvidia_provider_malformed_output_is_safe():
    provider = NvidiaSmiProvider(runner=lambda *_a, **_k: result("not,memory"))
    assert provider.snapshot().gpu_available is False


def test_runtime_residency_uses_ollama_and_vision_truth():
    provider = RuntimeResourceProvider(
        NvidiaSmiProvider(runner=lambda *_a, **_k: result("8192, 4000")),
        ollama_model="gemma", ollama_probe=lambda: ("gemma",),
        vision_probe=lambda: True)
    assert provider().active_workers == ("ollama_reasoning", "smolvlm_vision")


def test_runtime_residency_absent_workers_remain_absent():
    provider = RuntimeResourceProvider(
        NvidiaSmiProvider(runner=lambda *_a, **_k: result("8192, 7000")),
        ollama_model="gemma", ollama_probe=lambda: (), vision_probe=lambda: False)
    assert provider().active_workers == ()


def test_vision_plan_releases_reasoning_then_reuses_vision():
    plan = coordinator().plan(decision(), ResourceSnapshot(active_workers=(
        "ollama_reasoning", "smolvlm_vision")))
    assert [action.label for action in plan.actions] == [
        "release:ollama_reasoning", "reuse:smolvlm_vision"]


def test_vision_without_reasoning_does_not_plan_release():
    plan = coordinator().plan(decision(), ResourceSnapshot(active_workers=()))
    assert not any(action.kind == ResourceActionKind.RELEASE for action in plan.actions)


def test_no_model_route_has_no_actions_or_gpu_requirement():
    plan = coordinator().plan(decision(None, False),
                              ResourceSnapshot(active_workers=("ollama_reasoning",)))
    assert plan.ready and plan.actions == ()


def test_reasoning_and_vision_residency_are_reused_without_wrong_preemption():
    resources = coordinator()
    reasoning = decision("ollama_reasoning")
    reasoning_plan = resources.plan(reasoning,
        ResourceSnapshot(active_workers=("ollama_reasoning", "smolvlm_vision")))
    assert [action.label for action in reasoning_plan.actions] == ["reuse:ollama_reasoning"]


def test_prepare_releases_exactly_once_and_does_not_retry():
    calls = []
    resources = coordinator(release=lambda: (calls.append("release") or True))
    plan = resources.plan(decision(), ResourceSnapshot(active_workers=("ollama_reasoning",)))
    results = resources.prepare(plan)
    assert calls == ["release"] and results[0].success is True


def test_prepare_release_failure_is_controlled():
    def fail():
        raise RuntimeError("offline")
    resources = coordinator(release=fail)
    plan = resources.plan(decision(), ResourceSnapshot(active_workers=("ollama_reasoning",)))
    results = resources.prepare(plan)
    assert len(results) == 1 and results[0].success is False


def test_incompatible_gpu_leases_are_serialized_without_deadlock():
    resources = coordinator(ResourceSnapshot())
    first_acquired, release_first = Event(), Event()
    second_started, second_acquired = Event(), Event()

    def first():
        with resources.lease(decision("ollama_reasoning")):
            first_acquired.set()
            assert release_first.wait(2)

    def second():
        assert first_acquired.wait(2)
        second_started.set()
        with resources.lease(decision("smolvlm_vision")):
            second_acquired.set()

    first_thread = Thread(target=first); second_thread = Thread(target=second)
    first_thread.start(); second_thread.start()
    assert first_acquired.wait(2) and second_started.wait(2)
    assert not second_acquired.wait(.05)
    release_first.set()
    assert second_acquired.wait(2)
    first_thread.join(2); second_thread.join(2)
    assert not first_thread.is_alive() and not second_thread.is_alive()


def test_lease_is_released_when_worker_raises():
    resources = coordinator(ResourceSnapshot())
    with pytest.raises(RuntimeError):
        with resources.lease(decision("smolvlm_vision")):
            raise RuntimeError("worker failed")
    with resources.lease(decision("ollama_reasoning")):
        pass
