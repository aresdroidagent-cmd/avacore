from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Callable


class ModelCapability(str, Enum):
    NONE = "none"
    DIALOGUE = "dialogue"
    REASONING = "reasoning"
    VISION = "vision"
    CODING = "coding"
    REVIEW = "review"


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    required_capabilities: tuple[ModelCapability, ...] = ()
    preferred_capability: ModelCapability | None = None
    requires_model: bool = True
    latency_class: str = "interactive"
    risk_level: str = "low"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    runtime: str
    model_name: str | None
    capabilities: tuple[ModelCapability, ...]
    enabled: bool = True
    local: bool = True
    gpu_required: bool = False
    priority: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceSnapshot:
    gpu_available: bool = False
    gpu_total_mb: int | None = None
    gpu_free_mb: int | None = None
    active_workers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RouteDecision:
    task_type: str
    requires_model: bool
    required_capability: str | None
    worker_id: str | None
    model_name: str | None
    runtime: str | None
    reason: str
    resource_actions: tuple[str, ...] = ()
    fallback_used: bool = False


def default_workers(settings: Any) -> tuple[WorkerSpec, ...]:
    return (
        WorkerSpec("ollama_reasoning", "ollama", settings.ollama_model,
                   (ModelCapability.DIALOGUE, ModelCapability.REASONING),
                   gpu_required=True, priority=100),
        WorkerSpec("smolvlm_vision", "transformers", settings.vision_model,
                   (ModelCapability.VISION,), enabled=bool(settings.vision_enabled),
                   gpu_required=True, priority=100,
                   metadata={"identity_source": False, "quality":"limited"}),
        WorkerSpec("coding_placeholder", "unconfigured", None,
                   (ModelCapability.CODING,), enabled=False, priority=0),
        WorkerSpec("review_placeholder", "unconfigured", None,
                   (ModelCapability.REVIEW,), enabled=False, priority=0),
    )


def profile_for_operation(operation: str, *, command_requires_llm: bool | None = None) -> TaskProfile:
    name = operation.strip().casefold()
    no_model = {
        "telegram:/who":"structured local perception",
        "telegram:/idcheck":"structured local identity diagnostics",
        "telegram:/orbits":"deterministic Orbit diagnostics",
        "telegram:/tasks":"deterministic task diagnostics",
        "telegram:/questions":"deterministic question diagnostics",
        "telegram:/focus":"deterministic Spotlight diagnostics",
        "telegram:/workspace":"deterministic Workspace diagnostics",
    }
    if name in {"telegram:/see", "telegram:/camera", "vision.camera"}:
        return TaskProfile("vision.camera", (ModelCapability.VISION,), ModelCapability.VISION,
                           metadata={"reason":"explicit camera scene request"})
    if name in {"reply", "dialogue.reply"}:
        return TaskProfile("dialogue.reply", (ModelCapability.REASONING,), ModelCapability.REASONING,
                           metadata={"reason":"normal conversational reasoning"})
    if command_requires_llm is False or name in no_model:
        return TaskProfile(name, requires_model=False,
                           preferred_capability=ModelCapability.NONE,
                           metadata={"reason":no_model.get(name, "command declares requires_llm=false")})
    return TaskProfile(name, requires_model=bool(command_requires_llm),
                       metadata={"reason":"no configured model route"})


def profile_for_cognitive_task(task: Any) -> TaskProfile:
    mapping = {
        "vision": ModelCapability.VISION, "code": ModelCapability.CODING,
        "review": ModelCapability.REVIEW, "analyze": ModelCapability.REASONING,
        "research": ModelCapability.REASONING,
    }
    declared = []
    for value in getattr(task, "required_capabilities", ()) or ():
        try:
            declared.append(ModelCapability(str(value).casefold()))
        except ValueError:
            continue
    capability = declared[0] if declared else mapping.get(str(getattr(task, "task_type", "")).casefold())
    return TaskProfile(
        task_type=f"cognitive_task.{getattr(task, 'task_type', 'unknown')}",
        required_capabilities=(capability,) if capability else (),
        preferred_capability=capability,
        # Phase 5.0 provides recommendations only; Task Drive does not execute.
        requires_model=capability is not None,
        latency_class="background", risk_level=str(getattr(task, "risk_level", "low")),
        metadata={"task_id":getattr(task, "task_id", None), "recommendation_only":True},
    )


class ModelRouter:
    def __init__(self, workers: tuple[WorkerSpec, ...], *, enabled: bool = True,
                 history_limit: int = 50,
                 resource_provider: Callable[[], ResourceSnapshot] | None = None):
        self.workers = workers
        self.enabled = enabled
        self.history_limit = max(1, history_limit)
        self._history: deque[RouteDecision] = deque(maxlen=self.history_limit)
        self._lock = Lock()
        self.resource_provider = resource_provider or ResourceSnapshot

    @property
    def last_decision(self) -> RouteDecision | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self) -> list[RouteDecision]:
        with self._lock:
            return list(self._history)

    def resource_state(self) -> ResourceSnapshot:
        try:
            return self.resource_provider()
        except Exception:
            return ResourceSnapshot()

    def route(self, profile: TaskProfile,
              resources: ResourceSnapshot | None = None) -> RouteDecision:
        capability = profile.preferred_capability or (
            profile.required_capabilities[0] if profile.required_capabilities else None
        )
        if not self.enabled:
            decision = RouteDecision(profile.task_type, profile.requires_model,
                capability.value if capability else None, None, None, None,
                "model router disabled")
        elif not profile.requires_model or capability == ModelCapability.NONE:
            decision = RouteDecision(profile.task_type, False, None, None, None, None,
                str(profile.metadata.get("reason") or "operation is deterministic and needs no model"))
        elif capability is None:
            decision = RouteDecision(profile.task_type, True, None, None, None, None,
                "operation requires a model but declares no supported capability")
        else:
            candidates = sorted((worker for worker in self.workers
                if worker.enabled and capability in worker.capabilities),
                key=lambda worker: (-worker.priority, worker.worker_id))
            if not candidates:
                decision = RouteDecision(profile.task_type, True, capability.value,
                    None, None, None,
                    f"no enabled worker provides {capability.value}; no cross-capability fallback")
            else:
                worker = candidates[0]
                decision = RouteDecision(profile.task_type, True, capability.value,
                    worker.worker_id, worker.model_name, worker.runtime,
                    str(profile.metadata.get("reason") or
                        f"selected highest-priority enabled {capability.value} worker"),
                    resource_actions=())
        with self._lock:
            self._history.append(decision)
        return decision

    def debug_state(self) -> dict[str, Any]:
        return {
            "enabled":self.enabled,
            "workers":[asdict(worker) for worker in self.workers],
            "last_decision":asdict(self.last_decision) if self.last_decision else None,
            "resource_state":asdict(self.resource_state()),
            "history":[asdict(item) for item in self.history()],
            "history_limit":self.history_limit,
        }
