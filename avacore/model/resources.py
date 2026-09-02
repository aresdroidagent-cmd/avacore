from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import subprocess
from threading import Lock
from typing import Any, Callable, Iterator

from avacore.model.router import ResourceSnapshot, RouteDecision


_logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ResourceActionKind(str, Enum):
    RELEASE = "release"
    REUSE = "reuse"


@dataclass(frozen=True)
class ResourceAction:
    kind: ResourceActionKind
    worker_id: str

    @property
    def label(self) -> str:
        return f"{self.kind.value}:{self.worker_id}"


@dataclass(frozen=True)
class ResourceActionResult:
    action: ResourceAction
    success: bool
    detail: str


@dataclass(frozen=True)
class WorkerResourceState:
    worker_id: str
    resident: bool
    source: str


@dataclass(frozen=True)
class WorkerResourceProfile:
    worker_id: str
    gpu_required: bool
    preempt_before_start: tuple[str, ...] = ()
    release_supported: bool = False
    residency_preference: str = "reuse"


@dataclass(frozen=True)
class ResourcePlan:
    worker_id: str | None
    ready: bool
    actions: tuple[ResourceAction, ...]
    reason: str
    snapshot_before: ResourceSnapshot


@dataclass(frozen=True)
class ResourceExecution:
    worker_id: str | None
    started_at: str
    completed_at: str
    results: tuple[ResourceActionResult, ...] = ()


class NvidiaSmiProvider:
    def __init__(self, *, timeout: float = 2.0,
                 runner: Callable[..., Any] = subprocess.run):
        self.timeout = timeout
        self.runner = runner

    def snapshot(self) -> ResourceSnapshot:
        try:
            result = self.runner([
                "nvidia-smi", "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ], capture_output=True, text=True, timeout=self.timeout, check=False)
            if result.returncode != 0:
                return ResourceSnapshot()
            rows = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                total, free = (int(value.strip()) for value in line.split(",", 1))
                rows.append((total, free))
            if not rows:
                return ResourceSnapshot()
            return ResourceSnapshot(True, sum(x[0] for x in rows), sum(x[1] for x in rows))
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
            return ResourceSnapshot()


class RuntimeResourceProvider:
    def __init__(self, gpu_provider: NvidiaSmiProvider, *,
                 ollama_model: str,
                 ollama_probe: Callable[[], tuple[str, ...]],
                 vision_probe: Callable[[], bool]):
        self.gpu_provider = gpu_provider
        self.ollama_model = ollama_model
        self.ollama_probe = ollama_probe
        self.vision_probe = vision_probe

    def __call__(self) -> ResourceSnapshot:
        gpu = self.gpu_provider.snapshot()
        active: list[str] = []
        try:
            if self.ollama_model in self.ollama_probe():
                active.append("ollama_reasoning")
        except Exception:
            _logger.debug("Ollama residency probe unavailable", exc_info=True)
        try:
            if self.vision_probe():
                active.append("smolvlm_vision")
        except Exception:
            _logger.debug("Vision residency probe unavailable", exc_info=True)
        return ResourceSnapshot(gpu.gpu_available, gpu.gpu_total_mb, gpu.gpu_free_mb,
                                tuple(sorted(active)))


def default_resource_profiles(*, preempt_reasoning_for_vision: bool = True
                              ) -> tuple[WorkerResourceProfile, ...]:
    return (
        WorkerResourceProfile("ollama_reasoning", True, release_supported=True),
        WorkerResourceProfile("smolvlm_vision", True,
            preempt_before_start=(("ollama_reasoning",)
                                  if preempt_reasoning_for_vision else ()),
            release_supported=False),
        WorkerResourceProfile("coding_placeholder", True),
        WorkerResourceProfile("review_placeholder", True),
    )


class ResourceCoordinator:
    def __init__(self, profiles: tuple[WorkerResourceProfile, ...], *,
                 snapshot_provider: Callable[[], ResourceSnapshot],
                 release_adapters: dict[str, Callable[[], bool]] | None = None,
                 enabled: bool = True, history_limit: int = 50):
        self.profiles = {profile.worker_id: profile for profile in profiles}
        self.snapshot_provider = snapshot_provider
        self.release_adapters = dict(release_adapters or {})
        self.enabled = enabled
        self._gpu_lock = Lock()
        self._state_lock = Lock()
        self._plans: deque[ResourcePlan] = deque(maxlen=max(1, history_limit))
        self._executions: deque[ResourceExecution] = deque(maxlen=max(1, history_limit))

    def snapshot(self) -> ResourceSnapshot:
        try:
            return self.snapshot_provider()
        except Exception:
            _logger.warning("Resource snapshot unavailable", exc_info=True)
            return ResourceSnapshot()

    def plan(self, decision: RouteDecision,
             snapshot: ResourceSnapshot | None = None) -> ResourcePlan:
        state = snapshot or self.snapshot()
        profile = self.profiles.get(decision.worker_id or "")
        if not self.enabled or not decision.requires_model or decision.worker_id is None:
            plan = ResourcePlan(decision.worker_id, True, (),
                "no resource action required", state)
        elif profile is None:
            plan = ResourcePlan(decision.worker_id, False, (),
                "worker has no resource profile", state)
        else:
            actions = [ResourceAction(ResourceActionKind.RELEASE, worker_id)
                       for worker_id in profile.preempt_before_start
                       if worker_id in state.active_workers]
            if decision.worker_id in state.active_workers and profile.residency_preference == "reuse":
                actions.append(ResourceAction(ResourceActionKind.REUSE, decision.worker_id))
            reason = ("; ".join(action.label for action in actions)
                      if actions else "target will lazy-load through its normal worker call")
            plan = ResourcePlan(decision.worker_id, True, tuple(actions), reason, state)
        with self._state_lock:
            self._plans.append(plan)
        _logger.info("resource plan target=%s actions=%s", plan.worker_id,
                     [action.label for action in plan.actions])
        return plan

    def prepare(self, plan: ResourcePlan) -> tuple[ResourceActionResult, ...]:
        results = []
        for action in plan.actions:
            if action.kind == ResourceActionKind.REUSE:
                results.append(ResourceActionResult(action, True, "resident worker reused"))
                continue
            adapter = self.release_adapters.get(action.worker_id)
            if adapter is None:
                results.append(ResourceActionResult(action, False, "release adapter unavailable"))
                continue
            try:
                success = bool(adapter())
                detail = "released" if success else "release failed or worker already absent"
            except Exception:
                success, detail = False, "release failed"
                _logger.warning("Resource release failed worker=%s", action.worker_id,
                                exc_info=True)
            results.append(ResourceActionResult(action, success, detail))
            _logger.info("resource release worker=%s success=%s", action.worker_id, success)
        return tuple(results)

    @contextmanager
    def lease(self, decision: RouteDecision) -> Iterator[ResourcePlan]:
        profile = self.profiles.get(decision.worker_id or "")
        needs_gpu = bool(self.enabled and decision.requires_model and profile and profile.gpu_required)
        if not needs_gpu:
            yield self.plan(decision)
            return
        self._gpu_lock.acquire()
        started = _utc_now()
        try:
            plan = self.plan(decision, self.snapshot())
            results = self.prepare(plan)
            _logger.info("resource lease acquired worker=%s", decision.worker_id)
            yield plan
        finally:
            completed = _utc_now()
            execution = ResourceExecution(decision.worker_id, started, completed,
                                          results if "results" in locals() else ())
            with self._state_lock:
                self._executions.append(execution)
            self._gpu_lock.release()

    @property
    def last_plan(self) -> ResourcePlan | None:
        with self._state_lock:
            return self._plans[-1] if self._plans else None

    @property
    def last_execution(self) -> ResourceExecution | None:
        with self._state_lock:
            return self._executions[-1] if self._executions else None

    def debug_state(self) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {"enabled":self.enabled, "snapshot":asdict(snapshot),
                "worker_states":[asdict(WorkerResourceState(worker_id,
                    worker_id in snapshot.active_workers, "runtime_probe"))
                    for worker_id in sorted(self.profiles)],
                "last_plan":asdict(self.last_plan) if self.last_plan else None,
                "last_execution":asdict(self.last_execution) if self.last_execution else None}
