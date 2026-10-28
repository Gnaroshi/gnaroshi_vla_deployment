"""Small contracts for adding a policy or embodiment without rewriting the GUI."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Protocol
import numpy as np


class RobotRecoveryRequired(RuntimeError):
    """A typed device fault requiring an explicit, motion-free operator recovery."""
    def __init__(self, reason: str, *, status: dict | None = None):
        super().__init__(reason)
        self.status = dict(status or {})


@dataclass
class Observation:
    images: dict[str, np.ndarray]
    state: np.ndarray
    captured_ns: int
    camera_metadata: dict = field(default_factory=dict)
    captured_monotonic_ns: int | None = None


@dataclass
class Prediction:
    action: np.ndarray
    telemetry: dict = field(default_factory=dict)


@dataclass
class SessionConfig:
    task: str = "demo_task"
    method: str = "mock"
    instruction: str = "Run the mock deployment"
    target_hz: float = 60.0
    max_steps: int = 5000
    record_video: bool = True
    output_root: str = ".runtime/results"
    reference_manifest: str | None = None
    camera_roles: tuple[str,...] = ("primary","wrist")
    metadata: dict = field(default_factory=dict)

    def validate(self):
        if not np.isfinite(self.target_hz) or not 0 < self.target_hz <= 500:
            raise ValueError("target_hz must be finite and in (0,500]")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if not self.instruction.strip() or not self.task.strip():
            raise ValueError("task and instruction are required")
        if not self.camera_roles or len(set(self.camera_roles))!=len(self.camera_roles):
            raise ValueError("camera_roles must be nonempty and distinct")

    def asdict(self):
        return asdict(self)


class Policy(Protocol):
    """Calls are serialized by the session worker.

    For an opted-in sensor preview, connect may run on the session policy-loader
    while the worker reads cameras. Device control waits for connect to finish,
    and policy close waits for the loader to terminate.

    infer must return under a finite adapter-defined deadline. Cancellation never
    automatically retries an uncertain request or applies its eventual response.
    """
    metadata: dict
    def connect(self) -> None: ...
    def reset(self, episode_id: str) -> None: ...
    def infer(self, observation: Observation, instruction: str) -> Prediction: ...
    def close(self) -> None: ...


class Embodiment(Protocol):
    """stop must latch cancellation BEFORE waiting for a command lock.

    apply must check that latch inside the same lock used to issue a command.
    stop is callable from a separate operator thread. All other hardware calls
    are serialized on the session worker. home is cancellable and bounded.
    connect opens sensors/control but never homes or calibrates the gripper.
    Optional recover(cancel) verifies fresh healthy state and acknowledged stop,
    returning {"healthy": True, "stop_confirmed": True, ...}. It never unlocks,
    powers on, releases brakes, homes, activates the gripper or arms motion.
    Only RobotRecoveryRequired faults enable this explicit recovery path.
    """
    metadata: dict
    def connect(self) -> None: ...
    def observe(self) -> Observation: ...
    def home(self, cancel: Event) -> None: ...
    def begin_episode(self) -> None: ...
    def apply(self, prediction: Prediction) -> dict: ...
    def stop(self) -> bool: ...
    def close(self) -> None: ...


def action_vector(value, dimension: int = 7) -> np.ndarray:
    action = np.asarray(value, dtype=np.float64)
    if action.shape != (dimension,) or not np.isfinite(action).all():
        raise ValueError(f"action must have {dimension} finite values")
    return action
