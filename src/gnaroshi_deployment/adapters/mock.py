"""Deterministic CPU-only policy and embodiment for development and preflight."""
import threading
import time
import numpy as np
from ..contracts import Observation, Prediction, action_vector


class MockPolicy:
    metadata = {"kind": "mock", "action_representation": "relative_eef", "action_dim": 7}

    def __init__(self, delay_s=0.005):
        self.delay_s = delay_s
        self.steps = 0

    def connect(self):
        pass

    def reset(self, episode_id):
        self.steps = 0

    def infer(self, observation, instruction):
        time.sleep(self.delay_s)
        self.steps += 1
        action = np.zeros(7)
        action[0] = 0.0002 * np.sin(self.steps / 20)
        action[6] = 0.32 if self.steps % 120 > 60 else 0.02
        return Prediction(action, {"buffered": False, "rpc_ms": self.delay_s * 1000,
                                   "backbone_call_delta": 1, "chunk_start": self.steps-1,
                                   "response_horizon": 1})

    def close(self):
        pass


class MockEmbodiment:
    metadata = {"kind": "mock", "camera_serials": {"primary": "mock-primary", "wrist": "mock-wrist"}}

    def __init__(self):
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.state = np.zeros(7)
        self.applied = []
        self.home_count = 0

    def connect(self):
        pass

    def observe(self):
        images = {}
        for i, role in enumerate(("primary", "wrist")):
            image = np.zeros((240, 320, 3), np.uint8)
            image[:] = (32 + i * 18, 45, 59)
            x = int(150 + 70 * np.sin(time.monotonic()/2+i))
            image[95:145, max(0,x-25):min(320,x+25)] = (205, 66, 65)
            images[role] = image
        return Observation(images, self.state.copy(), time.time_ns(), self.metadata["camera_serials"],time.monotonic_ns())

    def home(self, cancel):
        if cancel.wait(0.03):
            raise InterruptedError("home cancelled")
        self.home_count += 1
        self.state[:] = 0

    def begin_episode(self):
        self.cancel.clear()

    def apply(self, prediction):
        action = action_vector(prediction.action)
        with self.lock:
            if self.cancel.is_set():
                raise InterruptedError("policy command cancelled")
            self.state[:6] += action[:6]
            self.state[6] = action[6]
            self.applied.append(action.copy())
        return {"raw_action": action.tolist(), "clipped_action": action.tolist(),
                "target_pose": self.state[:6].tolist(), "gripper_command": float(action[6])}

    def stop(self):
        self.cancel.set()
        with self.lock:
            return True

    def close(self):
        self.stop()
