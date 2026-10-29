"""Deterministic CPU-only policy and embodiment for development and preflight."""
import threading
import time
import numpy as np
from ..contracts import CameraError, Observation, Prediction, action_vector


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

    def __init__(self, camera_serials=None):
        self.metadata={**type(self).metadata,"camera_serials":dict(camera_serials or type(self).metadata["camera_serials"])}
        self.metadata["camera_configuration"]={"width":320,"height":240,"fps":30,"format":"rgb8"}
        self.cancel = threading.Event()
        self.lock = threading.Lock()
        self.state = np.zeros(7)
        self.applied = []
        self.home_count = 0
        self._frame_counter=0
        self._episode_active=False

    @property
    def camera_operation_pending(self):
        return False

    def list_camera_devices(self):
        return {"devices":[{"serial":serial,"name":name,"label":f"{name} [{serial}]",
                            "color_profiles":[dict(self.metadata["camera_configuration"])]}
                           for serial,name in (("mock-primary","Mock camera A"),
                                               ("mock-wrist","Mock camera B"),("mock-extra","Mock camera C"))],
                "selected":dict(self.metadata["camera_serials"]),
                "configuration":dict(self.metadata["camera_configuration"]),"warnings":[]}

    def configure_cameras(self, serials, cancel):
        if self._episode_active:
            raise ValueError("Stop the active rollout before changing cameras")
        if cancel.is_set():
            raise InterruptedError("Camera change cancelled")
        available={item["serial"] for item in self.list_camera_devices()["devices"]}
        if not isinstance(serials,dict) or set(serials)!=set(self.metadata["camera_serials"]):
            raise ValueError("Camera mapping must supply every configured role exactly once")
        if len(set(serials.values()))!=len(serials):
            raise ValueError("Select distinct connected mock cameras")
        if not set(serials.values())<=available:
            raise CameraError("Selected mock camera is not connected",category="camera_not_found",
                              status={"selected":dict(serials)})
        self.metadata["camera_serials"]=dict(serials)
        observation=self.observe()
        return {"selected":dict(serials),"configuration":dict(self.metadata["camera_configuration"]),
                "camera_metadata":observation.camera_metadata}

    def connect(self):
        pass

    def observe(self):
        images = {}
        self._frame_counter+=1
        metadata={}
        serial_order={"mock-primary":0,"mock-wrist":1,"mock-extra":2}
        for role,serial in self.metadata["camera_serials"].items():
            i=serial_order.get(serial,0)
            image = np.zeros((240, 320, 3), np.uint8)
            image[:] = (32 + i * 18, 45, 59)
            x = int(150 + 70 * np.sin(time.monotonic()/2+i))
            image[95:145, max(0,x-25):min(320,x+25)] = (205, 66, 65)
            images[role] = image
            metadata[role]={"serial":serial,"frame_number":self._frame_counter,"timestamp_domain":"mock",
                            "host_received_monotonic_ns":time.monotonic_ns()}
        return Observation(images, self.state.copy(), time.time_ns(), metadata,time.monotonic_ns())

    def home(self, cancel):
        if cancel.wait(0.03):
            raise InterruptedError("home cancelled")
        self.home_count += 1
        self.state[:] = 0

    def begin_episode(self):
        self.cancel.clear()
        self._episode_active=True

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
        self._episode_active=False
        with self.lock:
            return True

    def close(self):
        self.stop()
