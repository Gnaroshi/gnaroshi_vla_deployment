"""Camera-only preview must never connect controls before policy validation."""
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.adapters.ur5e import UR5eConfig, UR5eEmbodiment
from gnaroshi_deployment.contracts import SessionConfig
from gnaroshi_deployment.session import DeploymentSession


class ControlConnectionCancellation(unittest.TestCase):
    def test_close_rechecks_stop_after_control_is_assigned(self):
        entered, release = threading.Event(), threading.Event()
        class Control:
            disconnected = False
            def servoStop(self):
                return False
            def disconnect(self):
                self.disconnected = True
        control = Control()
        class ConnectingArm(UR5eEmbodiment):
            def connect(self):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("Test control connection did not finish")
                self.control = control
        config = UR5eConfig(robot_ip="127.0.0.1", camera_serials={"primary":"MOCK_P", "wrist":"MOCK_W"},
                           home_joints=(0.0,)*6)
        with tempfile.TemporaryDirectory() as temporary:
            session = DeploymentSession(SessionConfig(output_root=temporary, record_video=False),
                                        MockPolicy(), ConnectingArm(config), Path(__file__).parents[1])
            try:
                session.start()
                self.assertTrue(entered.wait(1))
                session.close()
                session._stop_thread.join(1)
                self.assertTrue(session._stop_results[session._stop_generation])
            finally:
                release.set()
                session.worker.join(3)
            self.assertFalse(session.worker.is_alive())
            self.assertTrue(control.disconnected)
            self.assertEqual(session.state, "ERROR")
            self.assertIn("stop was not confirmed", session.error)
            self.assertFalse(session.replacement_closed)
            self.assertFalse(session.can_replace)


class PreviewEmbodiment(MockEmbodiment):
    def __init__(self):
        super().__init__()
        self.preview_calls = self.preview_frames = self.control_calls = self.close_calls = 0

    def connect_preview(self):
        self.preview_calls += 1

    def observe_preview(self):
        self.preview_frames += 1
        observation = super().observe()
        observation.state = np.empty(0, dtype=np.float32)
        observation.preview_only = True
        return observation

    def connect(self):
        self.control_calls += 1

    def close(self):
        self.close_calls += 1
        super().close()


class LoadingPreview(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sessions, self.releases = [], []

    def tearDown(self):
        for event in self.releases:
            event.set()
        for session in self.sessions:
            session.close()
            session.worker.join(3)
            self.assertTrue(session.shutdown_complete)
        self.temporary.cleanup()

    def wait(self, predicate):
        deadline = time.monotonic()+3
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Preview transition timed out")
            time.sleep(.005)

    def make(self, policy, embodiment=None):
        config = SessionConfig(output_root=self.temporary.name, record_video=False,
            metadata={"preview_before_policy": True, "reload_selection": {"checkpoint": "mock_checkpoint"}})
        session = DeploymentSession(config, policy, embodiment or PreviewEmbodiment(), Path(__file__).parents[1])
        self.sessions.append(session)
        return session

    def test_preview_updates_while_policy_loads_and_controls_wait_for_validation(self):
        entered, release, validated = threading.Event(), threading.Event(), threading.Event()
        self.releases.append(release)
        loader_threads = []
        class LoadingPolicy(MockPolicy):
            loading_status = "Validating selected checkpoint metadata"
            def connect(self):
                loader_threads.append(threading.current_thread().name)
                entered.set()
                release.wait(2)
                validated.set()
        class GuardedEmbodiment(PreviewEmbodiment):
            def connect(self):
                if not validated.is_set():
                    raise AssertionError("Control connected before policy validation")
                super().connect()
        session = self.make(LoadingPolicy(), GuardedEmbodiment())
        session.start()
        self.assertTrue(entered.wait(1))
        self.wait(lambda: session.embodiment.preview_frames >= 2)
        self.assertEqual(session.state, "LOADING_POLICY")
        self.assertFalse(session.start_rollout())
        self.assertEqual(session.embodiment.control_calls, 0)
        self.assertTrue(session.latest.preview_only)
        self.assertEqual(session.latest.state.shape, (0,))
        self.assertTrue(session.latest.camera_metadata["preview_only"])
        release.set()
        self.wait(lambda: session.state == "READY")
        self.assertEqual(session.embodiment.control_calls, 1)
        self.assertEqual(loader_threads, ["policy-loader"])
        self.assertFalse(session._policy_loader.is_alive())
        events = [json.loads(line) for line in (session.folder/"events.jsonl").read_text().splitlines()]
        self.assertTrue(any("Validating selected" in row.get("detail", "") for row in events))

    def test_failed_policy_load_closes_sensors_after_loader_finishes(self):
        finished = threading.Event()
        class FailedPolicy(MockPolicy):
            def connect(self):
                try:
                    raise ValueError("Checkpoint metadata validation failed")
                finally:
                    finished.set()
            def close(self):
                if not finished.is_set():
                    raise AssertionError("Policy closed concurrently with loader")
        session = self.make(FailedPolicy())
        session.start()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertEqual(session.embodiment.preview_calls, 1)
        self.assertEqual(session.embodiment.control_calls, 0)
        self.assertEqual(session.embodiment.close_calls, 1)
        self.assertTrue(session.can_replace)
        error = next(json.loads(line) for line in (session.folder/"events.jsonl").read_text().splitlines()
                     if json.loads(line)["event"] == "error")
        self.assertEqual(error["selection"]["checkpoint"], "mock_checkpoint")

    def test_deferred_session_previews_without_loading_or_connecting_controls(self):
        class DeferredPolicy(MockPolicy):
            connect_calls = 0
            def connect(self):
                self.connect_calls += 1
        session = self.make(DeferredPolicy())
        session.config.metadata["defer_policy_load"] = True
        session.start()
        self.wait(lambda: session.state == "SELECTING" and session.latest is not None)
        self.assertEqual(session.policy.connect_calls, 0)
        self.assertEqual(session.embodiment.control_calls, 0)
        self.assertTrue(session.latest.preview_only)
        self.assertTrue(session.can_replace)
        self.assertFalse(session.start_rollout())
        session.close()
        self.wait(lambda: session.shutdown_complete)
        self.assertTrue(session.replacement_closed)

    def test_exit_waits_for_loader_then_closes_preview_without_controls(self):
        entered, finished = threading.Event(), threading.Event()
        class LoadingPolicy(MockPolicy):
            def connect(self):
                entered.set()
                session.closing.wait(2)
                finished.set()
                raise InterruptedError("Policy startup cancelled")
            def close(self):
                if not finished.is_set():
                    raise AssertionError("Loader must finish before close")
        session = self.make(LoadingPolicy())
        session.start()
        self.assertTrue(entered.wait(1))
        self.wait(lambda: session.latest is not None)
        session.close()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "CLOSED")
        self.assertIsNone(session.error)
        self.assertEqual(session.embodiment.control_calls, 0)
        self.assertEqual(session.embodiment.close_calls, 1)
        self.assertFalse(session._policy_loader.is_alive())

    def test_preview_failure_cancels_loader_and_is_not_a_reloadable_policy_error(self):
        finished = threading.Event()
        class LoadingPolicy(MockPolicy):
            def connect(self):
                session.closing.wait(2)
                finished.set()
                raise InterruptedError("Policy startup cancelled")
        class FailedPreview(PreviewEmbodiment):
            def observe_preview(self):
                raise RuntimeError("Camera frame unavailable")
            def close(self):
                if not finished.is_set():
                    raise AssertionError("Sensors closed before loader terminated")
                super().close()
        session = self.make(LoadingPolicy(), FailedPreview())
        session.start()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertIn("Camera frame unavailable", session.error)
        self.assertFalse(session.can_replace)
        self.assertEqual(session.embodiment.control_calls, 0)
        self.assertEqual(session.embodiment.close_calls, 1)

    def test_preview_observation_cannot_be_forwarded_to_policy(self):
        class PreviewAsState(PreviewEmbodiment):
            def observe(self):
                return self.observe_preview()
        session = self.make(MockPolicy(), PreviewAsState())
        session.start()
        self.wait(lambda: session.shutdown_complete)
        self.assertIn("cannot be used as a policy observation", session.error)
        self.assertEqual(session.embodiment.applied, [])


class CameraOnlyAdapter(unittest.TestCase):
    def test_preview_opens_only_cameras_and_full_connect_reuses_them(self):
        import gnaroshi_deployment.adapters.ur5e as driver
        pipelines, controls, receivers, grippers = [], [], [], []
        class Frame:
            def get_data(self): return np.zeros((2, 2, 3), np.uint8)
            def get_frame_number(self): return 1
            def get_timestamp(self): return 1.0
            def get_frame_timestamp_domain(self): return "mock"
        class Pipeline:
            def __init__(self):
                self.started = self.stopped = 0
                pipelines.append(self)
            def start(self, config): self.started += 1
            def stop(self): self.stopped += 1
            def wait_for_frames(self, **kwargs): return SimpleNamespace(get_color_frame=Frame)
        class Config:
            def enable_device(self, serial): pass
            def enable_stream(self, *args): pass
        class Control:
            def __init__(self, host): controls.append(self)
            def servoStop(self): return True
            def disconnect(self): pass
        class Receive:
            def __init__(self, host): receivers.append(self)
            def disconnect(self): pass
        class Gripper:
            def __init__(self, host): grippers.append(self)
            def get(self, name): return 0
            def halt(self): pass
            def close(self): pass
        fake_rs = SimpleNamespace(pipeline=Pipeline, config=Config,
            stream=SimpleNamespace(color=1), format=SimpleNamespace(rgb8=1))
        modules = {"pyrealsense2": fake_rs, "rtde_control": SimpleNamespace(RTDEControlInterface=Control),
            "rtde_receive": SimpleNamespace(RTDEReceiveInterface=Receive),
            "scipy.spatial.transform": SimpleNamespace(Rotation=object)}
        config = UR5eConfig(robot_ip="127.0.0.1", camera_serials={"primary":"mock-a", "wrist":"mock-b"},
                            home_joints=(0.0,)*6)
        environment = UR5eEmbodiment(config)
        with patch.dict("sys.modules", modules), patch.object(driver, "Gripper", Gripper):
            environment.connect_preview()
            observation = environment.observe_preview()
            self.assertEqual((len(controls), len(receivers), len(grippers)), (0, 0, 0))
            self.assertEqual(observation.state.shape, (0,))
            self.assertTrue(observation.preview_only)
            self.assertFalse(environment.armed.is_set())
            environment.connect()
            self.assertEqual(len(pipelines), 2)
            self.assertEqual((len(controls), len(receivers), len(grippers)), (1, 1, 1))
            environment.close()
        self.assertEqual([camera.started for camera in pipelines], [1, 1])
        self.assertEqual([camera.stopped for camera in pipelines], [1, 1])


if __name__ == "__main__":
    unittest.main()
