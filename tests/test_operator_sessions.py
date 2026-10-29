"""Operator device changes and recording recovery without real hardware."""
import hashlib
import builtins
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np

from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import CameraError, Observation, RobotRecoveryRequired, SessionConfig
from gnaroshi_deployment.records import RolloutRecorder, summary
from gnaroshi_deployment.session import DeploymentSession


class CountPolicy(MockPolicy):
    def __init__(self):
        super().__init__(.001)
        self.connections = self.resets = self.closes = 0

    def connect(self):
        self.connections += 1

    def reset(self, episode_id):
        self.resets += 1
        super().reset(episode_id)

    def close(self):
        self.closes += 1


class OperatorSessions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.sessions, self.releases = [], []

    def tearDown(self):
        for event in self.releases:
            event.set()
        for session in self.sessions:
            session.close()
            session.worker.join(4)
            self.assertTrue(session.shutdown_complete)
        self.temp.cleanup()

    def wait(self, predicate, timeout=4):
        deadline = time.monotonic()+timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Operator state transition timed out")
            time.sleep(.005)

    def make(self, embodiment=None, policy=None, metadata=None, state="READY"):
        cfg = SessionConfig(output_root=self.temp.name, target_hz=100, max_steps=100,
                            record_video=False, metadata=metadata or {})
        session = DeploymentSession(cfg, policy or CountPolicy(), embodiment or MockEmbodiment(),
                                    Path(__file__).parents[1])
        self.sessions.append(session)
        session.start()
        self.wait(lambda: session.state == state)
        return session

    def test_camera_scan_and_change_preserve_policy_notes_and_stationary_robot(self):
        session = self.make()
        session.save("Operator notes remain here")
        self.assertTrue(session.request_device_scan())
        self.wait(lambda: not session.device_status["scan_busy"])
        self.assertEqual(len(session.device_status["devices"]), 3)
        self.assertTrue(session.request_camera_update({"primary":"mock-extra", "wrist":"mock-wrist"}))
        self.assertFalse(session.start_rollout())
        self.assertFalse(session.home_robot())
        self.wait(lambda: not session.device_status["busy"])
        self.assertEqual(session.state, "READY")
        self.assertEqual(session.policy.connections, 1)
        self.assertEqual(session.policy.closes, 0)
        self.assertEqual(session.policy.resets, 1)
        self.assertEqual(session.embodiment.home_count, 0)
        self.assertEqual(session.embodiment.applied, [])
        self.assertEqual(session._notes, "Operator notes remain here")
        manifest = json.loads((session.folder/"manifest.json").read_text())
        self.assertEqual(manifest["embodiment"]["camera_serials"]["primary"], "mock-extra")

    def test_camera_selection_rejects_duplicates_active_rollout_and_missing_device(self):
        session = self.make()
        with self.assertRaises(ValueError):
            session.request_camera_update({"primary":"mock-extra", "wrist":"mock-extra"})
        session.start_rollout()
        self.wait(lambda: len(session.rows) > 1)
        self.assertFalse(session.request_camera_update({"primary":"mock-extra", "wrist":"mock-wrist"}))
        session.signal("retry")
        self.wait(lambda: session.state == "READY")
        self.assertTrue(session.request_camera_update({"primary":"missing", "wrist":"mock-wrist"}))
        self.wait(lambda: session.state == "CAMERA_ERROR" and not session.device_status["busy"])
        self.assertTrue(session.worker.is_alive())
        self.assertEqual(session.policy.connections, 1)
        self.assertEqual(session.device_status["selected"]["primary"], "mock-primary")
        self.assertTrue(session.request_camera_reconnect())
        self.wait(lambda: session.state == "READY")

    def test_camera_failure_saves_attempt_and_reconnect_does_not_restart_model_or_home(self):
        class FailingCamera(MockEmbodiment):
            broken = False
            def observe(self):
                if self._episode_active and len(self.applied) >= 2:
                    self.broken = True
                if self.broken:
                    raise CameraError("USB camera disconnected", category="camera_disconnected")
                return super().observe()
            def configure_cameras(self, serials, cancel):
                self.broken = False
                return super().configure_cameras(serials, cancel)
        session = self.make(embodiment=FailingCamera())
        session.start_rollout()
        self.wait(lambda: session.state == "CAMERA_ERROR" and len(session.results) == 1 and not session._rollout_active)
        result = session.results[0]
        self.assertEqual((result["outcome"], result["outcome_source"]), ("failure", "automatic_fault"))
        self.assertEqual(result["failure_category"], "camera_disconnected")
        recorded = session.folder/result["directory"]/"steps.jsonl"
        before = recorded.read_bytes()
        homes = session.embodiment.home_count
        self.assertTrue(session.request_camera_reconnect())
        self.wait(lambda: session.state == "READY")
        self.assertEqual(recorded.read_bytes(), before)
        self.assertEqual(session.embodiment.home_count, homes)
        self.assertEqual(session.policy.connections, 1)
        self.assertEqual(session.policy.closes, 0)
        self.assertEqual(len(session.results), 1)

    def test_wrong_initial_camera_can_be_repaired_before_policy_load(self):
        class Preview(MockEmbodiment):
            def connect_preview(self):
                raise CameraError("Initial camera is missing", category="camera_not_found")
            def observe_preview(self):
                observation = super().observe()
                observation.state = np.empty(0)
                return observation
        session = self.make(embodiment=Preview(), metadata={"preview_before_policy":True,
                            "defer_policy_load":True}, state="CAMERA_ERROR")
        self.assertEqual(session.policy.connections, 0)
        self.assertTrue(session.request_camera_update({"primary":"mock-extra", "wrist":"mock-wrist"}))
        self.wait(lambda: session.state == "SELECTING")
        self.assertEqual(session.policy.connections, 0)
        self.assertEqual(session.embodiment.home_count, 0)

    def test_camera_repair_during_model_load_returns_to_loading(self):
        release = threading.Event()
        self.releases.append(release)
        class LoadingPolicy(CountPolicy):
            def connect(self):
                super().connect()
                release.wait(3)
        class Preview(MockEmbodiment):
            broken = True
            def connect_preview(self): pass
            def observe_preview(self):
                if self.broken:
                    raise CameraError("Preview lost", category="camera_disconnected")
                observation = super().observe()
                observation.state = np.empty(0)
                return observation
            def configure_cameras(self, serials, cancel):
                self.broken = False
                return super().configure_cameras(serials, cancel)
        session = self.make(embodiment=Preview(), policy=LoadingPolicy(),
                            metadata={"preview_before_policy":True}, state="CAMERA_ERROR")
        self.assertTrue(session.request_camera_reconnect())
        self.wait(lambda: not session.device_status["busy"])
        self.assertEqual(session.state, "LOADING_POLICY")
        self.assertFalse(session.can_replace)
        release.set()
        self.wait(lambda: session.state == "READY")
        self.assertEqual(session.policy.connections, 1)

    def test_robot_fault_during_camera_change_preserves_recovery_flow(self):
        class FaultOnStop(MockEmbodiment):
            fault_on_stop = False
            def stop(self):
                if self.fault_on_stop:
                    self.fault_on_stop = False
                    raise RobotRecoveryRequired("Protective stop", category="protective_stop")
                return super().stop()
        session = self.make(embodiment=FaultOnStop())
        session.embodiment.fault_on_stop = True
        session.request_camera_reconnect()
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and not session.device_status["busy"])
        self.assertTrue(session.worker.is_alive())
        self.assertTrue(session._recovery_fault)
        self.assertEqual(session.policy.closes, 0)
        self.assertEqual(session.embodiment.home_count, 0)

    def test_stop_cancels_camera_transaction_and_cannot_start_motion(self):
        entered = threading.Event()
        class BlockingCameras(MockEmbodiment):
            block = True
            def configure_cameras(self, serials, cancel):
                if self.block:
                    entered.set()
                    if not cancel.wait(2):
                        raise TimeoutError("Test cancellation was not received")
                    raise InterruptedError("Operator cancelled camera operation")
                return super().configure_cameras(serials, cancel)
        session = self.make(embodiment=BlockingCameras())
        session.request_camera_reconnect()
        self.assertTrue(entered.wait(1))
        session.signal("stop")
        self.wait(lambda: session.state == "CAMERA_ERROR" and session.device_status["can_apply"])
        self.assertFalse(session.start_rollout())
        session.embodiment.block = False
        self.assertTrue(session.request_camera_reconnect())
        self.wait(lambda: session.state == "READY")
        self.assertEqual(session.embodiment.home_count, 0)
        self.assertEqual(session.embodiment.applied, [])

    def test_low_storage_blocks_start_before_home_and_can_be_rechecked(self):
        session = self.make()
        with patch("gnaroshi_deployment.session.shutil.disk_usage", return_value=SimpleNamespace(free=100, total=1000)):
            session._update_storage_health(force=True)
            self.assertFalse(session.start_rollout())
            self.assertFalse(session.operator_health["can_start"])
            self.assertEqual(session.embodiment.home_count, 0)
        session._update_storage_health(force=True)
        self.assertTrue(session.start_rollout())
        self.wait(lambda: len(session.rows) > 1)
        session.signal("retry")
        self.wait(lambda: session.state == "READY")

    def test_saved_annotation_and_recording_setting_preserve_outcome_and_raw_log(self):
        session = self.make()
        self.assertTrue(session.apply_recording(True))
        self.assertTrue(session.apply_recording(False))
        session.start_rollout()
        self.wait(lambda: len(session.rows) > 2)
        self.assertFalse(session.apply_recording(True))
        session.signal("success")
        self.wait(lambda: session.state == "READY")
        result = session.results[0]
        raw = session.folder/result["directory"]/"steps.jsonl"
        before = hashlib.sha256(raw.read_bytes()).hexdigest()
        self.assertTrue(session.annotate_result(result["id"], "Contact was delayed"))
        self.assertEqual(hashlib.sha256(raw.read_bytes()).hexdigest(), before)
        saved = json.loads((raw.parent/"result.json").read_text())
        self.assertEqual(saved["outcome"], "success")
        self.assertEqual(saved["operator_note"], "Contact was delayed")
        self.assertFalse(session.annotate_result(9999, "Unknown attempt"))

    def test_recording_dependency_failure_is_reported_before_changing_config_or_moving(self):
        session = self.make()
        actual_import = builtins.__import__
        def missing_cv2(name, *args, **kwargs):
            if name == "cv2":
                raise ImportError("Test: video dependency is unavailable")
            return actual_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=missing_cv2):
            with self.assertRaisesRegex(RuntimeError, "OpenCV"):
                session.apply_recording(True)
        self.assertFalse(session.config.record_video)
        self.assertEqual(session.embodiment.home_count, 0)
        self.assertEqual(session.embodiment.applied, [])


class RecordingMetrics(unittest.TestCase):
    def test_gpu_latency_is_available_per_executed_action_not_only_per_rpc(self):
        rows = [{"command_monotonic_s":i*.1, "deadline_miss":False, "buffered":i%10!=0,
                 "model_gpu_ms":300 if i%10==0 else None,
                 "rpc_ms":310 if i%10==0 else None, "backbone_call_delta":int(i%10==0)} for i in range(20)]
        result = summary(rows)
        self.assertEqual(result["latency"]["model_gpu_ms"]["mean_ms"], 300)
        self.assertEqual(result["latency"]["model_gpu_ms"]["amortized_ms_per_executed_action"], 30)
        self.assertEqual((result["backbone_calls"], result["rpc_requests"], result["client_buffered_steps"]), (2,2,18))

    def test_changed_video_dimensions_and_roles_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = RolloutRecorder(Path(temporary)/"rollout", 15, False)
            try:
                recorder.prepare({"primary":np.zeros((2,2,3), np.uint8)})
                with self.assertRaisesRegex(ValueError, "image size"):
                    recorder.prepare({"primary":np.zeros((4,4,3), np.uint8)})
                with self.assertRaisesRegex(ValueError, "roles changed"):
                    recorder.prepare({"wrist":np.zeros((2,2,3), np.uint8)})
            finally:
                recorder.close()
                recorder.close()


if __name__ == "__main__":
    unittest.main()
