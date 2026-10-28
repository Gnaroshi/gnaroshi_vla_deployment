"""Explicit recovery keeps the policy alive and cannot reuse a cancelled action."""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest
import numpy as np
from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import RobotRecoveryRequired, SessionConfig
from gnaroshi_deployment.gui import DeploymentGUI
from gnaroshi_deployment.session import DeploymentSession


class KeptPolicy(MockPolicy):
    def __init__(self):
        super().__init__(.001)
        self.connect_calls = self.close_calls = 0
        self.episodes = []

    def connect(self):
        self.connect_calls += 1

    def close(self):
        self.close_calls += 1

    def reset(self, episode):
        self.episodes.append(episode)
        super().reset(episode)


class RecoverableEmbodiment(MockEmbodiment):
    def __init__(self):
        super().__init__()
        self.fault_at = None
        self.released = False
        self.reject_stop = False
        self.recover_calls = self.close_calls = 0
        self.recovery_entered = threading.Event()
        self.recovery_release = None

    def fault(self, phase):
        if self.fault_at == phase:
            raise RobotRecoveryRequired("Robot protective stop remains active", status={"mode": "protective_stop"})

    def home(self, cancel):
        self.fault("home")
        return super().home(cancel)

    def begin_episode(self):
        self.fault("begin")
        return super().begin_episode()

    def observe(self):
        self.fault("observe")
        return super().observe()

    def observe_preview(self):
        observation = super().observe()
        observation.state = np.empty(0, np.float32)
        return observation

    def apply(self, prediction):
        if len(self.applied) >= 3:
            self.fault("apply")
        return super().apply(prediction)

    def stop(self):
        super().stop()
        return not self.reject_stop

    def recover(self, cancel):
        self.recover_calls += 1
        self.recovery_entered.set()
        if self.recovery_release is not None:
            self.recovery_release.wait(2)
        if cancel.is_set():
            raise InterruptedError("Robot recovery cancelled")
        if not self.released:
            raise RobotRecoveryRequired("Release protective stop on the teach pendant")
        return {"healthy": True, "stop_confirmed": self.stop()}

    def close(self):
        self.close_calls += 1


class RecoverySessions(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            release = getattr(session.embodiment, "recovery_release", None)
            if release is not None:
                release.set()
            session.close()
            session.worker.join(3)
            self.assertTrue(session.shutdown_complete)
        self.temporary.cleanup()

    def wait(self, predicate):
        deadline = time.monotonic()+3
        while not predicate():
            if time.monotonic()>deadline:
                self.fail("Recovery transition timed out")
            time.sleep(.005)

    def make(self, policy=None, embodiment=None):
        config = SessionConfig(output_root=self.temporary.name, record_video=False, target_hz=200)
        session = DeploymentSession(config, policy or KeptPolicy(), embodiment or RecoverableEmbodiment(),
                                    Path(__file__).parents[1])
        self.sessions.append(session)
        session.start()
        self.wait(lambda: session.state == "READY")
        return session

    def fault(self, session, phase="apply"):
        session.embodiment.fault_at = phase
        session.start_rollout()
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)

    def test_shutdown_waits_for_native_recovery_before_closing_devices(self):
        class Pending(RecoverableEmbodiment):
            recovery_pending = False
        embodiment = Pending()
        session = self.make(embodiment=embodiment)
        embodiment.recovery_pending = True
        try:
            session.close()
            self.wait(lambda: session._cleanup_started)
            time.sleep(.06)
            self.assertFalse(session.shutdown_complete)
            self.assertEqual((embodiment.close_calls, session.policy.close_calls), (0, 0))
            self.assertFalse(session.signal("stop"))
        finally:
            embodiment.recovery_pending = False
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual((embodiment.close_calls, session.policy.close_calls), (1, 1))

    def latched_stop_arm(self):
        class LatchedStop(RecoverableEmbodiment):
            fail_stop = False
            recovery_error = None
            def stop(self):
                super().stop()
                if self.fail_stop:
                    self.recovery_error = RobotRecoveryRequired("Protective stop interrupted the stop acknowledgement")
                    return False
                return True
        return LatchedStop()

    def test_protective_stop_during_operator_stop_keeps_policy_and_error_record(self):
        embodiment = self.latched_stop_arm()
        session = self.make(embodiment=embodiment)
        session.start_rollout()
        self.wait(lambda: len(session.rows) >= 3)
        embodiment.fail_stop = True
        session.signal("stop")
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
        self.assertEqual((session.policy.connect_calls, session.policy.close_calls), (1, 0))
        self.assertEqual([row["outcome"] for row in session.results], ["failure"])
        result = json.loads(next(session.folder.glob("rollout_*/result.json")).read_text())
        self.assertEqual(result["outcome"], "failure")
        self.assertTrue(result["recovery_required"])
        embodiment.fail_stop = False
        embodiment.released = True
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "READY")

    def test_manual_success_before_stop_fault_is_preserved_without_return_home(self):
        embodiment = self.latched_stop_arm()
        session = self.make(embodiment=embodiment)
        session.start_rollout()
        self.wait(lambda: len(session.rows) >= 3)
        home_count = embodiment.home_count
        embodiment.fail_stop = True
        session.signal("success")
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
        self.assertEqual(embodiment.home_count, home_count)
        self.assertEqual(session.results[0]["outcome"], "success")
        self.assertEqual(session.results[0]["outcome_home"], "blocked_recovery")
        self.assertEqual(json.loads((session.folder/"results.json").read_text())["success_rate"], 1)
        embodiment.fail_stop = False

    def test_predispatch_stop_fault_enters_recovery_without_homing_or_policy_close(self):
        entered, release = threading.Event(), threading.Event()
        embodiment = self.latched_stop_arm()
        original_observe = embodiment.observe
        def blocked_observe():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("Test observation was not released")
            return original_observe()
        embodiment.observe = blocked_observe
        session = self.make(embodiment=embodiment)
        try:
            self.assertTrue(entered.wait(1))
            session.start_rollout()
            embodiment.fail_stop = True
            session.signal("stop")
        finally:
            release.set()
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
        self.assertEqual(embodiment.home_count, 0)
        self.assertEqual(session.policy.close_calls, 0)
        embodiment.fail_stop = False

    def test_shutdown_waits_for_stop_thread_without_racing_disconnect(self):
        entered, release = threading.Event(), threading.Event()
        class BlockedStop(RecoverableEmbodiment):
            def stop(self):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("Test stop was not released")
                return super().stop()
        embodiment = BlockedStop()
        session = self.make(embodiment=embodiment)
        try:
            session.close()
            self.assertTrue(entered.wait(1))
            self.wait(lambda: session._cleanup_started)
            time.sleep(.06)
            self.assertFalse(session.shutdown_complete)
            self.assertEqual((embodiment.close_calls, session.policy.close_calls), (0, 0))
            self.assertFalse(session.signal("stop"))
        finally:
            release.set()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual((embodiment.close_calls, session.policy.close_calls), (1, 1))

    def test_device_fault_records_error_and_recovery_keeps_policy_without_motion(self):
        session = self.make()
        self.fault(session)
        self.assertTrue(session.worker.is_alive())
        self.assertEqual((session.policy.connect_calls, session.policy.close_calls), (1, 0))
        self.assertEqual([row["outcome"] for row in session.results], ["failure"])
        error_files = list(session.folder.glob("rollout_*/result.json"))
        self.assertEqual(len(error_files), 1)
        original = error_files[0].read_bytes()
        result = json.loads(original)
        self.assertEqual(result["outcome"], "failure")
        self.assertTrue(result["recovery_required"])
        self.assertEqual(result["steps"], 3)
        self.assertTrue((error_files[0].parent/"steps.jsonl").exists())
        self.assertFalse(session.start_rollout())
        self.assertFalse(session.signal("success"))
        self.assertFalse(session.can_replace)
        home_count, command_count = session.embodiment.home_count, len(session.embodiment.applied)
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "RECOVERY_REQUIRED")
        self.assertIn("teach pendant", session.error)
        session.embodiment.released = True
        session.embodiment.fault_at = None
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "READY")
        self.assertEqual(session.embodiment.home_count, home_count)
        self.assertEqual(len(session.embodiment.applied), command_count)
        self.assertEqual((session.policy.connect_calls, session.policy.close_calls), (1, 0))
        self.assertTrue(session.policy.episodes[-1].startswith("recovery_"))
        self.assertEqual(error_files[0].read_bytes(), original)
        self.assertIsNone(session.error)
        self.assertFalse(session.cancel.is_set())
        self.assertTrue(session.start_rollout())
        self.wait(lambda: session.state == "RUNNING" and len(session.rows) >= 2)
        session.signal("failure")
        self.wait(lambda: session.state == "READY")
        self.assertEqual([row["outcome"] for row in session.results], ["failure", "failure"])

    def test_home_begin_and_observe_faults_do_not_destroy_worker(self):
        for phase in ("home", "begin", "observe"):
            with self.subTest(phase=phase):
                session = self.make()
                self.fault(session, phase)
                self.assertTrue(session.worker.is_alive())
                self.assertEqual(session.policy.close_calls, 0)
                self.assertEqual([row["outcome"] for row in session.results], ["failure"])
                self.assertFalse(session.start_rollout())
                if phase != "observe":
                    self.assertEqual(len(list(session.folder.glob("rollout_*/result.json"))), 1)

    def test_old_false_stop_clears_only_after_fresh_verified_recovery(self):
        session = self.make()
        session.embodiment.reject_stop = True
        self.fault(session)
        self.assertTrue(session._stop_failed)
        session.embodiment.released = True
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "RECOVERY_REQUIRED")
        self.assertTrue(session._stop_failed)
        self.assertFalse(session._stop_confirmed)
        self.assertFalse(session.start_rollout())
        session.embodiment.reject_stop = False
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "READY")
        self.assertFalse(session._stop_failed)
        self.assertTrue(session._stop_confirmed)

    def test_recovery_waits_for_all_fault_stop_threads_to_finish(self):
        entered, release = threading.Event(), threading.Event()
        class SlowStop(RecoverableEmbodiment):
            def stop(self):
                entered.set()
                release.wait(2)
                return super().stop()
        session = self.make(embodiment=SlowStop())
        session.embodiment.fault_at = "apply"
        session.start_rollout()
        self.assertTrue(entered.wait(1))
        try:
            self.assertFalse(session.can_recover)
            self.assertFalse(session.recover_robot())
            self.assertEqual(session.embodiment.recover_calls, 0)
        finally:
            release.set()
        self.wait(lambda: session.can_recover)

    def test_stop_during_recovery_rejects_late_success_and_allows_later_explicit_retry(self):
        session = self.make()
        self.fault(session)
        session.embodiment.released = True
        session.embodiment.recovery_release = threading.Event()
        self.assertTrue(session.recover_robot())
        self.assertFalse(session.recover_robot())
        self.assertTrue(session.embodiment.recovery_entered.wait(1))
        self.assertTrue(session.signal("stop"))
        self.assertFalse(session.start_rollout())
        session.embodiment.recovery_release.set()
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
        self.assertFalse(any(episode.startswith("recovery_") for episode in session.policy.episodes))
        self.assertTrue(session.cancel.is_set())
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "READY")

    def test_exit_during_recovery_never_resets_or_rearms(self):
        session = self.make()
        self.fault(session)
        session.embodiment.released = True
        session.embodiment.recovery_release = threading.Event()
        self.assertTrue(session.recover_robot())
        self.assertTrue(session.embodiment.recovery_entered.wait(1))
        session.close()
        session.embodiment.recovery_release.set()
        self.wait(lambda: session.shutdown_complete)
        self.assertNotEqual(session.state, "READY")
        self.assertFalse(any(episode.startswith("recovery_") for episode in session.policy.episodes))
        self.assertEqual(session.policy.close_calls, 1)

    def test_untyped_policy_error_is_fatal_and_cannot_enable_recovery(self):
        class BadPolicy(KeptPolicy):
            def infer(self, observation, instruction):
                raise RuntimeError("Model transport failed")
        session = self.make(policy=BadPolicy())
        session.start_rollout()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertFalse(session.can_recover)
        self.assertFalse(session.recover_robot())

    def test_policy_failure_with_simultaneous_typed_stop_fault_remains_fatal(self):
        class BadPolicy(KeptPolicy):
            def infer(self, observation, instruction):
                raise RuntimeError("Model transport failed")
        embodiment = self.latched_stop_arm()
        session = self.make(policy=BadPolicy(), embodiment=embodiment)
        embodiment.fail_stop = True
        session.start_rollout()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertIn("Model transport failed", session.error)
        self.assertFalse(session.can_recover)
        self.assertFalse(session.recover_robot())
        self.assertEqual(session.policy.close_calls, 1)
        result = json.loads(next(session.folder.glob("rollout_*/result.json")).read_text())
        self.assertEqual(result["outcome"], "failure")
        self.assertFalse(result["recovery_required"])

    def test_typed_stop_fault_at_max_steps_keeps_policy_for_recovery(self):
        embodiment = self.latched_stop_arm()
        session = self.make(embodiment=embodiment)
        session.config.max_steps = 3
        embodiment.fail_stop = True
        session.start_rollout()
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
        self.assertEqual(session.policy.close_calls, 0)
        self.assertEqual(len(session.rows), 3)
        result = json.loads(next(session.folder.glob("rollout_*/result.json")).read_text())
        self.assertEqual(result["outcome"], "failure")
        self.assertTrue(result["recovery_required"])
        embodiment.fail_stop = False
        embodiment.released = True
        self.assertTrue(session.recover_robot())
        self.wait(lambda: session.state == "READY")

    def test_device_fault_type_from_policy_cannot_enable_robot_recovery(self):
        class BadPolicy(KeptPolicy):
            def infer(self, observation, instruction):
                raise RobotRecoveryRequired("Model adapter misclassified its failure")
        session = self.make(policy=BadPolicy())
        session.start_rollout()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertFalse(session.can_recover)
        self.assertFalse(session.recover_robot())

    def test_policy_reset_cancelled_by_stop_does_not_rearm(self):
        entered, release = threading.Event(), threading.Event()
        class ResetPolicy(KeptPolicy):
            def reset(self, episode):
                if episode.startswith("recovery_"):
                    entered.set()
                    release.wait(2)
                super().reset(episode)
        session = self.make(policy=ResetPolicy())
        self.fault(session)
        session.embodiment.released = True
        self.assertTrue(session.recover_robot())
        self.assertTrue(entered.wait(1))
        try:
            session.signal("stop")
            self.assertFalse(session.start_rollout())
        finally:
            release.set()
        self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
        self.assertTrue(session.cancel.is_set())
        self.assertEqual(session.policy.close_calls, 0)

    def test_stop_failure_without_typed_device_fault_cannot_enable_recovery(self):
        session = self.make()
        session.embodiment.reject_stop = True
        session.start_rollout()
        self.wait(lambda: session.state == "RUNNING" and len(session.rows) >= 2)
        session.signal("stop")
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertFalse(session.can_recover)
        self.assertFalse(session.recover_robot())

    def test_protective_stop_during_outcome_home_preserves_manual_outcome(self):
        class OutcomeHomeFault(RecoverableEmbodiment):
            def home(self, cancel):
                if self.home_count == 1:
                    raise RobotRecoveryRequired("Protective stop while returning home")
                return super().home(cancel)
        session = self.make(embodiment=OutcomeHomeFault())
        session.start_rollout()
        self.wait(lambda: session.state == "RUNNING" and len(session.rows) >= 2)
        session.signal("success")
        self.wait(lambda: session.can_recover)
        self.assertEqual([row["outcome"] for row in session.results], ["success"])
        saved = json.loads((session.folder/"results.json").read_text())
        self.assertEqual(saved["success_rate"], 1)
        result = json.loads(next(session.folder.glob("rollout_*/result.json")).read_text())
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["outcome_home"], "error")
        self.assertTrue(session.worker.is_alive())


@unittest.skipUnless(os.environ.get("DISPLAY"), "Recovery GUI tests need a display or Xvfb")
class RecoveryGUI(unittest.TestCase):
    def test_gui_exposes_explicit_recovery_and_keeps_same_session(self):
        try:
            root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        root.withdraw()
        with tempfile.TemporaryDirectory() as temporary:
            policy, embodiment = KeptPolicy(), RecoverableEmbodiment()
            session = DeploymentSession(SessionConfig(output_root=temporary, record_video=False, target_hz=200),
                                        policy, embodiment, Path(__file__).parents[1])
            app = DeploymentGUI(root, session)
            def wait(predicate):
                deadline = time.monotonic()+3
                while not predicate():
                    if time.monotonic()>deadline:
                        self.fail("Recovery GUI transition timed out")
                    root.update()
                    time.sleep(.005)
            try:
                wait(lambda: session.state == "READY")
                embodiment.fault_at = "apply"
                app.start()
                wait(lambda: session.can_recover)
                app.refresh()
                self.assertTrue(app.recover_button.grid_info())
                self.assertEqual(str(app.start_button.cget("state")), "disabled")
                self.assertEqual(str(app.success_button.cget("state")), "disabled")
                embodiment.released = True
                embodiment.fault_at = None
                self.assertTrue(app.recover_robot())
                wait(lambda: session.state == "READY")
                self.assertIs(app.session, session)
                self.assertEqual((policy.connect_calls, policy.close_calls), (1, 0))
            finally:
                app.close()
                session.worker.join(3)
                for callback in root.tk.call("after", "info"):
                    root.after_cancel(callback)
                root.destroy()


if __name__ == "__main__":
    unittest.main()
