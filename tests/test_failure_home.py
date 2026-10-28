"""Fault outcomes retain evidence; idle Home is explicit, isolated and cancellable."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import RobotRecoveryRequired, SessionConfig
from gnaroshi_deployment.records import RolloutRecorder
from gnaroshi_deployment.session import DeploymentSession


class CountedPolicy(MockPolicy):
    def __init__(self):
        super().__init__(.001)
        self.resets = []
        self.close_count = 0

    def reset(self, episode):
        self.resets.append(episode)
        super().reset(episode)

    def close(self):
        self.close_count += 1


class FaultEmbodiment(MockEmbodiment):
    def __init__(self, category="protective_stop", after=3):
        super().__init__()
        self.category, self.after = category, after
        self.fault_enabled = True
        self.released = False

    def apply(self, prediction):
        if self.fault_enabled and len(self.applied) >= self.after:
            raise RobotRecoveryRequired("Recorded device failure", category=self.category,
                status={"raw_action": prediction.action.tolist(), "action_translation_norm_m": .085})
        return super().apply(prediction)

    def observe_preview(self):
        result = super().observe()
        result.state = np.empty(0, np.float32)
        return result

    def recover(self, cancel):
        if not self.released:
            raise RobotRecoveryRequired("Release the fault first", category=self.category)
        self.fault_enabled = False
        return {"healthy": True, "stop_confirmed": True}


class FailureAndHome(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sessions = []

    def tearDown(self):
        for session in self.sessions:
            session.close()
            session.worker.join(3)
            self.assertTrue(session.shutdown_complete)
        self.temporary.cleanup()

    def wait(self, predicate):
        deadline = time.monotonic()+3
        while not predicate():
            if time.monotonic()>deadline:
                self.fail("State transition timed out")
            time.sleep(.005)

    def make(self, *, policy=None, embodiment=None, video=False):
        config = SessionConfig(output_root=self.temporary.name, record_video=video, target_hz=200)
        session = DeploymentSession(config, policy or CountedPolicy(), embodiment or MockEmbodiment(),
                                    Path(__file__).parents[1])
        self.sessions.append(session)
        session.start()
        self.wait(lambda: session.state == "READY")
        return session

    def test_robot_fault_categories_count_failure_and_recovery_never_duplicates_it(self):
        for category in ("protective_stop", "robot_disconnected", "ik_no_solution", "servo_rejected"):
            with self.subTest(category=category):
                arm = FaultEmbodiment(category)
                session = self.make(embodiment=arm)
                session.save("Operator's existing note")
                self.assertTrue(session.start_rollout())
                self.wait(lambda: session.can_recover)
                self.assertEqual(len(session.results), 1)
                result = session.results[0]
                self.assertEqual((result["outcome"], result["outcome_source"]), ("failure", "automatic_fault"))
                self.assertEqual(result["failure_category"], category)
                self.assertEqual(result["steps"], 3)
                self.assertEqual(result["outcome_home"], "not_requested")
                self.assertEqual(arm.home_count, 1)
                self.assertIn("85.00 mm", result["failure_note"])
                folder = session.folder/result["directory"]
                rows = [json.loads(line) for line in (folder/"steps.jsonl").read_text().splitlines()]
                self.assertEqual(len(rows), 4)
                self.assertEqual(rows[-1]["step_status"], "failed")
                self.assertEqual(rows[-1]["command_outcome"], "unconfirmed")
                self.assertNotIn("command_monotonic_s", rows[-1])
                self.assertEqual(len([row for row in rows if "command_monotonic_s" in row]), 3)
                self.assertEqual(json.loads((folder/"failed_step.json").read_text())["failure"]["category"], category)
                saved = json.loads((session.folder/"results.json").read_text())
                self.assertEqual((saved["successes"], saved["failures"], saved["success_rate"]), (0, 1, 0))
                self.assertEqual(saved["notes"], "Operator's existing note")
                self.assertIn(category, (session.folder/"notes.txt").read_text())
                old = (folder/"result.json").read_bytes()
                arm.released = True
                self.assertTrue(session.recover_robot())
                self.wait(lambda: session.state == "READY")
                session.save("Edited note")
                self.assertEqual(len(session.results), 1)
                self.assertEqual((folder/"result.json").read_bytes(), old)
                self.assertEqual(len(session.automatic_notes), 1)
                self.assertIn(category, (session.folder/"notes.txt").read_text())
                self.assertIn("Edited note", (session.folder/"notes.txt").read_text())

    def test_first_action_fault_retains_policy_input_videos_without_faking_command(self):
        import cv2
        session = self.make(embodiment=FaultEmbodiment("ik_no_solution", after=0), video=True)
        session.start_rollout()
        self.wait(lambda: session.can_recover)
        result = session.results[0]
        self.assertEqual(result["steps"], 0)
        self.assertIsNone(result["achieved_command_hz"])
        folder = session.folder/result["directory"]
        rows = [json.loads(line) for line in (folder/"steps.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["step_status"], "failed")
        self.assertNotIn("command_monotonic_s", rows[0])
        for role in ("primary", "wrist"):
            capture = cv2.VideoCapture(str(folder/f"{role}.mp4"))
            try:
                okay, image = capture.read()
                self.assertTrue(okay)
                self.assertEqual(image.shape[:2], (240, 320))
            finally:
                capture.release()
        frames = [json.loads(line) for line in (folder/"frames.jsonl").read_text().splitlines()]
        self.assertTrue(any(row["kind"] == "observation" for row in frames))
        self.assertTrue(any(row["kind"] == "video_frames" for row in frames))

    def test_failed_step_distinguishes_no_ik_attempt_from_acknowledged_arm_motion(self):
        cases = [("ik_no_solution", False, False, "not_attempted"),
                 ("robot_disconnected", True, True, "acknowledged"),
                 ("servo_command_failed", True, None, "unconfirmed")]
        for category, attempted, sent, expected in cases:
            with self.subTest(category=category):
                class StatusArm(FaultEmbodiment):
                    def apply(self, prediction):
                        raise RobotRecoveryRequired("Robot command failed", category=category,
                            status={"arm_command_attempted": attempted, "arm_command_sent": sent,
                                    "arm_command_acknowledged": sent is True,
                                    "gripper_command_sent": False})
                session = self.make(embodiment=StatusArm(after=0))
                session.start_rollout()
                self.wait(lambda: session.can_recover)
                result = session.results[0]
                failed = json.loads((session.folder/result["directory"]/"failed_step.json").read_text())
                self.assertEqual(failed["command_outcome"], expected)
                self.assertNotIn("command_monotonic_s", failed)
                self.assertEqual(result["steps"], 0)
                self.assertFalse(result["failure_status"]["gripper_command_sent"])

    def test_nan_prediction_keeps_failure_evidence_and_cannot_enable_recovery(self):
        class BadPolicy(CountedPolicy):
            def infer(self, observation, instruction):
                value = super().infer(observation, instruction)
                value.action[:] = np.nan
                return value
        session = self.make(policy=BadPolicy())
        session.start_rollout()
        self.wait(lambda: session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")
        self.assertEqual(session.results[0]["outcome"], "failure")
        folder = session.folder/session.results[0]["directory"]
        failed = json.loads((folder/"failed_step.json").read_text())
        self.assertEqual(failed["raw_action"], ["nan"]*7)
        self.assertEqual(failed["failure"]["category"], "invalid_action")
        self.assertEqual(session.results[0]["steps"], 0)
        self.assertFalse(session.can_recover)
        self.assertEqual(session.embodiment.applied, [])
        self.assertEqual(session.embodiment.home_count, 1)

    def test_inference_disconnect_preserves_observation_and_prior_executed_commands(self):
        class BrokenPolicy(CountedPolicy):
            def infer(self, observation, instruction):
                if self.steps >= 2:
                    raise ConnectionError("Policy transport disconnected")
                return super().infer(observation, instruction)
        session = self.make(policy=BrokenPolicy())
        session.start_rollout()
        self.wait(lambda: session.shutdown_complete)
        result = session.results[0]
        self.assertEqual((result["outcome"], result["failure_phase"], result["steps"]), ("failure", "infer", 2))
        rows = [json.loads(line) for line in (session.folder/result["directory"]/"steps.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1]["command_outcome"], "not_attempted")
        self.assertNotIn("raw_action", rows[-1])
        self.assertEqual(session.embodiment.home_count, 1)

    def test_recorder_close_failure_does_not_erase_recorded_actions_or_failure_result(self):
        original_close = RolloutRecorder.close
        def bad_close(recorder):
            original_close(recorder)
            raise RuntimeError("Encoder could not finalize")
        session = self.make()
        with patch.object(RolloutRecorder, "close", bad_close):
            session.start_rollout()
            self.wait(lambda: len(session.rows) >= 3)
            session.signal("stop")
            self.wait(lambda: session.shutdown_complete)
        result = session.results[0]
        self.assertEqual(result["outcome"], "failure")
        self.assertEqual(result["failure_category"], "recording_error")
        self.assertGreaterEqual(result["steps"], 3)
        self.assertTrue(result["recording_errors"])
        self.assertTrue((session.folder/result["directory"]/"steps.jsonl").exists())
        self.assertEqual(session.embodiment.home_count, 1)

    def test_start_home_failure_is_a_zero_command_attempt_failure(self):
        class BadHome(FaultEmbodiment):
            def home(self, cancel):
                raise RobotRecoveryRequired("Disconnected before home", category="robot_disconnected")
        session = self.make(embodiment=BadHome())
        session.start_rollout()
        self.wait(lambda: session.can_recover)
        result = session.results[0]
        self.assertEqual((result["outcome"], result["failure_phase"], result["steps"]), ("failure", "home", 0))
        self.assertEqual(session.policy.resets, [])

    def test_idle_home_does_not_open_rollout_reset_policy_or_use_start_home(self):
        class PoseOnlyHome(MockEmbodiment):
            idle_home_calls = 0
            def home_idle(self, cancel):
                self.idle_home_calls += 1
                if cancel.wait(.02):
                    raise InterruptedError("Idle home cancelled")
        arm = PoseOnlyHome()
        session = self.make(embodiment=arm)
        self.assertTrue(session.can_home)
        self.assertTrue(session.home_robot())
        self.assertFalse(session.home_robot())
        self.assertFalse(session.start_rollout())
        self.wait(lambda: session.state == "READY" and session.can_home)
        self.assertEqual(arm.idle_home_calls, 1)
        self.assertEqual(arm.home_count, 0)
        self.assertEqual(session.policy.resets, [])
        self.assertEqual(session.results, [])
        self.assertEqual(list(session.folder.glob("rollout_*")), [])
        session.start_rollout()
        self.wait(lambda: len(session.rows) >= 2)
        self.assertFalse(session.can_home)
        self.assertFalse(session.home_robot())
        session.signal("stop")
        self.wait(lambda: session.can_home)

    def test_fault_while_home_is_queued_retains_explicit_recovery_state(self):
        entered, release = threading.Event(), threading.Event()
        fault_ready, dispatch_release = threading.Event(), threading.Event()
        class FaultDuringQueuedHome(MockEmbodiment):
            fault_observed = False
            def observe(self):
                if not self.fault_observed:
                    self.fault_observed = True
                    entered.set()
                    if not release.wait(2):
                        raise TimeoutError("Test observation was not released")
                    raise RobotRecoveryRequired("Protective stop observed before Home dispatch",
                                                category="protective_stop")
                return super().observe()
            def recover(self, cancel):
                return {"healthy": True, "stop_confirmed": True}
        arm = FaultDuringQueuedHome()
        session = self.make(embodiment=arm)
        require_recovery = session._require_recovery
        def latch_before_dispatch(error):
            require_recovery(error)
            fault_ready.set()
            if not dispatch_release.wait(2):
                raise TimeoutError("Test recovery dispatch was not released")
        session._require_recovery = latch_before_dispatch
        try:
            self.assertTrue(entered.wait(1))
            self.assertTrue(session.home_robot())
            release.set()
            self.assertTrue(fault_ready.wait(1))
            self.wait(lambda: session.state == "RECOVERY_REQUIRED" and session.can_recover)
            self.assertTrue(session.cancel.is_set())
            self.assertTrue(session._recovery_fault)
            self.assertFalse(session.can_home)
            self.assertFalse(session.home_robot())
            self.assertFalse(session.start_rollout())
            self.assertEqual(arm.home_count, 0)
            self.assertEqual(session.policy.resets, [])
            self.assertEqual(session.results, [])
            self.assertEqual(len(session.automatic_notes), 1)
            # The stale Home is ahead of this newer Recover in the worker queue.
            self.assertTrue(session.recover_robot())
            self.assertEqual(session.state, "RECOVERING")
        finally:
            release.set()
            dispatch_release.set()
        self.wait(lambda: session.state == "READY" and session.can_home)
        self.assertFalse(session._recovery_fault)
        self.assertEqual(arm.home_count, 0)

    def test_stop_during_idle_home_waits_for_stop_ack_and_never_starts_episode(self):
        entered, stop_entered, release = threading.Event(), threading.Event(), threading.Event()
        class SlowHome(MockEmbodiment):
            def home_idle(self, cancel):
                entered.set()
                cancel.wait(2)
                raise InterruptedError("Idle home cancelled")
            def stop(self):
                stop_entered.set()
                release.wait(2)
                return super().stop()
        session = self.make(embodiment=SlowHome())
        try:
            self.assertTrue(session.home_robot())
            self.assertTrue(entered.wait(1))
            session.signal("stop")
            self.assertTrue(stop_entered.wait(1))
            self.assertFalse(session.can_home)
            self.assertFalse(session.start_rollout())
            time.sleep(.04)
            self.assertEqual(session.state, "STOPPING")
        finally:
            release.set()
        self.wait(lambda: session.can_home)
        self.assertEqual(session.policy.resets, [])
        self.assertEqual(session.results, [])

    def test_fault_during_idle_home_is_not_a_rollout_failure(self):
        class BadIdleHome(FaultEmbodiment):
            def home_idle(self, cancel):
                raise RobotRecoveryRequired("Protective stop while homing", category="protective_stop")
        session = self.make(embodiment=BadIdleHome())
        session.home_robot()
        self.wait(lambda: session.can_recover)
        self.assertFalse(session.can_home)
        self.assertEqual(session.results, [])
        self.assertEqual(list(session.folder.glob("rollout_*")), [])
        self.assertEqual(session.automatic_notes[0]["phase"], "idle_home")
        self.assertEqual(session.automatic_notes[0]["category"], "protective_stop")

    def test_idle_robot_fault_is_not_counted_as_an_attempt(self):
        class IdleFault(FaultEmbodiment):
            idle_fault = False
            def observe(self):
                if self.idle_fault:
                    raise RobotRecoveryRequired("Robot connection lost", category="robot_disconnected")
                return super().observe()
        arm = IdleFault()
        session = self.make(embodiment=arm)
        arm.idle_fault = True
        self.wait(lambda: session.can_recover)
        self.assertEqual(session.results, [])
        self.assertEqual(session.automatic_notes[0]["phase"], "idle_observe")
        self.assertFalse(session.can_home)


class RecorderClosure(unittest.TestCase):
    def test_pending_frame_error_still_closes_all_resources(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder = RolloutRecorder(Path(temporary)/"rollout", 15, False)
            writer = type("Writer", (), {"closed": False, "release": lambda self: setattr(self, "closed", True)})()
            recorder.writers["primary"] = writer
            with patch.object(recorder, "_emit_pending", side_effect=RuntimeError("Encoder failed")):
                with self.assertRaisesRegex(RuntimeError, "Encoder failed"):
                    recorder.close()
            self.assertTrue(recorder.log.closed)
            self.assertTrue(recorder.timestamps.closed)
            self.assertTrue(writer.closed)


if __name__ == "__main__":
    unittest.main()
