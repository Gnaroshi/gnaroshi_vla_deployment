"""CPU mock coverage for checkpoint replacement and its cancellation boundaries."""
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import tkinter as tk
import unittest

from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import SessionConfig
from gnaroshi_deployment.gui import DeploymentGUI
from gnaroshi_deployment.session import DeploymentSession


class CountPolicy(MockPolicy):
    def __init__(self):
        super().__init__(.001)
        self.connect_calls = self.close_calls = 0

    def connect(self):
        self.connect_calls += 1

    def close(self):
        self.close_calls += 1


class CountEmbodiment(MockEmbodiment):
    def __init__(self):
        super().__init__()
        self.connect_calls = self.close_calls = 0

    def connect(self):
        self.connect_calls += 1

    def close(self):
        self.close_calls += 1
        super().close()


class UnstartedCleanup(unittest.TestCase):
    def test_close_without_start_only_cleans_up_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            policy, embodiment = CountPolicy(), CountEmbodiment()
            session = DeploymentSession(SessionConfig(output_root=temporary, record_video=False),
                                        policy, embodiment, Path(__file__).parents[1])
            session.close()
            session.worker.join(3)
            self.assertTrue(session.shutdown_complete)
            self.assertTrue(session.replacement_closed)
            self.assertEqual(session.state, "CLOSED")
            self.assertEqual((policy.connect_calls, embodiment.connect_calls), (0, 0))
            self.assertEqual((policy.close_calls, embodiment.close_calls), (1, 1))
            session.close()
            self.assertEqual(session.state, "CLOSED")
            self.assertEqual((policy.close_calls, embodiment.close_calls), (1, 1))


@unittest.skipUnless(os.environ.get("DISPLAY"), "Checkpoint GUI tests need a display or Xvfb")
class CheckpointReload(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(f"Tk display unavailable: {error}")
        self.root.withdraw()
        self.temporary = tempfile.TemporaryDirectory()
        self.sessions, self.releases = [], []
        self.app = None

    def tearDown(self):
        for event in self.releases:
            event.set()
        if self.app is not None:
            self.app.close()
            self.wait(lambda: self.app.session.shutdown_complete and not self.app._reload_in_progress,
                      timeout=5)
            if self.app._reload_thread is not None:
                self.app._reload_thread.join(3)
                self.assertFalse(self.app._reload_thread.is_alive())
        for session in self.sessions:
            session.close()
            session.worker.join(3)
            self.assertTrue(session.shutdown_complete)
        try:
            # Each case owns its Tcl interpreter; do not leak scheduled refresh
            # callbacks into the next interpreter's event processing.
            for callback in self.root.tk.call("after", "info"):
                self.root.after_cancel(callback)
            self.root.destroy()
        except tk.TclError:
            pass
        self.temporary.cleanup()

    def wait(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Checkpoint transition timed out")
            try:
                self.root.update()
            except tk.TclError:
                pass
            time.sleep(.005)

    def settle(self, seconds=.08):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.root.update()
            time.sleep(.005)

    def make_session(self, config=None, policy=None, embodiment=None):
        config = config or SessionConfig(output_root=self.temporary.name, record_video=False,
            target_hz=200, max_steps=5000, metadata={"reload_selection": {"checkpoint": "checkpoint_a"}})
        session = DeploymentSession(config, policy or CountPolicy(), embodiment or CountEmbodiment(),
                                    Path(__file__).parents[1])
        self.sessions.append(session)
        return session

    def replacement(self, selection, previous):
        metadata = copy.deepcopy(previous.metadata)
        metadata["reload_selection"] = dict(selection)
        config = replace(previous, method="mock_reloaded",
                         output_root=str(Path(self.temporary.name) / selection["checkpoint"]),
                         metadata=metadata)
        return self.make_session(config=config)

    def launch(self, factory=None, policy=None, embodiment=None, wait_ready=True):
        session = self.make_session(policy=policy, embodiment=embodiment)
        self.app = DeploymentGUI(self.root, session,
            reload_fields=[{"key": "checkpoint", "label": "Checkpoint", "kind": "directory",
                            "value": "unused_default"}],
            reload_factory=factory or self.replacement)
        if wait_ready:
            self.wait(lambda: session.state == "READY")
        return session

    def test_factory_waits_for_old_stop_and_cleanup_and_runs_off_tk_thread(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        class DelayedStop(CountEmbodiment):
            def stop(self):
                entered.set()
                release.wait(3)
                return super().stop()
        calls = []
        main_thread = threading.get_ident()
        def factory(selection, previous):
            calls.append((threading.get_ident(), old.shutdown_complete, old.replacement_closed))
            return self.replacement(selection, previous)
        old = self.launch(factory=factory, embodiment=DelayedStop())
        self.assertEqual(self.app.reload_vars["checkpoint"].get(), "checkpoint_a")
        self.app.reload_vars["checkpoint"].set("checkpoint_b")
        self.assertTrue(self.app.load_checkpoint())
        self.assertTrue(entered.wait(1))
        self.settle()
        self.assertEqual(calls, [])
        self.assertFalse(self.app.load_checkpoint())
        self.assertFalse(old.replacement_closed)
        release.set()
        self.wait(lambda: self.app.session is not old and self.app.session.state == "READY"
                  and not self.app._reload_in_progress)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0][0], main_thread)
        self.assertEqual(calls[0][1:], (True, True))
        self.assertEqual(self.app.session.config.metadata["reload_selection"],
                         {"checkpoint": "checkpoint_b"})

    def test_checkpoint_panel_is_collapsed_and_restores_result_space_after_load(self):
        old = self.make_session()
        self.app = DeploymentGUI(self.root, old, reload_fields=[
            {"key": "checkpoint", "label": "Checkpoint", "kind": "directory"},
            {"key": "mode", "label": "Mode", "kind": "choice", "choices": ["default"], "value": "default"},
            {"key": "config", "label": "Config", "kind": "file"},
            {"key": "description", "label": "Description", "kind": "text"},
        ], reload_factory=self.replacement)
        self.wait(lambda: old.state == "READY")
        self.root.deiconify()
        self.root.geometry("1440x960")
        self.root.update()
        self.assertFalse(self.app._checkpoint_expanded)
        self.assertFalse(self.app.checkpoint_panel.grid_info())
        notebook = self.app.table.master.master
        for widget in (self.app.table, self.app.notes, self.app.details):
            notebook.select(widget.master)
            self.root.update_idletasks()
            self.assertGreaterEqual(widget.winfo_height(), 80)
            self.assertLessEqual(widget.winfo_rooty() + widget.winfo_height(),
                                 self.root.winfo_rooty() + self.root.winfo_height())
        self.app.checkpoint_toggle.invoke()
        self.root.update_idletasks()
        self.assertTrue(self.app._checkpoint_expanded)
        self.assertTrue(self.app.checkpoint_panel.grid_info())
        self.app.checkpoint_toggle.invoke()
        self.assertFalse(self.app.checkpoint_panel.grid_info())
        self.app.checkpoint_toggle.invoke()
        self.app.reload_vars["checkpoint"].set("checkpoint_b")
        self.assertTrue(self.app.load_checkpoint())
        self.assertFalse(self.app._checkpoint_expanded)
        self.wait(lambda: self.app.session is not old and self.app.session.state == "READY"
                  and not self.app._reload_in_progress)
        self.root.update_idletasks()
        self.assertFalse(self.app.checkpoint_panel.grid_info())
        self.assertEqual(self.app.reload_vars["checkpoint"].get(), "checkpoint_b")
        self.assertGreaterEqual(self.app.details.winfo_height(), 80)

    def test_reload_is_blocked_during_home_rollout_and_stopping(self):
        home_entered, home_release = threading.Event(), threading.Event()
        stop_entered, stop_release = threading.Event(), threading.Event()
        self.releases.extend((home_release, stop_release))
        class DelayedMotion(CountEmbodiment):
            def home(self, cancel):
                home_entered.set()
                home_release.wait(3)
                return super().home(cancel)
            def stop(self):
                stop_entered.set()
                stop_release.wait(3)
                return super().stop()
        calls = []
        def factory(selection, previous):
            calls.append(selection)
            return self.replacement(selection, previous)
        old = self.launch(factory=factory, embodiment=DelayedMotion())
        self.app.start()
        self.assertTrue(home_entered.wait(1))
        self.assertEqual(old.state, "HOMING")
        self.assertFalse(self.app.load_checkpoint())
        home_release.set()
        self.wait(lambda: old.state == "RUNNING")
        self.assertFalse(self.app.load_checkpoint())
        self.app.signal("stop")
        self.assertTrue(stop_entered.wait(1))
        self.assertEqual(old.state, "STOPPING")
        self.assertFalse(self.app.load_checkpoint())
        self.assertEqual(calls, [])
        stop_release.set()
        self.wait(lambda: old.state == "READY")

    def test_failed_stop_blocks_factory_even_after_worker_exits(self):
        class RejectStop(CountEmbodiment):
            def stop(self):
                super().stop()
                return False
        calls = []
        def factory(selection, previous):
            calls.append(selection)
            return self.replacement(selection, previous)
        old = self.launch(factory=factory, embodiment=RejectStop())
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: old.shutdown_complete and not self.app._reload_in_progress)
        self.assertEqual(old.state, "ERROR")
        self.assertFalse(old.can_replace)
        self.assertFalse(old.replacement_closed)
        self.assertFalse(self.app.load_checkpoint())
        self.assertEqual(calls, [])

    def test_factory_failure_keeps_old_records_and_allows_corrected_retry(self):
        calls = []
        def factory(selection, previous):
            calls.append(selection["checkpoint"])
            if selection["checkpoint"] == "missing_checkpoint":
                raise ValueError("Checkpoint is unavailable")
            return self.replacement(selection, previous)
        old = self.launch(factory=factory)
        self.app.notes.insert("1.0", "Preserve these experiment notes")
        self.app.reload_vars["checkpoint"].set("missing_checkpoint")
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: not self.app._reload_in_progress)
        self.assertIs(self.app.session, old)
        self.assertTrue(old.can_replace)
        self.assertIn("Checkpoint is unavailable", self.app.status.get())
        self.assertEqual((old.folder / "notes.txt").read_text(), "Preserve these experiment notes")
        self.app.reload_vars["checkpoint"].set("checkpoint_b")
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: self.app.session is not old and self.app.session.state == "READY"
                  and not self.app._reload_in_progress)
        self.assertEqual(calls, ["missing_checkpoint", "checkpoint_b"])
        self.assertEqual((old.folder / "notes.txt").read_text(), "Preserve these experiment notes")

    def test_factory_canonical_selection_is_preserved_and_shown(self):
        def factory(selection, previous):
            canonical = {"checkpoint": str((Path(self.temporary.name) / selection["checkpoint"]).resolve())}
            return self.replacement(canonical, previous)
        old = self.launch(factory=factory)
        self.app.reload_vars["checkpoint"].set("checkpoint_b")
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: self.app.session is not old and self.app.session.state == "READY"
                  and not self.app._reload_in_progress)
        canonical = str((Path(self.temporary.name) / "checkpoint_b").resolve())
        self.assertEqual(self.app.session.config.metadata["reload_selection"], {"checkpoint": canonical})
        self.assertEqual(self.app.session.config.metadata["reload_request"], {"checkpoint": "checkpoint_b"})
        self.assertEqual(self.app.reload_vars["checkpoint"].get(), canonical)

    def test_exit_during_factory_disposes_candidate_without_start_or_connect(self):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        candidates = []
        def factory(selection, previous):
            entered.set()
            release.wait(3)
            candidate = self.replacement(selection, previous)
            candidates.append(candidate)
            return candidate
        old = self.launch(factory=factory)
        self.assertTrue(self.app.load_checkpoint())
        self.wait(entered.is_set)
        self.app.close()
        self.assertTrue(self.app._reload_in_progress)
        release.set()
        self.wait(lambda: candidates and candidates[0].shutdown_complete
                  and not self.app._reload_in_progress)
        candidate = candidates[0]
        self.assertIs(self.app.session, old)
        self.assertEqual((candidate.policy.connect_calls, candidate.embodiment.connect_calls), (0, 0))
        self.assertEqual((candidate.policy.close_calls, candidate.embodiment.close_calls), (1, 1))
        self.assertTrue(candidate.replacement_closed)

    def test_failed_candidate_cleanup_blocks_retry_despite_safe_old_session(self):
        class RejectStop(CountEmbodiment):
            def stop(self):
                super().stop()
                return False
        candidates = []
        def factory(selection, previous):
            # Invalid task is rejected before start; its cleanup must still be safe.
            candidate = self.make_session(config=replace(previous, task="different_task"),
                                          embodiment=RejectStop())
            candidates.append(candidate)
            return candidate
        old = self.launch(factory=factory)
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: candidates and candidates[0].shutdown_complete
                  and not self.app._reload_in_progress)
        self.assertIs(self.app.session, old)
        self.assertTrue(old.can_replace)
        self.assertFalse(candidates[0].replacement_closed)
        self.assertFalse(self.app.load_checkpoint())
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].policy.connect_calls, 0)

    def test_saved_outcome_notes_and_media_survive_with_new_result_isolation(self):
        old = self.launch()
        self.app.start()
        self.wait(lambda: len(old.rows) >= 3)
        self.app.signal("success")
        self.wait(lambda: old.state == "READY" and len(old.results) == 1)
        self.app.notes.insert("1.0", "Notes for the first checkpoint")
        rollout = old.folder / old.results[0]["directory"]
        media = rollout / "primary.mp4"
        media.write_bytes(b"previously recorded media remains owned by its session")
        snapshot = {path: path.read_bytes() for path in rollout.iterdir() if path.is_file()}
        old_results = copy.deepcopy(old.results)
        self.app.reload_vars["checkpoint"].set("checkpoint_b")
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: self.app.session is not old and self.app.session.state == "READY"
                  and not self.app._reload_in_progress)
        new = self.app.session
        self.assertNotEqual(old.folder, new.folder)
        self.assertEqual(old.results, old_results)
        self.assertEqual(json.loads((old.folder / "results.json").read_text())["results"], old_results)
        self.assertEqual((old.folder / "notes.txt").read_text(), "Notes for the first checkpoint")
        self.assertTrue(all(path.read_bytes() == contents for path, contents in snapshot.items()))
        self.assertEqual(new.results, [])
        self.assertEqual(self.app.table.get_children(), ())
        self.assertEqual(self.app.notes.get("1.0", "end-1c"), "")
        self.assertIn(str(new.folder), self.app.folder_label.cget("text"))
        self.assertIn(new.config.method, self.root.title())
        self.app.start()
        self.wait(lambda: len(new.rows) >= 3)
        self.app.signal("failure")
        self.wait(lambda: new.state == "READY" and len(new.results) == 1)
        self.assertEqual(new.results[0]["outcome"], "failure")
        self.assertEqual(old.results, old_results)

    def test_policy_load_error_can_retry_after_safe_cleanup(self):
        class LoadError(CountPolicy):
            def connect(self):
                super().connect()
                raise ValueError("Unreadable checkpoint")
        old = self.launch(policy=LoadError(), wait_ready=False)
        self.wait(lambda: old.shutdown_complete)
        self.assertEqual(old.state, "ERROR")
        self.assertEqual(old.embodiment.connect_calls, 0)
        self.assertTrue(old.can_replace)
        self.assertTrue(self.app.load_checkpoint())
        self.wait(lambda: self.app.session is not old and self.app.session.state == "READY"
                  and not self.app._reload_in_progress)

    def test_runtime_error_cannot_be_rearmed_by_checkpoint_reload(self):
        class RuntimeErrorPolicy(CountPolicy):
            def infer(self, observation, instruction):
                raise RuntimeError("Prediction transport failed")
        old = self.launch(policy=RuntimeErrorPolicy())
        self.app.start()
        self.wait(lambda: old.shutdown_complete)
        self.assertEqual(old.state, "ERROR")
        self.assertTrue(old.replacement_closed)
        self.assertFalse(old.can_replace)
        self.assertFalse(self.app.load_checkpoint())


if __name__ == "__main__":
    unittest.main()
