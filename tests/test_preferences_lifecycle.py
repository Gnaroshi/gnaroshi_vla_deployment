"""Native presentation state across sessions and process exit; CPU mocks only."""
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import tkinter as tk
import unittest
from unittest.mock import patch

from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import SessionConfig
from gnaroshi_deployment.gui import DeploymentGUI
from gnaroshi_deployment.session import DeploymentSession
from gnaroshi_deployment.workspace import WorkspacePreferences


REPOSITORY = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("DISPLAY"), "Native presentation tests need a private display")
class PresentationLifecycle(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.config_directory = self.directory / "settings" / "profiles"
        self.preference_file = self.config_directory.parent / "ui_preferences.json"
        self.apps, self.roots, self.sessions = [], [], []

    def tearDown(self):
        for app in reversed(self.apps):
            try:
                app.close()
            except tk.TclError:
                pass
        for session in self.sessions:
            session.close()
            if session.worker.ident is not None:
                session.worker.join(3)
                self.assertFalse(session.worker.is_alive())
                self.assertTrue(session.shutdown_complete)
        for root in self.roots:
            try:
                for timer in root.tk.call("after", "info"):
                    root.after_cancel(timer)
                root.destroy()
            except tk.TclError:
                pass
        self.apps.clear()
        self.roots.clear()
        self.sessions.clear()
        gc.collect()  # Tcl owners must be released on the main thread.
        self.temporary.cleanup()

    def wait(self, root, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Mock presentation transition timed out")
            root.update()
            time.sleep(.005)
        root.update()

    def session(self, output_name):
        config = SessionConfig(task="Example task", method="Mock policy",
                               output_root=str(self.directory / output_name),
                               record_video=False, target_hz=200,
                               metadata={"defer_policy_load": True,
                                         "reload_selection": {"task": "Example task"}})
        session = DeploymentSession(config, MockPolicy(), MockEmbodiment(), REPOSITORY)
        self.sessions.append(session)
        return session

    def launch(self, output_name, *, reloadable=False):
        root = tk.Tk()
        root.withdraw()
        self.roots.append(root)
        options = {"config_directory": self.config_directory}
        if reloadable:
            options.update(reload_fields=({"key": "task", "label": "Task", "kind": "text"},),
                           reload_factory=lambda *_arguments: None)
        session = self.session(output_name)
        app = DeploymentGUI(root, session, **options)
        self.apps.append(app)
        self.wait(root, lambda: session.state == "SELECTING")
        return root, app

    def close_app(self, root, app):
        app.close()
        app.session.worker.join(3)
        self.assertTrue(app.session.shutdown_complete)
        for timer in root.tk.call("after", "info"):
            root.after_cancel(timer)
        root.destroy()

    def test_last_font_size_restores_for_a_new_output_session(self):
        for size, number in ((12, 1), (19, 2)):
            with self.subTest(size=size):
                root, app = self.launch(f"first_{number}")
                app.set_text_size(size)
                self.close_app(root, app)
                restored_root, restored = self.launch(f"next_{number}")
                self.assertEqual(restored.text_size, size)
                self.assertEqual(restored.fonts["body"].actual("size"), size)
                self.assertEqual(restored.preferences.path, self.preference_file)
                self.assertNotEqual(restored.session.folder, app.session.folder)
                self.close_app(restored_root, restored)

    def test_checkpoint_replacement_preserves_the_chosen_font(self):
        root, app = self.launch("first", reloadable=True)
        app.set_text_size(19)
        previous = app.session
        replacement = self.session("replacement")
        app.reload_factory = lambda _selection, _previous_config: replacement
        self.assertTrue(app.load_checkpoint())
        self.wait(root, lambda: app.session is replacement and replacement.state == "SELECTING"
                  and not app._reload_in_progress)
        self.assertTrue(previous.replacement_closed)
        self.assertEqual(app.text_size, 19)
        self.assertEqual(app.fonts["body"].actual("size"), 19)
        self.assertEqual(app.preferences.path, self.preference_file)
        self.close_app(root, app)
        restored_root, restored = self.launch("after_replacement")
        self.assertEqual(restored.fonts["body"].actual("size"), 19)

    def test_font_change_is_durable_before_the_next_gui_event(self):
        root, app = self.launch("immediate")
        app.set_text_size(19)
        # No update(), debounce wait or graceful close: this is the contract
        # needed when the desktop connection disappears immediately afterwards.
        stored = json.loads(self.preference_file.read_text())
        self.assertEqual(stored["values"]["text_size"], 19)
        self.assertEqual(app.fonts["body"].actual("size"), 19)

    def test_owned_mock_process_exit_keeps_its_last_font_choice(self):
        preferences = WorkspacePreferences(self.preference_file)
        preferences.set("text_size", 12)
        self.assertTrue(preferences.flush())
        marker = self.directory / "child_choice.json"
        script = r'''
import json, os, sys, tkinter as tk
from pathlib import Path
from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import SessionConfig
from gnaroshi_deployment.gui import DeploymentGUI
from gnaroshi_deployment.session import DeploymentSession
root = tk.Tk()
root.withdraw()
config = SessionConfig(output_root=sys.argv[2], record_video=False,
                       metadata={"defer_policy_load": True})
session = DeploymentSession(config, MockPolicy(), MockEmbodiment(), Path(sys.argv[4]))
app = DeploymentGUI(root, session, config_directory=Path(sys.argv[1]))
app.set_text_size(21)
Path(sys.argv[3]).write_text(json.dumps({"rendered_size": app.fonts["body"].actual("size")}))
os._exit(0)  # Only this disposable CPU mock process; no real GUI is signalled.
'''
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        environment["PYTHONPATH"] = str(REPOSITORY / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run([sys.executable, "-c", script, str(self.config_directory),
                                 str(self.directory / "child_output"), str(marker), str(REPOSITORY)],
                                env=environment, text=True, capture_output=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(marker.read_text())["rendered_size"], 21)
        self.assertEqual(json.loads(self.preference_file.read_text())["values"]["text_size"], 21)
        root, app = self.launch("after_exit")
        self.assertEqual(app.fonts["body"].actual("size"), 21)

    def test_corrupt_preferences_are_preserved_until_an_explicit_font_choice(self):
        self.preference_file.parent.mkdir(parents=True)
        self.preference_file.write_text("{broken")
        root, app = self.launch("malformed")
        self.assertEqual(self.preference_file.read_text(), "{broken")
        self.assertTrue(app.preferences.error)
        self.close_app(root, app)
        self.assertEqual(self.preference_file.read_text(), "{broken")
        root, app = self.launch("explicit_choice_after_malformed")
        app.set_text_size(18)
        self.assertEqual(json.loads(self.preference_file.read_text())["values"]["text_size"], 18)
        self.assertIsNone(app.preferences.error)

    def test_write_failure_keeps_the_choice_and_does_not_replace_saved_settings(self):
        saved = WorkspacePreferences(self.preference_file)
        saved.set("text_size", 12)
        self.assertTrue(saved.flush())
        previous = self.preference_file.read_bytes()
        root, app = self.launch("write_failure")
        original_open = Path.open

        def denied_preference_temp(path, *arguments, **keywords):
            if path.parent == self.preference_file.parent and path.name.startswith(".ui_preferences.json."):
                raise PermissionError("presentation storage is read-only: " + str(path))
            return original_open(path, *arguments, **keywords)

        with patch.object(Path, "open", new=denied_preference_temp):
            app.set_text_size(18)
            self.assertEqual(app.fonts["body"].actual("size"), 18)
            self.assertEqual(app.preferences.get("text_size"), 18)
            self.assertIn("read-only", app.preferences.error)
            self.assertEqual(self.preference_file.read_bytes(), previous)
            feedback = app.presentation_status.get()
            self.assertRegex(feedback.lower(), r"could not|unable|not saved")
            self.assertRegex(feedback.lower(), r"permissions|storage|space")
            self.assertNotIn(str(self.preference_file.parent), feedback)
            self.assertNotIn("read-only:", feedback)
            self.assertEqual(app.presentation_label.cget("textvariable"), str(app.presentation_status))
            self.assertEqual(app.presentation_label.winfo_manager(), "grid")
        app.set_text_size(18)
        self.assertIsNone(app.preferences.error)
        self.assertEqual(json.loads(self.preference_file.read_text())["values"]["text_size"], 18)
        self.assertEqual(app.presentation_status.get(), "")
        self.assertEqual(app.presentation_label.winfo_manager(), "")


if __name__ == "__main__":
    unittest.main()
