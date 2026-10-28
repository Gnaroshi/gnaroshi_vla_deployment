"""Native UI states, keyboard isolation and failure notes; CPU mock only."""
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
import unittest

from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
from gnaroshi_deployment.contracts import RobotRecoveryRequired, SessionConfig
from gnaroshi_deployment.gui import DeploymentGUI, THEME
from gnaroshi_deployment.session import DeploymentSession


@unittest.skipUnless(os.environ.get("DISPLAY"), "Native GUI tests need a private display")
class GUIOperations(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.temporary = tempfile.TemporaryDirectory()
        self.app = None

    def tearDown(self):
        if self.app:
            self.app.close()
            self.app.session.worker.join(3)
            self.assertTrue(self.app.session.shutdown_complete)
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.temporary.cleanup()

    def wait(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Mock UI transition timed out")
            self.root.update()
            time.sleep(.005)
        self.root.update()

    def launch(self, *, embodiment=None, policy=None, selecting=False):
        metadata = {"defer_policy_load": selecting}
        config = SessionConfig(task="CPU demo", method="Mock · no hardware", output_root=self.temporary.name,
                               record_video=False, target_hz=200, metadata=metadata)
        session = DeploymentSession(config, policy or MockPolicy(.001), embodiment or MockEmbodiment(), Path(__file__).parents[1])
        self.app = DeploymentGUI(self.root, session)
        self.wait(lambda: session.state == ("SELECTING" if selecting else "READY"))
        return session

    def test_home_button_queues_pose_only_move_without_a_rollout(self):
        session = self.launch()
        self.app.refresh()
        self.assertEqual(str(self.app.home_button.cget("state")), "normal")
        self.app.home_button.invoke()
        self.assertEqual(session.state, "HOMING")
        self.app.refresh()
        self.assertEqual(str(self.app.home_button.cget("state")), "disabled")
        self.assertEqual(str(self.app.start_button.cget("state")), "disabled")
        self.wait(lambda: session.state == "READY")
        self.assertEqual(session.embodiment.home_count, 1)
        self.assertEqual(session.results, [])
        self.assertFalse(list(session.folder.glob("rollout_*")))
        self.assertIn("Home", self.app.control_hint.cget("text"))

    def test_home_shortcut_rejects_modifiers_text_fields_and_physical_key_repeat(self):
        session = self.launch()
        for widget in (self.app.notes, self.app.hz_entry, self.app.steps_entry, ttk.Combobox(self.root), ttk.Spinbox(self.root)):
            self.app.keypress(SimpleNamespace(keysym="h", widget=widget, state=0))
        for modifier in (1, 4, 8, 64):
            self.app.keypress(SimpleNamespace(keysym="h", widget=self.root, state=modifier))
        self.assertEqual(session.embodiment.home_count, 0)
        event = SimpleNamespace(keysym="h", widget=self.root, state=0)
        self.app.keypress(event)
        self.app.keypress(event)
        self.wait(lambda: session.state == "READY")
        self.app.keypress(event)
        self.assertEqual(session.embodiment.home_count, 1)
        self.app.keyrelease(event)
        # X11 synthetic repeat release/press pairs remain the same physical hold.
        self.app.keypress(event)
        self.assertEqual(session.embodiment.home_count, 1)
        self.app.keyrelease(event)
        self.wait(lambda: "h" not in self.app._shortcut_down)
        self.app.keypress(event)
        self.wait(lambda: session.state == "READY")
        self.assertEqual(session.embodiment.home_count, 2)

    def test_real_home_key_event_and_active_rollout_blocking(self):
        session = self.launch()
        self.root.deiconify()
        self.root.focus_force()
        self.root.update()
        self.root.event_generate("<KeyPress-h>")
        self.root.event_generate("<KeyRelease-h>")
        self.wait(lambda: session.embodiment.home_count == 1 and session.state == "READY")
        self.app.start()
        self.wait(lambda: session.state == "RUNNING")
        before = session.embodiment.home_count
        self.assertFalse(self.app.home_robot())
        self.app.refresh()
        self.assertEqual(str(self.app.home_button.cget("state")), "disabled")
        self.assertEqual(session.embodiment.home_count, before)
        self.app.signal("stop")
        self.wait(lambda: session.state == "READY")

    def test_selection_does_not_enable_home_or_start(self):
        session = self.launch(selecting=True)
        self.app.refresh()
        self.assertEqual(str(self.app.home_button.cget("state")), "disabled")
        self.assertEqual(str(self.app.start_button.cget("state")), "disabled")
        self.assertFalse(self.app.home_robot())
        self.assertEqual(session.embodiment.home_count, 0)

    def test_automatic_failure_reason_is_visible_and_manual_notes_are_preserved(self):
        class Fault( MockEmbodiment):
            def apply(self, prediction):
                if len(self.applied) >= 2:
                    raise RobotRecoveryRequired("Protective stop detail /private/diagnostic", category="protective_stop",
                                                status={"action_translation_norm_m": .0848})
                return super().apply(prediction)
            def recover(self, cancel):
                return {"healthy": True, "stop_confirmed": True}
        session = self.launch(embodiment=Fault())
        self.app.notes.insert("1.0", "Operator note remains editable")
        self.app.start()
        self.wait(lambda: session.can_recover and len(session.results) == 1)
        self.app.refresh()
        self.assertEqual(session.results[0]["outcome"], "failure")
        self.assertIn("Failure 1", self.app.counts.get())
        item = self.app.table.get_children()[0]
        self.assertEqual(self.app.table.item(item, "values")[-1], "Protective stop")
        self.app.table.selection_set(item)
        self.app.show_result_note()
        self.assertIn("84.80 mm", self.app.failure_note.get())
        self.assertNotIn("/private", self.app.failure_note.get())
        self.assertIn("Protective stop", self.app.failure_notes.get("1.0", "end"))
        self.assertNotIn("/private", self.app.failure_notes.get("1.0", "end"))
        self.assertIn("/private", self.app.details.get("1.0", "end"))
        self.assertEqual(self.app.notes.get("1.0", "end-1c"), "Operator note remains editable")
        self.assertEqual(str(self.app.home_button.cget("state")), "disabled")
        self.app.save()
        saved = (session.folder / "notes.txt").read_text()
        self.assertIn("Operator note remains editable", saved)
        self.assertIn("protective_stop", saved)

    def test_primary_view_hides_technical_paths_and_theme_has_readable_contrast(self):
        session = self.launch()
        self.root.deiconify()
        self.root.update()
        self.app.refresh()
        self.assertFalse(self.app.folder_label.winfo_viewable())
        self.assertNotIn(str(session.folder), self.app.status.get())
        self.assertEqual(str(session.folder), json.loads(self.app.details.get("1.0", "end"))["results"])
        def luminance(color):
            c = [int(color[n:n+2], 16)/255 for n in (1, 3, 5)]
            c = [v/12.92 if v <= .04045 else ((v+.055)/1.055)**2.4 for v in c]
            return sum(a*b for a, b in zip(c, (.2126, .7152, .0722)))
        for foreground, background in (("text", "canvas"), ("muted", "canvas"), ("text", "panel"), ("ink", "accent"), ("ink", "danger")):
            light, dark = sorted((luminance(THEME[foreground]), luminance(THEME[background])), reverse=True)
            self.assertGreaterEqual((light+.05)/(dark+.05), 4.5)


if __name__ == "__main__":
    unittest.main()
