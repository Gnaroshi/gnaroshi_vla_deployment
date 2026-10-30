"""Native UI states, keyboard isolation and failure notes; CPU mock only."""
import json
import gc
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk
from tkinter import font as tkfont
import unittest
from unittest.mock import patch

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
        self.app = None
        self.root = None
        gc.collect()
        self.temporary.cleanup()

    def wait(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                self.fail("Mock UI transition timed out")
            self.root.update()
            time.sleep(.005)
        self.root.update()

    def settle_layout(self):
        # Tk resolves nested frames and an aspect-aware camera split over
        # successive configure events, independently of the session worker.
        deadline=time.monotonic()+.12
        while time.monotonic()<deadline:
            self.root.update()
            time.sleep(.005)

    def launch(self, *, embodiment=None, policy=None, selecting=False, config_metadata=None, **gui_options):
        metadata = {"defer_policy_load": selecting,**(config_metadata or {})}
        config = SessionConfig(task="CPU demo", method="Mock · no hardware", output_root=self.temporary.name,
                               record_video=False, target_hz=200, metadata=metadata)
        session = DeploymentSession(config, policy or MockPolicy(.001), embodiment or MockEmbodiment(), Path(__file__).parents[1])
        self.app = DeploymentGUI(self.root, session, **gui_options)
        self.wait(lambda: session.state == ("SELECTING" if selecting else "READY"))
        return session

    def test_home_button_queues_pose_only_move_without_a_rollout(self):
        session = self.launch()
        self.app.refresh()
        self.assertNotIn("Loading",self.app.status.get())
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

    def test_saved_session_scoreboard_excludes_idle_faults_and_preserves_manual_success(self):
        session = self.launch()
        session.results[:] = [
            {"outcome":"success", "failure_category":"protective_stop", "outcome_source":"manual"},
            {"outcome":"success", "outcome_source":"manual"},
            {"outcome":"failure", "outcome_source":None},
            {"outcome":"failure", "outcome_source":"automatic_fault"},
            {"outcome":"discarded", "outcome_source":"automatic_fault"},
        ]
        self.app._update_scoreboard(session.results)
        self.assertEqual({key:value.get() for key,value in self.app.score_vars.items()},
                         {"success":"2", "failure":"2", "rate":"50.0%", "total":"4"})
        self.assertEqual(self.app.automatic_count.get(), "Automatic failures: 1")
        self.assertEqual(self.app.score_scope.get(), "Saved in this session")
        session.results.clear()
        self.app._update_scoreboard([])
        self.assertEqual(self.app.score_vars["rate"].get(), "—")

    def test_semantic_fonts_and_operator_zoom_change_actual_rendered_text(self):
        self.launch()
        score_font = tkfont.Font(root=self.root, font=self.app.style.lookup("Success.Score.TLabel", "font"))
        self.assertEqual(score_font.actual("size"), 24)
        self.assertEqual(score_font.actual("weight"), "bold")
        self.assertEqual(self.app.fonts["body"].actual("size"), 14)
        self.app.set_text_size(21)
        self.root.update()
        note_font = tkfont.Font(root=self.root, font=self.app.notes.cget("font"))
        self.assertEqual(note_font.actual("size"), 21)
        self.assertEqual(self.app.fonts["score"].actual("size"), 24)
        self.assertGreaterEqual(self.app.style.lookup("Treeview", "rowheight"), note_font.metrics("linespace")+10)
        self.assertEqual(self.app.preferences.get("text_size"), 21)
        self.assertGreaterEqual(self.app.table.column("result","width"),self.app.fonts["body"].measure("success")+16)

    def test_close_cancels_pending_presentation_save_timer(self):
        self.launch()
        self.app._schedule_preferences()
        pending = self.app._preferences_job
        self.assertIsNotNone(pending)
        self.app.close()
        self.assertNotIn(pending,self.root.tk.call("after","info"))
        self.assertIsNone(self.app._preferences_job)

    def test_primary_counts_and_actions_fit_small_window_with_large_text(self):
        self.launch()
        self.root.deiconify()
        self.root.geometry("960x640+0+0")
        self.app.set_text_size(21)
        self.root.update()
        self.app._layout_workspace()
        self.root.update()
        self.assertGreaterEqual(self.app.workspace_canvas.winfo_height(),180)
        for widget in (*self.app.run_buttons, *self.app.score_labels.values(),self.app.automatic_count_label,self.app.state_label,
                       self.app.exit_button, self.app.larger_text_button):
            x, y = widget.winfo_rootx()-self.root.winfo_rootx(), widget.winfo_rooty()-self.root.winfo_rooty()
            self.assertGreaterEqual(x, 0, str(widget))
            self.assertGreaterEqual(y, 0, str(widget))
            self.assertLessEqual(x+widget.winfo_width(), self.root.winfo_width(), str(widget))
            self.assertLessEqual(y+widget.winfo_height(), self.root.winfo_height(), str(widget))
        self.assertLessEqual(self.app.scoreboard.winfo_reqwidth(),self.root.winfo_width())
        details_index=next(index for index,tab in enumerate(self.app.tabs.tabs()) if self.app.tabs.tab(tab,"text")=="Details")
        if self.app.view_picker.winfo_viewable():
            self.app.view_picker.current(details_index)
            self.app.view_picker.event_generate("<<ComboboxSelected>>")
        else:
            self.app.tabs.select(self.app.details_tab)
        self.root.update()
        self.assertEqual(self.app.tabs.tab(self.app.tabs.select(), "text"), "Details")

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

    def test_explicit_device_scan_camera_change_and_swap_draft_preserve_policy_and_notes(self):
        class Cameras(MockEmbodiment):
            scans = 0
            def list_camera_devices(self):
                self.scans += 1
                return super().list_camera_devices()
        class Policy(MockPolicy):
            connects = 0
            def connect(self):
                self.connects += 1
        robot, policy = Cameras(), Policy(.001)
        session = self.launch(embodiment=robot, policy=policy)
        self.app.notes.insert("1.0", "Keep this experiment note across device changes")
        self.app.refresh()
        self.assertEqual(robot.scans, 0)
        self.assertTrue(self.app.refresh_devices())
        self.wait(lambda: not session.device_status["scan_busy"] and len(session.device_status["devices"])==3)
        self.app.refresh()
        self.assertIn("mock-extra", self.app.camera_selectors["primary"].cget("values"))
        self.app.camera_vars["primary"].set("mock-extra")
        self.app._camera_draft_changed()
        self.assertTrue(self.app.apply_cameras())
        self.wait(lambda: session.state=="READY" and session.device_status["selected"]["primary"]=="mock-extra")
        self.app.refresh()
        self.assertEqual(policy.connects, 1)
        self.assertEqual(robot.home_count, 0)
        self.assertEqual(robot.applied, [])
        self.assertIn("Keep this experiment", self.app.notes.get("1.0", "end"))
        self.assertEqual(session.results, [])
        previous = dict(session.device_status["selected"])
        self.assertTrue(self.app.swap_camera_roles())
        self.assertEqual(self.app.camera_vars["primary"].get(), previous["wrist"])
        self.assertEqual(session.device_status["selected"], previous)
        self.assertIn("draft", self.app.device_hint.get())

    def test_duplicate_camera_draft_and_active_changes_do_not_reconfigure_streams(self):
        session = self.launch()
        self.app.refresh()
        original = dict(session.device_status["selected"])
        self.app.camera_vars["primary"].set(original["wrist"])
        self.assertFalse(self.app.apply_cameras())
        self.assertIn("different camera", self.app.device_hint.get())
        self.assertEqual(session.device_status["selected"], original)
        self.app.camera_vars["primary"].set(original["primary"])
        self.app.start()
        self.wait(lambda: session.state=="RUNNING")
        self.app.refresh()
        self.assertEqual(str(self.app.apply_cameras_button.cget("state")), "disabled")
        self.assertFalse(self.app.apply_cameras())
        self.assertFalse(self.app.restart_cameras())
        self.assertEqual(session.device_status["selected"], original)
        self.app.signal("stop")
        self.wait(lambda: session.state=="READY")

    def test_operator_health_exposes_storage_recording_and_stale_frames_before_start(self):
        session = self.launch()
        report = {"can_start":False, "disk_free_bytes":2**29, "disk_warning":"Free storage is low",
                  "recording":{"enabled":True,"healthy":False,"error":"writer unavailable"},
                  "cameras":{"primary":{"age_ms":1600,"frame_number":44},"wrist":{"age_ms":30}}}
        with patch.object(type(session), "operator_health", property(lambda _session:report)):
            self.app.refresh()
            self.assertEqual(str(self.app.start_button.cget("state")), "disabled")
            self.assertIn("0.5 GiB", self.app.health.get())
            self.assertIn("writer unavailable", self.app.health.get())
            self.assertIn("stale", self.app.camera_health.get())
            self.assertIn("44", self.app.camera_health.get())
        self.assertEqual(session.embodiment.home_count, 0)

    def saved_rollout(self):
        self.app.start()
        self.wait(lambda: self.app.session.state=="RUNNING" and len(self.app.session.rows)>=2)
        before = len(self.app.session.results)
        self.app.signal("success")
        self.wait(lambda: self.app.session.state=="READY" and len(self.app.session.results)==before+1)
        self.app.refresh()
        return self.app.session.results[-1]

    def test_selected_rollout_annotation_is_autosaved_without_changing_outcome_or_media(self):
        session = self.launch()
        first, second = self.saved_rollout(), self.saved_rollout()
        items = self.app.table.get_children()
        self.app.table.selection_set(items[0])
        self.app.show_result_note()
        self.app.result_annotation.set("Ball grasp succeeded; operator outcome selected later")
        self.app.schedule_annotation()
        self.app.table.selection_set(items[1])
        self.app.show_result_note()
        self.assertEqual(session.results[0]["operator_note"], "Ball grasp succeeded; operator outcome selected later")
        self.assertEqual((first["outcome"], second["outcome"]), ("success", "success"))
        self.assertEqual(self.app.result_annotation.get(), "")
        self.app.result_annotation.set("Second observation")
        self.app.schedule_annotation()
        self.wait(lambda: session.results[1].get("operator_note")=="Second observation")
        saved = json.loads((session.folder / "results.json").read_text())
        self.assertEqual(saved["successes"], 2)
        self.assertEqual(saved["failures"], 0)

    def test_record_browser_opens_only_selected_contained_paths_and_handles_missing_video(self):
        session = self.launch()
        result = self.saved_rollout()
        folder = session.folder / result["directory"]
        video = folder / "primary.mp4"
        self.assertFalse(self.app.open_result_video())
        self.assertIn("Cannot open video", self.app.status.get())
        video.write_bytes(b"record-browser fixture; not decoded")
        self.app.video_role.set("primary")
        with patch.object(self.app, "_launch_path", return_value=True) as launch:
            self.assertTrue(self.app.open_result_folder())
            launch.assert_called_with(folder.resolve())
            self.assertTrue(self.app.open_result_video())
            launch.assert_called_with(video.resolve())
            self.assertTrue(self.app.copy_result_path())
            self.assertEqual(self.root.clipboard_get(), str(folder.resolve()))
            result["directory"] = "../outside"
            self.assertFalse(self.app.open_result_folder())
            self.assertIn("outside", self.app.status.get())
            self.assertEqual(launch.call_count, 2)

    def test_help_is_available_while_typing_and_save_failure_keeps_operator_text(self):
        session = self.launch()
        self.app.keypress(SimpleNamespace(keysym="F1", widget=self.app.notes, state=0))
        self.assertTrue(self.app._help_window.winfo_exists())
        self.assertEqual(session.embodiment.home_count, 0)
        self.app.notes.insert("1.0", "Unsaved text must remain visible")
        with patch.object(session, "save", side_effect=OSError("storage unavailable")):
            self.assertFalse(self.app.save())
            self.assertIn("Unsaved text", self.app.notes.get("1.0", "end"))
            self.assertIn("Could not save", self.app.status.get())
        self.assertTrue(self.app.save())

    def test_long_selection_forms_and_device_controls_remain_accessible_on_laptop_window(self):
        fields = [{"key":f"setting_{number}","label":f"Setting {number}","value":"draft"} for number in range(17)]
        fields[0]["readonly"] = True
        self.launch(selecting=True, reload_fields=fields, reload_factory=lambda *_args:None)
        self.root.geometry("1280x900+0+0")
        self.root.deiconify()
        self.settle_layout()
        self.assertEqual(self.app.tabs.tab(self.app.tabs.select(),"text"),"Policy")
        camera_bounds=[(widget.winfo_rooty(),widget.winfo_height()) for widget in self.app.camera_labels]
        self.app.tabs.select(self.app.rollouts_tab)
        self.root.update()
        self.app.tabs.select(self.app.policy_tab)
        self.settle_layout()
        self.assertEqual(camera_bounds,[(widget.winfo_rooty(),widget.winfo_height()) for widget in self.app.camera_labels])
        self.assertGreater(self.app.camera_labels[0].winfo_height(), 80)
        self.assertTrue(self.app.load_button.winfo_viewable())
        self.assertEqual(str(self.app.reload_widgets[0][0].cget("state")), "readonly")
        last_field = self.app.reload_widgets[-1][0]
        last_field.focus_force()
        self.root.update()
        form_canvas = self.app._selection_form.scroll_canvas
        self.assertGreaterEqual(last_field.winfo_rooty(),form_canvas.winfo_rooty())
        self.assertLessEqual(last_field.winfo_rooty()+last_field.winfo_height(),form_canvas.winfo_rooty()+form_canvas.winfo_height())
        self.app.tabs.select(self.app.devices_tab)
        self.root.update()
        canvas = next(widget for widget in self.app.devices_tab.winfo_children() if isinstance(widget, tk.Canvas))
        canvas.yview_moveto(1)
        self.root.update()
        button = self.app.reconnect_cameras_button
        self.assertGreaterEqual(button.winfo_rooty(), canvas.winfo_rooty())
        self.assertLess(button.winfo_rooty()+button.winfo_height(), canvas.winfo_rooty()+canvas.winfo_height()+1)

    def test_large_selection_header_and_full_draft_choice_remain_readable(self):
        long_task = "A task with a deliberately long name · 한글 작업 이름도 끝까지 확인"
        fields = [{"key":"task","label":"Task","kind":"choice","choices":[long_task],"value":long_task}]
        fields += [{"key":f"setting_{number}","label":f"Setting {number}","value":"draft","advanced":number>8}
                   for number in range(1,17)]
        self.launch(selecting=True,reload_fields=fields,reload_factory=lambda *_args:None)
        self.root.geometry("960x640+0+0")
        self.root.deiconify()
        self.app.set_text_size(21)
        self.root.update()
        for button in (self.app.checkpoint_toggle,self.app.help_button,self.app.exit_button,self.app.larger_text_button):
            self.assertGreaterEqual(button.winfo_width(),button.winfo_reqwidth())
            self.assertLessEqual(button.winfo_rootx()+button.winfo_width(),self.root.winfo_rootx()+self.root.winfo_width())
        choice,variable,hint = self.app._selection_choice_hints[0]
        self.assertEqual(variable.get(),long_task)
        self.assertTrue(hint.winfo_viewable())
        self.assertGreater(hint.winfo_height(),self.app.fonts["caption"].metrics("linespace"))
        form_canvas=self.app._selection_form.scroll_canvas
        self.assertGreater(float(form_canvas.cget("scrollregion").split()[3]),form_canvas.winfo_height())
        self.assertTrue(self.app.load_button.winfo_viewable())
        self.assertGreaterEqual(self.app.workspace_canvas.winfo_height(),180)
        self.assertTrue(self.app.context_button.winfo_viewable())
        self.assertFalse(self.app.identity_label.winfo_viewable())
        self.app.context_button.invoke()
        self.root.update()
        self.assertTrue(self.app.identity_label.winfo_viewable())

    def test_recording_checkbox_changes_only_ready_configuration_without_moving_robot(self):
        session = self.launch()
        self.assertFalse(session.config.record_video)
        self.app.record_video.set(True)
        self.app.apply_settings()
        self.assertTrue(session.config.record_video)
        self.assertTrue(self.app.current_selection()["record_video"])
        self.assertEqual(session.embodiment.home_count, 0)
        self.assertEqual(session.embodiment.applied, [])

    def test_compact_actions_are_pointer_reachable_and_policy_does_not_resize_cameras(self):
        fields=[{"key":f"setting_{number}","label":f"Setting {number}","value":"draft","advanced":number>8}
                for number in range(17)]
        long_task="A deliberately long task title · 긴 실제 작업 이름도 보존"
        session=self.launch(selecting=True,reload_fields=fields,reload_factory=lambda *_args:None,
                            config_metadata={"display_task":long_task},
                            config_loader=lambda *_args:{},config_saver=lambda *_args:None)
        self.root.deiconify()
        for width,height,size in ((1685,1376,12),(1366,768,14),(960,640,21),(960,640,24)):
            with self.subTest(window=(width,height),text_size=size):
                self.root.geometry(f"{width}x{height}+0+0")
                self.app.set_text_size(size)
                self.app.tabs.select(self.app.policy_tab)
                self.settle_layout()
                self.assertIn(long_task,self.app.instruction_label.cget("text"))
                self.assertEqual(self.app.method_label.cget("text"),"No policy loaded")
                if size==12:
                    self.assertLessEqual(self.app.workspace_canvas.winfo_rooty()-self.root.winfo_rooty(),160)
                    self.assertLessEqual(self.app.camera_labels[0].winfo_rooty()-self.root.winfo_rooty(),180)
                if size==21:
                    self.assertGreaterEqual(self.app.workspace_canvas.winfo_height(),280)
                for button in self.app.run_buttons:
                    self.assertEqual(button.winfo_width(),button.winfo_reqwidth())
                    target=self.root.winfo_containing(button.winfo_rootx()+button.winfo_width()//2,
                                                     button.winfo_rooty()+button.winfo_height()//2)
                    self.assertIs(target,button,"A frame must not cover an apparently mapped action")
                    self.assertLessEqual(button.winfo_rootx()+button.winfo_width(),self.root.winfo_rootx()+width)
                def cameras():
                    return [(label.winfo_rootx(),label.winfo_rooty(),label.winfo_width(),label.winfo_height())
                            for label in self.app.camera_labels]
                before=cameras()
                self.app._show_load_error("load failed","raw diagnostic is confined to Details")
                self.settle_layout()
                self.assertEqual(cameras(),before)
                self.assertTrue(self.app.load_error.winfo_viewable())
                self.assertTrue(self.app.load_button.winfo_viewable())
                self.assertLessEqual(self.app.load_button.winfo_rooty()+self.app.load_button.winfo_height(),
                                     self.app.workspace_canvas.winfo_rooty()+self.app.workspace_canvas.winfo_height())
                self.assertNotIn("raw diagnostic",self.app.status.get())
                self.app.tabs.select(self.app.rollouts_tab)
                self.settle_layout()
                self.assertEqual(cameras(),before)
                self.assertLess(self.app.measurement_panel.winfo_height(),self.app.main_pane.winfo_height()/2)
        self.assertEqual(session.embodiment.home_count,0)
        self.assertEqual(session.embodiment.applied,[])

    def test_saved_success_counts_reflow_at_maximum_text_without_hiding_policy_actions(self):
        fields=[{"key":f"setting_{number}","label":f"Setting {number}","value":"draft","advanced":number>8}
                for number in range(17)]
        session=self.launch(reload_fields=fields,reload_factory=lambda *_args:None)
        self.root.deiconify()
        self.root.geometry("960x640+0+0")
        self.app.set_text_size(24)
        self.settle_layout()
        for number in range(12):
            self.app.start()
            self.wait(lambda: session.state=="RUNNING" and len(session.rows)>=2)
            self.app.signal("success")
            self.wait(lambda: session.state=="READY" and len(session.results)==number+1)
        self.wait(lambda:self.app.score_vars["success"].get()=="12")
        self.app.tabs.select(self.app.policy_tab)
        self.app.toggle_advanced_fields()
        self.settle_layout()
        self.assertEqual(self.app.score_vars["success"].get(),"12")
        self.assertEqual(self.app.score_vars["rate"].get(),"100.0%")
        canvas=self.app.workspace_canvas
        self.assertEqual(self.app.workspace_content.winfo_height(),canvas.winfo_height())
        self.assertLessEqual(self.app.load_button.winfo_rooty()+self.app.load_button.winfo_height(),
                             canvas.winfo_rooty()+canvas.winfo_height())
        for widget in (*self.app.score_labels.values(),self.app.automatic_count_label):
            self.assertLessEqual(widget.winfo_rootx()+widget.winfo_width(),self.root.winfo_rootx()+self.root.winfo_width())
        last=self.app.reload_widgets[-1][0]
        last.focus_force()
        self.settle_layout()
        viewport=self.app._selection_form.scroll_canvas
        self.assertGreaterEqual(viewport.winfo_height(),last.winfo_reqheight())
        self.assertGreaterEqual(last.winfo_rooty(),viewport.winfo_rooty())
        self.assertLessEqual(last.winfo_rooty()+last.winfo_height(),viewport.winfo_rooty()+viewport.winfo_height())
        self.assertEqual(len(session.results),12)


if __name__ == "__main__":
    unittest.main()
