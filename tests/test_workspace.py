"""Presentation persistence and native splitter interaction without devices."""
import gc
import json
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
from unittest.mock import patch

from gnaroshi_deployment.workspace import (
    ResizablePane, WorkspacePreferences, clamp_window_geometry, geometry_string,
)


class Preferences(unittest.TestCase):
    def test_invalid_preferences_are_nondestructive_until_explicit_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"preferences.json"
            for text in ("{broken", '{"schema_version":2}', '{"schema_version":true}'):
                path.write_text(text)
                preferences = WorkspacePreferences(path)
                self.assertEqual(preferences.get("text_size", 14), 14)
                self.assertTrue(preferences.error)
                self.assertFalse(preferences.flush())
                self.assertEqual(path.read_text(), text)
            preferences.set("text_size", 18)
            self.assertTrue(preferences.flush())
            self.assertEqual(WorkspacePreferences(path).get("text_size"), 18)
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_valid_values_and_panes_roundtrip_and_unknown_fields_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"preferences.json"
            path.write_text(json.dumps({"schema_version":1, "unknown":"ignored", "values":{
                "text_size":True, "window":{"width":1440,"height":960,"x":-3000,"y":2000,"extra":"ignored"},
                "panes":{"main":{"fractions":{"horizontal":.72,"vertical":float("inf")},
                                  "collapsed":["second","nonsense"],"orientation":"horizontal"}},
                "huge":"x"*1001}}))
            preferences = WorkspacePreferences(path)
            self.assertEqual(preferences.get("text_size",14),14)
            self.assertIsNone(preferences.get("huge"))
            self.assertNotIn("extra", preferences.get("window"))
            self.assertEqual(preferences.pane("main")["fractions"],{"horizontal":.72})
            preferences.set("text_size",20)
            preferences.set_pane("main",{"fractions":{"horizontal":.66,"vertical":.5},"collapsed":[],"orientation":"vertical"})
            self.assertTrue(preferences.flush())
            loaded = WorkspacePreferences(path)
            self.assertEqual(loaded.get("text_size"),20)
            self.assertEqual(loaded.pane("main")["fractions"]["vertical"],.5)
            with self.assertRaises(ValueError): preferences.set("text_size",0)

    def test_failed_write_reports_error_and_keeps_in_memory_presentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            preferences = WorkspacePreferences(Path(temporary)/"preferences.json")
            preferences.set("text_size",18)
            with patch.object(Path,"mkdir",side_effect=PermissionError("test readonly directory")):
                self.assertFalse(preferences.flush())
            self.assertIn("readonly",preferences.error)
            self.assertEqual(preferences.get("text_size"),18)
            self.assertTrue(preferences.flush())

    def test_window_restore_is_visible_and_does_not_mutate_saved_geometry(self):
        desired={"width":2400,"height":1600,"x":-5000,"y":2000}
        original=desired.copy()
        restored=clamp_window_geometry(desired,1280,800)
        self.assertEqual(restored,{"width":1280,"height":800,"x":0,"y":0})
        self.assertEqual(desired,original)
        self.assertEqual(geometry_string(restored),"1280x800+0+0")
        self.assertEqual(clamp_window_geometry({"width":True,"height":"bad"},500,300),
                         {"width":500,"height":300,"x":0,"y":0})


@unittest.skipUnless(os.environ.get("DISPLAY"),"Native workspace tests require a private display")
class NativeWorkspace(unittest.TestCase):
    def setUp(self):
        self.root=tk.Tk()
        self.root.geometry("1100x700")
        self.root.focus_force()
        self.temporary=tempfile.TemporaryDirectory()
        self.path=Path(self.temporary.name)/"preferences.json"
        self.preferences=WorkspacePreferences(self.path)
        self.pane=ResizablePane(self.root,key="main",preferences=self.preferences,first_label="Observations",
                                second_label="Results",first_min=240,second_min=260,
                                default_fraction=.6,show_header=False)
        self.pane.pack(fill="both",expand=True)
        self.entry=ttk.Entry(self.pane.second)
        self.entry.pack()
        self.tick()

    def tearDown(self):
        self.pane.close()
        self.root.destroy()
        self.entry=self.pane=self.root=None
        gc.collect()
        self.temporary.cleanup()

    def tick(self, seconds=.01):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            self.root.update()
            time.sleep(.001)
        self.root.update_idletasks()

    def pointer(self, name, x_root=None, y_root=None, **kwargs):
        divider=self.pane.separator
        divider.event_generate(name,x=3,y=3,
            rootx=divider.winfo_rootx()+3 if x_root is None else int(x_root),
            rooty=divider.winfo_rooty()+3 if y_root is None else int(y_root),**kwargs)
        self.tick()

    def test_actual_pointer_drag_captures_then_releases_and_persists(self):
        start=self.pane.actual_fraction
        self.pointer("<ButtonPress-1>")
        self.assertIs(self.pane.grab_current(),self.pane.separator)
        self.pointer("<B1-Motion>",x_root=self.pane.paned.winfo_rootx()+820)
        self.pointer("<ButtonRelease-1>",x_root=self.pane.paned.winfo_rootx()+820)
        self.assertIsNone(self.pane.grab_current())
        self.assertGreater(self.pane.actual_fraction,start+.1)
        self.tick(.35)
        stored=WorkspacePreferences(self.path).pane("main")["fractions"]["horizontal"]
        self.assertAlmostEqual(stored,self.pane.desired_fraction)

    def test_keyboard_resize_clamp_reset_and_separator_metadata(self):
        divider=self.pane.separator
        self.root.focus_force();divider.focus_set();self.tick()
        self.assertEqual(divider.winfo_class(),"TSeparator")
        self.assertTrue(divider.cget("takefocus"))
        initial=self.pane.snapshot()["position"]
        divider.event_generate("<KeyPress-Right>");self.tick()
        self.assertGreater(self.pane.snapshot()["position"],initial)
        divider.event_generate("<KeyPress-Right>",state=1);self.tick()
        self.assertGreater(self.pane.snapshot()["position"],initial+50)
        divider.event_generate("<KeyPress-Home>");self.tick()
        self.assertEqual(self.pane.snapshot()["position"],self.pane.snapshot()["minimum"])
        divider.event_generate("<KeyPress-End>");self.tick()
        self.assertEqual(self.pane.snapshot()["position"],self.pane.snapshot()["maximum"])
        divider.event_generate("<KeyPress-Return>");self.tick()
        self.assertEqual(self.pane.desired_fraction,.6)
        self.assertIn("Observations",self.pane.snapshot()["separator_label"])

    def test_clamped_small_viewport_preserves_desired_fraction_on_return(self):
        self.pane.set_fraction(.78);self.tick()
        desired=self.pane.desired_fraction
        self.root.geometry("480x400");self.tick()
        snapshot=self.pane.snapshot()
        self.assertLessEqual(snapshot["position"],snapshot["maximum"])
        self.assertGreaterEqual(snapshot["position"],snapshot["minimum"])
        self.assertAlmostEqual(self.pane.desired_fraction,desired)
        self.assertTrue(self.pane.first.winfo_viewable())
        self.assertTrue(self.pane.second.winfo_viewable())
        self.root.geometry("1400x800");self.tick()
        self.assertAlmostEqual(self.pane.actual_fraction,desired,delta=.01)

    def test_collapse_removes_focus_and_expand_remains_reachable(self):
        self.root.focus_force();self.entry.focus_set();self.tick()
        self.assertTrue(self.pane.collapse("second"));self.tick()
        self.assertFalse(self.entry.winfo_viewable())
        button=self.pane.buttons["second"]
        self.assertTrue(button.winfo_viewable())
        self.assertIn("Show",button.cget("text"))
        self.assertIsNot(self.root.focus_get(),self.entry)
        self.assertTrue(self.pane.expand("second"));self.tick()
        self.assertTrue(self.entry.winfo_viewable())
        self.assertIs(self.root.focus_get(),self.pane.separator)

    def test_temporary_content_absence_does_not_overwrite_user_collapse(self):
        self.pane.collapse("second");self.tick()
        self.pane.set_visible("second",False,persist=False);self.tick()
        self.pane.set_visible("second",True,persist=False);self.tick()
        self.assertFalse(self.entry.winfo_viewable())
        self.assertEqual(self.pane.collapsed,("second",))
        self.pane.expand("second");self.tick()
        self.assertTrue(self.entry.winfo_viewable())

    def test_orientation_preserves_independent_desired_split(self):
        self.pane.set_fraction(.73);self.tick()
        self.pane.set_orientation("vertical");self.tick()
        self.assertEqual(self.pane.desired_fraction,.6)
        self.pane.set_fraction(.45);self.tick()
        self.pane.separator.focus_set()
        start=self.pane.snapshot()["position"]
        self.pane.separator.event_generate("<KeyPress-Down>");self.tick()
        self.assertGreater(self.pane.snapshot()["position"],start)
        self.pane.set_orientation("horizontal");self.tick()
        self.assertAlmostEqual(self.pane.desired_fraction,.73)

    def test_orientation_change_never_places_a_full_surface_over_the_other_pane(self):
        self.pane.set_orientation("vertical");self.tick()
        self.assertLessEqual(self.pane.separator.winfo_height(),12)
        self.assertEqual(self.pane.separator.winfo_width(),self.pane.paned.winfo_width())
        self.pane.set_orientation("horizontal");self.tick()
        self.assertLessEqual(self.pane.separator.winfo_width(),12)
        self.assertEqual(self.pane.separator.winfo_height(),self.pane.paned.winfo_height())
        self.assertEqual(self.pane.separator.winfo_x(),self.pane.paned.sash_coord(0)[0])

    def test_nested_content_requests_keep_grip_on_actual_native_boundary(self):
        nested=ResizablePane(self.pane.second,orient="vertical",key="inspector",first_min=70,
                              second_min=150,show_header=False)
        nested.pack(fill="both",expand=True)
        caption=ttk.Label(nested.first,text="Measurements",wraplength=420)
        caption.pack(fill="x")
        notebook=ttk.Notebook(nested.second)
        notebook.pack(fill="both",expand=True)
        tab=ttk.Frame(notebook);notebook.add(tab,text="Results")
        editor=tk.Text(tab,width=80,height=8);editor.pack(fill="both",expand=True)
        self.tick()
        caption.configure(text="A long status that updates the nested content request and wraps to a new line. "*5)
        self.pane.set_orientation("vertical");self.tick()
        self.pane.set_orientation("horizontal");self.tick()
        for current in (self.pane,nested):
            x,y=current.paned.sash_coord(0)
            self.assertEqual(current.separator.winfo_x(),x)
            self.assertEqual(current.separator.winfo_y(),y)
            expected=current.first.winfo_width() if current.orient=="horizontal" else current.first.winfo_height()
            self.assertEqual(current.snapshot()["position"],expected)
        nested.close()

    def test_double_click_and_presentation_restore_after_recreation(self):
        self.pane.set_fraction(.75);self.tick()
        divider=self.pane.separator
        for sequence, timestamp in (("<ButtonPress-1>",1000),("<ButtonRelease-1>",1001),
                                    ("<ButtonPress-1>",1100),("<ButtonRelease-1>",1101)):
            self.pointer(sequence,time=timestamp)
        self.assertEqual(self.pane.desired_fraction,.6)
        self.pane.set_fraction(.71)
        self.pane.collapse("second");self.tick()
        self.pane.close();self.pane.destroy()
        self.pane=ResizablePane(self.root,key="main",preferences=WorkspacePreferences(self.path),
                                first_label="Observations",second_label="Results",show_header=False)
        self.pane.pack(fill="both",expand=True)
        self.entry=ttk.Entry(self.pane.second);self.entry.pack()
        self.tick()
        self.assertAlmostEqual(self.pane.desired_fraction,.71)
        self.assertFalse(self.entry.winfo_viewable())
        self.assertTrue(self.pane.buttons["second"].winfo_viewable())

    def test_unavailable_content_moves_hidden_focus_to_visible_pane_only(self):
        self.root.focus_force();self.entry.focus_set();self.tick()
        self.pane.set_visible("second",False,persist=False);self.tick()
        self.assertFalse(self.entry.winfo_viewable())
        self.assertIs(self.root.focus_get(),self.pane.first)
        self.assertFalse(self.pane.header.winfo_viewable())

    def test_lost_pointer_capture_and_escape_restore_without_leftover_jobs(self):
        original=self.pane.desired_fraction
        self.pointer("<ButtonPress-1>")
        self.pointer("<B1-Motion>",x_root=self.pane.paned.winfo_rootx()+780)
        self.entry.grab_set();self.tick(.08)
        self.assertIsNone(self.pane._drag)
        self.assertAlmostEqual(self.pane.desired_fraction,original)
        self.entry.grab_release()
        self.pointer("<ButtonPress-1>")
        self.pointer("<B1-Motion>",x_root=self.pane.paned.winfo_rootx()+790)
        self.pane.separator.event_generate("<KeyPress-Escape>");self.tick()
        self.assertIsNone(self.pane.grab_current())
        self.assertAlmostEqual(self.pane.desired_fraction,original)
        self.pane.close()
        self.assertIsNone(self.pane._layout_job)
        self.assertIsNone(self.pane._save_job)
        self.assertIsNone(self.pane._drag_watch_job)

    def test_escape_only_consumes_an_active_resize_and_preserves_app_shortcut(self):
        escaped=[]
        self.root.bind("<KeyPress-Escape>",lambda event:escaped.append(True))
        self.root.focus_force();self.pane.separator.focus_set();self.tick()
        self.pane.separator.event_generate("<KeyPress-Escape>");self.tick()
        self.assertEqual(len(escaped),1)
        self.pointer("<ButtonPress-1>")
        self.pointer("<B1-Motion>",x_root=self.pane.paned.winfo_rootx()+790)
        self.pane.separator.event_generate("<KeyPress-Escape>");self.tick()
        self.assertEqual(len(escaped),1)
        self.assertIsNone(self.pane._drag)


if __name__=="__main__":
    unittest.main()
