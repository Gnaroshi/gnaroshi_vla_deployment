"""Recorded-scene integrity, ambiguous matches and native operator workflow.

Fixtures are synthetic RGB images. Native checks use the CPU mock embodiment;
none of these tests connects to cameras, a policy server or robot hardware.
"""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import tkinter as tk
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from gnaroshi_deployment.scene_matching import index_rollouts, load_reference, match_objects, validate_objects
from gnaroshi_deployment.records import write_json


def fixture(root, *, method="method_a", task="CPU demo", rollout=1, image=None):
    session = Path(root) / task / method / "session_demo"
    folder = session / f"rollout_{rollout:04d}"
    folder.mkdir(parents=True)
    write_json(session / "manifest.json", {"config": {"task": task, "method": method,
               "metadata": {"checkpoint": "weights/example", "display_method": method}}})
    write_json(folder / "result.json", {"id": rollout, "outcome": "success"})
    if image is None:
        image = np.full((240, 320, 3), 30, np.uint8)
        image[70:110, 60:100] = np.random.default_rng(18).integers(40, 255, (40, 40, 3), dtype=np.uint8)
    path = folder / "initial_primary.png"
    Image.fromarray(image).save(path)
    snapshot = {"schema_version": 1, "stage": "after_home_before_first_action", "state": [0.0]*7,
                "camera_serials": {"primary": "mock-primary"},
                "images": {"primary": {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "width": image.shape[1], "height": image.shape[0]}}}
    write_json(folder / "initial_state.json", snapshot)
    return folder, image


class ReferenceIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_catalog_browses_both_methods_and_keeps_bad_records_visible_as_warnings(self):
        fixture(self.root)
        fixture(self.root, method="method_b")
        bad = self.root / "broken"; bad.mkdir()
        (bad / "result.json").write_text("not valid JSON")
        records, warnings = index_rollouts(self.root)
        self.assertEqual({row["method"] for row in records}, {"method_a", "method_b"})
        self.assertEqual(len(warnings), 1)
        self.assertIn("broken", warnings[0])

    def test_first_observation_is_exact_and_fingerprint_ignores_method_and_location(self):
        folder, expected = fixture(self.root)
        fixture(self.root, method="method_b", image=expected)
        before = {p: p.read_bytes() for p in folder.iterdir()}
        records, _ = index_rollouts(self.root)
        first, images = load_reference(self.root, records[0])
        second, _ = load_reference(self.root, records[1])
        self.assertEqual(first["source_kind"], "lossless_first_observation")
        np.testing.assert_array_equal(images["primary"], expected)
        self.assertEqual(first["reference_fingerprint"], second["reference_fingerprint"])
        self.assertEqual(before, {p: p.read_bytes() for p in folder.iterdir()})

    def test_annotation_does_not_break_portable_layout_identity(self):
        folder, _ = fixture(self.root)
        record = index_rollouts(self.root)[0][0]
        reference, _ = load_reference(self.root, record)
        write_json(folder / "result.json", {"id": 1, "outcome": "success", "operator_note": "A later note"})
        updated, _ = load_reference(self.root, reference)
        self.assertEqual(reference["reference_fingerprint"], updated["reference_fingerprint"])
        self.assertNotEqual(reference["result_sha256"], updated["result_sha256"])

    def test_corrupt_image_checksum_and_wrong_snapshot_stage_fail_closed(self):
        folder, _ = fixture(self.root)
        record = index_rollouts(self.root)[0][0]
        image_path = folder / "initial_primary.png"
        image_path.write_bytes(image_path.read_bytes()+b"tampered")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_reference(self.root, record)
        document = json.loads((folder / "initial_state.json").read_text())
        document["stage"] = "after_first_action"
        write_json(folder / "initial_state.json", document)
        with self.assertRaisesRegex(ValueError, "first policy observation"):
            load_reference(self.root, record)

    def test_escaping_reference_and_snapshot_symlinks_are_rejected(self):
        folder, _ = fixture(self.root)
        record = index_rollouts(self.root)[0][0]
        with self.assertRaisesRegex(ValueError, "leaves"):
            load_reference(self.root, {**record, "result_relative_path": "../result.json"})
        external = self.root / "other.png"
        shutil.copyfile(folder / "initial_primary.png", external)
        (folder / "initial_primary.png").unlink()
        (folder / "initial_primary.png").symlink_to(external)
        with self.assertRaisesRegex(ValueError, "leaves"):
            load_reference(self.root, record)

    def test_scan_bounds_and_cancellation(self):
        fixture(self.root)
        fixture(self.root, method="method_b")
        rows, warnings = index_rollouts(self.root, limit=1)
        self.assertEqual(len(rows), 1)
        self.assertTrue(warnings)
        cancel = threading.Event(); cancel.set()
        with self.assertRaises(InterruptedError):
            index_rollouts(self.root, cancel=cancel)

    def test_regions_must_stay_inside_declared_camera(self):
        _, image = fixture(self.root)
        region = {"id": "first", "label": "Object", "role": "primary", "box": [60, 70, 40, 40]}
        self.assertEqual(validate_objects([region], {"primary": image}), [region])
        for patch_item in ({"box": [-1, 0, 20, 20]}, {"box": [0, 0, 11, 20]}, {"role": "missing"}, {"label": ""}):
            with self.assertRaises(ValueError):
                validate_objects([{**region, **patch_item}], {"primary": image})
        with self.assertRaises(ValueError):
            validate_objects([region, region], {"primary": image})

    @unittest.skipUnless(shutil.which("ffmpeg"), "Legacy frame extraction requires ffmpeg")
    def test_legacy_source_is_explicit_and_loads_first_frame_and_state(self):
        folder, image = fixture(self.root)
        encoded = subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-nostdin", "-y",
                                  "-i", str(folder / "initial_primary.png"), "-frames:v", "1",
                                  "-c:v", "libx264", "-pix_fmt", "yuv420p", str(folder / "primary.mp4")],
                                 capture_output=True, timeout=10)
        self.assertEqual(encoded.returncode, 0, encoded.stderr.decode())
        (folder / "initial_state.json").unlink()
        (folder / "steps.jsonl").write_text(json.dumps({"state": [.25]*7})+"\n")
        record = index_rollouts(self.root)[0][0]
        reference, images = load_reference(self.root, record)
        self.assertEqual(reference["source_kind"], "legacy_decoded_first_video_frame")
        self.assertEqual(reference["initial_state"], [.25]*7)
        self.assertEqual(images["primary"].shape, image.shape)


class ObjectPlacement(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((180, 240, 3), 35, np.uint8)
        self.patch = np.random.default_rng(12).integers(20, 255, (32, 32, 3), dtype=np.uint8)
        self.reference[40:72, 60:92] = self.patch
        self.objects = [{"id": "object_1", "label": "Target", "role": "primary", "box": [60, 40, 32, 32]}]

    def test_translation_reports_correct_direction_and_original_is_aligned(self):
        live = np.full_like(self.reference, 35)
        live[48:80, 81:113] = self.patch
        row = match_objects(self.reference, live, self.objects)[0]
        self.assertTrue(row["matched"])
        self.assertFalse(row["aligned"])
        np.testing.assert_allclose(row["offset_px"], [21, 8], atol=.1)
        exact = match_objects(self.reference, self.reference, self.objects)[0]
        self.assertTrue(exact["aligned"])

    def test_duplicate_object_is_ambiguous_not_a_confident_alignment(self):
        live = self.reference.copy()
        live[100:132, 160:192] = self.patch
        row = match_objects(self.reference, live, self.objects)[0]
        self.assertEqual(row["status"], "ambiguous match")
        self.assertFalse(row["matched"])

    def test_missing_or_untextured_object_is_not_accepted(self):
        row = match_objects(self.reference, np.full_like(self.reference, 35), self.objects)[0]
        self.assertFalse(row["matched"])
        plain = np.full_like(self.reference, 35)
        row = match_objects(plain, plain, self.objects)[0]
        self.assertIn("texture too weak", row["status"])
        self.assertFalse(row["aligned"])

    def test_resolution_and_invalid_thresholds_reject_evidence(self):
        with self.assertRaisesRegex(ValueError, "resolution"):
            match_objects(self.reference, self.reference[:100], self.objects)
        for kwargs in ({"tolerance_px": float("nan")}, {"tolerance_px": 0}, {"min_score": 1.5}):
            with self.assertRaises(ValueError):
                match_objects(self.reference, self.reference, self.objects, **kwargs)


@unittest.skipUnless(os.environ.get("DISPLAY"), "Native workflow tests need a private display")
class SceneWorkflow(unittest.TestCase):
    def setUp(self):
        from gnaroshi_deployment.adapters.mock import MockEmbodiment, MockPolicy
        from gnaroshi_deployment.contracts import SessionConfig
        from gnaroshi_deployment.gui import DeploymentGUI
        from gnaroshi_deployment.session import DeploymentSession
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.reference_folder, self.image = fixture(self.directory / "results")
        self.record = index_rollouts(self.directory / "results")[0][0]
        self.root = tk.Tk(); self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda kind,error,traceback:self.errors.append((kind,error))
        config = SessionConfig(task="CPU demo", method="method_b", output_root=str(self.directory / "output"),
                               record_video=False)
        self.session = DeploymentSession(config, MockPolicy(), MockEmbodiment(), Path(__file__).parents[1])
        self.app = DeploymentGUI(self.root, self.session, results_root=self.directory / "results")
        self.panel = self.app.scene_panel
        self.wait(lambda:self.session.state == "READY" and not self.panel._busy and bool(self.panel.records))
        self.panel._loaded(load_reference(self.panel.results_root, self.record))
        self.observation = SimpleNamespace(images={"primary":self.image.copy(), "wrist":self.image.copy()}, state=np.zeros(7),
                                            captured_ns=time.time_ns(), camera_metadata={"primary":{"serial":"mock-primary"}, "wrist":{"serial":"mock-wrist"}})
        self.panel.refresh(self.observation)

    def tearDown(self):
        self.app.close(); self.session.worker.join(4)
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        try: self.root.destroy()
        except tk.TclError: pass
        self.assertEqual(self.errors, [])
        self.assertTrue(self.session.shutdown_complete)
        self.app = self.panel = self.root = None
        gc.collect()
        self.temporary.cleanup()

    def wait(self, predicate, timeout=5):
        deadline = time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>deadline: self.fail("Native scene workflow timed out")
            self.root.update(); time.sleep(.005)
        self.root.update()

    def tick(self):
        deadline = time.monotonic()+.18
        while time.monotonic()<deadline:
            self.root.update(); time.sleep(.005)

    def test_explicit_pairing_is_bidirectional_without_any_robot_motion(self):
        self.assertEqual(self.panel.reference["method"], "method_a")
        self.assertEqual(self.session.config.method, "method_b")
        self.assertTrue(self.panel.use_reference())
        context = self.app.trial_context["scene_match"]
        self.assertTrue(context["alignment"]["operator_confirmed"])
        self.assertIn("3-D", context["alignment"]["limitation"])
        self.assertTrue(self.panel.before_start())
        self.assertEqual(self.session.embodiment.home_count, 0)
        self.assertEqual(self.session.embodiment.applied, [])
        self.panel.clear_pairing()
        self.assertNotIn("scene_match", self.app.trial_context)
        self.assertIsNotNone(self.panel.reference)

    def test_any_camera_or_task_change_and_stale_frame_clear_pairing(self):
        self.assertTrue(self.panel.use_reference())
        changed = copy.deepcopy(self.observation)
        changed.camera_metadata["primary"]["serial"] = "mock-other"
        self.panel.refresh(changed)
        self.assertFalse(self.panel._armed)
        self.assertNotIn("scene_match", self.app.trial_context)
        self.assertFalse(self.panel.use_reference())
        self.panel.refresh(self.observation)
        self.assertTrue(self.panel.use_reference())
        self.session.config.task = "Different task"
        self.panel.refresh(self.observation)
        self.assertFalse(self.panel._armed)
        self.assertFalse(self.panel.use_reference())
        self.session.config.task = "CPU demo"
        self.observation.captured_ns -= 4_000_000_000
        self.panel.refresh(self.observation)
        self.assertIn("stale", self.panel._compatibility()[1])

    def test_unchecked_other_view_cannot_be_labelled_fully_aligned(self):
        self.panel.objects = [{"id":"a", "label":"A", "role":"primary", "box":[60,70,40,40]},
                              {"id":"b", "label":"B", "role":"wrist", "box":[0,0,20,20]}]
        self.panel.matches = [{**self.panel.objects[0], "aligned":True, "matched":True}]
        self.panel._match_measured_ns = time.time_ns()
        self.panel._match_parameters = (8.0,.65)
        context = self.panel.reference_context()
        self.assertEqual(context["alignment"]["status"], "needs_review")
        self.assertEqual(context["alignment"]["unchecked_object_ids"], ["b"])
        self.panel._match_measured_ns -= 4_000_000_000
        self.assertEqual(self.panel.reference_context()["alignment"]["status"], "not_recently_checked")

    def test_save_load_layout_is_portable_preserves_sources_and_requires_rearming(self):
        self.panel.objects = [{"id":"a", "label":"A", "role":"primary", "box":[60,70,40,40]}]
        self.panel.layout_id.set("layout_demo")
        self.panel.mode.set("Side by side"); self.panel.opacity.set(.7)
        self.panel.tolerance.set("5"); self.panel.min_score.set("0.8")
        destination = self.directory / "layout.json"
        before = (self.reference_folder / "initial_state.json").read_bytes()
        with patch("gnaroshi_deployment.scene_matching.filedialog.asksaveasfilename", return_value=str(destination)):
            self.panel.save_layout()
        self.assertTrue(destination.is_file())
        self.panel.objects = []
        self.panel.mode.set("Overlay"); self.panel.opacity.set(.1)
        self.panel.tolerance.set("100"); self.panel.min_score.set("0.3")
        with patch("gnaroshi_deployment.scene_matching.filedialog.askopenfilename", return_value=str(destination)):
            self.panel.load_layout()
        self.wait(lambda:not self.panel._busy)
        self.assertEqual(len(self.panel.objects), 1)
        self.assertFalse(self.panel._armed)
        self.assertEqual(before, (self.reference_folder / "initial_state.json").read_bytes())
        self.assertEqual(self.panel.layout_id.get(), "layout_demo")
        self.assertEqual(self.panel.mode.get(), "Side by side")
        self.assertAlmostEqual(self.panel.opacity.get(), .7)
        self.assertEqual(self.panel.tolerance.get(), "5")
        self.assertEqual(self.panel.min_score.get(), "0.8")

    def test_latest_manual_operation_wins_and_stale_worker_cannot_replace_reference(self):
        gate = threading.Event()
        seen = []
        self.panel._work(lambda:gate.wait(2), lambda result:seen.append("stale"))
        self.panel._work(lambda:"latest", seen.append)
        gate.set()
        self.wait(lambda:not self.panel._busy)
        self.assertEqual(seen, ["latest"])

    def test_resize_max_font_scroll_and_object_drag_keep_controls_reachable(self):
        self.root.deiconify(); self.root.geometry("960x720")
        self.app.tabs.select(self.app.scene_tab)
        self.app.set_text_size(24)
        self.tick(); self.panel.refresh(self.observation); self.tick()
        for widget in (*self.panel._toolbar_buttons, self.panel.draw_button, self.panel.use_button,
                       self.panel.remove_button, self.panel.check_button):
            right = widget.winfo_rootx()+widget.winfo_width()
            self.assertLessEqual(right, self.panel.winfo_rootx()+self.panel.winfo_width()+2, str(widget))
            self.assertGreater(widget.winfo_width(), 30)
        self.panel.pane.set_fraction(.23)
        self.panel.content.scroll_canvas.yview_moveto(1)
        self.tick()
        self.assertGreater(self.panel.content.scroll_canvas.yview()[0], 0)
        self.panel.content.scroll_canvas.yview_moveto(.3)
        self.panel.draw_region(); self.tick()
        self.assertGreater(self.panel.canvas.winfo_height(), 20)
        ox,oy,width,height,iw,ih = self.panel._draw_geometry
        def event(x,y):return SimpleNamespace(x=round(ox+x*width/iw),y=round(oy+y*height/ih))
        self.panel._drag_start(event(60,70)); self.panel._drag_move(event(100,110)); self.panel._drag_end(event(100,110))
        self.assertEqual(len(self.panel.objects), 1)
        np.testing.assert_allclose(self.panel.objects[0]["box"], [60,70,40,40], atol=2)
        self.assertEqual(self.session.embodiment.home_count, 0)
        self.root.geometry("1440x1000"); self.app.set_text_size(14)
        self.app.main_pane.set_fraction(.38)
        self.tick(); self.panel.refresh(self.observation); self.tick()
        self.assertGreaterEqual(self.panel.canvas.winfo_width(), 320)
        self.assertEqual(int(self.panel.editor.grid_info()["column"]), 1)
        self.app.main_pane.set_fraction(.26)
        self.tick(); self.panel.refresh(self.observation); self.tick()
        self.assertEqual(int(self.panel.editor.grid_info()["column"]), 1)
        self.assertGreater(self.panel.canvas.winfo_height(), 240)
        self.assertGreater(self.panel.canvas.winfo_width(), 300)
        self.assertLessEqual(self.panel.use_button.winfo_rootx()+self.panel.use_button.winfo_width(),
                             self.panel.winfo_rootx()+self.panel.winfo_width()+2)
        capture = os.environ.get("SCENE_MATCH_SCREENSHOTS")
        if capture:
            from PIL import ImageGrab
            directory = Path(capture); directory.mkdir(parents=True, exist_ok=True)
            self.panel.content.scroll_canvas.yview_moveto(0); self.tick()
            ImageGrab.grab().save(directory / "scene_wide_14pt.png")
            self.root.geometry("960x720"); self.app.set_text_size(24); self.tick()
            self.panel.refresh(self.observation); self.tick()
            ImageGrab.grab().save(directory / "scene_narrow_24pt.png")


if __name__ == "__main__":
    unittest.main()
