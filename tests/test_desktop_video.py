"""Local desktop delivery and CPU playback compatibility, without a live display."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from gnaroshi_deployment import desktop
from gnaroshi_deployment.desktop import compatible_video, desktop_environment, external_application_environment, open_local_path, recorded_video


class DesktopVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def video(self, name="primary.mp4"):
        path = self.root / name
        path.write_bytes(b"local video fixture")
        return path

    def test_recording_lookup_accepts_legacy_and_new_extensions_but_not_escape(self):
        older = self.video()
        self.assertEqual(recorded_video(self.root, "primary"), older)
        newer = self.video("primary.webm")
        self.assertEqual(recorded_video(self.root, "primary"), newer)
        with self.assertRaises(ValueError):
            recorded_video(self.root, "../primary")
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "wrist.webm").write_bytes(b"outside")
        (self.root / "wrist.webm").symlink_to(outside / "wrist.webm")
        with self.assertRaisesRegex(ValueError, "outside"):
            recorded_video(self.root, "wrist")

    def test_opener_uses_recovered_desktop_environment_and_reports_real_error(self):
        path = self.video("primary.webm")
        environment = {"DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus",
                       "DEPLOYMENT_GUI_TK_PRELOAD": "/gui/libtk8.6.so",
                       "LD_PRELOAD": "/foreign/libinstrument.so:/gui/libtk8.6.so"}
        with patch("gnaroshi_deployment.desktop.desktop_environment", return_value=environment) as probe, \
             patch("gnaroshi_deployment.desktop.shutil.which", return_value="/usr/bin/xdg-open"), \
             patch("gnaroshi_deployment.desktop._run_owned", return_value=subprocess.CompletedProcess([], 3, "", "no application for video/webm")) as launch:
            with self.assertRaisesRegex(RuntimeError, "exit 3.*no application"):
                open_local_path(path, environment=environment)
            self.assertEqual(launch.call_args.args[0], ["/usr/bin/xdg-open", str(path)])
            probe.assert_called_once_with(environment, cancel=None)
            self.assertEqual(launch.call_args.kwargs["environment"], {
                "DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus", "LD_PRELOAD": "/foreign/libinstrument.so"})
            self.assertIsNot(launch.call_args.kwargs["environment"], environment)
            self.assertEqual(environment["LD_PRELOAD"], "/foreign/libinstrument.so:/gui/libtk8.6.so")
            self.assertEqual(launch.call_args.kwargs["timeout"], 8)

    def test_external_application_filter_removes_only_marked_gui_libraries_and_matching_scripts(self):
        environment = {"DEPLOYMENT_GUI_TK_PRELOAD": "/gui/libtk8.6.so",
                       "LD_PRELOAD": "/foreign/libinstrument.so /gui/libtk8.6.so",
                       "DEPLOYMENT_GUI_TK_SCRIPTS": "/gui/tk8.6", "TK_LIBRARY": "/gui/tk8.6",
                       "DEPLOYMENT_GUI_TCL_SCRIPTS": "/gui/tcl8.6", "TCL_LIBRARY": "/gui/tcl8.6",
                       "DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus"}
        original = dict(environment)
        filtered = external_application_environment(environment)
        self.assertEqual(filtered, {"LD_PRELOAD": "/foreign/libinstrument.so", "DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus"})
        self.assertEqual(environment, original)
        self.assertIsNot(filtered, environment)

    def test_external_application_filter_handles_multiple_owned_libraries_and_preserves_unmarked_tk(self):
        environment = {"DEPLOYMENT_GUI_TK_PRELOAD": "/gui/libtk8.6.so:/gui/libtcl8.6.so",
                       "LD_PRELOAD": "/gui/libtk8.6.so /foreign/libtk8.6.so:/gui/libtcl8.6.so /foreign/libtcl8.7.so",
                       "DEPLOYMENT_GUI_TK_SCRIPTS": "/gui/tk8.6", "TK_LIBRARY": "/gui/tk8.6",
                       "DEPLOYMENT_GUI_TCL_SCRIPTS": "/gui/tcl8.6", "TCL_LIBRARY": "/foreign/tcl8.6"}
        filtered = external_application_environment(environment)
        self.assertEqual(filtered, {"LD_PRELOAD": "/foreign/libtk8.6.so:/foreign/libtcl8.7.so", "TCL_LIBRARY": "/foreign/tcl8.6"})
        own_only = {"DEPLOYMENT_GUI_TK_PRELOAD": "/gui/libtk8.6.so /gui/libtcl8.6.so",
                    "LD_PRELOAD": "/gui/libtk8.6.so:/gui/libtcl8.6.so"}
        self.assertEqual(external_application_environment(own_only), {})

    def test_external_application_filter_without_markers_preserves_environment_exactly(self):
        environment = {"LD_PRELOAD": "/foreign/libtk8.6.so  : /foreign/libinstrument.so",
                       "TK_LIBRARY": "/foreign/tk8.6", "TCL_LIBRARY": "/foreign/tcl8.6",
                       "DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus"}
        original = dict(environment)
        filtered = external_application_environment(environment)
        self.assertEqual(filtered, original)
        self.assertEqual(environment, original)
        self.assertIsNot(filtered, environment)

    def test_converter_receives_filtered_environment_and_keeps_parent_gui_environment(self):
        environment = {"DEPLOYMENT_GUI_TK_PRELOAD": "/gui/libtk8.6.so",
                       "LD_PRELOAD": "/gui/libtk8.6.so:/foreign/libinstrument.so",
                       "DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus"}
        source = self.video()
        def converter(command, **kwargs):
            Path(command[-1]).write_bytes(b"compatibility fixture")
            return subprocess.CompletedProcess(command, 0, "", "")
        with patch.dict(os.environ, environment, clear=True), patch.object(desktop, "_run_owned", side_effect=converter) as launch:
            playable = compatible_video(source, self.root / ".runtime" / "playback")
            self.assertTrue(playable.is_file())
            self.assertEqual(launch.call_args.kwargs["environment"], {
                "LD_PRELOAD": "/foreign/libinstrument.so", "DISPLAY": ":98", "DBUS_SESSION_BUS_ADDRESS": "test-bus"})
            self.assertEqual(dict(os.environ), environment)

    def test_ssh_environment_recovers_desktop_session_fields_without_changing_process(self):
        original = {"PATH": "/usr/bin", "DISPLAY": ":99"}
        raw = b"DISPLAY=:1\0XAUTHORITY=/run/user/example/Xauthority\0XDG_CURRENT_DESKTOP=ubuntu:GNOME\0DBUS_SESSION_BUS_ADDRESS=desktop-bus\0PRIVATE_VALUE=do-not-copy\0"
        probes = []
        def run(command, **kwargs):
            if command[0] == "pgrep":
                return subprocess.CompletedProcess(command, 0, "123\n", "")
            env = kwargs["environment"]
            probes.append(env)
            return subprocess.CompletedProcess(command, 0 if env.get("DISPLAY") == ":1" else 1, "", "")
        with patch("gnaroshi_deployment.desktop._run_owned", side_effect=run), \
             patch("gnaroshi_deployment.desktop.Path.read_bytes", return_value=raw):
            recovered = desktop_environment(original)
        self.assertEqual(original, {"PATH": "/usr/bin", "DISPLAY": ":99"})
        self.assertEqual(recovered["DISPLAY"], ":1")
        self.assertEqual(recovered["XDG_CURRENT_DESKTOP"], "ubuntu:GNOME")
        self.assertEqual(recovered["DBUS_SESSION_BUS_ADDRESS"], "desktop-bus")
        self.assertNotIn("PRIVATE_VALUE", recovered)
        self.assertEqual(len(probes), 2)

    def test_open_folder_does_not_transcode_and_external_uri_is_not_launched(self):
        with patch("gnaroshi_deployment.desktop.desktop_environment", return_value={"DISPLAY": ":98"}), \
             patch("gnaroshi_deployment.desktop.shutil.which", return_value="/usr/bin/xdg-open"), \
             patch("gnaroshi_deployment.desktop.compatible_video") as convert, \
             patch("gnaroshi_deployment.desktop._run_owned", return_value=subprocess.CompletedProcess([], 0, "", "")) as launch:
            self.assertEqual(open_local_path(self.root), self.root)
            convert.assert_not_called()
            with self.assertRaises(FileNotFoundError):
                open_local_path("https://invalid.example/video.mp4")
            self.assertEqual(launch.call_count, 1)

    def test_conversion_timeout_and_failure_leave_original_and_no_partial_cache(self):
        original = self.video()
        content = original.read_bytes()
        cache = self.root / ".runtime" / "playback"
        with patch("gnaroshi_deployment.desktop._run_owned", side_effect=subprocess.TimeoutExpired("converter", 0.1)):
            with self.assertRaisesRegex(RuntimeError, "original is unchanged"):
                compatible_video(original, cache, timeout=0.1)
        self.assertEqual(original.read_bytes(), content)
        self.assertEqual(list(cache.iterdir()), [])
        with patch("gnaroshi_deployment.desktop._run_owned", return_value=subprocess.CompletedProcess([], 1, "", "cannot encode VP8")):
            with self.assertRaisesRegex(RuntimeError, "cannot encode VP8"):
                compatible_video(original, cache)
        self.assertEqual(list(cache.iterdir()), [])
        with self.assertRaisesRegex(ValueError, ".runtime"):
            # TMPDIR itself may be a project's .runtime directory on a robot.
            # This path is rejected before directory creation or conversion.
            compatible_video(original, Path(self.root.anchor) / "unscoped-playback-cache")

    def test_real_cpu_conversion_preserves_frames_fps_source_and_reuses_cache(self):
        import cv2
        source = self.root / "primary.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (64, 48))
        self.assertTrue(writer.isOpened(), "The test environment must support existing MPEG-4 recording")
        for index in range(7):
            frame = np.zeros((48, 64, 3), np.uint8)
            frame[:, :, 2] = index * 30
            writer.write(frame)
        writer.release()
        original = source.read_bytes()
        cache = self.root / ".runtime" / "playback"
        playable = compatible_video(source, cache)
        capture = cv2.VideoCapture(str(playable))
        try:
            self.assertTrue(capture.isOpened())
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS), 12.5, places=3)
            frames = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
            self.assertEqual(len(frames), 7)
            self.assertGreater(float(frames[-1][:, :, 2].mean()), float(frames[0][:, :, 2].mean()) + 150)
        finally:
            capture.release()
        self.assertEqual(source.read_bytes(), original)
        with patch("gnaroshi_deployment.desktop._run_owned") as converter:
            self.assertEqual(compatible_video(source, cache), playable)
            converter.assert_not_called()

    def test_cancelled_before_launch_does_not_probe_desktop_or_start_any_process(self):
        cancel = threading.Event()
        cancel.set()
        with patch.object(desktop, "desktop_environment") as probe, \
             patch.object(desktop.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                open_local_path(self.root, cancel=cancel)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                compatible_video(self.video(), self.root / ".runtime" / "playback", cancel=cancel)
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                desktop._run_owned([sys.executable, "-c", "raise AssertionError('must not start')"], timeout=1, cancel=cancel)
            probe.assert_not_called()
            launch.assert_not_called()

    def _converter_fixture(self):
        script = self.root / "owned_converter_fixture.py"
        script.write_text('''import json, os, pathlib, signal, subprocess, sys, time
record, destination = map(pathlib.Path, sys.argv[1:])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)', __file__])
def stop(signum, frame):
    if child.poll() is None:
        child.terminate()
    child.wait(timeout=2)
    record.with_suffix('.reaped').write_text(str(child.returncode))
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
destination.write_bytes(b'partial playback file')
record.write_text(json.dumps({'parent': os.getpid(), 'child': child.pid, 'pgid': os.getpgrp()}))
while True:
    time.sleep(.05)
''')
        record = self.root / "owned_processes.json"
        def cleanup():
            if record.is_file():
                owned = json.loads(record.read_text())
                for pid in (owned["parent"], owned["child"]):
                    proc = Path(f"/proc/{pid}")
                    try:
                        # Limit exceptional cleanup to this exact fixture's group.
                        if str(script).encode() in (proc / "cmdline").read_bytes() and os.getpgid(pid) == owned["pgid"]:
                            os.killpg(owned["pgid"], signal.SIGKILL)
                            break
                    except (FileNotFoundError, ProcessLookupError):
                        pass
        self.addCleanup(cleanup)
        return script, record

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "POSIX process-group lifecycle test")
    def test_actual_conversion_cancel_reaps_owned_parent_child_and_partial_cache(self):
        source = self.video()
        original = source.read_bytes()
        cache = self.root / ".runtime" / "playback"
        script, record = self._converter_fixture()
        runner = desktop._run_owned
        cancel = threading.Event()
        started = threading.Event()
        def cancel_when_ready():
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not record.exists():
                time.sleep(.01)
            if record.exists():
                started.set()
            cancel.set()
        trigger = threading.Thread(target=cancel_when_ready)
        trigger.start()
        def substitute_converter(command, **kwargs):
            self.assertEqual(command[2], "--convert-video")
            return runner([sys.executable, str(script), str(record), command[-1]], **kwargs)
        before = time.monotonic()
        try:
            with patch.object(desktop, "_run_owned", side_effect=substitute_converter):
                with self.assertRaisesRegex(RuntimeError, "cancelled"):
                    compatible_video(source, cache, timeout=5, cancel=cancel)
        finally:
            trigger.join(4)
        self.assertFalse(trigger.is_alive())
        self.assertTrue(started.is_set(), "The real converter must have started before cancellation")
        self.assertLess(time.monotonic()-before, 4)
        owned = json.loads(record.read_text())
        self.assertEqual(owned["parent"], owned["pgid"])
        self.assertTrue(record.with_suffix(".reaped").is_file(), "The converter must wait for its child")
        self.assertTrue(all(not Path(f"/proc/{owned[key]}").exists() for key in ("parent", "child")), "Owned processes must be gone, including zombies")
        self.assertEqual(list(cache.iterdir()), [])
        self.assertEqual(source.read_bytes(), original)

    @unittest.skipUnless(os.name == "posix" and Path("/proc").is_dir(), "POSIX process-group lifecycle test")
    def test_actual_conversion_timeout_reaps_owned_parent_child_and_partial_cache(self):
        source = self.video()
        original = source.read_bytes()
        cache = self.root / ".runtime" / "playback"
        script, record = self._converter_fixture()
        runner = desktop._run_owned
        def substitute_converter(command, **kwargs):
            self.assertEqual(command[2], "--convert-video")
            return runner([sys.executable, str(script), str(record), command[-1]], **kwargs)
        before = time.monotonic()
        with patch.object(desktop, "_run_owned", side_effect=substitute_converter):
            with self.assertRaisesRegex(RuntimeError, "exceeded.*original is unchanged"):
                compatible_video(source, cache, timeout=.4)
        self.assertLess(time.monotonic()-before, 3)
        owned = json.loads(record.read_text())
        self.assertEqual(owned["parent"], owned["pgid"])
        self.assertTrue(record.with_suffix(".reaped").is_file())
        self.assertTrue(all(not Path(f"/proc/{owned[key]}").exists() for key in ("parent", "child")), "Owned processes must be gone, including zombies")
        self.assertEqual(list(cache.iterdir()), [])
        self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
