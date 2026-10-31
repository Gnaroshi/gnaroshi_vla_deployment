"""Read-only rollout references and image-plane placement assistance.

This module never acquires a camera or controls a robot. It compares already
available observations with recorded first observations. Pixel alignment is an
operator aid, not a claim of identical 3-D object poses or a motion interlock.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from PIL import Image, ImageTk

from .records import write_json


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json(path, limit=2 * 1024 * 1024):
    path = Path(path)
    if path.stat().st_size > limit:
        raise ValueError(f"Record is too large: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def _contained(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Reference path leaves the selected results folder")
    return path


def _serials(metadata):
    result = {}
    for role, item in (metadata or {}).items():
        if isinstance(item, dict):
            serial = item.get("serial") or item.get("serial_number") or item.get("device_serial")
            if serial is not None:
                result[str(role)] = str(serial)
    return result


def _background_operation(output, generation, token, operation):
    """Worker owns only data, never a Tk widget or a bound GUI callback."""
    try: output.put((generation, token, operation(), None))
    except Exception as error: output.put((generation, token, None, f"{type(error).__name__}: {error}"))


def index_rollouts(root, *, cancel=None, limit=2000, max_directories=20000, timeout_s=8):
    """Bounded, symlink-safe discovery across methods/checkpoints, newest first."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError("Choose an existing results folder")
    cancel = cancel or threading.Event()
    records, warnings, cache = [], [], {}
    deadline, visited = time.monotonic() + timeout_s, 0
    for directory, names, files in os.walk(root, followlinks=False):
        visited += 1
        if cancel.is_set():
            raise InterruptedError("Reference scan cancelled")
        if time.monotonic() >= deadline or visited > max_directories or len(records) >= limit:
            warnings.append("Scan limit reached; choose a narrower results folder.")
            break
        names[:] = sorted(name for name in names if not name.startswith(".")
                          and not (Path(directory) / name).is_symlink())
        if "result.json" not in files:
            continue
        try:
            folder = _contained(root, directory)
            result_path = _contained(root, folder / "result.json")
            result = _json(result_path)
            manifest = {}
            ancestor = folder.parent
            while ancestor.is_relative_to(root):
                if ancestor not in cache:
                    candidate = ancestor / "manifest.json"
                    cache[ancestor] = _json(_contained(root, candidate)) if candidate.is_file() else None
                if cache[ancestor] is not None:
                    manifest = cache[ancestor]
                    break
                if ancestor == root:
                    break
                ancestor = ancestor.parent
            config = manifest.get("config", {})
            metadata = config.get("metadata", {})
            checkpoint = metadata.get("checkpoint") or metadata.get("controller") or ""
            if isinstance(checkpoint, dict):
                checkpoint = checkpoint.get("path", "")
            label = Path(str(checkpoint)).name or "unspecified"
            controller = metadata.get("selection_identity", {}).get("controller", {})
            if isinstance(controller, dict) and controller.get("sha256"):
                label += " / " + str(controller["sha256"])[:10]
            relative = result_path.relative_to(root).as_posix()
            records.append({"record_id": _digest(relative.encode()),
                            "result_relative_path": relative,
                            "result_sha256": _digest(result_path.read_bytes()),
                            "task": str(config.get("task", result.get("task", "unknown"))),
                            "method": str(metadata.get("display_method") or config.get("method", "unknown")),
                            "checkpoint": label, "rollout_id": result.get("id", "?"),
                            "outcome": str(result.get("outcome", "unknown")),
                            "created_ns": result_path.stat().st_mtime_ns,
                            "camera_serials": result.get("camera_serials", {}) or metadata.get("robot", {}).get("camera_serials", {})
                                or manifest.get("embodiment", {}).get("camera_serials", {}),
                            "state_description": metadata.get("state", "") or manifest.get("embodiment", {}).get("state", ""),
                            "home_joints": metadata.get("robot", {}).get("home_joints"),
                            "source_kind": "snapshot" if (folder / "initial_state.json").is_file() else "legacy_video"})
        except (OSError, ValueError, TypeError, AttributeError) as error:
            warnings.append(f"Skipped {Path(directory).name}: {error}")
    records.sort(key=lambda value: (value["created_ns"], value["record_id"]), reverse=True)
    return records, warnings[:20]


def load_reference(root, record):
    """Read lossless snapshots; older recordings explicitly use a decoded frame."""
    root = Path(root).resolve()
    result_path = _contained(root, root / record["result_relative_path"])
    folder = result_path.parent
    if _digest(record["result_relative_path"].encode()) != record["record_id"]:
        raise ValueError("The reference identifier does not match its result path")
    _json(result_path)
    snapshot = folder / "initial_state.json"
    images, hashes, state, serials = {}, {}, [], dict(record.get("camera_serials", {}))
    if snapshot.is_file():
        data = _json(_contained(root, snapshot))
        if data.get("stage") != "after_home_before_first_action":
            raise ValueError("Snapshot is not marked as the first policy observation")
        state = data.get("state", [])
        serials.update(_serials(data.get("camera_metadata", {})))
        serials.update(data.get("camera_serials", {}))
        if len(data.get("images", {})) > 8:
            raise ValueError("A reference supports at most eight camera views")
        for role, description in data.get("images", {}).items():
            if not isinstance(description, dict) or not role.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Invalid snapshot camera role")
            path = _contained(folder, folder / description["path"])
            if path.stat().st_size > 32 * 1024 * 1024:
                raise ValueError("Snapshot image exceeds the supported size")
            raw = path.read_bytes()
            if _digest(raw) != description.get("sha256"):
                raise ValueError(f"The {role} snapshot checksum does not match")
            with Image.open(path) as source:
                if max(source.size) > 4096:
                    raise ValueError("Unsupported reference image dimensions")
                source.load()
                images[role] = np.asarray(source.convert("RGB")).copy()
            hashes[role] = _digest(images[role].tobytes())
        source_kind = "lossless_first_observation"
    else:
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise ValueError("Legacy references need ffmpeg; new initial snapshots load directly")
        for path in sorted(folder.glob("*.mp4"))[:8]:
            path = _contained(folder, path)
            if not path.stem.replace("_", "").replace("-", "").isalnum():
                continue
            decoded = subprocess.run([executable, "-v", "error", "-nostdin", "-i", str(path),
                                      "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                                     capture_output=True, timeout=5, check=False)
            if decoded.returncode == 0 and decoded.stdout:
                if len(decoded.stdout) > 32 * 1024 * 1024:
                    raise ValueError("Decoded reference frame exceeds the supported size")
                with Image.open(io.BytesIO(decoded.stdout)) as source:
                    if max(source.size) > 4096:
                        raise ValueError("Unsupported reference image dimensions")
                    images[path.stem] = np.asarray(source.convert("RGB")).copy()
                hashes[path.stem] = _digest(images[path.stem].tobytes())
        steps = folder / "steps.jsonl"
        if steps.is_file():
            with _contained(folder, steps).open() as stream:
                first = json.loads(stream.readline(1024 * 1024))
            state = first.get("state", [])
        frames = folder / "frames.jsonl"
        if frames.is_file():
            with _contained(folder, frames).open() as stream:
                first = json.loads(stream.readline(1024 * 1024))
            serials.update(_serials(first.get("camera_metadata", {})))
        source_kind = "legacy_decoded_first_video_frame"
    if not images:
        raise ValueError("This rollout has no initial images or readable video")
    state_array = np.asarray(state, dtype=float)
    if state_array.ndim != 1 or state_array.size > 1024 or not np.isfinite(state_array).all():
        raise ValueError("Unsupported saved robot state")
    state = state_array.tolist()
    for image in images.values():
        if image.ndim != 3 or image.shape[2] != 3 or max(image.shape[:2]) > 4096:
            raise ValueError("Unsupported reference image dimensions")
    reference = {**record, "source_kind": source_kind, "initial_state": state,
                 "state_description": data.get("identity", {}).get("state_layout", record.get("state_description", "")) if snapshot.is_file() else record.get("state_description", ""),
                 "result_sha256": _digest(result_path.read_bytes()),
                 "camera_serials": {str(k): str(v) for k, v in serials.items()},
                 "image_sha256_by_role": hashes}
    fingerprint_data = {"task": record["task"], "camera_serials": reference["camera_serials"],
                        "images": hashes, "shapes": {role:list(image.shape) for role,image in images.items()}}
    reference["reference_fingerprint"] = _digest(json.dumps(fingerprint_data, sort_keys=True).encode())
    return reference, images


def validate_objects(objects, images):
    if not isinstance(objects, list) or len(objects) > 12:
        raise ValueError("A layout supports at most 12 object regions")
    checked, identifiers = [], set()
    for item in objects:
        role, label, identifier = item.get("role"), str(item.get("label", "")).strip(), str(item.get("id", ""))
        if role not in images or not label or len(label) > 100 or not identifier or identifier in identifiers:
            raise ValueError("Each object needs a camera, unique ID and short label")
        box = item.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError("Object regions use integer x, y, width and height")
        x, y, w, h = box
        ih, iw = images[role].shape[:2]
        if min(x, y) < 0 or min(w, h) < 12 or x + w > iw or y + h > ih:
            raise ValueError("Draw an object region of at least 12 pixels inside the reference")
        identifiers.add(identifier)
        checked.append({"id": identifier, "label": label, "role": role, "box": list(box)})
    return checked


def match_objects(reference, live, objects, *, tolerance_px=8.0, min_score=.65):
    """Multiscale appearance matching with a distinct-peak ambiguity check.

    Similarity is not a probability. A repeated/textureless object is reported
    as ambiguous rather than silently accepting a visually plausible location.
    """
    import cv2
    if reference.shape != live.shape or reference.dtype != np.uint8 or live.dtype != np.uint8:
        raise ValueError("Reference and live camera resolution must match")
    if not math.isfinite(tolerance_px) or tolerance_px <= 0 or not 0 < min_score <= 1:
        raise ValueError("Use a positive pixel tolerance and a similarity threshold in (0, 1]")
    factor = min(1.0, 960 / reference.shape[1])
    size = (max(1, round(reference.shape[1] * factor)), max(1, round(reference.shape[0] * factor)))
    ref = cv2.resize(reference, size) if factor < 1 else reference
    image = cv2.resize(live, size) if factor < 1 else live
    rows = []
    for item in objects:
        x, y, w, h = item["box"]
        sx, sy, sw, sh = [round(value * factor) for value in (x, y, w, h)]
        template = ref[sy:sy + sh, sx:sx + sw]
        row = {**item, "matched": False, "aligned": False, "status": "not found"}
        if min(template.shape[:2]) < 8 or np.std(cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)) < 3:
            rows.append({**row, "status": "texture too weak; use visual overlay"})
            continue
        best = None
        for scale in (.9, 1.0, 1.1):
            width, height = max(8, round(sw * scale)), max(8, round(sh * scale))
            if width >= image.shape[1] or height >= image.shape[0]:
                continue
            sample = cv2.resize(template, (width, height))
            heat = cv2.matchTemplate(image, sample, cv2.TM_CCOEFF_NORMED)
            _, score, _, point = cv2.minMaxLoc(heat)
            alternative = heat.copy()
            px, py = point
            radius_x, radius_y = max(5, width // 2), max(5, height // 2)
            alternative[max(0, py-radius_y):py+radius_y+1, max(0, px-radius_x):px+radius_x+1] = -1
            second = float(alternative.max())
            if best is None or score > best[0]:
                best = (score, point, width, height, second)
        if best is None:
            rows.append(row)
            continue
        score, (px, py), width, height, second = best
        box = [px / factor, py / factor, width / factor, height / factor]
        dx, dy = box[0] + box[2] / 2 - (x + w / 2), box[1] + box[3] / 2 - (y + h / 2)
        matched = score >= min_score and score - second >= .06
        aligned = matched and math.hypot(dx, dy) <= tolerance_px and abs(width / sw - 1) <= .05
        rows.append({**row, "score": float(score), "peak_margin": float(score-second),
                     "live_box": box, "offset_px": [dx, dy], "matched": matched, "aligned": aligned,
                     "status": "aligned in image" if aligned else "move to outline" if matched
                     else "ambiguous match" if score >= min_score else "not found"})
    return rows


class SceneMatchingPanel(ttk.Frame):
    """A session-independent, bidirectional saved-rollout placement workbench."""
    def __init__(self, parent, app, results_root=None):
        super().__init__(parent)
        self.app = app
        self.results_root = Path(results_root or app.session.config.output_root).expanduser().resolve()
        self.reference = None
        self.images, self.objects, self.matches = {}, [], []
        self.records, self._visible_records = [], {}
        self.latest = None
        self._queue = queue.Queue()
        self._cancel = threading.Event()
        self._busy, self._generation, self._last_match = False, 0, 0.0
        self._pending_work = None
        self._callbacks, self._next_work_token = {}, 0
        self._photos, self._draw_geometry = {}, None
        self._reference_preview = None
        self._drag = None
        self._armed = False
        self._match_stamp = None
        self._match_measured_ns = None
        self._match_parameters = None
        self._render_key = None
        self._last_text_size = None
        self._controls_geometry = None
        self._last_compatibility_note = None
        self._live_signature = None
        self._scan_errors = []
        self._closed = False
        self.status = tk.StringVar(value="Choose a saved rollout, then match the objects to its initial scene.")
        self.summary = tk.StringVar(value="No reference selected")
        self.robot_summary = tk.StringVar(value="Robot pose comparison appears with live state feedback.")
        self.search = tk.StringVar(value="")
        self.task_filter = tk.StringVar(value="All tasks")
        self.method_filter = tk.StringVar(value="All methods")
        self.role = tk.StringVar(value=app.preferences.get("scene_camera_role", "primary"))
        self.mode = tk.StringVar(value=app.preferences.get("scene_view", "Overlay"))
        self.opacity = tk.DoubleVar(value=app.preferences.get("scene_blend_percent", 50)/100)
        self.layout_id = tk.StringVar(value="")
        self.object_label = tk.StringVar(value="Object 1")
        self.tolerance = tk.StringVar(value="8")
        self.min_score = tk.StringVar(value="0.65")
        self.grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1); parent.rowconfigure(0, weight=1)
        content = self.content = app._scrollable_content(self, fill_height=True)
        content.minimum_height = 690
        content.columnconfigure(0, weight=1); content.rowconfigure(2, weight=1)
        self.status_label = ttk.Label(content, textvariable=self.status, wraplength=430, style="Caption.TLabel")
        self.status_label.grid(row=0, sticky="ew", pady=(0, 8))
        top = self.toolbar = ttk.Frame(content); top.grid(row=1, sticky="ew", pady=(0, 8))
        self.browse_button = ttk.Button(top, text="Results folder", width=0, command=self.choose_root)
        self.scan_button = ttk.Button(top, text="Saved rollouts", width=0, command=self.scan)
        self.open_layout_button = ttk.Button(top, text="Load layout", width=0, command=self.load_layout)
        self.clear_pairing_button = ttk.Button(top, text="Clear pairing", width=0, command=self.clear_pairing)
        self._toolbar_buttons = (self.browse_button, self.scan_button, self.open_layout_button, self.clear_pairing_button)
        for column, button in enumerate(self._toolbar_buttons): button.grid(row=0, column=column, sticky="ew", padx=(0, 4))
        self.pane = app._pane(content, key="scene_matching", orient="vertical", first_label="Saved rollouts",
                              second_label="Placement", first_min=125, second_min=430,
                              default_fraction=.25, collapsible=(True, False), show_header=False)
        self.pane.grid(row=2, sticky="nsew")
        self._build_browser(self.pane.first)
        self._build_matching(self.pane.second)
        self._poll_job = app.root.after(80, self._poll)
        self._scan_job = app.root.after_idle(self._initial_scan)

    def _initial_scan(self):
        self._scan_job = None
        if not self._closed: self.scan()

    def _build_browser(self, parent):
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        filters = ttk.Frame(parent); filters.grid(row=0, sticky="ew", pady=(0, 4))
        for column in (0, 1): filters.columnconfigure(column, weight=1)
        self.task_picker = ttk.Combobox(filters, textvariable=self.task_filter, values=("All tasks",), state="readonly", width=10)
        self.method_picker = ttk.Combobox(filters, textvariable=self.method_filter, values=("All methods",), state="readonly", width=10)
        self.task_picker.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.method_picker.grid(row=0, column=1, sticky="ew")
        self.task_picker.bind("<<ComboboxSelected>>", lambda _event:self._filter())
        self.method_picker.bind("<<ComboboxSelected>>", lambda _event:self._filter())
        search_row = ttk.Frame(parent); search_row.grid(row=1, sticky="ew", pady=(0, 4))
        search_row.columnconfigure(1, weight=1)
        ttk.Label(search_row, text="Search", style="Caption.TLabel").grid(row=0, column=0, padx=(0, 8))
        entry = ttk.Entry(search_row, textvariable=self.search)
        entry.grid(row=0, column=1, sticky="ew")
        entry.bind("<KeyRelease>", lambda _event:self._filter())
        entry.accessible_name = "Search saved task, method, checkpoint or rollout"
        listing = ttk.Frame(parent); listing.grid(row=2, sticky="nsew")
        listing.columnconfigure(0, weight=1); listing.rowconfigure(0, weight=1)
        self.table = ttk.Treeview(listing, columns=("task", "method", "checkpoint", "run", "outcome", "date"), show="headings", height=4, selectmode="browse")
        for name, title, width in (("task", "Task", 85), ("method", "Method", 120), ("checkpoint", "Checkpoint", 110),
                                    ("run", "Run", 45), ("outcome", "Result", 75), ("date", "Saved / edited", 130)):
            self.table.heading(name, text=title); self.table.column(name, width=width, minwidth=45, stretch=name in {"method", "checkpoint"})
        self.table.grid(row=0, column=0, sticky="nsew")
        y = ttk.Scrollbar(listing, orient="vertical", command=self.table.yview); y.grid(row=0, column=1, sticky="ns")
        x = ttk.Scrollbar(listing, orient="horizontal", command=self.table.xview); x.grid(row=1, column=0, sticky="ew")
        self.table.configure(yscrollcommand=y.set, xscrollcommand=x.set)
        self.table.bind("<<TreeviewSelect>>", self._select_reference)

    def _build_matching(self, parent):
        self.matching_surface = parent
        parent.columnconfigure(0, weight=1); parent.rowconfigure(2, weight=1)
        self.summary_label = ttk.Label(parent, textvariable=self.summary, style="Caption.TLabel", wraplength=430)
        self.summary_label.grid(row=0, sticky="ew", pady=(4, 4))
        modes = self.view_controls = ttk.Frame(parent); modes.grid(row=1, sticky="ew", pady=(0, 4))
        modes.columnconfigure(1, weight=1)
        self.role_picker = ttk.Combobox(modes, textvariable=self.role, state="readonly", width=8)
        self.role_picker.grid(row=0, column=0, sticky="w", padx=(0, 4))
        self.role_picker.bind("<<ComboboxSelected>>", lambda _event:self._view_changed())
        self.mode_picker = ttk.Combobox(modes, textvariable=self.mode, state="readonly", width=12,
                                      values=("Overlay", "Side by side", "Reference", "Live"))
        self.mode_picker.grid(row=0, column=1, sticky="ew")
        self.mode_picker.bind("<<ComboboxSelected>>", lambda _event:self._view_changed())
        ttk.Label(modes, text="Blend", style="Caption.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Scale(modes, from_=0, to=1, variable=self.opacity, command=self._blend_changed).grid(row=1, column=1, sticky="ew")
        self.canvas = tk.Canvas(parent, width=1, height=220, background="#181E26", highlightthickness=1, highlightbackground="#354253")
        self.canvas.grid(row=2, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event:self._render())
        self.canvas.bind("<ButtonPress-1>", self._drag_start)
        self.canvas.bind("<B1-Motion>", self._drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._drag_end)
        self.robot_label = ttk.Label(parent, textvariable=self.robot_summary, style="Caption.TLabel", wraplength=430)
        self.robot_label.grid(row=3, sticky="ew", pady=(4, 4))
        editor = self.editor = ttk.Frame(parent); editor.grid(row=4, sticky="ew"); editor.columnconfigure(1, weight=1)
        ttk.Label(editor, text="Object").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(editor, textvariable=self.object_label, width=1).grid(row=0, column=1, sticky="ew")
        self.draw_button = ttk.Button(editor, text="Draw region", width=0, command=self.draw_region)
        self.draw_button.grid(row=0, column=2, padx=(4, 0))
        object_list = self.object_list = ttk.Frame(parent); object_list.grid(row=5, sticky="ew", pady=(4, 4)); object_list.columnconfigure(0, weight=1)
        object_list.rowconfigure(0, weight=1)
        self.objects_table = ttk.Treeview(object_list, columns=("object", "offset", "score", "match"), show="headings", height=3, selectmode="browse")
        for key, label, width in (("object", "Object", 85), ("offset", "Offset (px)", 105), ("score", "Similarity", 85), ("match", "Placement", 170)):
            self.objects_table.heading(key, text=label); self.objects_table.column(key, width=width, minwidth=50, stretch=key=="match")
        self.objects_table.grid(row=0, column=0, sticky="nsew")
        object_scroll = ttk.Scrollbar(object_list, orient="vertical", command=self.objects_table.yview)
        object_scroll.grid(row=0, column=1, sticky="ns")
        object_xscroll = ttk.Scrollbar(object_list, orient="horizontal", command=self.objects_table.xview)
        object_xscroll.grid(row=1, column=0, sticky="ew")
        self.objects_table.configure(yscrollcommand=object_scroll.set, xscrollcommand=object_xscroll.set)
        self.objects_table.tag_configure("aligned", foreground="#A7D7C8")
        self.objects_table.tag_configure("unmatched", foreground="#E8CC93")
        options = self.options = ttk.Frame(parent); options.grid(row=6, sticky="ew")
        options.columnconfigure(0, weight=1); options.columnconfigure(1, weight=1)
        self.tolerance_fields = ttk.Frame(options)
        ttk.Label(self.tolerance_fields, text="Tolerance px", style="Caption.TLabel").pack(side="left")
        ttk.Entry(self.tolerance_fields, textvariable=self.tolerance, width=4).pack(side="left", padx=(4, 8))
        self.similarity_fields = ttk.Frame(options)
        ttk.Label(self.similarity_fields, text="Similarity", style="Caption.TLabel").pack(side="left")
        ttk.Entry(self.similarity_fields, textvariable=self.min_score, width=5).pack(side="left", padx=4)
        self.remove_button = ttk.Button(options, text="Remove region", width=0, command=self.remove_region)
        self.check_button = ttk.Button(options, text="Check placement", width=0, command=lambda:self._submit_match(force=True))
        footer = self.footer = ttk.Frame(parent); footer.grid(row=7, sticky="ew", pady=(8, 0)); footer.columnconfigure(1, weight=1)
        ttk.Label(footer, text="Layout ID").grid(row=0, column=0, padx=(0, 8))
        ttk.Entry(footer, textvariable=self.layout_id, width=1).grid(row=0, column=1, sticky="ew")
        ttk.Button(footer, text="Save layout", width=0, command=self.save_layout).grid(row=0, column=2, padx=(4, 0))
        self.use_button = ttk.Button(footer, text="Use for next rollout", style="Accent.TButton", command=self.use_reference)
        self.use_button.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(6, 0))

    def _set_status(self, value):
        if self.status.get() != value:
            self.status.set(value)

    def _work(self, operation, callback, *, busy_text=None):
        if self._closed:
            return False
        if self._busy:
            self._pending_work = (operation, callback, busy_text)
            return False
        self._busy = True
        generation = self._generation
        token = self._next_work_token
        self._next_work_token += 1
        self._callbacks[token] = callback
        if busy_text: self._set_status(busy_text)
        threading.Thread(target=_background_operation, args=(self._queue, generation, token, operation),
                         name="scene-reference-worker", daemon=True).start()
        return True

    def _poll(self):
        if self._closed: return
        try:
            while True:
                generation, token, result, error = self._queue.get_nowait()
                callback = self._callbacks.pop(token, None)
                self._busy = False
                if self._pending_work is not None:
                    operation, next_callback, busy_text = self._pending_work
                    self._pending_work = None
                    self._work(operation, next_callback, busy_text=busy_text)
                    continue
                if generation != self._generation or callback is None: continue
                if error: self._set_status(error)
                else: callback(result)
        except queue.Empty: pass
        self._poll_job = self.app.root.after(80, self._poll)

    def choose_root(self):
        chosen = filedialog.askdirectory(parent=self.app.root, initialdir=self.results_root, title="Recorded results")
        if chosen:
            self.results_root = Path(chosen).resolve()
            self._clear_reference()
            self.scan()

    def scan(self):
        self.pane.expand("first", persist=False)
        root, cancel = self.results_root, self._cancel
        return self._work(lambda:index_rollouts(root, cancel=cancel), self._scanned, busy_text="Reading saved rollouts…")

    def _scanned(self, payload):
        self.records, self._scan_errors = payload
        tasks = sorted({row["task"] for row in self.records})
        methods = sorted({row["method"] for row in self.records})
        self.task_picker.configure(values=("All tasks", *tasks)); self.method_picker.configure(values=("All methods", *methods))
        if self.task_filter.get() not in {"All tasks", *tasks}: self.task_filter.set("All tasks")
        if self.method_filter.get() not in {"All methods", *methods}: self.method_filter.set("All methods")
        self._filter()
        self._set_status(f"{len(self.records)} recorded rollouts. Select either method's initial scene."
                         if self.records else "No recorded initial scenes found. Choose the folder containing saved results.")
        if self._scan_errors: self._set_status(self.status.get() + " " + self._scan_errors[0])

    def _filter(self):
        needle = self.search.get().strip().lower()
        old = self.table.selection()
        self._visible_records = {row["record_id"]:row for row in self.records
            if self.task_filter.get() in {"All tasks", row["task"]}
            and self.method_filter.get() in {"All methods", row["method"]}
            and (not needle or needle in " ".join(str(row[k]) for k in ("task", "method", "checkpoint", "rollout_id", "outcome")).lower())}
        self.table.delete(*self.table.get_children())
        for key, row in self._visible_records.items():
            date = time.strftime("%m-%d %H:%M", time.localtime(row["created_ns"]/1e9))
            self.table.insert("", "end", iid=key, values=(*tuple(row[k] for k in ("task", "method", "checkpoint", "rollout_id", "outcome")), date))
        if old and old[0] in self._visible_records: self.table.selection_set(old)

    def _select_reference(self, _event=None):
        selected = self.table.selection()
        if not selected or selected[0] not in self._visible_records: return
        record = self._visible_records[selected[0]]
        if self.reference and record["record_id"] == self.reference["record_id"]: return
        root = self.results_root
        self._work(lambda:load_reference(root, record), self._loaded, busy_text="Loading the recorded initial scene…")

    def _loaded(self, payload):
        self._clear_reference()
        self.reference, self.images = payload
        self._controls_geometry = None
        roles = tuple(self.images)
        self.role_picker.configure(values=roles)
        if self.role.get() not in roles: self.role.set(roles[0])
        self.layout_id.set("layout_" + self.reference["record_id"][:8])
        legacy = self.reference["source_kind"].startswith("legacy")
        self.summary.set(f"{self.reference['task']} · {self.reference['method']} · {self.reference['checkpoint']} · Run {self.reference['rollout_id']}\n"
                         + ("Legacy video frame · compressed image, approximate initial reference" if legacy
                            else "Exact first policy observation · captured after Home"))
        self._set_status("Match the saved scene visually, or draw regions around individual objects.")
        self.pane.collapse("first", persist=False)
        self._render()

    def _clear_reference(self):
        self._generation += 1
        self.reference, self.images, self.objects, self.matches = None, {}, [], []
        self._armed, self._match_stamp, self._render_key = False, None, None
        self._reference_preview = None
        self._match_measured_ns = None
        self.objects_table.delete(*self.objects_table.get_children()) if hasattr(self, "objects_table") else None
        if hasattr(self.app, "set_trial_context"): self.app.set_trial_context({"scene_match": None})

    def bind_session(self, session):
        self._generation += 1
        self._armed = False
        self.matches = []
        self.latest = None
        self._live_signature = None
        self._match_measured_ns = None
        if self.reference:
            self._set_status("Policy changed. Check the current scene and explicitly use the reference again.")

    def _view_changed(self):
        self._generation += 1
        self._controls_geometry = None
        self.app.preferences.set("scene_camera_role", self.role.get())
        self.app.preferences.set("scene_view", self.mode.get())
        self.app.preferences.flush()
        self.matches = []
        self._match_stamp = None
        self._match_measured_ns = None
        self._render_key = None
        self._render()

    def _blend_changed(self, _value=None):
        self.app.preferences.set("scene_blend_percent", round(self.opacity.get()*100))
        self._render()

    def clear_pairing(self):
        self._armed = False
        self.app.set_trial_context({"scene_match": None})
        self._set_status("Pairing cleared. The reference remains visible for comparison.")

    def _layout_controls(self):
        width = max(200, self.content.winfo_width()-16)
        key = (width, self.app.text_size)
        if key == self._controls_geometry: return
        self._controls_geometry = key
        button_width = max(button.winfo_reqwidth() for button in self._toolbar_buttons)+4
        columns = max(1, min(4, width//button_width))
        if columns == 3: columns = 2  # Two complete rows avoid an orphan action.
        for column in range(4): self.toolbar.columnconfigure(column, weight=1 if column<columns else 0, uniform="scene_tools" if column<columns else "")
        for index,button in enumerate(self._toolbar_buttons):
            button.grid(row=index//columns, column=index%columns, sticky="ew", padx=(0,4 if index%columns<columns-1 else 0), pady=(0,4))
        wide = width >= 800+max(0, self.app.text_size-14)*20
        surface = self.matching_surface
        # Treeview's stretch column rewrites its requested width after a
        # stacked layout. Do not let that historical request consume the image
        # column when the inspector becomes wide.
        surface.columnconfigure(0, weight=1, minsize=320 if wide else 0)
        surface.columnconfigure(1, weight=1 if wide else 0, minsize=340 if wide else 0)
        if wide: self.objects_table.column("match", width=170)
        self.summary_label.grid_configure(columnspan=2 if wide else 1)
        self.view_controls.grid_configure(row=1, column=0)
        self.canvas.grid_configure(row=2, column=0, rowspan=3 if wide else 1)
        self.robot_label.grid_configure(row=5 if wide else 3, column=0, columnspan=2 if wide else 1)
        for widget,row,narrow_row in ((self.editor,1,4), (self.object_list,2,5), (self.options,3,6), (self.footer,4,7)):
            widget.grid_configure(row=row if wide else narrow_row, column=1 if wide else 0, padx=(12,0) if wide else 0)
        self.object_list.grid_configure(sticky="nsew" if wide else "ew")
        form_width = max(280, self.footer.winfo_reqwidth(), self.object_list.winfo_reqwidth()) if wide else width
        image_width = max(240, width-form_width-12) if wide else width
        shape = self.images.get(self.role.get())
        aspect = shape.shape[0]/shape.shape[1] if shape is not None else .75
        self.canvas.configure(height=max(240, min(520, round(image_width*aspect))))
        narrow = self.tolerance_fields.winfo_reqwidth()+self.similarity_fields.winfo_reqwidth()+8 > form_width
        self.tolerance_fields.grid(row=0, column=0, columnspan=2 if narrow else 1, sticky="w")
        self.similarity_fields.grid(row=1 if narrow else 0, column=0 if narrow else 1, columnspan=2 if narrow else 1, sticky="w")
        action_row = 2 if narrow else 1
        stack_actions = self.remove_button.winfo_reqwidth()+self.check_button.winfo_reqwidth()+8 > form_width
        self.remove_button.grid(row=action_row, column=0, columnspan=2 if stack_actions else 1, sticky="ew", padx=(0,0 if stack_actions else 4), pady=(4,0))
        self.check_button.grid(row=action_row+1 if stack_actions else action_row, column=0 if stack_actions else 1, columnspan=2 if stack_actions else 1, sticky="ew", pady=(4,0))

    def _compatibility(self):
        if not self.reference or self.latest is None: return False, "Waiting for a saved scene and live preview"
        captured = getattr(self.latest, "captured_ns", None)
        if not isinstance(captured, int) or not 0 <= time.time_ns()-captured <= 3_000_000_000:
            return False, "Live image is stale; wait for current camera feedback"
        task = self.app.session.config.task
        if task != self.reference["task"]:
            return False, "Reference task differs from the selected policy task"
        role = self.role.get()
        if role not in self.latest.images or role not in self.images: return False, "Selected camera is unavailable"
        serials = {**getattr(self.app.session.embodiment, "metadata", {}).get("camera_serials", {}), **_serials(self.latest.camera_metadata)}
        identity_known = True
        for camera_role, image in self.images.items():
            if camera_role not in self.latest.images: return False, f"Reference camera {camera_role} is unavailable"
            if self.latest.images[camera_role].shape != image.shape: return False, f"Camera {camera_role} resolution changed; matching is disabled"
            expected, actual = self.reference.get("camera_serials", {}).get(camera_role), serials.get(camera_role)
            if expected and actual and str(expected) != str(actual): return False, f"Camera {camera_role} identity changed; select the matching camera role"
            if expected and not actual: return False, f"Live camera {camera_role} identity is unavailable"
            identity_known = identity_known and bool(expected)
        return True, "Camera identity not recorded; visually verify the fixed camera position" if not identity_known else "Camera identity and resolution match; physical mount position is not measured"

    def refresh(self, observation=None):
        if self._closed: return
        self.latest = observation if observation is not None else self.app.session.latest
        if self.latest is not None:
            signature = (self.app.session.config.task, tuple(sorted((role, image.shape, _serials(self.latest.camera_metadata).get(role)) for role,image in self.latest.images.items())))
            if self._live_signature is not None and signature != self._live_signature:
                self._generation += 1
                self.matches = []
                self._match_stamp = self._match_measured_ns = None
                self._armed = False
                self.app.set_trial_context({"scene_match": None})
                self._set_status("Task or camera configuration changed. Check placement and associate the reference again.")
            self._live_signature = signature
        if self.app.text_size != self._last_text_size:
            self._last_text_size = self.app.text_size
        width = max(200, self.winfo_width() - 28)
        for label in (self.status_label, self.summary_label, self.robot_label):
            if int(label.cget("wraplength")) != width: label.configure(wraplength=width)
        self._layout_controls()
        # Reserve the controls' actual requested heights before giving the
        # remaining area to the aspect-preserving image. Font scaling must not
        # squeeze the placement canvas to zero; the outer surface scrolls.
        row_height = self.app.fonts["body"].metrics("linespace")+max(10,round(12*self.app.text_size/14))
        first_min = max(round(125*self.app.text_size/14), self.pane.first.winfo_reqheight()-2*row_height)
        second_min = max(round(430*self.app.text_size/14), self.pane.second.winfo_reqheight())
        minimum = max(round(690*self.app.text_size/14),
                      (first_min if self.pane.snapshot()["first_visible"] else 0)+second_min+self.toolbar.winfo_reqheight()+self.status_label.winfo_reqheight()+48)
        if self.content.minimum_height != minimum:
            self.content.minimum_height = minimum
            self.pane.set_minimums(first=first_min, second=second_min)
            self.content.resize_viewport()
        valid, reason = self._compatibility()
        idle = self.app.session.state in {"READY", "SELECTING", "CAMERA_ERROR", "RECOVERY_REQUIRED"}
        state = "normal" if valid and idle else "disabled"
        if str(self.use_button.cget("state")) != state: self.use_button.configure(state=state)
        if self._armed and not valid:
            self._armed = False
            self.app.set_trial_context({"scene_match": None})
            self._set_status(reason + "; previous alignment association cleared")
        elif self.reference and not valid and reason != self._last_compatibility_note:
            self._set_status(reason)
        self._last_compatibility_note = reason
        self._robot_comparison()
        if not self.winfo_viewable(): return
        self._render()
        if valid and idle: self._submit_match()

    def _robot_comparison(self):
        state = np.asarray(getattr(self.latest, "state", []))
        reference = np.asarray(self.reference.get("initial_state", []) if self.reference else [])
        if not state.size:
            text = "Camera preview only · load a policy for robot pose feedback."
        elif state.shape == reference.shape and state.size:
            description = getattr(self.app.session.embodiment, "metadata", {}).get("state", "")
            reference_description = self.reference.get("state_description", "")
            if state.size == 7 and description == reference_description and description.startswith("TCP xyz meters + Euler xyz radians"):
                translation = float(np.linalg.norm(state[:3]-reference[:3]))*1000
                angle = np.abs((state[3:6]-reference[3:6]+np.pi)%(2*np.pi)-np.pi).max()*180/np.pi
                text = f"Robot Δposition {translation:.1f} mm · max Euler Δ {angle:.1f}° · gripper Δ {abs(state[6]-reference[6]):.3f}\nHome/reset → align objects → associate reference → Start. Start homes again; the actual first observation is recorded."
            else:
                text = f"Robot state max |Δ| {np.max(np.abs(state-reference)):.4g} · units follow the embodiment. Image alignment cannot verify depth."
        else:
            text = "Saved robot state is unavailable or uses a different dimension. Image alignment cannot verify depth."
        if self.robot_summary.get() != text: self.robot_summary.set(text)

    def _submit_match(self, force=False):
        if self._busy or self._closed or not self.objects or not self._compatibility()[0]: return False
        stamp = (self.role.get(), self.latest.captured_ns)
        if not force and (stamp == self._match_stamp or time.monotonic()-self._last_match < .4): return False
        role = self.role.get()
        objects = copy.deepcopy([item for item in self.objects if item["role"] == role])
        if not objects: return False
        try: tolerance, score = float(self.tolerance.get()), float(self.min_score.get())
        except ValueError:
            self._set_status("Enter numeric pixel tolerance and similarity threshold")
            return False
        reference, live = self.images[role].copy(), self.latest.images[role].copy()
        self._last_match, self._match_stamp = time.monotonic(), stamp
        captured_ns = self.latest.captured_ns
        return self._work(lambda:(match_objects(reference, live, objects, tolerance_px=tolerance, min_score=score), captured_ns, (tolerance, score)), self._matched)

    def _matched(self, payload):
        rows, self._match_measured_ns, self._match_parameters = payload
        self.matches = rows
        self._update_object_rows()
        self._render_key = None
        self._render()

    def _update_object_rows(self):
        rows = {item["id"]:item for item in self.matches}
        wanted = {item["id"] for item in self.objects}
        for item in self.objects_table.get_children():
            if item not in wanted: self.objects_table.delete(item)
        for region in self.objects:
            item = rows.get(region["id"], {**region, "status": "not checked · " + region["role"], "matched": False, "aligned": False})
            offset = item.get("offset_px")
            text = f"{offset[0]:+.1f}, {offset[1]:+.1f}" if offset and item["matched"] else "—"
            values = (item["label"], text, f"{item['score']:.2f}" if "score" in item else "—", item["status"])
            if self.objects_table.exists(item["id"]): self.objects_table.item(item["id"], values=values, tags=("aligned" if item["aligned"] else "unmatched",))
            else: self.objects_table.insert("", "end", iid=item["id"], values=values, tags=("aligned" if item["aligned"] else "unmatched",))

    def draw_region(self):
        if not self.reference:
            self._set_status("Select a saved rollout first")
            return
        self.mode.set("Reference")
        self._set_status("Drag a box around one object in the recorded image. Include its boundary, with little background.")
        self._render()

    def _image_point(self, x, y):
        if not self._draw_geometry: return None
        ox, oy, width, height, iw, ih = self._draw_geometry
        if not ox <= x <= ox+width or not oy <= y <= oy+height: return None
        return (max(0, min(iw-1, round((x-ox)*iw/width))), max(0, min(ih-1, round((y-oy)*ih/height))))

    def _drag_start(self, event):
        if self.mode.get() != "Reference" or not self.reference: return
        point = self._image_point(event.x, event.y)
        if point:
            self._drag = (point, event.x, event.y)
            self.canvas.delete("selection")

    def _drag_move(self, event):
        if not self._drag: return
        self.canvas.delete("selection")
        self.canvas.create_rectangle(self._drag[1], self._drag[2], event.x, event.y, outline="#E8CC93", width=2, tags="selection")

    def _drag_end(self, event):
        if not self._drag: return
        start, end = self._drag[0], self._image_point(event.x, event.y)
        self._drag = None; self.canvas.delete("selection")
        if end is None: return
        x, y = min(start[0], end[0]), min(start[1], end[1])
        box = [x, y, abs(start[0]-end[0]), abs(start[1]-end[1])]
        item = {"id": f"object_{time.time_ns()}", "label": self.object_label.get(), "role": self.role.get(), "box": box}
        try: self.objects = validate_objects([*self.objects, item], self.images)
        except ValueError as error:
            self._set_status(str(error)); return
        self.object_label.set(f"Object {len(self.objects)+1}")
        self._generation += 1
        self.matches = []
        self._match_stamp = self._match_measured_ns = None
        self._update_object_rows()
        self._armed = False
        self.app.set_trial_context({"scene_match": None})
        self.mode.set("Overlay")
        self._submit_match(force=True)
        self._render_key = None; self._render()
        self._set_status("Desired: green. Detected: amber. Draw another region in Reference. Similarity is not a probability.")

    def remove_region(self):
        selected = self.objects_table.selection()
        if selected:
            self._generation += 1
            self.objects = [item for item in self.objects if item["id"] != selected[0]]
            self.matches = [item for item in self.matches if item["id"] != selected[0]]
            self.objects_table.delete(selected[0])
            self._armed = False
            self.app.set_trial_context({"scene_match": None})
            self._render_key = None; self._render()

    def _render(self):
        if self._closed or not hasattr(self, "canvas"): return
        role, mode = self.role.get(), self.mode.get()
        reference = self.images.get(role)
        live = getattr(self.latest, "images", {}).get(role)
        if reference is None:
            self.canvas.delete("all")
            self.canvas.create_text(max(1,self.canvas.winfo_width())/2, max(1,self.canvas.winfo_height())/2,
                                    text="Select a recorded initial scene", fill="#B7C0CC", font=self.app.fonts["body"], width=max(150,self.canvas.winfo_width()-24))
            return
        compatible = live is not None and live.shape == reference.shape
        stamp = getattr(self.latest, "captured_ns", None) if mode != "Reference" and compatible else None
        key = (role, mode, stamp, round(self.opacity.get(), 2), self.canvas.winfo_width(), self.canvas.winfo_height(),
               tuple((m["id"], m.get("live_box", []).__repr__(), m.get("aligned")) for m in self.matches), len(self.objects))
        if key == self._render_key: return
        self._render_key = key
        side_by_side = mode == "Side by side" and compatible
        iw, ih = reference.shape[1]*(2 if side_by_side else 1), reference.shape[0]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        factor = min(cw/iw, ch/ih)
        width, height = max(1, round(iw*factor)), max(1, round(ih*factor))
        ox, oy = (cw-width)/2, (ch-height)/2
        # Blend only at display resolution. Reblending entire 4K observations
        # on the Tk thread can stall operator controls and has no visual value.
        reference_width = max(1, round(width/2)) if side_by_side else width
        preview_key = (role, reference_width, height)
        if self._reference_preview is None or self._reference_preview[0] != preview_key:
            self._reference_preview = (preview_key, Image.fromarray(reference).resize((reference_width,height), Image.Resampling.BILINEAR))
        rendered = self._reference_preview[1]
        if compatible and mode != "Reference":
            live_width = max(1, width-reference_width) if side_by_side else width
            live_preview = Image.fromarray(live).resize((live_width,height), Image.Resampling.BILINEAR)
            if mode == "Live": rendered = live_preview
            elif mode == "Overlay": rendered = Image.blend(rendered, live_preview, self.opacity.get())
            elif side_by_side:
                combined = Image.new("RGB", (width,height))
                combined.paste(rendered, (0,0)); combined.paste(live_preview, (reference_width,0))
                rendered = combined
        photo = ImageTk.PhotoImage(rendered, master=self.app.root)
        self._photos["image"] = photo
        self.canvas.delete("all")
        self.canvas.create_image(ox, oy, anchor="nw", image=photo)
        self._draw_geometry = (ox, oy, width, height, iw, ih)
        for item in self.objects:
            if item["role"] != role: continue
            x, y, w, h = item["box"]
            self.canvas.create_rectangle(ox+x*factor, oy+y*factor, ox+(x+w)*factor, oy+(y+h)*factor,
                                         outline="#A7D7C8", width=2)
            self.canvas.create_text(ox+x*factor+3, oy+y*factor+3, anchor="nw", text=item["label"], fill="#A7D7C8", font=self.app.fonts["caption"])
        if mode != "Reference":
            for item in self.matches:
                if not item.get("matched") or item["role"] != role: continue
                x, y, w, h = item["live_box"]
                if mode == "Side by side": x += reference.shape[1]
                self.canvas.create_rectangle(ox+x*factor, oy+y*factor, ox+(x+w)*factor, oy+(y+h)*factor,
                                             outline="#A7D7C8" if item["aligned"] else "#E8CC93", width=2, dash=(5, 3))

    def reference_context(self):
        if not self.reference: return None
        layout = self.layout_id.get().strip()
        if not layout or len(layout) > 120: raise ValueError("Enter a layout ID of 1–120 characters")
        valid, reason = self._compatibility()
        matches = copy.deepcopy(self.matches)
        checked_ids = {item["id"] for item in matches}
        complete = checked_ids == {item["id"] for item in self.objects}
        status = "visual_only" if not self.objects else "aligned_in_image" if valid and matches and complete and all(item["aligned"] for item in matches) else "needs_review"
        if self.objects and (self._match_measured_ns is None or time.time_ns()-self._match_measured_ns > 3_000_000_000):
            status = "not_recently_checked"
        try: parameters = (float(self.tolerance.get()), float(self.min_score.get()))
        except ValueError: raise ValueError("Enter numeric pixel tolerance and similarity threshold") from None
        if not math.isfinite(parameters[0]) or parameters[0] <= 0 or not 0 < parameters[1] <= 1:
            raise ValueError("Use positive pixel tolerance and similarity in (0, 1]")
        if self.objects and self._match_parameters != parameters:
            status = "not_recently_checked"
        return {"schema_version": 1, "layout_id": layout, "results_root": str(self.results_root),
                "reference": copy.deepcopy({key:value for key,value in self.reference.items() if key != "reference_fingerprint"}),
                "reference_fingerprint": self.reference["reference_fingerprint"], "objects": copy.deepcopy(self.objects),
                "display": {"role": self.role.get(), "mode": self.mode.get(), "blend": self.opacity.get()},
                "alignment": {"role": self.role.get(), "measured_ns": self._match_measured_ns, "status": status,
                              "matches": matches, "camera_compatible": valid, "camera_note": reason,
                              "operator_confirmed": self._armed, "pixel_tolerance": self.tolerance.get(),
                              "similarity_threshold": self.min_score.get(),
                              "unchecked_object_ids": sorted({item["id"] for item in self.objects}-checked_ids),
                              "confirmation_scope": "Operator associated this reference; physical pose equality is not certified",
                              "limitation": "Image-plane assistance; depth and identical 3-D poses are not verified"}}

    def use_reference(self):
        valid, reason = self._compatibility()
        if not valid:
            self._set_status(reason); return False
        if self.app.session.state not in {"READY", "SELECTING"}:
            self._set_status("Associate a reference while idle"); return False
        try:
            self._armed = True
            context = self.reference_context()
            accepted = self.app.set_trial_context({"scene_match": context})
            if accepted is False: raise RuntimeError("The current session cannot accept a reference yet")
        except (ValueError, RuntimeError) as error:
            self._armed = False; self._set_status(str(error)); return False
        self._set_status(f"Layout {context['layout_id']} is associated. Start homes again; its actual first observation will be saved. The pairing does not certify equal 3-D placement.")
        return True

    def before_start(self):
        if not self._armed: return True
        valid, reason = self._compatibility()
        if not valid:
            self._set_status(reason); return False
        self.app.set_trial_context({"scene_match": self.reference_context()})
        return True

    def save_layout(self):
        try:
            context = self.reference_context()
            if context is None: raise ValueError("Select a saved rollout first")
            directory = self.results_root / ".layouts"
            directory.mkdir(exist_ok=True)
            filename = "".join(char if char.isalnum() or char in "_-" else "_" for char in context["layout_id"])+".json"
            path = filedialog.asksaveasfilename(parent=self.app.root, defaultextension=".json", initialdir=directory, initialfile=filename, title="Save placement layout")
            if path:
                write_json(Path(path), {"schema_version": 1, "kind": "recorded_scene_layout", **context})
                self._set_status("Layout saved; source recordings were not changed")
        except (OSError, ValueError) as error: self._set_status(str(error))

    def load_layout(self):
        path = filedialog.askopenfilename(parent=self.app.root, initialdir=self.results_root / ".layouts" if (self.results_root / ".layouts").is_dir() else self.results_root, filetypes=(("Scene layout", "*.json"),), title="Load placement layout")
        if not path: return
        root = self.results_root
        def operation():
            document = _json(path)
            if document.get("kind") != "recorded_scene_layout" or document.get("schema_version") != 1:
                raise ValueError("Choose a recorded scene layout")
            # Resolve against the current chosen root so a copied layout remains
            # portable and cannot silently redirect the browser elsewhere.
            reference, images = load_reference(root, document["reference"])
            if document.get("reference_fingerprint") != reference["reference_fingerprint"]:
                raise ValueError("The layout's initial images or camera identity changed")
            objects = validate_objects(document.get("objects", []), images)
            display = document.get("display", {})
            role = display.get("role", document.get("alignment", {}).get("role", next(iter(images))))
            mode, blend = display.get("mode", "Overlay"), float(display.get("blend", .5))
            tolerance = float(document.get("alignment", {}).get("pixel_tolerance", 8))
            score = float(document.get("alignment", {}).get("similarity_threshold", .65))
            if role not in images or mode not in {"Overlay", "Side by side", "Reference", "Live"} or not 0 <= blend <= 1:
                raise ValueError("Unsupported saved camera or display setting")
            if not math.isfinite(tolerance) or tolerance <= 0 or not 0 < score <= 1:
                raise ValueError("Unsupported saved matching parameters")
            return document, reference, images, objects, (role, mode, blend, tolerance, score)
        def loaded(payload):
            document, reference, images, objects, settings = payload
            self._loaded((reference, images)); self.objects = objects
            self.layout_id.set(document["layout_id"])
            role, mode, blend, tolerance, score = settings
            self.role.set(role); self.mode.set(mode); self.opacity.set(blend)
            self.tolerance.set(f"{tolerance:g}"); self.min_score.set(f"{score:g}")
            self._update_object_rows()
            self._submit_match(force=True); self._render_key = None; self._render()
            self._set_status("Layout loaded. Compare the live scene, then use it for the next rollout.")
        self._work(operation, loaded, busy_text="Reading saved placement layout…")

    def prepare_close(self):
        if self._closed: return
        self._closed = True
        self._cancel.set()
        self._pending_work = None
        self._callbacks.clear()
        self._generation += 1
        self.app.preferences.flush()
        for name in ("_poll_job", "_scan_job"):
            timer = getattr(self, name, None)
            if timer is not None:
                self.app.root.after_cancel(timer)
                setattr(self, name, None)
        self._photos.clear()

    def close(self):
        self.prepare_close()
