"""Desktop GUI; the session worker owns model and robot operations."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
from PIL import Image, ImageTk
from .reference import ReferenceCatalog
from .records import summary, write_json
from .session import DeploymentSession
from .workspace import WorkspacePreferences, ResizablePane, clamp_window_geometry, geometry_string


# Semantic working palette, following Gnaroshi MDS ui-ux guidance. Device and
# policy-specific identity belongs to the owning project's configuration.
THEME = {
    "canvas": "#11151B", "panel": "#181E26", "inset": "#222B38",
    "border": "#354253", "text": "#F4F7FA", "muted": "#B7C0CC",
    "accent": "#A7D7C8", "danger": "#E7A0A6", "warning": "#E8CC93",
    "disabled": "#768391", "selection": "#304C55", "ink": "#11151B",
}
SPACE = {"small": 4, "button_y": 6, "control": 8, "group": 12, "section": 16}


class DeploymentGUI:
    def __init__(self, root, session, *, reload_fields=None, reload_factory=None,
                 selection_resolver=None, config_loader=None, config_saver=None, config_directory=None,
                 preferences_path=None):
        self.root, self.session = root, session
        self.reload_fields = tuple(reload_fields or ())
        self.reload_factory = reload_factory
        self.selection_resolver = selection_resolver
        self.config_loader, self.config_saver = config_loader, config_saver
        self.config_directory = Path(config_directory).expanduser() if config_directory is not None else None
        preference_file = preferences_path or session.config.metadata.get("ui_preferences_path")
        if preference_file is None:
            preference_file = (self.config_directory.parent/"ui_preferences.json" if self.config_directory is not None
                               else Path(session.config.output_root)/".ui_preferences.json")
        self.preferences = WorkspacePreferences(Path(preference_file))
        self.text_size = max(12, min(24, int(self.preferences.get("text_size", 14))))
        self.presentation_status = tk.StringVar(value="")
        self.reference_enabled = tk.BooleanVar(value=bool(self.preferences.get("reference_visible", False)))
        self._workspace_panes = []
        self._layout_job = None
        self._preferences_job = None
        self._preferences_changed = False
        self._selection_cells = []
        self._selection_choice_hints = []
        self._scroll_regions = []
        self._last_displayed_state = None
        self._measurement_layout_seen = None
        self._compact_chrome = False
        self._short_context_open = False
        self._advanced_fields_open = False
        self._selection_context = copy.deepcopy(session.config.metadata.get("reload_selection", {}))
        self._load_error_session = None
        if bool(self.reload_fields) != bool(reload_factory):
            raise ValueError("Checkpoint selection requires both reload_fields and reload_factory")
        keys = [field["key"] for field in self.reload_fields]
        if len(keys) != len(set(keys)) or any(not key for key in keys):
            raise ValueError("Reload field keys must be nonempty and distinct")
        if any(field.get("kind", "text") not in {"directory", "file", "text", "choice"} for field in self.reload_fields):
            raise ValueError("Unknown reload field kind")
        self.reload_vars, self.reload_widgets = {}, []
        self._reload_events = queue.Queue()
        self._reload_cancel = threading.Event()
        self._reload_thread = None
        self._reload_in_progress = False
        self._reload_blocked = False
        self._reload_cleanup_session = None
        self.photos, self.reference_images = [], []
        self.notes_job = None
        self.closing = False
        self._shortcut_down = set()
        self._shortcut_release_jobs = {}
        self._failure_notes_seen = None
        self._event_detail = ""
        self._load_error_detail = None
        self._last_save_error = None
        self._annotation_job = None
        self._annotation_drafts = {}
        self._annotation_original = ""
        self._utility_events = queue.Queue()
        self._help_window = None
        self._camera_draft_dirty = False
        self._camera_actual_seen = self._camera_mapping()
        self.camera_title_vars = {}
        root.title(f"{session.config.method} | {session.config.task} deployment")
        restored_window = clamp_window_geometry(self.preferences.get("window", {"width":1440,"height":960}),
                                               root.winfo_screenwidth(), root.winfo_screenheight(),
                                               min_width=960, min_height=640)
        root.geometry(geometry_string(restored_window))
        root.minsize(min(960, root.winfo_screenwidth()), min(640, root.winfo_screenheight()))
        self._configure_theme()
        self.state = tk.StringVar(value="CONNECTING")
        self.status = tk.StringVar(value="Preparing deployment")
        self.metrics = tk.StringVar(value="No rollout measurements")
        self.counts = tk.StringVar(value="No saved rollouts")
        self.failure_note = tk.StringVar(value="Select a saved rollout to view its result.")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(3, weight=1)
        heading = self.heading = ttk.Frame(root, padding=(SPACE["section"], SPACE["control"]))
        heading.grid(row=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        self.header_tools = ttk.Frame(heading)
        self.heading_label = ttk.Label(heading, text=self._display_task(), style="Heading.TLabel")
        self.heading_label.grid(row=0, column=0, sticky="w")
        self.active_method = "No policy loaded" if session.config.metadata.get("defer_policy_load") else self._display_method()
        self.method_label = ttk.Label(heading,text=self.active_method,style="Caption.TLabel")
        self.state_label = ttk.Label(heading, textvariable=self.state, style="State.TLabel", padding=(10, 4))
        self.state_label.grid(row=0, column=1, padx=SPACE["group"])
        if self.reload_fields:
            self.checkpoint_toggle = ttk.Button(heading, text="Policy", width=0, command=self.toggle_checkpoint_panel)
            self.checkpoint_toggle.grid(row=0, column=2, padx=(0, SPACE["control"]))
        self.help_button = ttk.Button(heading, text="Help", width=0, command=self.show_help)
        self.help_button.grid(row=0, column=3 if self.reload_fields else 2, padx=(0, SPACE["control"]))
        self.context_button = ttk.Button(heading,text="Context",width=0,command=self.toggle_context)
        self.zoom_controls = ttk.Frame(heading)
        self.zoom_controls.grid(row=0, column=4 if self.reload_fields else 3, padx=(0, SPACE["control"]))
        self.smaller_text_button = ttk.Button(self.zoom_controls, text="A-", width=0, command=lambda:self.change_text_size(-1))
        self.smaller_text_button.pack(side="left")
        self.text_size_label = ttk.Label(self.zoom_controls, text=f"{self.text_size} pt", style="Caption.TLabel", anchor="center")
        self.text_size_label.pack(side="left", padx=SPACE["small"])
        self.larger_text_button = ttk.Button(self.zoom_controls, text="A+", width=0, command=lambda:self.change_text_size(1))
        self.larger_text_button.pack(side="left")
        self.exit_button = ttk.Button(heading, text="Exit", width=0, command=self.close)
        self.exit_button.grid(row=0, column=5 if self.reload_fields else 4)
        self.instruction_label = ttk.Label(heading, text=f"{self._display_task()} · {session.config.instruction}", wraplength=1000, style="Muted.TLabel")
        self.instruction_label.grid(row=1, column=0, columnspan=6 if self.reload_fields else 5, sticky="w", pady=(SPACE["small"], 0))
        self.active_identity = tk.StringVar(value="Waiting for the selected policy")
        self.identity_label = ttk.Label(heading, textvariable=self.active_identity, style="Caption.TLabel", wraplength=1100)
        self.identity_label.grid(row=2, column=0, columnspan=6 if self.reload_fields else 5, sticky="w", pady=(SPACE["small"], 0))
        self.inspector_toggle = ttk.Button(heading, text="Inspector", width=0, command=self.toggle_inspector)
        self.toolbar = ttk.Frame(root, padding=(SPACE["section"], SPACE["small"]))
        self.toolbar.grid(row=1, sticky="ew")
        self.toolbar.columnconfigure(0, weight=1)
        controls = self.controls = ttk.Frame(self.toolbar)
        controls.grid(row=0, column=0, sticky="w")
        self.action_groups = tuple(ttk.Frame(controls) for _ in range(3))
        self.action_rows = tuple(ttk.Frame(controls) for _ in range(3))
        self._scoreboard(self.toolbar)
        body = ttk.Frame(root, padding=(SPACE["section"], 0, SPACE["section"], SPACE["control"]))
        body.grid(row=3, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        self.workspace_canvas = tk.Canvas(body, background=THEME["canvas"], highlightthickness=0, width=1, height=1)
        self.workspace_canvas.grid(row=0,column=0,sticky="nsew")
        self.workspace_scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.workspace_canvas.yview)
        self.workspace_scrollbar.grid(row=0,column=1,sticky="ns")
        self.workspace_canvas.configure(yscrollcommand=self.workspace_scrollbar.set)
        self.workspace_content = ttk.Frame(self.workspace_canvas)
        self.workspace_content.columnconfigure(0,weight=1)
        self.workspace_content.rowconfigure(0,weight=1)
        self._workspace_window = self.workspace_canvas.create_window(0,0,anchor="nw",window=self.workspace_content)
        self.workspace_canvas.bind("<Configure>",self._workspace_resized)
        root.bind("<FocusIn>",lambda event:self._reveal_focus(event,self.workspace_canvas,self.workspace_content),add="+")
        self.main_pane = self._pane(self.workspace_content, key="main", orient="horizontal", first_label="Cameras", second_label="Results",
                                    first_min=360, second_min=420, default_fraction=.62, collapsible=(False, True), show_header=False)
        self.main_pane.grid(row=0, column=0, sticky="nsew")
        main = self.main_pane.first
        main.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        self.visual_pane = self._pane(main, key="live_reference", orient="vertical", first_label="Live cameras", second_label="Start reference",
                                      first_min=180, second_min=210, default_fraction=.72, show_header=False,
                                      collapsible=(False, True))
        self.visual_pane.grid(row=1, column=0, sticky="nsew")
        for container in (self.visual_pane.first, self.visual_pane.second):
            container.columnconfigure(0, weight=1)
            container.rowconfigure(0, weight=1)
        self.camera_labels = self._image_pair(self.visual_pane.first, "Live observations", row=0)
        self._live_size = None
        self.visual_pane.first.bind("<Configure>",self._live_resized,add="+")
        self._references(self.visual_pane.second)
        side = self.main_pane.second
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=0)
        self.inspector_content = side
        self.measurement_panel = ttk.Frame(side, padding=(SPACE["control"], SPACE["small"]))
        self.measurement_panel.grid(row=0, column=0, sticky="ew")
        self._inspector_width = None
        side.bind("<Configure>", self._inspector_resized, add="+")
        run, outcome, robot = self.action_groups
        self.start_button = ttk.Button(run, text="Start [N]", width=0, style="Accent.TButton", command=self.start, state="disabled")
        self.stop_button = ttk.Button(run, text="Stop [X]", width=0, style="Stop.TButton", command=lambda:self.signal("stop"), state="disabled")
        self.success_button = ttk.Button(outcome, text="Success [S]", width=0, command=lambda:self.signal("success"), state="disabled")
        self.failure_button = ttk.Button(outcome, text="Failure [F]", width=0, command=lambda:self.signal("failure"), state="disabled")
        self.retry_button = ttk.Button(outcome, text="Retry [R]", width=0, command=lambda:self.signal("retry"), state="disabled")
        self.home_button = ttk.Button(robot, text="Home [H]", width=0, command=self.home_robot, state="disabled")
        self.recover_button = ttk.Button(robot, text="Recover robot", width=0, command=self.recover_robot, state="disabled")
        self.run_buttons = (self.start_button, self.success_button, self.failure_button,
                            self.stop_button, self.retry_button, self.home_button, self.recover_button)
        for group, buttons in ((run,(self.start_button,self.stop_button)),
                               (outcome,(self.success_button,self.failure_button,self.retry_button)),
                               (robot,(self.home_button,self.recover_button))):
            for number, button in enumerate(buttons):
                button.pack(side="left", padx=(0,SPACE["small"]) if number<len(buttons)-1 else 0)
        self.control_hint = ttk.Label(self.measurement_panel, text="Start: home, then a fresh rollout.", style="Caption.TLabel", wraplength=420)
        self.control_hint.pack(anchor="w", fill="x", padx=SPACE["control"], pady=(SPACE["small"], 0))
        self.measurement_label = ttk.Label(self.measurement_panel, textvariable=self.metrics, justify="left", wraplength=420)
        self.measurement_label.pack(anchor="w", fill="x", padx=SPACE["control"], pady=SPACE["small"])
        self.inspector_content.columnconfigure(0, weight=1)
        self.inspector_content.rowconfigure(2, weight=1)
        self.view_selector = ttk.Frame(self.inspector_content, padding=(SPACE["control"], SPACE["small"]))
        self.view_selector.columnconfigure(1, weight=1)
        ttk.Label(self.view_selector, text="View").grid(row=0, column=0, padx=(0,SPACE["control"]))
        self.selected_view = tk.StringVar(value="Rollouts")
        self.view_picker = ttk.Combobox(self.view_selector, textvariable=self.selected_view,
                                       values=(("Policy",) if self.reload_fields else ())+("Rollouts","Notes","Devices","Settings","Details"), state="readonly", width=12)
        self.view_picker.grid(row=0, column=1, sticky="ew")
        tabs = self.tabs = ttk.Notebook(self.inspector_content)
        tabs.grid(row=2, sticky="nsew")
        self.view_picker.bind("<<ComboboxSelected>>", lambda _event:tabs.select(self.view_picker.current()))
        tabs.bind("<<NotebookTabChanged>>", self._view_changed)
        frames, tab_frames = {}, {}
        for title in (("Policy",) if self.reload_fields else ())+("Rollouts", "Notes", "Devices", "Settings", "Details"):
            if title in {"Rollouts", "Devices", "Settings"}:
                outer = ttk.Frame(tabs)
                inner = self._scrollable_content(outer)
            else:
                outer = inner = ttk.Frame(tabs, padding=SPACE["control"])
            tabs.add(outer, text=title)
            frames[title], tab_frames[title] = inner, outer
        results, notes, devices, settings, details = (frames[title] for title in ("Rollouts", "Notes", "Devices", "Settings", "Details"))
        self.devices_tab = tab_frames["Devices"]
        self.details_tab = tab_frames["Details"]
        self.rollouts_tab = tab_frames["Rollouts"]
        self.policy_tab = tab_frames.get("Policy")
        if self.reload_fields:
            frames["Policy"].columnconfigure(0, weight=1)
            frames["Policy"].rowconfigure(0, weight=1)
            self._checkpoint_panel(frames["Policy"])
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1, minsize=120)
        self.table = ttk.Treeview(results, columns=("run", "result", "steps", "hz", "reason"), show="headings", height=5, selectmode="browse")
        for column, title, width in (("run", "Run", 40), ("result", "Outcome", 70), ("steps", "Steps", 48), ("hz", "Hz", 48), ("reason", "Reason", 160)):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, minwidth=width if column != "reason" else 80, stretch=column == "reason", anchor="w" if column == "reason" else "center")
        self.table.tag_configure("failure", foreground=THEME["danger"])
        self.table.tag_configure("success", foreground=THEME["accent"])
        self.table.grid(row=0, column=0, sticky="nsew")
        self._size_result_columns()
        self.table.bind("<<TreeviewSelect>>", self.show_result_note)
        self.failure_note_label = ttk.Label(results, textvariable=self.failure_note, wraplength=420, style="Caption.TLabel")
        self.failure_note_label.grid(row=1, column=0, sticky="ew", pady=(SPACE["control"], 0))
        self.result_summary = tk.StringVar(value="")
        self.result_summary_label = ttk.Label(results, textvariable=self.result_summary, wraplength=420, style="Caption.TLabel")
        self.result_summary_label.grid(row=2, sticky="ew", pady=(SPACE["small"], 0))
        result_actions = ttk.Frame(results)
        result_actions.grid(row=3, sticky="ew", pady=(SPACE["control"], 0))
        self.open_result_button = ttk.Button(result_actions, text="Open folder", command=self.open_result_folder)
        self.open_result_button.pack(side="left")
        self.copy_result_button = ttk.Button(result_actions, text="Copy path", command=self.copy_result_path)
        self.copy_result_button.pack(side="left", padx=SPACE["small"])
        video_actions = ttk.Frame(results)
        video_actions.grid(row=4, sticky="ew", pady=(SPACE["small"], 0))
        self.video_role = tk.StringVar(value=session.config.camera_roles[0])
        self.video_role_selector = ttk.Combobox(video_actions, textvariable=self.video_role, values=session.config.camera_roles, state="readonly", width=10)
        self.video_role_selector.pack(side="left")
        self.open_video_button = ttk.Button(video_actions, text="Open video", command=self.open_result_video)
        self.open_video_button.pack(side="left", padx=SPACE["small"])
        annotation = ttk.Frame(results)
        annotation.grid(row=5, sticky="ew", pady=(SPACE["control"], 0))
        annotation.columnconfigure(0, weight=1)
        ttk.Label(annotation, text="Selected rollout annotation", style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        self.result_annotation = tk.StringVar(value="")
        self.annotation_entry = ttk.Entry(annotation, textvariable=self.result_annotation)
        self.annotation_entry.grid(row=1, column=0, sticky="ew", pady=(SPACE["small"], 0))
        self.annotation_entry.bind("<KeyRelease>", self.schedule_annotation)
        self.annotation_button = ttk.Button(annotation, text="Save", command=self.save_result_annotation)
        self.annotation_button.grid(row=1, column=1, padx=(SPACE["small"], 0), pady=(SPACE["small"], 0))
        self._annotation_result_id = None
        self._devices_panel(devices)
        notes.columnconfigure(0, weight=1)
        notes.rowconfigure(0, weight=1)
        self.notes_pane = self._pane(notes, key="notes", orient="vertical", first_label="Experiment notes",
                                     second_label="Automatic fault notes", first_min=90, second_min=70,
                                     default_fraction=.65, collapsible=(False,True),show_header=False)
        self.notes_pane.grid(row=0, column=0, sticky="nsew")
        self.notes = self._text_area(self.notes_pane.first, height=4,title="Experiment notes")
        self.notes.bind("<KeyRelease>", self.schedule_notes)
        self.failure_notes = self._text_area(self.notes_pane.second, height=4,title="Automatic fault notes")
        self.failure_notes.configure(state="disabled")
        self.hz = tk.StringVar(value=str(session.config.target_hz))
        self.steps = tk.StringVar(value=str(session.config.max_steps))
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="Change while ready", style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, SPACE["group"]))
        ttk.Label(settings, text="Target Hz").grid(row=1, column=0, sticky="w")
        ttk.Label(settings, text="Max steps").grid(row=2, column=0, sticky="w", pady=SPACE["control"])
        self.hz_entry = ttk.Entry(settings, textvariable=self.hz, width=12)
        self.hz_entry.grid(row=1, column=1, sticky="ew", padx=(SPACE["control"], 0))
        self.steps_entry = ttk.Entry(settings, textvariable=self.steps, width=12)
        self.steps_entry.grid(row=2, column=1, sticky="ew", padx=(SPACE["control"], 0))
        self.apply_button = ttk.Button(settings, text="Apply settings", command=self.apply_settings, state="disabled")
        self.apply_button.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(SPACE["control"], 0))
        self.record_video = tk.BooleanVar(value=session.config.record_video)
        self.record_video_toggle = ttk.Checkbutton(settings, text="Record rollout videos", variable=self.record_video)
        self.record_video_toggle.grid(row=3, column=0, columnspan=2, sticky="w", pady=(SPACE["control"], 0))
        ttk.Separator(settings).grid(row=7, column=0, columnspan=2, sticky="ew", pady=SPACE["section"])
        ttk.Label(settings, text="Text size").grid(row=8, column=0, sticky="w")
        self.text_size_choice = tk.StringVar(value=str(self.text_size))
        self.text_size_picker = ttk.Combobox(settings, textvariable=self.text_size_choice,
                                             values=tuple(str(value) for value in range(12,25)), state="readonly", width=12)
        self.text_size_picker.grid(row=8, column=1, sticky="ew", padx=(SPACE["control"], 0))
        self.text_size_picker.bind("<<ComboboxSelected>>", lambda _event:self.set_text_size(self.text_size_choice.get()))
        ttk.Label(settings, text="Ctrl + / − to resize text · Ctrl 0 to reset", style="Caption.TLabel", wraplength=420).grid(
            row=9, column=0, columnspan=2, sticky="w", pady=SPACE["control"])
        self.reset_layout_button = ttk.Button(settings, text="Reset panel sizes", command=self.reset_layout)
        self.reset_layout_button.grid(row=10, column=0, columnspan=2, sticky="ew")
        self.reference_toggle = ttk.Checkbutton(settings, text="Show training reference", variable=self.reference_enabled,
                                                command=self.toggle_reference)
        self.reference_toggle.grid(row=11,column=0,columnspan=2,sticky="w",pady=(SPACE["control"],0))
        self.reference_toggle.configure(state="normal" if self.catalog else "disabled")
        self.presentation_label = ttk.Label(settings,textvariable=self.presentation_status,style="Caption.TLabel",wraplength=420,
                                            foreground=THEME["warning"])
        self.presentation_label.grid(row=12,column=0,columnspan=2,sticky="ew",pady=(SPACE["small"],0))
        self.presentation_label.grid_remove()
        self.health = tk.StringVar(value="Waiting for device and storage status")
        ttk.Label(settings, text="Storage / recording / cameras", style="Muted.TLabel").grid(row=5, column=0, columnspan=2, sticky="w", pady=(SPACE["section"], SPACE["small"]))
        ttk.Label(settings, textvariable=self.health, wraplength=340, justify="left").grid(row=6, column=0, columnspan=2, sticky="ew")
        details.columnconfigure(0, weight=1)
        details.rowconfigure(0, weight=1)
        details_text = ttk.Frame(details)
        details_text.grid(row=0, column=0, sticky="nsew")
        self.details = self._text_area(details_text, height=6)
        # Paths/provenance stay in the explicit Details tab, rather than the
        # primary operating surface. Keep this label for project integrations.
        self.folder_label = ttk.Label(details, text=str(session.folder), style="Muted.TLabel", wraplength=365)
        self.folder_label.grid(row=1, column=0, sticky="ew", pady=(SPACE["control"], 0))
        records = ttk.Frame(results)
        records.grid(row=6,sticky="ew",pady=(SPACE["control"],0))
        self.delete_button = ttk.Button(records, text="Delete last [D]", command=self.delete, state="disabled")
        self.delete_button.pack(side="right")
        ttk.Button(notes,text="Save notes [W]",command=self.save).grid(row=1,column=0,sticky="e",pady=(SPACE["control"],0))
        self.status_label = ttk.Label(root, textvariable=self.status, wraplength=1380, padding=(SPACE["section"], SPACE["small"]), style="Caption.TLabel")
        self.status_label.grid(row=4, sticky="ew")
        root.bind("<KeyPress>", self.keypress)
        root.bind("<KeyRelease>", self.keyrelease)
        root.bind("<FocusOut>", self._clear_shortcuts)
        root.bind("<Configure>", self._resize_copy)
        for binding, delta in (("<Control-plus>",1), ("<Control-equal>",1), ("<Control-minus>",-1)):
            root.bind(binding, lambda _event, delta=delta:(self.change_text_size(delta), "break")[-1])
        root.bind("<Control-Key-0>", lambda _event:(self.set_text_size(14), "break")[-1])
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._rows_seen = None
        self._details_seen = None
        self._layout_workspace()
        session.start()
        root.after(80, self.refresh)

    def _display_task(self):
        explicit = self.session.config.metadata.get("display_task")
        return str(explicit) if explicit else str(self.session.config.task).replace("_", " ").title()

    def _text_area(self, parent, *, height, title=None):
        parent.columnconfigure(0, weight=1)
        row = 1 if title else 0
        parent.rowconfigure(row, weight=1)
        if title:
            ttk.Label(parent,text=title,style="Caption.TLabel").grid(row=0,column=0,columnspan=2,sticky="w",pady=(0,SPACE["small"]))
        text = tk.Text(parent, wrap="word", width=24, height=height, **self._text_style())
        text.grid(row=row, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=text.yview)
        scrollbar.grid(row=row, column=1, sticky="ns")
        text.configure(yscrollcommand=scrollbar.set)
        return text

    def _display_method(self):
        metadata = getattr(self.session.policy, "metadata", {})
        return str(metadata.get("display_method") or self.session.config.metadata.get("display_method")
                   or self.session.config.method).replace("_", " ")

    def _pane(self, parent, **options):
        pane = ResizablePane(parent, preferences=self.preferences, **options)
        self._workspace_panes.append(pane)
        return pane

    def _scoreboard(self, parent):
        self.scoreboard = ttk.Frame(parent)
        self.scoreboard.grid(row=0,column=1,sticky="e",padx=(SPACE["section"],0))
        self.score_context = ttk.Frame(self.scoreboard)
        self.score_context.pack(side="left",padx=(0,SPACE["control"]))
        self.score_scope = tk.StringVar(value="Saved in this session")
        ttk.Label(self.score_context,text="Session",style="Caption.TLabel").pack(anchor="w")
        self.automatic_count = tk.StringVar(value="Automatic failures: 0")
        self.automatic_display = tk.StringVar(value="Automatic failures: 0")
        self.automatic_count_label = ttk.Label(self.score_context,textvariable=self.automatic_display,style="Caption.TLabel")
        self.automatic_count_label.pack(anchor="w")
        self.score_vars, self.score_labels, self.score_cells, self.score_captions = {}, {}, {}, {}
        for key,label,style in (("success","Success","Success.Score.TLabel"),("failure","Failure","Failure.Score.TLabel"),
                                ("rate","SR","Score.TLabel"),("total","Total","Score.TLabel")):
            cell=self.score_cells[key]=ttk.Frame(self.scoreboard)
            cell.pack(side="left",padx=(0,SPACE["group"]))
            self.score_vars[key]=variable=tk.StringVar(value="—" if key=="rate" else "0")
            self.score_labels[key]=ttk.Label(cell,textvariable=variable,style=style)
            self.score_labels[key].pack(side="left")
            self.score_captions[key]=ttk.Label(cell,text=label,style="Caption.TLabel")
            self.score_captions[key].pack(side="left",padx=(SPACE["small"],0))

    def _update_scoreboard(self, results):
        successes = sum(row.get("outcome") == "success" for row in results)
        failures = sum(row.get("outcome") == "failure" for row in results)
        total = successes+failures
        automatic = sum(row.get("outcome") == "failure" and row.get("outcome_source") == "automatic_fault" for row in results)
        rate = f"{100*successes/total:.1f}%" if total else "—"
        for key, value in (("success",successes), ("failure",failures), ("total",total), ("rate",rate)):
            self.score_vars[key].set(str(value))
        self.automatic_count.set(f"Automatic failures: {automatic}")
        self.automatic_display.set(f"Auto {automatic}" if getattr(self,"_compact_counts",False) else self.automatic_count.get())
        self.counts.set(f"Success {successes}  ·  Failure {failures}  ·  SR {rate}")

    def set_text_size(self, value):
        value = max(12, min(24, int(value)))
        self._preferences_changed = True
        self.text_size = value
        self.preferences.set("text_size", value)
        self._flush_presentation_preferences()
        self._configure_theme()
        if hasattr(self, "text_size_label"):
            self.text_size_label.configure(text=f"{value} pt")
        if hasattr(self, "text_size_choice"):
            self.text_size_choice.set(str(value))
        if hasattr(self, "table"):
            self._size_result_columns()
        self._schedule_layout()
        self._schedule_preferences()
        return value

    def _size_result_columns(self):
        for column, samples in (("run",("Run","999")),("result",("Outcome","success","failure")),
                                ("steps",("Steps","9999")),("hz",("Hz","200.0"))):
            width = max(self.fonts["body"].measure(text) for text in samples)+2*SPACE["control"]
            self.table.column(column,width=width,minwidth=width,stretch=False)

    def change_text_size(self, delta):
        return self.set_text_size(self.text_size+delta)

    def toggle_context(self):
        self._short_context_open = not self._short_context_open
        self._schedule_layout()

    def reset_layout(self):
        self._preferences_changed = True
        for pane in self._workspace_panes:
            pane.reset()
        self._schedule_layout()

    def _schedule_layout(self):
        if self.closing:
            return
        if self._layout_job is not None:
            self.root.after_cancel(self._layout_job)
        self._layout_job = self.root.after_idle(self._layout_workspace)

    def _inspector_resized(self, event):
        if event.width != self._inspector_width:
            self._inspector_width = event.width
            self._schedule_layout()

    def _live_resized(self,event):
        size=(event.width,event.height)
        if size!=self._live_size:
            self._live_size=size
            self._schedule_layout()

    def _workspace_resized(self, event=None):
        if self.closing or not hasattr(self,"workspace_canvas"):
            return
        width, height = self.workspace_canvas.winfo_width(), self.workspace_canvas.winfo_height()
        # Each inspector form owns its scrolling. Inflating this outer canvas
        # would move the fixed policy actions below the visible workspace.
        content_height = max(height, 220)
        self.workspace_canvas.itemconfigure(self._workspace_window,width=max(1,width),height=content_height)
        self.workspace_canvas.configure(scrollregion=(0,0,max(1,width),content_height))
        if content_height>height:
            self.workspace_scrollbar.grid()
        else:
            self.workspace_scrollbar.grid_remove()
            self.workspace_canvas.yview_moveto(0)

    def _schedule_preferences(self):
        if self.closing:
            return
        if self._preferences_job is not None:
            self.root.after_cancel(self._preferences_job)
        self._preferences_job = self.root.after(350, self._save_preferences)

    def _save_preferences(self):
        if self._preferences_job is not None:
            self.root.after_cancel(self._preferences_job)
        self._preferences_job = None
        if not self.root.winfo_exists():
            return
        # Invalid/unknown preference files remain available for inspection until
        # the operator deliberately changes a presentation setting.
        if self.preferences.error and not self._preferences_changed:
            return
        self.preferences.set("window", {"width": self.root.winfo_width(), "height": self.root.winfo_height(),
                                       "x": self.root.winfo_x(), "y": self.root.winfo_y()})
        self._flush_presentation_preferences()

    def _flush_presentation_preferences(self):
        self.preferences.flush()
        self.presentation_status.set("Could not save display settings. Check permissions or free space." if self.preferences.error else "")
        if hasattr(self,"presentation_label"):
            if self.presentation_status.get():
                self.presentation_label.grid()
            else:
                self.presentation_label.grid_remove()

    def toggle_inspector(self):
        if self.closing:
            return
        if "second" not in self.main_pane.collapsed:
            self.inspector_toggle.focus_set()
        self.main_pane.toggle("second")
        self.main_pane.header.grid_remove()
        self._schedule_layout()

    def toggle_reference(self):
        self.preferences.set("reference_visible",self.reference_enabled.get())
        self._preferences_changed=True
        self._flush_presentation_preferences()
        self._sync_reference_visibility()
        self._schedule_layout()

    def _view_changed(self, _event=None):
        if not self.tabs.select():
            return
        title=self.tabs.tab(self.tabs.select(),"text")
        self.selected_view.set(title)
        self._checkpoint_expanded=title=="Policy"
        if title=="Policy":
            self.measurement_panel.grid_remove()
        else:
            self.measurement_panel.grid()
        self._schedule_layout()

    def _layout_workspace(self):
        self._layout_job = None
        if self.closing or not hasattr(self,"main_pane") or not self.root.winfo_exists():
            return
        width=max(1,self.root.winfo_width()); available=max(1,width-2*SPACE["section"]); scale=self.text_size/14
        self._compact_chrome=True
        self.heading.configure(padding=(SPACE["section"],SPACE["small"]))
        self.main_pane.header.grid_remove()
        self.visual_pane.header.grid_remove()
        self.fonts["score"].configure(size=max(self.text_size,min(24,round(self.text_size*24/14))))
        self._compact_counts = available<1200 and self.text_size>18
        self.automatic_display.set(self.automatic_count.get().replace("Automatic failures:","Auto") if self._compact_counts else self.automatic_count.get())
        for key in self.score_cells:
            self.score_labels[key].pack_configure(side="top" if self._compact_counts else "left")
            self.score_captions[key].pack_configure(side="top" if self._compact_counts else "left",padx=0 if self._compact_counts else (SPACE["small"],0))
        self.score_captions["failure"].configure(text="Fail" if self._compact_counts else "Failure")
        # Controls stay in three task groups. Their natural widths, not seven
        # stretched cells, determine where a coherent group wraps.
        group_widths=[group.winfo_reqwidth() for group in self.action_groups]
        score_width=self.scoreboard.winfo_reqwidth()
        controls_width=sum(group_widths)+2*SPACE["section"]
        combined=controls_width+score_width+SPACE["section"]<=available
        self.toolbar.columnconfigure(0,weight=1 if combined else 0)
        self.controls.grid(row=0,column=0,columnspan=1 if combined else 2,sticky="w")
        if combined:
            self.scoreboard.grid(row=0,column=1,sticky="e",padx=(SPACE["section"],0),pady=0)
            control_space=available-score_width-SPACE["section"]
        else:
            self.scoreboard.grid(row=1,column=0,columnspan=2,sticky="w",padx=0,pady=(SPACE["small"],0))
            control_space=available
        row=column=used=0
        for action_row in self.action_rows:action_row.grid_remove()
        for group,requested in zip(self.action_groups,group_widths):
            if column and used+requested>control_space:
                row,column,used=row+1,0,0
            self.action_rows[row].grid(row=row,column=0,sticky="w")
            group.grid(in_=self.action_rows[row],row=0,column=column,sticky="w",padx=(0,SPACE["section"]),pady=(SPACE["small"],0) if row else 0)
            group.lift()
            used+=requested+SPACE["section"];column+=1
        if not combined and row and used+score_width<=available:
            self.scoreboard.grid(in_=self.action_rows[row],row=0,column=column,sticky="w",padx=0,pady=0)
        self.scoreboard.lift()
        tools=([self.checkpoint_toggle] if self.reload_fields else [])+[self.inspector_toggle,self.help_button,self.context_button,self.zoom_controls,self.exit_button]
        compact=sum(w.winfo_reqwidth()+SPACE["control"] for w in tools)+self.state_label.winfo_reqwidth()+250>available
        self.heading_label.grid_configure(row=0,column=0,columnspan=1,sticky="w")
        self.state_label.grid_configure(row=0,column=1,columnspan=1,sticky="e")
        self.context_button.configure(text="Hide context" if self._short_context_open else "Context")
        if compact:
            self.heading_label.grid_configure(columnspan=5)
            self.state_label.grid_configure(column=5)
            self.header_tools.grid(row=1,column=0,columnspan=6,sticky="w")
            row=column=used=0
            for widget in tools:
                requested=widget.winfo_reqwidth()+SPACE["control"]
                if column and used+requested>available:row,column,used=row+1,0,0
                widget.grid_configure(in_=self.header_tools,row=row,column=column,columnspan=1,sticky="w",pady=0)
                used+=requested;column+=1
            context_row=2
            heading_width=available-self.state_label.winfo_reqwidth()-SPACE["group"]
        else:
            self.header_tools.grid_remove()
            for index,widget in enumerate(tools,start=2):
                widget.grid_configure(in_=self.heading,row=0,column=index,columnspan=1,sticky="e",pady=0)
            context_row=1
            heading_width=available-sum(w.winfo_reqwidth()+SPACE["control"] for w in tools)-self.state_label.winfo_reqwidth()-SPACE["group"]
        task_width=min(self.fonts["heading"].measure(self._display_task()),max(180,round(heading_width*.5)))
        task=self._display_task()
        while task and self.fonts["heading"].measure(task)>task_width:
            task=task[:-2].rstrip()+"…"
        self.heading_label.configure(text=task,wraplength=0)
        method_width=max(80,heading_width-task_width-SPACE["group"])
        method=self.active_method
        while method and self.fonts["caption"].measure(method)>method_width:
            method=method[:-2].rstrip()+"…"
        self.method_label.configure(text=method)
        self.method_label.grid(row=0,column=0,columnspan=5 if compact else 1,sticky="w",padx=(task_width+SPACE["group"],0))
        self.identity_label.grid_configure(row=context_row,column=0,columnspan=6,sticky="w",pady=0)
        self.instruction_label.grid_configure(row=context_row+1,column=0,columnspan=6,sticky="w",pady=(SPACE["small"],0))
        if self._short_context_open:
            self.identity_label.grid();self.instruction_label.grid()
        else:
            self.identity_label.grid_remove();self.instruction_label.grid_remove()
        self.identity_label.configure(wraplength=max(300,available))
        self.instruction_label.configure(wraplength=max(300,available))
        self.status_label.configure(wraplength=max(300,available))
        self.main_pane.set_orientation("horizontal",persist=False)
        self.main_pane.set_minimums(first=280,second=min(round(370*scale),round(available*.55)))
        self._workspace_resized()
        pane_width=max(220,self.main_pane.second.winfo_width()-2*SPACE["control"])
        tab_width=sum(self.fonts["body"].measure(self.tabs.tab(tab,"text"))+2*SPACE["control"]+4 for tab in self.tabs.tabs())
        self.tabs.configure(style="Compact.TNotebook" if pane_width<tab_width else "TNotebook")
        if pane_width<tab_width:self.view_selector.grid(row=1,column=0,sticky="ew")
        else:self.view_selector.grid_remove()
        self.control_hint.configure(wraplength=pane_width)
        self.measurement_label.configure(wraplength=pane_width)
        for label in (self.failure_note_label,self.result_summary_label,self.folder_label):label.configure(wraplength=pane_width)
        self._fit_secondary_labels(self.inspector_content,pane_width)
        if self._selection_cells:self._layout_checkpoint_form()
        if hasattr(self,"camera_pane"):
            panel=self.visual_pane.first
            w,h=panel.winfo_width(),panel.winfo_height()
            # Choose the orientation that displays more of two 4:3 views while
            # preserving their aspect ratio; the divider remains user-resizable.
            horizontal=min(max(1,(w-12)/2)/4,max(1,h-28)/3)**2
            vertical=min(max(1,w)/4,max(1,(h-12)/2-28)/3)**2
            direction="vertical" if vertical>horizontal*1.10 else "horizontal" if horizontal>vertical*1.10 else self.camera_pane.orient
            self.camera_pane.set_orientation(direction,persist=False)
        if self.catalog and self.reference_enabled.get():
            reference_caption=next(child for child in self.reference_labels[0].master.winfo_children()
                                   if isinstance(child,ttk.Label))
            reference_chrome=(self.reference_title.master.winfo_reqheight()+5+2*SPACE["small"]
                              +reference_caption.winfo_reqheight()+4)
            self.visual_pane.set_minimums(first=180,second=140+reference_chrome)

    def _fit_secondary_labels(self, parent, available):
        for child in parent.winfo_children():
            if isinstance(child,ttk.Label) and child.cget("wraplength") and self.root.winfo_pixels(child.cget("wraplength"))>0:
                parent_width = child.master.winfo_width()
                limit = min(available,parent_width-2*SPACE["control"]) if parent_width>120 else available
                if self.root.winfo_pixels(child.cget("wraplength")) != max(100,limit):
                    child.configure(wraplength=max(100,limit))
            if child.winfo_children():
                self._fit_secondary_labels(child,available)

    def _layout_checkpoint_form(self):
        if not hasattr(self, "_selection_form"):
            return
        width = max(1, self._selection_form.winfo_width())
        cell_minimum = round(300*self.text_size/14)
        columns = max(1, min(3, width//cell_minimum))
        for column in range(3):
            self._selection_form.columnconfigure(column, weight=1 if column<columns else 0,
                                                 uniform="selection" if column<columns else "")
        index = 0
        for cell, advanced in self._selection_cells:
            if advanced and not self._advanced_fields_open:
                cell.grid_remove()
                continue
            row, column = divmod(index, columns)
            cell.grid(row=row+2, column=column, sticky="ew",
                      padx=(0, SPACE["control"]) if column<columns-1 else 0, pady=(0, SPACE["control"]))
            index += 1
        for widget, variable, hint in self._selection_choice_hints:
            width = max(80,widget.winfo_width()-2*SPACE["control"])
            hint.configure(wraplength=width)
            if self.fonts["body"].measure(variable.get())>width-20:
                hint.grid(row=2,column=0,columnspan=2,sticky="ew",pady=(SPACE["small"],0))
            else:
                hint.grid_remove()

    def toggle_advanced_fields(self):
        self._advanced_fields_open = not self._advanced_fields_open
        self.advanced_fields_button.configure(text="Basic fields" if self._advanced_fields_open else "Advanced")
        self._layout_checkpoint_form()

    def _scrollable_content(self, parent, *, height=None):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        canvas = tk.Canvas(parent, background=THEME["canvas"], highlightthickness=0,
                           width=1, height=height if height is not None else 1)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        inner = ttk.Frame(canvas, padding=SPACE["control"])
        window = canvas.create_window(0, 0, anchor="nw", window=inner)
        inner.bind("<Configure>", lambda _event:canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda event:canvas.itemconfigure(window, width=max(1,event.width)))
        self._scroll_regions.append((canvas,inner))
        # Use the visible scrollbar rather than global wheel bindings that can
        # steal scrolling from notes or accidentally change a camera combobox.
        inner.scroll_canvas = canvas
        return inner

    def _reveal_focus(self, event, canvas, content):
        widget = event.widget
        if self.closing or not str(widget).startswith(str(content)+".") or not canvas.winfo_viewable():
            return
        if canvas is self.workspace_canvas:
            # Reveal a nested form first, then reveal its newly visible control
            # in the workspace viewport. Reversing these steps loses focus offscreen.
            for inner_canvas,inner_content in self._scroll_regions:
                if str(widget).startswith(str(inner_content)+"."):
                    self._reveal_focus(event,inner_canvas,inner_content)
        bounds = canvas.bbox("all")
        if not bounds or canvas.winfo_height()<2:
            return
        top = widget.winfo_rooty()-content.winfo_rooty()
        bottom = top+widget.winfo_height()
        current = canvas.canvasy(0)
        height = canvas.winfo_height()
        target = max(0,top-SPACE["control"]) if top<current else bottom-height+SPACE["control"] if bottom>current+height else current
        if target != current:
            canvas.yview_moveto(target/max(1,bounds[3]))

    def _devices_panel(self, parent):
        parent.columnconfigure(1, weight=1)
        self.device_hint = tk.StringVar(value="Refresh lists connected cameras; it does not start a policy or move the robot.")
        ttk.Label(parent, textvariable=self.device_hint, wraplength=370, style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, SPACE["control"]))
        self.scan_devices_button = ttk.Button(parent, text="Refresh connected cameras", command=self.refresh_devices)
        self.scan_devices_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, SPACE["group"]))
        self.camera_vars, self.camera_selectors, self.camera_device_names = {}, {}, {}
        mapping = self._camera_mapping()
        for row, role in enumerate(self.session.config.camera_roles, start=2):
            ttk.Label(parent, text=role.title()).grid(row=row*2, column=0, sticky="w", padx=(0, SPACE["control"]))
            variable = tk.StringVar(value=str(mapping.get(role, "")))
            selector = ttk.Combobox(parent, textvariable=variable, values=[variable.get()] if variable.get() else (), state="readonly", width=18)
            selector.grid(row=row*2, column=1, sticky="ew")
            selector.bind("<<ComboboxSelected>>", self._camera_draft_changed)
            self.camera_vars[role], self.camera_selectors[role] = variable, selector
            self.camera_device_names[role] = tk.StringVar(value="Current assignment")
            ttk.Label(parent, textvariable=self.camera_device_names[role], style="Muted.TLabel", wraplength=280).grid(row=row*2+1, column=1, sticky="w", pady=(0, SPACE["control"]))
        row = 4 + 2*len(self.camera_vars)
        actions = ttk.Frame(parent)
        actions.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(SPACE["small"], 0))
        self.swap_cameras_button = ttk.Button(actions, text="Swap roles" if len(self.camera_vars)==2 else "Swap first two roles", command=self.swap_camera_roles)
        self.swap_cameras_button.pack(side="left")
        self.apply_cameras_button = ttk.Button(actions, text="Apply cameras", command=self.apply_cameras)
        self.apply_cameras_button.pack(side="right")
        self.reconnect_cameras_button = ttk.Button(parent, text="Restart camera preview", command=self.restart_cameras)
        self.reconnect_cameras_button.grid(row=row+1, column=0, columnspan=2, sticky="ew", pady=(SPACE["control"], 0))
        self.camera_health = tk.StringVar(value="No camera observations yet")
        ttk.Label(parent, textvariable=self.camera_health, wraplength=370, justify="left").grid(row=row+2, column=0, columnspan=2, sticky="ew", pady=(SPACE["group"], 0))
        ttk.Label(parent, text="Camera changes preserve the loaded policy. Apply only while idle; Start is always a separate action.", style="Muted.TLabel", wraplength=370).grid(row=row+3, column=0, columnspan=2, sticky="ew", pady=(SPACE["control"], 0))

    def _camera_mapping(self):
        metadata = getattr(self.session.embodiment, "metadata", {})
        mapping = metadata.get("camera_serials", {}) if isinstance(metadata, dict) else {}
        return {role:str(mapping.get(role, "")) for role in self.session.config.camera_roles}

    def _camera_draft_changed(self, _event=None):
        self._camera_draft_dirty = True
        self.device_hint.set("Camera selection is a draft. Apply cameras to change the live streams.")

    def refresh_devices(self):
        if self.closing or self._reload_in_progress:
            return False
        operation = getattr(self.session, "request_device_scan", None)
        try:
            if callable(operation) and operation():
                self.device_hint.set("Listing connected camera identities...")
                return True
            self.device_hint.set("Camera discovery is unavailable or busy. Wait until the current operation ends.")
        except Exception as error:
            self.device_hint.set(f"Camera discovery failed: {error}")
        return False

    def swap_camera_roles(self):
        if self.closing or self._reload_in_progress or len(self.camera_vars)<2:
            return False
        status = getattr(self.session, "device_status", {})
        if not status.get("can_apply", False):
            return False
        left, right = list(self.camera_vars.values())[:2]
        first, second = left.get(), right.get()
        left.set(second)
        right.set(first)
        self._camera_draft_changed()
        return True

    def apply_cameras(self):
        if self.closing or self._reload_in_progress:
            return False
        mapping = {role:variable.get().strip() for role,variable in self.camera_vars.items()}
        if not all(mapping.values()) or len(set(mapping.values())) != len(mapping):
            self.device_hint.set("Choose a different camera for every configured role.")
            return False
        operation = getattr(self.session, "request_camera_update", None)
        try:
            if callable(operation) and operation(mapping):
                self._camera_draft_dirty = True
                self.device_hint.set("Applying camera assignments. Policy remains loaded; no motion is requested.")
                return True
            self.device_hint.set("Camera changes require an idle session and acknowledged stop.")
        except Exception as error:
            self.device_hint.set(f"Camera change failed: {error}")
        return False

    def restart_cameras(self):
        if self.closing or self._reload_in_progress:
            return False
        operation = getattr(self.session, "request_camera_reconnect", None)
        try:
            if callable(operation) and operation():
                self.device_hint.set("Restarting the current camera streams; loaded policy is preserved.")
                return True
            self.device_hint.set("Camera restart is unavailable while another operation is active.")
        except Exception as error:
            self.device_hint.set(f"Camera restart failed: {error}")
        return False

    def _sync_camera_selection(self, mapping):
        if not self.reload_fields:
            return
        selected = self.current_selection()
        selected["camera_serials"] = dict(mapping)
        for role, serial in mapping.items():
            if f"{role}_serial" in selected:
                selected[f"{role}_serial"] = serial
        selected.setdefault("resolved_config", {}).setdefault("robot", {})["camera_serials"] = dict(mapping)
        try:
            if self.selection_resolver:
                selected = self.selection_resolver(selected, "devices")
            self._apply_selection(selected)
        except Exception as error:
            # The camera transaction already succeeded. Keep its actual identity
            # even if a project-specific draft cannot currently be resolved.
            self._apply_selection(selected)
            self.session.record_event("camera_selection_sync_error", detail=str(error))
            self.device_hint.set("Cameras changed. Check the selected task before saving its configuration.")

    @staticmethod
    def _finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)

    def _update_device_status(self, latest):
        status = getattr(self.session, "device_status", {})
        status = status if isinstance(status, dict) else {}
        active = self.closing or self._reload_in_progress or status.get("busy")
        can_apply = not active and status.get("can_apply", False)
        can_scan = not active and not status.get("scan_busy") and status.get("can_scan", False)
        self.scan_devices_button.configure(text="Listing cameras..." if status.get("scan_busy") else "Refresh connected cameras", state="normal" if can_scan else "disabled")
        self.apply_cameras_button.configure(state="normal" if can_apply else "disabled")
        self.reconnect_cameras_button.configure(state="normal" if can_apply else "disabled")
        self.swap_cameras_button.configure(state="normal" if can_apply and len(self.camera_vars)>1 else "disabled")
        devices = status.get("devices", [])
        devices = [row for row in devices if isinstance(row, dict) and row.get("serial")]
        names = {str(row["serial"]):str(row.get("name") or row.get("label") or "Camera") for row in devices}
        actual = status.get("selected") or self._camera_mapping()
        actual = {role:str(actual.get(role, "")) for role in self.camera_vars}
        changed = self._camera_actual_seen is not None and actual != self._camera_actual_seen
        if changed or self._camera_actual_seen is None:
            if changed or not self._camera_draft_dirty:
                for role, value in actual.items():
                    self.camera_vars[role].set(value)
                self._camera_draft_dirty = False
            if changed:
                self._sync_camera_selection(actual)
                self.device_hint.set("Camera assignments applied. The loaded policy is unchanged.")
            self._camera_actual_seen = dict(actual)
        for role, selector in self.camera_selectors.items():
            selected = self.camera_vars[role].get()
            values = sorted(set(names) | {serial for serial in actual.values() if serial} | ({selected} if selected else set()))
            selector.configure(values=values, state="readonly" if can_apply else "disabled")
            current = actual.get(role, "")
            name = names.get(selected, "Current camera" if selected==current else "Not in the refreshed device list")
            suffix = " · live" if selected==current else " · draft"
            self.camera_device_names[role].set(name + suffix)
        if status.get("error"):
            self.device_hint.set(f"Camera operation failed: {status['error']}")
        elif status.get("warnings"):
            self.device_hint.set("; ".join(map(str,status["warnings"])))
        elif not status.get("supported", bool(status)):
            self.device_hint.set("This embodiment does not expose camera selection. Existing streams remain available.")
        health = getattr(self.session, "operator_health", {})
        health = health if isinstance(health, dict) else {}
        camera_health = health.get("cameras", {})
        rows = []
        capture_age = None
        if latest is not None and latest.captured_monotonic_ns is not None:
            capture_age = max(0, (time.monotonic_ns()-latest.captured_monotonic_ns)/1e6)
        for role in self.camera_vars:
            item = camera_health.get(role, {})
            age = item.get("age_ms", capture_age)
            frame = item.get("frame_number")
            camera_meta = latest.camera_metadata.get(role, {}) if latest is not None else {}
            if not isinstance(camera_meta, dict):
                camera_meta = {}
            if frame is None:
                frame = camera_meta.get("frame_number")
            age_text = f"{age:.0f} ms old" if self._finite_number(age) else "waiting for frame"
            if self._finite_number(age) and age>1000:
                age_text += " · stale"
            if frame is not None:
                age_text += f" · frame {frame}"
            source_age = item.get("source_age_ms")
            if self._finite_number(source_age):
                age_text += f" · source {source_age:.0f} ms"
            fps = item.get("fps")
            if self._finite_number(fps):
                age_text += f" · preview {fps:.1f} fps"
            rows.append(f"{role.title()}: {age_text}")
            if role in self.camera_title_vars:
                self.camera_title_vars[role].set(role.title())
        self.camera_health.set("\n".join(rows))
        storage = health.get("disk_free_bytes")
        messages = [f"Free storage {storage/2**30:.1f} GiB" if self._finite_number(storage) else "Storage status unavailable"]
        recording = health.get("recording", {})
        if not recording.get("enabled", self.session.config.record_video):
            messages.append("Video recording OFF")
        elif recording.get("error") or recording.get("healthy") is False:
            messages.append("Recording error: " + str(recording.get("error") or "writer unavailable"))
        elif recording.get("writers"):
            messages.append(f"{'Last recording' if recording.get('closed') else 'Recording'}: {recording.get('frames', 0)} video frames")
        else:
            messages.append("Video recording ON · writers open at rollout start")
        if health.get("disk_error") or health.get("disk_warning"):
            warning = health.get("disk_error") or health.get("disk_warning")
            messages.append("Output storage is running low" if warning is True else str(warning))
        self.health.set("\n".join(messages))
        return health

    def _configure_theme(self):
        self.root.configure(bg=THEME["canvas"])
        available_families = set(tkfont.families(root=self.root))
        family = getattr(self, "_font_family", "Noto Sans CJK KR" if "Noto Sans CJK KR" in available_families
                         else tkfont.nametofont("TkDefaultFont", root=self.root).actual("family"))
        self._font_family = family
        if not hasattr(self, "fonts"):
            self.fonts = {role:tkfont.Font(root=self.root, family=family) for role in
                          ("body", "caption", "heading", "state", "score")}
        for role, size, weight in (("body",self.text_size,"normal"),
                                  ("caption",max(12,self.text_size-1),"normal"),
                                  ("heading",self.text_size+4,"bold"),
                                  ("state",self.text_size,"bold"),
                                  ("score",max(self.text_size,min(24,round(self.text_size*24/14))),"bold")):
            self.fonts[role].configure(family=family, size=size, weight=weight)
        tkfont.nametofont("TkDefaultFont", root=self.root).configure(family=family, size=self.text_size)
        tkfont.nametofont("TkTextFont", root=self.root).configure(family=family, size=self.text_size)
        self.style = style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=THEME["canvas"], foreground=THEME["text"], font=self.fonts["body"])
        style.configure("TFrame", background=THEME["canvas"])
        style.configure("TLabel", background=THEME["canvas"], foreground=THEME["text"])
        style.configure("Muted.TLabel", foreground=THEME["muted"])
        style.configure("Caption.TLabel", foreground=THEME["muted"], font=self.fonts["caption"])
        style.configure("Heading.TLabel", font=self.fonts["heading"])
        style.configure("State.TLabel", background=THEME["inset"], font=self.fonts["state"])
        style.configure("Score.TLabel", font=self.fonts["score"])
        style.configure("Success.Score.TLabel", foreground=THEME["accent"])
        style.configure("Failure.Score.TLabel", foreground=THEME["danger"])
        for name, color in (("Ready", THEME["accent"]), ("Fault", THEME["danger"]), ("Pending", THEME["warning"])):
            style.configure(f"{name}.State.TLabel", foreground=color)
        button_y=max(4,round((44-self.fonts["body"].metrics("linespace")-6)/2))
        style.configure("TButton", background=THEME["inset"], foreground=THEME["text"], padding=(SPACE["control"], button_y), bordercolor=THEME["border"], lightcolor=THEME["inset"], darkcolor=THEME["inset"], borderwidth=1, focusthickness=2, focuscolor=THEME["accent"])
        style.map("TButton", background=[("active", THEME["border"])], foreground=[("disabled", THEME["disabled"])])
        for name, color in (("Accent", THEME["accent"]), ("Stop", THEME["danger"])):
            style.configure(f"{name}.TButton", background=color, foreground=THEME["ink"], bordercolor=color)
            style.map(f"{name}.TButton", background=[("disabled", THEME["inset"]), ("active", color)], foreground=[("disabled", THEME["disabled"])])
        style.configure("TLabelframe", background=THEME["canvas"], bordercolor=THEME["border"], relief="solid", borderwidth=1)
        style.configure("TLabelframe.Label", background=THEME["canvas"], foreground=THEME["muted"])
        style.configure("TEntry", fieldbackground=THEME["panel"], foreground=THEME["text"], bordercolor=THEME["border"], insertcolor=THEME["text"], padding=SPACE["small"])
        style.map("TEntry", fieldbackground=[("disabled", THEME["inset"])], foreground=[("disabled", THEME["disabled"])])
        style.configure("TCombobox", fieldbackground=THEME["panel"], background=THEME["inset"], foreground=THEME["text"], arrowcolor=THEME["text"], bordercolor=THEME["border"], padding=SPACE["small"])
        style.map("TCombobox", fieldbackground=[("readonly", THEME["panel"])], foreground=[("disabled", THEME["disabled"]), ("readonly", THEME["text"])], selectbackground=[("readonly", THEME["selection"])], selectforeground=[("readonly", THEME["text"])])
        self.root.option_add("*TCombobox*Listbox.background", THEME["panel"])
        self.root.option_add("*TCombobox*Listbox.foreground", THEME["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", THEME["selection"])
        style.configure("Treeview", rowheight=self.fonts["body"].metrics("linespace")+max(10,round(12*self.text_size/14)),
                        font=self.fonts["body"], background=THEME["panel"], fieldbackground=THEME["panel"], foreground=THEME["text"], borderwidth=0)
        style.map("Treeview", background=[("selected", THEME["selection"])], foreground=[("selected", THEME["text"])])
        style.configure("Treeview.Heading", background=THEME["inset"], foreground=THEME["muted"], padding=(SPACE["small"], SPACE["control"]), borderwidth=0)
        style.configure("TNotebook", background=THEME["canvas"], borderwidth=0, tabmargins=0)
        style.configure("TNotebook.Tab", background=THEME["panel"], foreground=THEME["muted"], padding=(SPACE["control"], SPACE["control"]))
        style.map("TNotebook.Tab", background=[("selected", THEME["inset"])], foreground=[("selected", THEME["text"])])
        style.layout("Compact.TNotebook.Tab", [])

    def _text_style(self):
        return {"background": THEME["panel"], "foreground": THEME["text"], "insertbackground": THEME["text"],
                "font": self.fonts["body"],
                "selectbackground": THEME["selection"], "selectforeground": THEME["text"], "relief": "flat",
                "highlightthickness": 1, "highlightbackground": THEME["border"], "highlightcolor": THEME["accent"], "padx": SPACE["control"], "pady": SPACE["control"]}

    def _resize_copy(self, event):
        if event.widget is self.root:
            self._schedule_layout()
            self._schedule_preferences()

    def _checkpoint_panel(self, parent):
        self.checkpoint_panel = panel = ttk.Frame(parent, padding=SPACE["control"])
        panel.grid(row=0, column=0, sticky="nsew")
        panel.rowconfigure(0,weight=1)
        panel.columnconfigure(0, weight=1)
        form = self._selection_form = self._scrollable_content(panel, height=235)
        form.bind("<Configure>", lambda _event:self._layout_checkpoint_form(),add="+")
        utilities=ttk.Frame(form)
        utilities.grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,SPACE["control"]))
        selected = self.session.config.metadata.get("reload_selection", {})
        for index, field in enumerate(self.reload_fields):
            row, column = divmod(index, 3)
            form.columnconfigure(column, weight=1, uniform="selection")
            cell = ttk.Frame(form)
            cell.grid(row=row, column=column, sticky="ew", padx=(0, SPACE["control"]) if column != 2 else 0, pady=(0, SPACE["control"]))
            self._selection_cells.append((cell, bool(field.get("advanced"))))
            cell.columnconfigure(0, weight=1)
            key, kind = field["key"], field.get("kind", "text")
            value = selected.get(key, field.get("value", ""))
            self.reload_vars[key] = variable = tk.StringVar(value=str(value))
            ttk.Label(cell, text=field.get("label", key), style="Muted.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, SPACE["small"]))
            if kind == "choice":
                widget = ttk.Combobox(cell, textvariable=variable, values=field.get("choices", ()), state="readonly", width=1)
                widget.bind("<<ComboboxSelected>>", lambda _event, key=key:self.resolve_selection(key))
                hint = ttk.Label(cell,textvariable=variable,style="Caption.TLabel",wraplength=250)
                self._selection_choice_hints.append((widget,variable,hint))
                variable.trace_add("write",lambda *_args:self._schedule_layout())
                widget.bind("<Configure>",lambda _event:self._schedule_layout(),add="+")
            else:
                widget = ttk.Entry(cell, textvariable=variable, width=1)
                if field.get("readonly"):
                    widget.configure(state="readonly")
            widget.grid(row=1, column=0, sticky="ew")
            self.reload_widgets.append((widget, "readonly" if kind == "choice" or field.get("readonly") else "normal"))
            if kind in {"directory", "file"}:
                button = ttk.Button(cell, text="Browse", command=lambda field=field:self.browse_checkpoint(field))
                button.grid(row=1, column=1, padx=(SPACE["small"], 0))
                self.reload_widgets.append((button, "disabled" if field.get("readonly") else "normal"))
        self.advanced_fields_button = ttk.Button(utilities, text="Advanced", width=0, command=self.toggle_advanced_fields)
        if any(advanced for _cell, advanced in self._selection_cells):
            self.advanced_fields_button.pack(side="left",padx=(0,SPACE["control"]))
        files = ttk.Frame(panel)
        files.grid(row=2, column=0, sticky="ew", pady=(SPACE["control"], 0))
        files.columnconfigure(0, weight=1)
        config_menu=tk.Menu(utilities,tearoff=False)
        self.config_menu_button=ttk.Menubutton(utilities,text="Config",width=0,menu=config_menu)
        for title, callback, enabled in (("Load config", self.load_config, self.config_loader),
                                         ("Save config", self.save_config, self.config_saver)):
            if enabled:
                config_menu.add_command(label=title,command=callback)
        if self.config_loader or self.config_saver:
            self.config_menu_button.pack(side="left")
            self.reload_widgets.append((self.config_menu_button,"normal"))
        self.load_button = ttk.Button(files, text="Load checkpoint", style="Accent.TButton", command=self.load_checkpoint, state="disabled")
        self.load_button.grid(row=0, column=1, sticky="e")
        self.load_error = tk.Text(form, width=1, height=1, wrap="word", **self._text_style())
        self.load_error.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0,SPACE["control"]))
        self.load_error.configure(state="disabled")
        self.load_error.grid_remove()
        self._record_selection(self.session, self.current_selection())
        self._layout_checkpoint_form()
        self._set_checkpoint_expanded(bool(self.session.config.metadata.get("defer_policy_load", False)))

    def _set_checkpoint_expanded(self, expanded):
        self._checkpoint_expanded=bool(expanded)
        if expanded:
            self.main_pane.expand("second")
            self.tabs.select(self.policy_tab)
        elif self.policy_tab is not None and self.tabs.select()==str(self.policy_tab):
            self.tabs.select(self.rollouts_tab)
        self.main_pane.header.grid_remove()
        self.checkpoint_toggle.configure(text="Policy")
        self._schedule_layout()

    def toggle_checkpoint_panel(self):
        if not self.closing:
            self._set_checkpoint_expanded(not (self.tabs.select()==str(self.policy_tab)))

    def current_selection(self):
        selection = copy.deepcopy(self._selection_context)
        selection.update({key:value.get() for key,value in self.reload_vars.items()})
        return selection

    def _apply_selection(self, selection):
        if not isinstance(selection, dict):
            raise TypeError("Selection callbacks must return a dictionary")
        self._selection_context = copy.deepcopy(selection)
        for key, value in selection.items():
            if key in self.reload_vars:
                self.reload_vars[key].set(str(value))

    def resolve_selection(self, changed_key):
        try:
            selected = self.current_selection()
            if self.selection_resolver:
                selected = self.selection_resolver(selected, changed_key)
            self._apply_selection(selected)
            return True
        except Exception as error:
            self._selection_error("Selection", error)
            return False

    def _show_load_error(self, message, detail=None):
        self._load_error_detail = detail or message
        summary = "Load failed. Open Details."
        self.status.set("Policy could not be loaded · open Details.")
        if self.reload_fields:
            self.load_error.configure(state="normal")
            self.load_error.delete("1.0", "end")
            self.load_error.insert("1.0", summary)
            self.load_error.configure(state="disabled")
            self.load_error.grid()
            self._set_checkpoint_expanded(True)

    def _selection_error(self, operation, error):
        detail = traceback.format_exc()
        message = f"{operation} failed: {type(error).__name__}: {error}"
        self.session.record_event("selection_error", operation=operation, detail=message,
                                  traceback=detail, selection=self.current_selection())
        self._show_load_error(message, detail)

    def _selection_editable(self):
        return (not self.closing and not self._reload_in_progress and not self._reload_blocked
                and self.session.can_replace)

    def load_config(self, path=None):
        if not self.config_loader or not self._selection_editable():
            return False
        if path is None:
            path = filedialog.askopenfilename(parent=self.root, title="Load deployment config",
                initialdir=str(self.config_directory) if self.config_directory else None,
                filetypes=[("JSON", "*.json"), ("All files", "*")])
        if not path:
            return False
        try:
            selected = self.config_loader(Path(path))
            if self.selection_resolver:
                selected = self.selection_resolver(selected, "config")
            self._apply_selection(selected)
            self.session.record_event("config_loaded", path=str(path), selection=selected)
            self.status.set(f"Loaded configuration: {path}. Load checkpoint to apply it")
            return True
        except Exception as error:
            self._selection_error("Config load", error)
            return False

    def save_config(self, path=None):
        if not self.config_saver or not self._selection_editable() or not self.resolve_selection("save"):
            return False
        if path is None:
            path = filedialog.asksaveasfilename(parent=self.root, title="Save deployment config",
                initialdir=str(self.config_directory) if self.config_directory else None,
                defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not path:
            return False
        try:
            selected = self.current_selection()
            self.config_saver(Path(path), selected)
            self.session.record_event("config_saved", path=str(path), selection=selected)
            self.status.set(f"Configuration saved: {path}")
            return True
        except Exception as error:
            self._selection_error("Config save", error)
            return False

    @staticmethod
    def _record_selection(session, selection):
        # The project factory may resolve paths and select a canonical checkpoint.
        session.config.metadata.setdefault("reload_selection", dict(selection))
        session.config.metadata["reload_request"] = dict(selection)
        session.manifest["config"] = session.config.asdict()
        write_json(session.folder / "manifest.json", session.manifest)

    def browse_checkpoint(self, field):
        if self.closing or self._reload_in_progress or self._reload_blocked or not self.session.can_replace:
            return
        options = {"parent": self.root, "title": field.get("label", field["key"])}
        current = self.reload_vars[field["key"]].get().strip()
        if current:
            options["initialdir"] = current if field["kind"] == "directory" else str(Path(current).parent)
        chosen = (filedialog.askdirectory if field["kind"] == "directory" else filedialog.askopenfilename)(**options)
        if chosen:
            self.reload_vars[field["key"]].set(chosen)
            self.resolve_selection(field["key"])

    def load_checkpoint(self):
        if not self.reload_factory or self.closing or self._reload_in_progress or self._reload_blocked or not self.session.can_replace:
            return False
        if not self.resolve_selection("load"):
            return False
        if not self.save():
            return False
        selection = self.current_selection()
        self.session.record_event("reload_requested", selection=selection)
        previous, previous_config = self.session, copy.deepcopy(self.session.config)
        self._reload_in_progress = True
        self._set_checkpoint_expanded(False)
        self.load_error.grid_remove()
        self.status.set("Closing the current session before loading the selected checkpoint")
        self._reload_thread = threading.Thread(target=self._reload, args=(previous, selection, previous_config),
                                               name="checkpoint-loader", daemon=True)
        self._reload_thread.start()
        return True

    def _reload(self, previous, selection, previous_config):
        candidate = None
        handed_off = False
        try:
            previous.close()
            previous.worker.join()
            if not previous.can_replace or not previous.replacement_closed:
                raise RuntimeError("Session did not close with confirmed stop and cleanup; checkpoint replacement is blocked")
            if self._reload_cancel.is_set():
                return
            self._reload_events.put(("status", "Loading the selected checkpoint"))
            candidate = self.reload_factory(dict(selection), previous_config)
            if not isinstance(candidate, DeploymentSession) or candidate is previous:
                raise TypeError("reload_factory must return a new unstarted DeploymentSession")
            if candidate.worker.ident is not None:
                raise RuntimeError("reload_factory returned an already-started session")
            candidate.config.validate()
            if candidate.config.camera_roles != previous.config.camera_roles:
                raise ValueError("Checkpoint changes must keep the configured camera roles")
            if candidate.folder == previous.folder:
                raise ValueError("A replacement must use its own result directory")
            self._record_selection(candidate, selection)
            if self._reload_cancel.is_set():
                return
            decision, acknowledged = {"accepted": False}, threading.Event()
            self._reload_events.put(("ready", candidate, decision, acknowledged))
            while not acknowledged.wait(.05):
                if self._reload_cancel.is_set():
                    break
            handed_off = decision["accepted"]
        except Exception as error:
            detail = traceback.format_exc()
            message = f"Checkpoint load failed: {type(error).__name__}: {error}"
            previous.record_event("reload_error", detail=message, traceback=detail, selection=selection)
            self._reload_events.put(("error", message, detail))
        finally:
            try:
                if isinstance(candidate, DeploymentSession) and candidate is not previous and not handed_off:
                    self._reload_cleanup_session = candidate
                    candidate.close()
                    candidate.worker.join()
                    if not candidate.replacement_closed:
                        self._reload_events.put(("blocked", "Replacement cleanup or stop was not confirmed; checkpoint replacement is blocked"))
                    else:
                        self._reload_cleanup_session = None
            except Exception as error:
                self._reload_events.put(("blocked", f"Replacement cleanup failed: {error}; checkpoint replacement is blocked"))
            finally:
                self._reload_events.put(("done",))

    def _bind_session(self, session):
        self.session = session
        self._last_displayed_state = None
        self.root.title(f"{session.config.method} | {session.config.task} deployment")
        self.heading_label.configure(text=self._display_task())
        self.instruction_label.configure(text=f"{self._display_task()} · {session.config.instruction}")
        self.folder_label.configure(text=str(session.folder))
        self.hz.set(str(session.config.target_hz))
        self.steps.set(str(session.config.max_steps))
        self.record_video.set(session.config.record_video)
        self._apply_selection(session.config.metadata.get("reload_selection", {}))
        self._set_checkpoint_expanded(False)
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self._rows_seen, self._details_seen = None, None
        self._failure_notes_seen = None
        self._load_error_detail = None
        self._last_save_error = None
        self._event_detail = ""
        self.failure_note.set("Select a saved rollout to view its result.")
        self.result_summary.set("")
        self.result_annotation.set("")
        self._annotation_result_id = None
        self._annotation_original = ""
        self._camera_actual_seen = self._camera_mapping()
        self._camera_draft_dirty = False
        self.video_role.set(session.config.camera_roles[0])
        self.video_role_selector.configure(values=session.config.camera_roles)
        self.photos = []
        for label in self.camera_labels:
            label.configure(image="", text="Waiting for frame")
        self._load_reference_catalog()
        for label in self.reference_labels:
            label.configure(image="", text="Waiting for frame")
        self.show_reference()
        self.state.set("CONNECTING")
        self.status.set("Opening selected policy and embodiment")
        session.start()

    def _reload_updates(self):
        while True:
            try:
                event = self._reload_events.get_nowait()
            except queue.Empty:
                return
            kind = event[0]
            if kind == "ready":
                _, candidate, decision, acknowledged = event
                try:
                    if not self.closing:
                        self._bind_session(candidate)
                        decision["accepted"] = True
                except Exception as error:
                    self.status.set(f"Checkpoint load failed: {error}")
                finally:
                    acknowledged.set()
            elif kind == "done":
                self._reload_in_progress = False
            elif kind == "blocked":
                self._reload_blocked = True
                self._show_load_error(event[1])
            elif kind == "error" and not self.closing:
                self._show_load_error(event[1], event[2] if len(event)>2 else None)
            elif not self.closing:
                self.status.set(event[1])

    def _image_pair(self,parent,title,row,roles=None):
        panel = ttk.Frame(parent, padding=SPACE["small"])
        panel.grid(row=row, column=0, sticky="nsew")
        roles=roles or self.session.config.camera_roles
        live = title == "Live observations"
        if len(roles) == 2:
            pane = self._pane(panel, key="observations" if live else "reference_cameras", orient="horizontal",
                              first_label=roles[0].title(), second_label=roles[1].title(), first_min=140,
                              second_min=140, default_fraction=.5, collapsible=(True,True), show_header=False)
            pane.pack(fill="both", expand=True)
            containers = (pane.first, pane.second)
            if live:
                self.camera_pane = pane
        else:
            containers = []
            for column in range(len(roles)):
                panel.columnconfigure(column, weight=1)
                frame = ttk.Frame(panel)
                frame.grid(row=0, column=column, sticky="nsew")
                containers.append(frame)
            panel.rowconfigure(0, weight=1)
        labels=[]
        for container, role in zip(containers, roles):
            container.columnconfigure(0, weight=1)
            container.rowconfigure(1, weight=1)
            if live:
                self.camera_title_vars[role] = variable = tk.StringVar(value=role.title())
                ttk.Label(container, textvariable=variable, style="Caption.TLabel").grid(row=0,column=0,sticky="w",pady=(0,4))
            else:
                ttk.Label(container,text=role.title(), style="Caption.TLabel").grid(row=0,column=0,sticky="w",pady=(0,4))
            label=tk.Label(container,bg=THEME["panel"],fg=THEME["muted"],text="Waiting for frame",width=1,height=1, font=self.fonts["body"])
            label.grid(row=1,column=0,sticky="nsew")
            if live:
                label.bind("<Double-Button-1>", lambda _event, role=role:self.zoom_live(role))
            labels.append(label)
        if not live:
            self.reference_images_panel = panel
        return labels

    def _references(self,parent):
        frame=ttk.Frame(parent)
        self._reference_parent = parent
        self._reference_error = None
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0,weight=1)
        frame.rowconfigure(1,weight=1)
        toolbar=ttk.Frame(frame)
        toolbar.grid(row=0,sticky="ew",pady=(0,5))
        self.reference_title = ttk.Label(toolbar, text="Reference", style="Caption.TLabel")
        self.reference_title.pack(side="left", padx=(0, 12))
        self.reference_var=tk.StringVar(value="No reference manifest configured")
        self.reference_buttons=[]
        self._load_reference_catalog()
        for label,operation in (("‹",lambda:self.move_reference(-1)),("›",lambda:self.move_reference(1)),("Random",self.random_reference),("Zoom",self.zoom_reference)):
            button=ttk.Button(toolbar,text=label,width=2 if label in {"‹","›"} else 0,command=operation,state="normal" if self.catalog else "disabled")
            button.pack(side="left",padx=2)
            self.reference_buttons.append(button)
        self.reference_counter = ttk.Label(toolbar, textvariable=self.reference_var, style="Muted.TLabel")
        self.reference_counter.pack(side="left", padx=8)
        self.reference_labels=self._image_pair(frame,"Episode start · never policy input",1,roles=("primary","wrist"))
        self.reference_panel = self.reference_images_panel
        self._sync_reference_visibility()
        self.show_reference()

    def _load_reference_catalog(self):
        self.catalog = None
        self.reference_images = []
        self._reference_error = None
        self.reference_var.set("")
        if self.session.config.reference_manifest:
            try:
                self.catalog = ReferenceCatalog(self.session.config.reference_manifest, self.session.config.task,
                                                self.session.config.instruction)
            except Exception as error:
                self._reference_error = str(error)
                self.reference_var.set("Unavailable · see Details")
        for button in self.reference_buttons:
            button.configure(state="normal" if self.catalog else "disabled")
        if hasattr(self, "reference_panel"):
            self._sync_reference_visibility()
        if hasattr(self,"reference_toggle"):
            self.reference_toggle.configure(state="normal" if self.catalog else "disabled")

    def _sync_reference_visibility(self):
        if self.catalog and self.reference_enabled.get():
            self.reference_panel.grid()
            self.visual_pane.set_visible("second", True, persist=False)
            self.reference_title.configure(text="Reference")
            for button in self.reference_buttons:
                button.pack(side="left", padx=2, before=self.reference_counter)
        else:
            self.reference_panel.grid_remove()
            self.visual_pane.set_visible("second", False, persist=False)
            self.reference_title.configure(text="No training reference selected" if not self._reference_error else "Training reference unavailable")
            for button in self.reference_buttons:
                button.pack_forget()

    def show_reference(self):
        if self.catalog:
            self.reference_images=self.catalog.images()
            self.reference_var.set(f"{self.catalog.index+1} / {len(self.catalog.samples)}")

    def move_reference(self,offset):
        if self.catalog:
            self.catalog.move(offset)
            self.show_reference()

    def random_reference(self):
        if self.catalog:
            self.catalog.randomize()
            self.show_reference()

    def zoom_reference(self):
        if not self.catalog:
            return
        window=tk.Toplevel(self.root)
        window.title("Training episode start")
        window.photos=[]
        for column,image in enumerate(self.reference_images):
            preview=image.copy()
            preview.thumbnail((700,600))
            photo=ImageTk.PhotoImage(preview,master=window)
            window.photos.append(photo)
            tk.Label(window,image=photo).grid(row=0,column=column)

    def zoom_live(self, role):
        with self.session.lock:
            latest = self.session.latest
            if latest is None or role not in latest.images:
                return False
            image = latest.images[role].copy()
            metadata = copy.deepcopy(latest.camera_metadata.get(role, {}))
        window = tk.Toplevel(self.root)
        window.title(f"{role.title()} · observation snapshot")
        preview = Image.fromarray(image)
        preview.thumbnail((1100, 800))
        window.photo = ImageTk.PhotoImage(preview, master=window)
        ttk.Label(window, image=window.photo).pack()
        ttk.Label(window, text="Snapshot only · live inference continues in the main window", padding=8).pack()
        ttk.Label(window, text=json.dumps(metadata, ensure_ascii=False), wraplength=1000, padding=8).pack()
        window.bind("<Escape>", lambda _event:(window.destroy(), "break")[-1])
        return True

    def show_help(self):
        if self._help_window is not None and self._help_window.winfo_exists():
            self._help_window.lift()
            return
        window = self._help_window = tk.Toplevel(self.root)
        window.title("Deployment controls")
        window.configure(bg=THEME["canvas"])
        text = ("N  Start: home, then a fresh rollout\n"
                "S / F  Mark Success / Failure, save, then return home\n"
                "X / R  Stop and discard this attempt; remain stationary\n"
                "H  Go to task home while ready\n"
                "W  Save experiment notes\n"
                "D  Confirm deletion of the last saved rollout\n"
                "Esc  Stop and close deployment\n\n"
                "Shortcuts are ignored while editing text or selecting a setting.\n"
                "Hold a key once; repeated key events do not repeat commands.\n\n"
                "Devices: refresh connected cameras, select each role, then Apply.\n"
                "Swap changes the draft only; it never moves the robot.\n"
                "Double-click a live image to inspect an observation snapshot.\n\n"
                "Rollouts: select a saved result to open its folder or video and\n"
                "add an annotation. Annotations never change the recorded outcome.\n\n"
                "Success timing is chosen by the operator. Recorded duration is\n"
                "the observed control interval, not an automatic task completion time.")
        ttk.Label(window, text=text, padding=SPACE["section"], justify="left").pack()
        ttk.Button(window, text="Close help", command=window.destroy).pack(pady=(0, SPACE["section"]))
        window.bind("<Escape>", lambda _event:(window.destroy(), "break")[-1])

    def start(self):
        if self.closing or self._reload_in_progress:
            return False
        if self.session.start_rollout():
            self.status.set("Moving to task home, then resetting policy and starting rollout")
            return True
        return False

    def signal(self,action):
        if self.session.signal(action):
            self.status.set(f"{action.title()} requested; cancelling pending policy action")

    def recover_robot(self):
        if self.closing or self._reload_in_progress:
            return False
        if self.session.recover_robot():
            self.status.set("Checking robot health and stop acknowledgement; no home or motion requested")
            return True
        return False

    def apply_settings(self):
        try:
            if self.closing or self._reload_in_progress:
                raise RuntimeError("Wait for checkpoint loading to finish")
            self.session.apply_settings(self.hz.get(),self.steps.get())
            recording = getattr(self.session, "apply_recording", None)
            if callable(recording):
                recording(bool(self.record_video.get()))
            self._selection_context.update(target_hz=self.session.config.target_hz, max_steps=self.session.config.max_steps)
            self._selection_context["record_video"] = self.session.config.record_video
            if isinstance(self._selection_context.get("resolved_config"), dict):
                self._selection_context["resolved_config"].update(target_hz=self.session.config.target_hz,
                    max_steps=self.session.config.max_steps, record_video=self.session.config.record_video)
            self.status.set("Runtime settings applied")
        except Exception as error:
            self.status.set(str(error))

    def schedule_notes(self,_event=None):
        if self.notes_job:
            self.root.after_cancel(self.notes_job)
        self.notes_job=self.root.after(500,self.save)

    def save(self):
        if self.notes_job:
            self.root.after_cancel(self.notes_job)
        self.notes_job=None
        try:
            self._remember_annotation()
            if not self._flush_annotations():
                return False
            self.session.save(self.notes.get("1.0","end-1c"))
            self._last_save_error = None
            self.status.set("Results and notes saved")
            return True
        except Exception as error:
            self._last_save_error = f"{type(error).__name__}: {error}"
            self.status.set("Could not save notes. Text remains in the window; inspect storage in Settings.")
            return False

    def delete(self):
        if not self.closing and not self._reload_in_progress and self.session.state=="READY" and self.session.results and messagebox.askyesno("Delete last result?","Delete the last saved result and its videos?",parent=self.root):
            identifier = self.session.results[-1]["id"]
            if self.session.delete_previous():
                self._annotation_drafts.pop(identifier, None)
                if self._annotation_result_id == identifier:
                    self._annotation_result_id = None
                    self._annotation_original = ""
                    self.result_annotation.set("")

    def _home_description(self):
        if callable(getattr(self.session.embodiment, "home_idle", None)):
            return "Home moves only the arm; gripper stays unchanged."
        return "Home uses the connected robot's configured home behavior."

    def home_robot(self):
        if self.closing or self._reload_in_progress or not self.session.can_home:
            return False
        if self.session.home_robot():
            self.status.set("Moving home. " + self._home_description() + " No rollout will start.")
            return True
        return False

    def keypress(self, event):
        key = event.keysym.lower()
        if key == "f1":
            self.show_help()
            return "break"
        widget = event.widget
        widget_class = widget.winfo_class() if hasattr(widget, "winfo_class") else ""
        if isinstance(widget, (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Combobox, ttk.Spinbox)) or widget_class in {"Entry", "TEntry", "Text", "Spinbox", "TSpinbox", "TCombobox", "Listbox"}:
            return
        # Shift/Ctrl/Alt/Super shortcuts belong to text editing and the desktop.
        # NumLock/CapsLock do not change this app's unmodified-key contract.
        if getattr(event, "state", 0) & 0xED:
            return
        pending = self._shortcut_release_jobs.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        if key in self._shortcut_down:
            return "break"
        actions = {"n": self.start, "h": self.home_robot, "x": lambda:self.signal("stop"),
                   "s": lambda:self.signal("success"), "f": lambda:self.signal("failure"),
                   "r": lambda:self.signal("retry"), "w": self.save, "d": self.delete, "escape": self.close}
        if key in actions:
            self._shortcut_down.add(key)
            actions[key]()
            return "break"

    def keyrelease(self, event):
        key = event.keysym.lower()
        pending = self._shortcut_release_jobs.pop(key, None)
        if pending is not None:
            self.root.after_cancel(pending)
        # X11 key repeat can emit a synthetic release/press pair. Deferring the
        # release lets the next press cancel it and keeps one physical hold inert.
        self._shortcut_release_jobs[key] = self.root.after(15, lambda:self._release_shortcut(key))

    def _release_shortcut(self, key):
        self._shortcut_release_jobs.pop(key, None)
        self._shortcut_down.discard(key)

    def _clear_shortcuts(self, _event=None):
        for pending in self._shortcut_release_jobs.values():
            self.root.after_cancel(pending)
        self._shortcut_release_jobs.clear()
        self._shortcut_down.clear()

    @staticmethod
    def _result_reason(result):
        category = result.get("failure_category") or result.get("category")
        labels = {"protective_stop": "Protective stop", "emergency_stop": "Emergency stop",
                  "robot_disconnected": "Connection lost", "robot_safety_stop": "Safety stop",
                  "robot_not_ready": "Robot not ready", "control_program_stopped": "Control stopped",
                  "ik_no_solution": "No joint solution", "ik_request_failed": "Joint solver failed",
                  "invalid_action": "Invalid action", "invalid_action_target": "Invalid target",
                  "servo_rejected": "Command rejected", "servo_command_failed": "Command failed",
                  "gripper_command_failed": "Gripper failed", "stop_unconfirmed": "Stop unconfirmed",
                  "stop_command_failed": "Stop failed", "policy_error": "Policy failed",
                  "state_feedback_error": "State feedback lost", "status_feedback_error": "Status unavailable",
                  "recording_error": "Recording failed", "robot_fault": "Robot fault",
                  "camera_error": "Camera failed", "camera_not_found": "Camera missing",
                  "camera_not_connected": "Camera disconnected", "camera_frame_timeout": "Camera timed out",
                  "camera_stale_frame": "Camera frame stalled", "camera_frame_invalid": "Invalid camera frame",
                  "camera_capture_failed": "Camera capture failed", "camera_profile_mismatch": "Unexpected camera image",
                  "camera_discovery_failed": "Camera discovery failed", "camera_profile_unsupported": "Camera mode unavailable",
                  "camera_connect_timeout": "Camera connection timed out", "camera_connect_failed": "Camera connection failed",
                  "camera_operation_pending": "Camera driver busy", "camera_stop_failed": "Camera stop failed",
                  "camera_change_failed": "Camera change failed"}
        if category:
            return labels.get(category, "Unexpected failure")
        return "Operator" if result.get("outcome") == "failure" else ""

    @classmethod
    def _result_note(cls, result):
        if result.get("failure_reason") or result.get("failure_note") or result.get("category"):
            text = cls._result_reason(result) + "."
            status = result.get("failure_status") or result.get("status") or {}
            motion = status.get("action_translation_norm_m") if isinstance(status, dict) else None
            if isinstance(motion, (int, float)):
                text += f" Requested movement: {motion*1000:.2f} mm."
            return text + "\nAvailable records saved · see Details."
        return "Marked as failure by the operator." if result.get("outcome") == "failure" else "Marked as success by the operator."

    def show_result_note(self, _event=None):
        result = self._selected_result()
        if result is not None:
            self.failure_note.set(self._result_note(result))
            source = str(result.get("outcome_source", "manual")).replace("_", " ")
            duration = result.get("active_duration_s")
            text = f"{source.title()} outcome · {result.get('steps', 0)} executed actions"
            if self._finite_number(duration):
                text += f" · {duration:.2f} s observed"
            home = result.get("outcome_home")
            if home:
                text += f"\nReturn home: {home}"
            self.result_summary.set(text)
            if self._annotation_result_id != result["id"]:
                self._remember_annotation()
                self._flush_annotations()
                self.result_annotation.set(self._annotation_drafts.get(result["id"], str(result.get("operator_note", ""))))
                self._annotation_result_id = result["id"]
                self._annotation_original = str(result.get("operator_note", ""))

    def _selected_result(self):
        selected = self.table.selection()
        if not selected:
            return None
        values = self.table.item(selected[0], "values")
        if not values:
            return None
        with self.session.lock:
            result = next((copy.deepcopy(row) for row in self.session.results if str(row["id"]) == str(values[0])), None)
        return result

    def _selected_result_folder(self):
        result = self._selected_result()
        if result is None:
            return None
        folder = self.session.folder.resolve()
        relative = Path(str(result.get("directory", "")))
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("Saved rollout directory is outside this session")
        selected = (folder / relative).resolve()
        if selected.parent != folder or not selected.is_dir():
            raise ValueError("Saved rollout directory is unavailable or outside this session")
        return selected

    def _launch_path(self, path):
        path = Path(path).resolve(strict=True)
        if sys.platform == "darwin":
            program = shutil.which("open")
        elif sys.platform == "win32":
            program = None
        else:
            program = shutil.which("xdg-open")
        if not program:
            self.status.set("No desktop opener is available. Use Copy path to open this record manually.")
            return False
        events = self._utility_events
        def launch():
            try:
                result = subprocess.run([program, str(path)], stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=8)
                if result.returncode:
                    events.put("The desktop could not open this record. Use Copy path instead.")
            except (OSError, subprocess.TimeoutExpired):
                events.put("The desktop opener did not respond. Use Copy path instead.")
        threading.Thread(target=launch, name="record-opener", daemon=True).start()
        self.status.set("Opening selected record in the desktop application")
        return True

    def open_result_folder(self):
        try:
            folder = self._selected_result_folder() or self.session.folder.resolve(strict=True)
            return self._launch_path(folder)
        except Exception as error:
            self.status.set(f"Cannot open result: {error}")
            return False

    def open_result_video(self):
        try:
            folder = self._selected_result_folder()
            role = self.video_role.get()
            if folder is None or role not in self.session.config.camera_roles:
                raise ValueError("Select a saved rollout and camera role first")
            path = (folder / f"{role}.mp4").resolve(strict=True)
            if path.parent != folder:
                raise ValueError("Video path is outside the saved rollout")
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError("The selected camera has no saved video")
            return self._launch_path(path)
        except Exception as error:
            self.status.set(f"Cannot open video: {error}")
            return False

    def copy_result_path(self):
        try:
            folder = self._selected_result_folder() or self.session.folder.resolve(strict=True)
            self.root.clipboard_clear()
            self.root.clipboard_append(str(folder))
            self.status.set("Selected result path copied")
            return True
        except Exception as error:
            self.status.set(f"Cannot copy result path: {error}")
            return False

    def save_result_annotation(self):
        if self.closing or self._reload_in_progress:
            return False
        self._remember_annotation()
        return self._flush_annotations()

    def _remember_annotation(self):
        if self._annotation_result_id is not None and self.result_annotation.get() != self._annotation_original:
            self._annotation_drafts[self._annotation_result_id] = self.result_annotation.get()

    def schedule_annotation(self, _event=None):
        self._remember_annotation()
        if self._annotation_job:
            self.root.after_cancel(self._annotation_job)
        self._annotation_job = self.root.after(500, self._flush_annotations)

    def _flush_annotations(self):
        if self._annotation_job:
            self.root.after_cancel(self._annotation_job)
            self._annotation_job = None
        if not self._annotation_drafts:
            return True
        operation = getattr(self.session, "annotate_result", None)
        if not callable(operation):
            return False
        try:
            for identifier, note in list(self._annotation_drafts.items()):
                if not operation(identifier, note):
                    raise RuntimeError("Saved rollout is unavailable for annotation")
                self._annotation_drafts.pop(identifier, None)
                if identifier == self._annotation_result_id:
                    self._annotation_original = note
            self.status.set("Rollout annotation saved; its outcome is unchanged")
            return True
        except Exception as error:
            self.status.set(f"Could not save annotation: {error}")
        return False

    def _show_images(self,images,labels):
        for image,label in zip(images,labels):
            preview=image.copy() if isinstance(image,Image.Image) else Image.fromarray(image)
            width,height=max(20,label.winfo_width()),max(20,label.winfo_height())
            factor=min(width/preview.width,height/preview.height)
            size=(max(1,round(preview.width*factor)),max(1,round(preview.height*factor)))
            if size!=preview.size:
                preview=preview.resize(size,Image.Resampling.BILINEAR)
            photo=ImageTk.PhotoImage(preview,master=self.root)
            self.photos.append(photo)
            label.configure(image=photo,text="")

    def refresh(self):
        try:
            while True:
                self.status.set(self._utility_events.get_nowait())
        except queue.Empty:
            pass
        try:
            while True:
                event=self.session.events.get_nowait()
                if event.get("detail"):
                    self._event_detail = str(event["detail"])
                    if self.session.state not in {"ERROR", "RECOVERY_REQUIRED", "RECOVERING", "LOADING_POLICY", "CONNECTING", "CONNECTING_PREVIEW"}:
                        self.status.set(event["detail"])
        except queue.Empty:
            pass
        self._reload_updates()
        with self.session.lock:
            state=self.session.state
            rows=list(self.session.rows)
            results=list(self.session.results)
            latest=self.session.latest
        health = self._update_device_status(latest)
        policy_metadata = getattr(self.session.policy, "metadata", {})
        if state == "SELECTING":
            self.active_identity.set("No policy loaded · camera preview only")
        else:
            identity = [self._display_method()]
            mode = policy_metadata.get("gate_mode") or self.session.config.metadata.get("gate_mode")
            calibrated = policy_metadata.get("gate_calibrated", self.session.config.metadata.get("gate_calibrated"))
            if mode and mode != "fixed_chunk":
                identity.append(str(mode).replace("_", " ").upper())
                identity.append("Calibrated" if calibrated else "Uncalibrated candidate")
            self.active_identity.set(" · ".join(identity))
        active_method="No policy loaded" if state=="SELECTING" else self._display_method()
        if active_method!=self.active_method:
            self.active_method=active_method
            self._schedule_layout()
        if state == "ERROR" and self.session._startup_error and self._load_error_session is not self.session:
            self._load_error_session = self.session
            self._show_load_error(self.session.error, getattr(self.session, "error_detail", None))
        displayed_state = "LOADING" if self._reload_in_progress and not self.closing else state
        state_labels = {"SELECTING": "Choose policy", "READY": "Ready", "LOADING": "Loading policy",
                        "LOADING_POLICY": "Loading policy", "CONNECTING_PREVIEW": "Opening cameras",
                        "CONNECTING": "Connecting", "RECOVERY_REQUIRED": "Recovery needed",
                        "RECOVERING": "Checking robot", "WAITING_OUTCOME": "Choose outcome",
                        "HOMING": "Going home", "STARTING": "Starting", "RUNNING": "Running",
                        "STOPPING": "Stopping", "ERROR": "Stopped with error", "CLOSED": "Closed",
                        "CAMERA_ERROR": "Camera unavailable", "RECONFIGURING": "Changing cameras"}
        self.state.set(state_labels.get(displayed_state, displayed_state.replace("_", " ").capitalize()))
        self.state_label.configure(style=("Ready.State.TLabel" if displayed_state == "READY" else
                                         "Fault.State.TLabel" if displayed_state in {"ERROR", "RECOVERY_REQUIRED", "CAMERA_ERROR"} else
                                         "Pending.State.TLabel" if displayed_state not in {"SELECTING", "CLOSED"} else "State.TLabel"))
        if state == "ERROR" and not self._reload_in_progress:
            self.status.set("Deployment stopped; records kept. Open Details before retrying.")
        elif state == "RECOVERY_REQUIRED":
            self.status.set("Robot fault: rollout stopped and its record is preserved. Resolve the condition, then Recover robot.")
        elif state == "RECOVERING":
            self.status.set("Checking robot health and confirmed stop. No motion is requested.")
        elif state == "LOADING_POLICY":
            self.status.set("Loading the selected policy. Camera preview remains available.")
        elif state == "CONNECTING_PREVIEW":
            self.status.set("Opening camera preview. Policy and robot control remain disconnected.")
        elif state == "SELECTING":
            self.status.set("Select a task and checkpoint, then Load checkpoint.")
        elif state == "CAMERA_ERROR":
            self.status.set("Camera unavailable. Open Devices to change assignments or restart preview; the loaded policy is preserved.")
        elif state == "RECONFIGURING":
            self.status.set("Changing camera streams. No motion is requested; Start remains a separate action.")
        elif state == "READY" and self._last_displayed_state in {None,"CONNECTING","LOADING_POLICY","CONNECTING_PREVIEW","SELECTING"}:
            self.status.set("Ready. Start prepares a fresh rollout.")
        self._last_displayed_state = state
        ready=state=="READY" and not self.closing and not self._reload_in_progress
        recovery=state in {"RECOVERY_REQUIRED", "RECOVERING"} or self.session._recovery_fault
        active=state in {"STARTING","HOMING","RUNNING","WAITING_OUTCOME","STOPPING", "RECOVERY_REQUIRED", "RECOVERING"}
        self.start_button.configure(state="normal" if ready and health.get("can_start", True) else "disabled")
        for button in (self.stop_button,self.retry_button):
            button.configure(state="normal" if active else "disabled")
        self.recover_button.configure(text="Checking robot..." if state == "RECOVERING" else "Recover robot",
                                      state="normal" if self.session.can_recover and not self.closing and not self._reload_in_progress else "disabled")
        home_allowed = self.session.can_home and not self.closing and not self._reload_in_progress
        self.home_button.configure(text="Going home..." if state == "HOMING" and not self.session._rollout_active else "Home [H]",
                                   state="normal" if home_allowed else "disabled")
        automatic_notes = self.session.automatic_notes
        latest_failure = automatic_notes[-1] if automatic_notes else None
        if recovery:
            reason = self._result_reason(latest_failure) if latest_failure else "Robot fault"
            self.control_hint.configure(text=f"{reason}. Resolve the robot condition.\nThen Recover robot before Home / Start.", foreground=THEME["danger"])
        elif state == "HOMING":
            self.control_hint.configure(text="Moving to task home. Stop cancels the move.\n" + self._home_description(), foreground=THEME["muted"])
        elif ready and not health.get("can_start", True):
            self.control_hint.configure(text="Start unavailable: " + str(health.get("reason") or "Check device and storage status") + "\nOpen Devices / Settings to resolve it.", foreground=THEME["warning"])
        else:
            self.control_hint.configure(text="Start homes; Home is an idle move.",foreground=THEME["muted"])
        if recovery or state=="HOMING" or ready and not health.get("can_start",True):
            self.control_hint.pack(anchor="w",fill="x",padx=SPACE["control"],pady=(SPACE["small"],0),before=self.measurement_label)
        else:
            self.control_hint.pack_forget()
        for button in (self.success_button,self.failure_button):
            button.configure(state="normal" if state in {"RUNNING","WAITING_OUTCOME"} else "disabled")
        for widget in (self.apply_button,self.hz_entry,self.steps_entry,self.record_video_toggle):
            widget.configure(state="normal" if ready else "disabled")
        self.delete_button.configure(state="normal" if ready and results else "disabled")
        if self.reload_fields:
            self.checkpoint_toggle.configure(state="disabled" if self.closing else "normal")
            allowed = not self.closing and not self._reload_in_progress and not self._reload_blocked and self.session.can_replace
            for widget, enabled_state in self.reload_widgets:
                widget.configure(state=enabled_state if allowed else "disabled")
            self.load_button.configure(state="normal" if allowed else "disabled")
        self.notes.configure(state="disabled" if self.closing or self._reload_in_progress else "normal")
        annotation_allowed = not self.closing and not self._reload_in_progress and bool(self.table.selection()) and callable(getattr(self.session, "annotate_result", None))
        self.annotation_entry.configure(state="normal" if annotation_allowed else "disabled")
        self.annotation_button.configure(state="normal" if annotation_allowed else "disabled")
        self.open_video_button.configure(state="normal" if self.table.selection() else "disabled")
        stats=summary(rows)
        latency=stats["latency"]["policy_ms"]
        hz=stats["achieved_command_hz"]
        self.metrics.set(f"Target {self.session.config.target_hz:g} Hz   ·   Actual {hz:.2f} Hz" if hz is not None else f"Target {self.session.config.target_hz:g} Hz   ·   Actual —")
        self.metrics.set(self.metrics.get() + f"   ·   Actions {stats['steps']}   ·   {stats['active_duration_s']:.1f} s")
        measurement_content = (self.control_hint.cget("text"),self.metrics.get())
        if measurement_content != self._measurement_layout_seen:
            self._measurement_layout_seen = measurement_content
            self._schedule_layout()
        self._update_scoreboard(results)
        rendered = [(r["id"], r["outcome"], r["steps"], f"{r['achieved_command_hz']:.1f}" if r["achieved_command_hz"] is not None else "—", self._result_reason(r)) for r in results]
        if rendered != self._rows_seen:
            previous_selection = self.table.selection()
            previous_id = self.table.item(previous_selection[0], "values")[0] if previous_selection else None
            self.table.delete(*self.table.get_children())
            selected_item = None
            for row in rendered:
                item = self.table.insert("", "end", values=row, tags=(row[1],))
                if str(row[0]) == str(previous_id):
                    selected_item = item
            if selected_item:
                self.table.selection_set(selected_item)
                self.show_result_note()
            elif rendered:
                latest_item = self.table.get_children()[-1]
                self.table.selection_set(latest_item)
                self.show_result_note()
            else:
                self.failure_note.set("Select a saved rollout to view its result.")
                self.result_summary.set("")
                self.result_annotation.set("")
                self._annotation_result_id = None
                self._annotation_original = ""
            self._rows_seen = rendered
        failure_notes = "\n\n".join((f"Rollout {note['rollout_id']}: " if note.get("rollout_id") is not None else "Outside rollout: ") + self._result_note(note) for note in automatic_notes) or "No automatic failures recorded."
        if failure_notes != self._failure_notes_seen:
            self.failure_notes.configure(state="normal")
            self.failure_notes.delete("1.0", "end")
            self.failure_notes.insert("1.0", failure_notes)
            self.failure_notes.configure(state="disabled")
            self._failure_notes_seen = failure_notes
        details = json.dumps({"policy": self.session.policy.metadata, "embodiment": self.session.embodiment.metadata,
                              "selection": self.session.config.metadata.get("reload_selection", {}),
                              "results": str(self.session.folder), "latest_status_detail": self._event_detail,
                              "error": self.session.error, "load_error": self._load_error_detail,
                              "save_error": self._last_save_error, "device_status": getattr(self.session, "device_status", {}),
                              "operator_health": health, "selected_rollout": self._selected_result(),
                              "failure_results": [r for r in results if r.get("failure_reason")],
                              "automatic_notes": automatic_notes,
                              "reference_error": self._reference_error,
                              "measurements": stats}, indent=2)
        if details != self._details_seen:
            self.details.configure(state="normal")
            self.details.delete("1.0", "end")
            self.details.insert("1.0", details)
            self.details.configure(state="disabled")
            self._details_seen = details
        self.photos=[]
        if latest:
            images = [latest.images.get(r) for r in self.session.config.camera_roles]
            if all(image is not None for image in images):
                self._show_images(images,self.camera_labels)
        if self.reference_images:
            self._show_images(self.reference_images,self.reference_labels)
        if (self.closing and self.session.shutdown_complete and not self._reload_in_progress
                and (self._reload_cleanup_session is None or self._reload_cleanup_session.shutdown_complete)
                and (self._reload_thread is None or not self._reload_thread.is_alive())):
            self.root.destroy()
            return
        self.root.after(100,self.refresh)

    def close(self):
        if self.closing:
            return
        self.save()
        self._save_preferences()
        for pane in self._workspace_panes:
            pane.close()
        for job_name in ("_layout_job", "_preferences_job"):
            job = getattr(self, job_name, None)
            if job is not None:
                self.root.after_cancel(job)
                setattr(self, job_name, None)
        self.closing=True
        self._clear_shortcuts()
        self._reload_cancel.set()
        self.status.set("Stopping policy commands, finalizing recordings and closing devices")
        self.session.close()
