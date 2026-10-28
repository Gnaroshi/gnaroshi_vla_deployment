"""Desktop GUI; the session worker owns model and robot operations."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from .reference import ReferenceCatalog
from .records import summary, write_json
from .session import DeploymentSession


class DeploymentGUI:
    def __init__(self, root, session, *, reload_fields=None, reload_factory=None,
                 selection_resolver=None, config_loader=None, config_saver=None, config_directory=None):
        self.root, self.session = root, session
        self.reload_fields = tuple(reload_fields or ())
        self.reload_factory = reload_factory
        self.selection_resolver = selection_resolver
        self.config_loader, self.config_saver = config_loader, config_saver
        self.config_directory = Path(config_directory).expanduser() if config_directory is not None else None
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
        root.title(f"{session.config.method} | {session.config.task} deployment")
        root.geometry("1440x960")
        root.minsize(1100, 780)
        root.configure(bg="#f4f5f6")
        root.option_add("*Font", "{DejaVu Sans} 11")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure("TFrame", background="#f4f5f6")
        style.configure("TLabel", background="#f4f5f6", foreground="#24272b")
        style.configure("TButton", padding=(12, 9))
        style.configure("TButton",background="#ffffff",foreground="#24272b",bordercolor="#d5dade",lightcolor="#ffffff",darkcolor="#ffffff")
        style.configure("TLabelframe",background="#ffffff",bordercolor="#dde2e6",relief="solid")
        style.configure("TLabelframe.Label",background="#f4f5f6",foreground="#59665f",font=("DejaVu Sans",10))
        style.configure("Accent.TButton", background="#23664e", foreground="white")
        style.configure("Stop.TButton", background="#b33d46", foreground="white")
        style.configure("Treeview", rowheight=27, font=("DejaVu Sans", 10),background="#ffffff",fieldbackground="#ffffff")
        self.state = tk.StringVar(value="CONNECTING")
        self.status = tk.StringVar(value="Opening selected policy and embodiment")
        self.metrics = tk.StringVar(value="No rollout measurements")
        self.counts = tk.StringVar(value="Saved: 0   Success: 0   Failure: 0   SR: —")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        heading = ttk.Frame(root, padding=18)
        heading.grid(row=0, sticky="ew")
        heading.columnconfigure(0, weight=1)
        self.heading_label = ttk.Label(heading, text=f"{session.config.task.title()}  /  {session.config.method}", font=("DejaVu Sans", 18, "bold"))
        self.heading_label.grid(row=0, column=0, sticky="w")
        ttk.Label(heading, textvariable=self.state, font=("DejaVu Sans", 12, "bold")).grid(row=0, column=1, padx=16)
        if self.reload_fields:
            self.checkpoint_toggle = ttk.Button(heading, text="Task / checkpoint...", command=self.toggle_checkpoint_panel)
            self.checkpoint_toggle.grid(row=0, column=2, padx=(0, 8))
        ttk.Button(heading, text="Exit", command=self.close).grid(row=0, column=3 if self.reload_fields else 2)
        self.instruction_label = ttk.Label(heading, text=session.config.instruction, wraplength=1200)
        self.instruction_label.grid(row=1, column=0, columnspan=4 if self.reload_fields else 3, sticky="w", pady=(12, 0))
        if self.reload_fields:
            self._checkpoint_panel(heading)
        body = ttk.Frame(root, padding=(18,0,18,12))
        body.grid(row=1, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=380)
        body.rowconfigure(0, weight=1)
        main = ttk.Frame(body)
        main.grid(row=0, column=0, sticky="nsew", padx=(0,18))
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=2)
        main.rowconfigure(1, weight=1)
        self.camera_labels = self._image_pair(main, "Live policy observations", row=0)
        self._references(main)
        side = ttk.Frame(body)
        side.grid(row=0, column=1, sticky="nsew")
        side.columnconfigure(0, weight=1)
        side.rowconfigure(4, weight=1)
        controls = ttk.LabelFrame(side, text="Rollout", padding=10)
        controls.grid(row=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        controls.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(controls, text="Start [N] · home first", style="Accent.TButton", command=self.start, state="disabled")
        self.start_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,6))
        self.stop_button = ttk.Button(controls, text="Stop & discard [X]", style="Stop.TButton", command=lambda:self.signal("stop"), state="disabled")
        self.stop_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,6))
        self.success_button = ttk.Button(controls, text="Success [S]", command=lambda:self.signal("success"), state="disabled")
        self.success_button.grid(row=2, column=0, sticky="ew", padx=(0,3))
        self.failure_button = ttk.Button(controls, text="Failure [F]", command=lambda:self.signal("failure"), state="disabled")
        self.failure_button.grid(row=2, column=1, sticky="ew", padx=(3,0))
        self.retry_button = ttk.Button(controls, text="Stop for retry [R]", command=lambda:self.signal("retry"), state="disabled")
        self.retry_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6,0))
        self.recover_button = ttk.Button(controls, text="Recover robot", command=self.recover_robot, state="disabled")
        self.recover_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6,0))
        self.recover_button.grid_remove()
        self.control_hint = ttk.Label(controls, text="Success / Failure: save then return home\nStop / Retry: discard and remain stationary", font=("DejaVu Sans", 9))
        self.control_hint.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8,0))
        settings = ttk.LabelFrame(side, text="Runtime · change while ready", padding=10)
        settings.grid(row=1, sticky="ew", pady=10)
        self.hz = tk.StringVar(value=str(session.config.target_hz))
        self.steps = tk.StringVar(value=str(session.config.max_steps))
        ttk.Label(settings, text="Target Hz").grid(row=0,column=0,sticky="w")
        ttk.Label(settings, text="Max steps").grid(row=1,column=0,sticky="w",pady=6)
        self.hz_entry = ttk.Entry(settings, textvariable=self.hz, width=12)
        self.hz_entry.grid(row=0,column=1,padx=8)
        self.steps_entry = ttk.Entry(settings, textvariable=self.steps, width=12)
        self.steps_entry.grid(row=1,column=1,padx=8)
        self.apply_button = ttk.Button(settings, text="Apply", command=self.apply_settings, state="disabled")
        self.apply_button.grid(row=0,column=2,rowspan=2)
        metrics = ttk.LabelFrame(side, text="Measured rollout", padding=10)
        metrics.grid(row=2,sticky="ew")
        ttk.Label(metrics,textvariable=self.metrics,justify="left",font=("DejaVu Sans",11)).pack(anchor="w")
        ttk.Label(side,textvariable=self.counts,wraplength=375).grid(row=3,sticky="ew",pady=10)
        tabs = ttk.Notebook(side)
        tabs.grid(row=4,sticky="nsew")
        results, notes, details = (ttk.Frame(tabs,padding=6) for _ in range(3))
        for frame,title in ((results,"Rollouts"),(notes,"Notes"),(details,"Details")):
            tabs.add(frame,text=title)
        results.columnconfigure(0,weight=1)
        results.rowconfigure(0,weight=1)
        self.table = ttk.Treeview(results,columns=("run","result","steps","hz"),show="headings",height=5)
        for column,title,width in (("run","Run",42),("result","Outcome",85),("steps","Steps",62),("hz","Hz",62)):
            self.table.heading(column,text=title)
            self.table.column(column,width=width,anchor="center")
        self.table.grid(sticky="nsew")
        notes.columnconfigure(0,weight=1)
        notes.rowconfigure(0,weight=1)
        self.notes = tk.Text(notes,wrap="word",width=24,height=6,relief="flat",padx=8,pady=8)
        self.notes.grid(sticky="nsew")
        self.notes.bind("<KeyRelease>",self.schedule_notes)
        self.details = tk.Text(details,wrap="word",width=24,height=6,relief="flat",font=("DejaVu Sans",9))
        self.details.pack(fill="both",expand=True)
        records = ttk.Frame(side)
        records.grid(row=5,sticky="ew",pady=(10,0))
        ttk.Button(records,text="Save [W]",command=self.save).pack(side="left")
        self.delete_button = ttk.Button(records,text="Delete last [D]",command=self.delete,state="disabled")
        self.delete_button.pack(side="left",padx=6)
        ttk.Label(root,textvariable=self.status,wraplength=1350,padding=(18,6)).grid(row=2,sticky="ew")
        self.folder_label = ttk.Label(root,text=str(session.folder),padding=(18,0,18,12),font=("DejaVu Sans",9))
        self.folder_label.grid(row=3,sticky="ew")
        root.bind("<KeyPress>",self.keypress)
        root.protocol("WM_DELETE_WINDOW",self.close)
        self._rows_seen = None
        self._details_seen = None
        session.start()
        root.after(80,self.refresh)

    def _checkpoint_panel(self, parent):
        self.checkpoint_panel = panel = ttk.LabelFrame(parent, text="Policy selection · load while stopped", padding=8)
        panel.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(12, 0))
        panel.columnconfigure(1, weight=1)
        selected = self.session.config.metadata.get("reload_selection", {})
        for row, field in enumerate(self.reload_fields):
            key, kind = field["key"], field.get("kind", "text")
            value = selected.get(key, field.get("value", ""))
            self.reload_vars[key] = variable = tk.StringVar(value=str(value))
            ttk.Label(panel, text=field.get("label", key)).grid(row=row, column=0, sticky="w", padx=(0, 8))
            if kind == "choice":
                widget = ttk.Combobox(panel, textvariable=variable, values=field.get("choices", ()), state="readonly")
                widget.bind("<<ComboboxSelected>>", lambda _event, key=key:self.resolve_selection(key))
            else:
                widget = ttk.Entry(panel, textvariable=variable)
            widget.grid(row=row, column=1, sticky="ew", pady=2)
            self.reload_widgets.append((widget, "readonly" if kind == "choice" else "normal"))
            if kind in {"directory", "file"}:
                button = ttk.Button(panel, text="Browse", command=lambda field=field:self.browse_checkpoint(field))
                button.grid(row=row, column=2, padx=(6, 0))
                self.reload_widgets.append((button, "normal"))
        self.load_button = ttk.Button(panel, text="Load checkpoint", command=self.load_checkpoint, state="disabled")
        self.load_button.grid(row=0, column=3, rowspan=max(1, len(self.reload_fields)), padx=(10, 0), sticky="ns")
        files = ttk.Frame(panel)
        files.grid(row=len(self.reload_fields), column=0, columnspan=4, sticky="w", pady=(6, 0))
        for title, callback, enabled in (("Load config JSON", self.load_config, self.config_loader),
                                         ("Save config JSON", self.save_config, self.config_saver)):
            if enabled:
                button = ttk.Button(files, text=title, command=callback)
                button.pack(side="left", padx=(0, 6))
                self.reload_widgets.append((button, "normal"))
        self.load_error = tk.Text(panel, height=4, wrap="word", relief="flat", font=("DejaVu Sans", 9))
        self.load_error.grid(row=len(self.reload_fields)+1, column=0, columnspan=4, sticky="ew", pady=(6, 0))
        self.load_error.configure(state="disabled")
        self.load_error.grid_remove()
        self._record_selection(self.session, self.current_selection())
        self._set_checkpoint_expanded(bool(self.session.config.metadata.get("defer_policy_load", False)))

    def _set_checkpoint_expanded(self, expanded):
        self._checkpoint_expanded = expanded
        if expanded:
            self.checkpoint_panel.grid()
        else:
            self.checkpoint_panel.grid_remove()
        self.checkpoint_toggle.configure(text="Hide task / checkpoint" if expanded else "Task / checkpoint...")

    def toggle_checkpoint_panel(self):
        if not self.closing:
            self._set_checkpoint_expanded(not self._checkpoint_expanded)

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
        self.status.set(message)
        if self.reload_fields:
            self.load_error.configure(state="normal")
            self.load_error.delete("1.0", "end")
            self.load_error.insert("1.0", detail or message)
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
        self.save()
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
        self.root.title(f"{session.config.method} | {session.config.task} deployment")
        self.heading_label.configure(text=f"{session.config.task.title()}  /  {session.config.method}")
        self.instruction_label.configure(text=session.config.instruction)
        self.folder_label.configure(text=str(session.folder))
        self.hz.set(str(session.config.target_hz))
        self.steps.set(str(session.config.max_steps))
        self._apply_selection(session.config.metadata.get("reload_selection", {}))
        self._set_checkpoint_expanded(False)
        self.notes.configure(state="normal")
        self.notes.delete("1.0", "end")
        self._rows_seen, self._details_seen = None, None
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
        panel = ttk.LabelFrame(parent,text=title,padding=8)
        panel.grid(row=row,sticky="nsew",pady=(0,10))
        roles=roles or self.session.config.camera_roles
        for column in range(len(roles)):
            panel.columnconfigure(column,weight=1)
        panel.rowconfigure(1,weight=1)
        labels=[]
        for column,role in enumerate(roles):
            ttk.Label(panel,text=role.title()).grid(row=0,column=column,sticky="w",pady=(0,4))
            label=tk.Label(panel,bg="#202b36",fg="#d0d6dc",text="Waiting for frame",width=1,height=1)
            label.grid(row=1,column=column,sticky="nsew",padx=(0,4) if column==0 else (4,0))
            labels.append(label)
        return labels

    def _references(self,parent):
        frame=ttk.Frame(parent)
        frame.grid(row=1,sticky="nsew")
        frame.columnconfigure(0,weight=1)
        frame.rowconfigure(1,weight=1)
        toolbar=ttk.Frame(frame)
        toolbar.grid(row=0,sticky="ew",pady=(0,5))
        ttk.Label(toolbar,text="Training start reference").pack(side="left",padx=(0,12))
        self.reference_var=tk.StringVar(value="No reference manifest configured")
        self.reference_buttons=[]
        self._load_reference_catalog()
        for label,operation in (("‹",lambda:self.move_reference(-1)),("›",lambda:self.move_reference(1)),("Random",self.random_reference),("Zoom",self.zoom_reference)):
            button=ttk.Button(toolbar,text=label,width=7,command=operation,state="normal" if self.catalog else "disabled")
            button.pack(side="left",padx=2)
            self.reference_buttons.append(button)
        ttk.Label(toolbar,textvariable=self.reference_var).pack(side="left",padx=8)
        self.reference_labels=self._image_pair(frame,"Episode start · never policy input",1,roles=("primary","wrist"))
        self.show_reference()

    def _load_reference_catalog(self):
        self.catalog = None
        self.reference_images = []
        self.reference_var.set("No reference manifest configured")
        if self.session.config.reference_manifest:
            try:
                self.catalog = ReferenceCatalog(self.session.config.reference_manifest, self.session.config.task,
                                                self.session.config.instruction)
            except Exception as error:
                self.reference_var.set(str(error))
        for button in self.reference_buttons:
            button.configure(state="normal" if self.catalog else "disabled")

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
            self._selection_context.update(target_hz=self.session.config.target_hz, max_steps=self.session.config.max_steps)
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
        self.session.save(self.notes.get("1.0","end-1c"))
        self.status.set("Results and notes saved")

    def delete(self):
        if not self.closing and not self._reload_in_progress and self.session.state=="READY" and self.session.results and messagebox.askyesno("Delete last result?","Delete the last saved result and its videos?",parent=self.root):
            self.session.delete_previous()

    def keypress(self,event):
        key=event.keysym.lower()
        if isinstance(event.widget,(tk.Entry,tk.Text,tk.Spinbox,ttk.Entry,ttk.Combobox)):
            return
        actions={"n":self.start,"x":lambda:self.signal("stop"),"s":lambda:self.signal("success"),"f":lambda:self.signal("failure"),"r":lambda:self.signal("retry"),"w":self.save,"d":self.delete,"escape":self.close}
        if key in actions:
            actions[key]()

    def _show_images(self,images,labels):
        for image,label in zip(images,labels):
            preview=image.copy() if isinstance(image,Image.Image) else Image.fromarray(image)
            preview.thumbnail((max(20,label.winfo_width()),max(20,label.winfo_height())))
            photo=ImageTk.PhotoImage(preview,master=self.root)
            self.photos.append(photo)
            label.configure(image=photo,text="")

    def refresh(self):
        try:
            while True:
                event=self.session.events.get_nowait()
                if event.get("detail"):
                    self.status.set(event["detail"])
        except queue.Empty:
            pass
        self._reload_updates()
        with self.session.lock:
            state=self.session.state
            rows=list(self.session.rows)
            results=list(self.session.results)
            latest=self.session.latest
        if state == "ERROR" and self.session._startup_error and self._load_error_session is not self.session:
            self._load_error_session = self.session
            self._show_load_error(self.session.error, getattr(self.session, "error_detail", None))
        self.state.set("LOADING" if self._reload_in_progress and not self.closing else state)
        ready=state=="READY" and not self.closing and not self._reload_in_progress
        recovery=state in {"RECOVERY_REQUIRED", "RECOVERING"} or self.session._recovery_fault
        active=state in {"STARTING","HOMING","RUNNING","WAITING_OUTCOME","STOPPING", "RECOVERY_REQUIRED", "RECOVERING"}
        self.start_button.configure(state="normal" if ready else "disabled")
        for button in (self.stop_button,self.retry_button):
            button.configure(state="normal" if active else "disabled")
        if recovery:
            self.retry_button.grid_remove()
            self.recover_button.grid()
        else:
            self.recover_button.grid_remove()
            self.retry_button.grid()
        self.recover_button.configure(text="Checking robot..." if state == "RECOVERING" else "Recover robot",
                                      state="normal" if self.session.can_recover and not self.closing else "disabled")
        self.control_hint.configure(text="Release fault on teach pendant, then Recover robot\nRecovery stays stationary; Start is separate" if recovery else
                                    "Success / Failure: save then return home\nStop / Retry: discard and remain stationary")
        for button in (self.success_button,self.failure_button):
            button.configure(state="normal" if state in {"RUNNING","WAITING_OUTCOME"} else "disabled")
        for widget in (self.apply_button,self.hz_entry,self.steps_entry):
            widget.configure(state="normal" if ready else "disabled")
        self.delete_button.configure(state="normal" if ready and results else "disabled")
        if self.reload_fields:
            self.checkpoint_toggle.configure(state="disabled" if self.closing else "normal")
            allowed = not self.closing and not self._reload_in_progress and not self._reload_blocked and self.session.can_replace
            for widget, enabled_state in self.reload_widgets:
                widget.configure(state=enabled_state if allowed else "disabled")
            self.load_button.configure(state="normal" if allowed else "disabled")
        self.notes.configure(state="disabled" if self.closing or self._reload_in_progress else "normal")
        stats=summary(rows)
        latency=stats["latency"]["policy_ms"]
        hz=stats["achieved_command_hz"]
        self.metrics.set(f"Target {self.session.config.target_hz:g} Hz   ·   Actual {hz:.2f} Hz" if hz is not None else f"Target {self.session.config.target_hz:g} Hz   ·   Actual —")
        self.metrics.set(self.metrics.get()+f"\nSteps {stats['steps']}   ·   Active {stats['active_duration_s']:.1f} s\nPolicy / step {latency['mean_ms']:.1f} ms (mean)" if latency else self.metrics.get()+"\nNo policy latency measurements")
        rpc=stats["latency"]["rpc_ms"]
        if rpc:
            self.metrics.set(self.metrics.get()+f"\nRPC {rpc['mean_ms']:.1f} ms · {rpc['count']} requests")
        calls=stats["full_calls"] if stats["full_calls"] is not None else "—"
        self.metrics.set(self.metrics.get()+f"\nBackbone calls {calls}   ·   Buffered {stats['buffered_steps']}")
        successes=sum(r["outcome"]=="success" for r in results)
        rate=f"{100*successes/len(results):.1f}%" if results else "—"
        self.counts.set(f"Saved {len(results)}  ·  Success {successes}  ·  Failure {len(results)-successes}  ·  SR {rate}")
        rendered=[(r["id"],r["outcome"],r["steps"],f"{r['achieved_command_hz']:.1f}" if r["achieved_command_hz"] is not None else "—") for r in results]
        if rendered!=self._rows_seen:
            self.table.delete(*self.table.get_children())
            for row in rendered:
                self.table.insert("","end",values=row)
            self._rows_seen=rendered
        details=json.dumps({"policy":self.session.policy.metadata,"embodiment":self.session.embodiment.metadata,
                            "selection":self.session.config.metadata.get("reload_selection", {}),
                            "results":str(self.session.folder)},indent=2)
        if details!=self._details_seen:
            self.details.configure(state="normal")
            self.details.delete("1.0","end")
            self.details.insert("1.0",details)
            self.details.configure(state="disabled")
            self._details_seen=details
        self.photos=[]
        if latest:
            self._show_images([latest.images[r] for r in self.session.config.camera_roles],self.camera_labels)
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
        self.closing=True
        self._reload_cancel.set()
        self.status.set("Stopping policy commands, finalizing recordings and closing devices")
        self.session.close()
