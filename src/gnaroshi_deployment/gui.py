"""Desktop GUI; the session worker owns model and robot operations."""
from __future__ import annotations
import json
import queue
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk
from .reference import ReferenceCatalog
from .records import summary


class DeploymentGUI:
    def __init__(self, root, session):
        self.root, self.session = root, session
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
        ttk.Label(heading, text=f"{session.config.task.title()}  /  {session.config.method}", font=("DejaVu Sans", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(heading, textvariable=self.state, font=("DejaVu Sans", 12, "bold")).grid(row=0, column=1, padx=16)
        ttk.Button(heading, text="Exit", command=self.close).grid(row=0, column=2)
        ttk.Label(heading, text=session.config.instruction, wraplength=1200).grid(row=1, column=0, columnspan=3, sticky="w", pady=(12, 0))
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
        self.start_button = ttk.Button(controls, text="Start [N] · home first", style="Accent.TButton", command=self.start)
        self.start_button.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0,6))
        self.stop_button = ttk.Button(controls, text="Stop & discard [X]", style="Stop.TButton", command=lambda:self.signal("stop"))
        self.stop_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,6))
        self.success_button = ttk.Button(controls, text="Success [S]", command=lambda:self.signal("success"))
        self.success_button.grid(row=2, column=0, sticky="ew", padx=(0,3))
        self.failure_button = ttk.Button(controls, text="Failure [F]", command=lambda:self.signal("failure"))
        self.failure_button.grid(row=2, column=1, sticky="ew", padx=(3,0))
        self.retry_button = ttk.Button(controls, text="Stop for retry [R]", command=lambda:self.signal("retry"))
        self.retry_button.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6,0))
        ttk.Label(controls, text="Success / Failure: save then return home\nStop / Retry: discard and remain stationary", font=("DejaVu Sans", 9)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8,0))
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
        self.apply_button = ttk.Button(settings, text="Apply", command=self.apply_settings)
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
        self.delete_button = ttk.Button(records,text="Delete last [D]",command=self.delete)
        self.delete_button.pack(side="left",padx=6)
        ttk.Label(root,textvariable=self.status,wraplength=1350,padding=(18,6)).grid(row=2,sticky="ew")
        ttk.Label(root,text=str(session.folder),padding=(18,0,18,12),font=("DejaVu Sans",9)).grid(row=3,sticky="ew")
        root.bind("<KeyPress>",self.keypress)
        root.protocol("WM_DELETE_WINDOW",self.close)
        self._rows_seen = None
        self._details_seen = None
        session.start()
        root.after(80,self.refresh)

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
        self.catalog=None
        if self.session.config.reference_manifest:
            try:
                self.catalog=ReferenceCatalog(self.session.config.reference_manifest,self.session.config.task,self.session.config.instruction)
            except Exception as error:
                self.reference_var.set(str(error))
        for label,operation in (("‹",lambda:self.move_reference(-1)),("›",lambda:self.move_reference(1)),("Random",self.random_reference),("Zoom",self.zoom_reference)):
            ttk.Button(toolbar,text=label,width=7,command=operation,state="normal" if self.catalog else "disabled").pack(side="left",padx=2)
        ttk.Label(toolbar,textvariable=self.reference_var).pack(side="left",padx=8)
        self.reference_labels=self._image_pair(frame,"Episode start · never policy input",1,roles=("primary","wrist"))
        self.show_reference()

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
        if self.session.start_rollout():
            self.status.set("Moving to task home, then resetting policy and starting rollout")

    def signal(self,action):
        if self.session.signal(action):
            self.status.set(f"{action.title()} requested; cancelling pending policy action")

    def apply_settings(self):
        try:
            self.session.apply_settings(self.hz.get(),self.steps.get())
            self.status.set("Runtime settings applied")
        except Exception as error:
            self.status.set(str(error))

    def schedule_notes(self,_event=None):
        if self.notes_job:
            self.root.after_cancel(self.notes_job)
        self.notes_job=self.root.after(500,self.save)

    def save(self):
        self.notes_job=None
        self.session.save(self.notes.get("1.0","end-1c"))
        self.status.set("Results and notes saved")

    def delete(self):
        if self.session.state=="READY" and self.session.results and messagebox.askyesno("Delete last result?","Delete the last saved result and its videos?",parent=self.root):
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
        with self.session.lock:
            state=self.session.state
            rows=list(self.session.rows)
            results=list(self.session.results)
            latest=self.session.latest
        self.state.set(state)
        ready=state=="READY"
        active=state in {"STARTING","HOMING","RUNNING","WAITING_OUTCOME","STOPPING"}
        self.start_button.configure(state="normal" if ready else "disabled")
        for button in (self.stop_button,self.retry_button):
            button.configure(state="normal" if active else "disabled")
        for button in (self.success_button,self.failure_button):
            button.configure(state="normal" if state in {"RUNNING","WAITING_OUTCOME"} else "disabled")
        for widget in (self.apply_button,self.hz_entry,self.steps_entry):
            widget.configure(state="normal" if ready else "disabled")
        self.delete_button.configure(state="normal" if ready and results else "disabled")
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
        details=json.dumps({"policy":self.session.policy.metadata,"embodiment":self.session.embodiment.metadata,"results":str(self.session.folder)},indent=2)
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
        if self.closing and not self.session.worker.is_alive():
            self.root.destroy()
            return
        self.root.after(100,self.refresh)

    def close(self):
        if self.closing:
            return
        self.save()
        self.closing=True
        self.status.set("Stopping policy commands, finalizing recordings and closing devices")
        self.session.close()
