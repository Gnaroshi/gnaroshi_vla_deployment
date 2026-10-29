"""Local presentation preferences and a reusable keyboard-resizable workspace.

These controls never inspect policy, robot or recording data. A caller chooses
the preference file and owns the surrounding application's workflow.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk
import uuid


def clamp_window_geometry(value, screen_width, screen_height, *, min_width=960, min_height=680):
    """Restore a readable, visible window without rewriting the desired size."""
    value = value if isinstance(value, dict) else {}
    sw, sh = max(1, int(screen_width)), max(1, int(screen_height))
    def integer(key, default):
        result = value.get(key, default)
        return result if type(result) is int else default
    width = min(sw, max(min(min_width, sw), integer("width", min(1440, sw))))
    height = min(sh, max(min(min_height, sh), integer("height", min(960, sh))))
    x = min(max(0, integer("x", (sw-width)//2)), max(0, sw-width))
    y = min(max(0, integer("y", (sh-height)//2)), max(0, sh-height))
    return {"width": width, "height": height, "x": x, "y": y}


def geometry_string(value):
    return f"{value['width']}x{value['height']}+{value['x']}+{value['y']}"


class WorkspacePreferences:
    """A bounded, optional JSON preference file, written only after a change.

    A corrupt/unsupported file is left intact when read. Explicit presentation
    changes may replace it atomically. The original session/result data is never
    a preference source and is never modified here.
    """
    LIMIT = 128 * 1024

    def __init__(self, path=None):
        self.path = Path(path).expanduser() if path is not None else None
        self._values = {}
        self._dirty = False
        self.error = None
        if self.path is not None and self.path.exists():
            try:
                if self.path.stat().st_size > self.LIMIT:
                    raise ValueError("Presentation preferences exceed the size limit")
                stored = json.loads(self.path.read_text())
                if not isinstance(stored, dict) or type(stored.get("schema_version")) is not int or stored.get("schema_version") != 1:
                    raise ValueError("Unsupported presentation preference schema")
                values = stored.get("values", {})
                if not isinstance(values, dict) or len(values) > 128:
                    raise ValueError("Presentation preferences must be a bounded object")
                # Unknown top-level fields are ignored for forward compatibility.
                self._values = self._clean(values)
            except (OSError, ValueError, TypeError, RecursionError) as error:
                self.error = str(error)

    @staticmethod
    def _clean(values):
        result = {}
        for key, value in values.items():
            if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", key):
                continue
            if key == "text_size":
                if type(value) is int and 10 <= value <= 24:
                    result[key] = value
                continue
            if key == "window":
                if isinstance(value, dict):
                    result[key] = {name:value[name] for name in ("width", "height", "x", "y")
                                   if type(value.get(name)) is int and abs(value[name]) <= 100000}
                continue
            if key == "panes":
                if isinstance(value, dict):
                    panes = {}
                    for name, state in list(value.items())[:32]:
                        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", name) or not isinstance(state, dict):
                            continue
                        fractions = {}
                        for orientation in ("horizontal", "vertical"):
                            number = state.get("fractions", {}).get(orientation) if isinstance(state.get("fractions"), dict) else None
                            if type(number) in (int, float) and math.isfinite(number) and 0 < number < 1:
                                fractions[orientation] = float(number)
                        collapsed = state.get("collapsed", [])
                        collapsed = [side for side in ("first", "second") if side in collapsed] if isinstance(collapsed, list) else []
                        panes[name] = {"fractions":fractions, "collapsed":collapsed,
                                       "orientation":state.get("orientation") if state.get("orientation") in ("horizontal", "vertical") else "horizontal"}
                    result[key] = panes
                continue
            # The application may store small presentation flags without
            # exposing arbitrary nested payloads or executable configuration.
            if value is None or type(value) in (str, bool, int):
                if not isinstance(value, str) or len(value) <= 1000:
                    result[key] = value
        return result

    def get(self, key, default=None):
        return copy.deepcopy(self._values.get(key, default))

    def set(self, key, value):
        clean = self._clean({key:value})
        if key not in clean:
            raise ValueError(f"Invalid presentation preference: {key}")
        if self._values.get(key) != clean[key]:
            self._values[key] = clean[key]
            self._dirty = True

    def snapshot(self):
        return copy.deepcopy(self._values)

    def pane(self, key):
        return self.get("panes", {}).get(key, {})

    def set_pane(self, key, value):
        panes = self.get("panes", {})
        panes[key] = value
        self.set("panes", panes)

    def flush(self):
        if self.path is None or not self._dirty:
            return False
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps({"schema_version":1, "values":self._values}, indent=2, allow_nan=False)+"\n"
            if len(payload.encode()) > self.LIMIT:
                raise ValueError("Presentation preferences exceed the size limit")
            with temporary.open("x") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
            self._dirty = False
            self.error = None
            return True
        except (OSError, ValueError) as error:
            self.error = str(error)
            return False
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class ResizablePane(ttk.Frame):
    """Two stable child frames with pointer, keyboard and persisted resizing.

    Focus the separator and use Arrow keys, Shift+Arrow, Home/End or Return.
    Double click resets the split. Space toggles the second pane when allowed.
    Native pane geometry owns placement; all resize interaction lives here.
    """
    def __init__(self, master, *, orient="horizontal", preferences=None, key="pane",
                 first_label="First pane", second_label="Second pane", first_min=240,
                 second_min=240, default_fraction=.6, collapsible=(False, True),
                 show_header=True, show_controls=None, **kwargs):
        super().__init__(master, **kwargs)
        if orient not in ("horizontal", "vertical"):
            raise ValueError("Pane orientation must be horizontal or vertical")
        if not 0 < default_fraction < 1 or not math.isfinite(default_fraction):
            raise ValueError("Default pane fraction must be finite and between zero and one")
        if len(collapsible) != 2:
            raise ValueError("Two collapse capabilities are required")
        self.preferences = preferences or WorkspacePreferences()
        self.key = str(key)
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", self.key):
            raise ValueError("Pane preference key must be a simple identifier")
        self.orient = orient
        self.first_label, self.second_label = str(first_label), str(second_label)
        self.first_min, self.second_min = self._minimum(first_min), self._minimum(second_min)
        self.default_fraction = float(default_fraction)
        self._ratios = {"horizontal":self.default_fraction, "vertical":self.default_fraction}
        state = self.preferences.pane(self.key)
        self._ratios.update(state.get("fractions", {}))
        self._collapsible = {side:bool(allowed) for side, allowed in zip(("first", "second"), collapsible)}
        self._collapsed = {side for side in state.get("collapsed", []) if self._collapsible.get(side)}
        if len(self._collapsed) == 2:
            self._collapsed.discard("first")
        self._hidden = set()
        self._show_header = show_header if show_controls is None else bool(show_controls)
        self._layout_job = self._save_job = self._sync_job = None
        self._drag_watch_job = None
        self._focus_target = None
        self._closed = False
        self._drag = None
        self._focused = False
        self._position = self._low = self._high = 0
        self._available = 0
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.header = ttk.Frame(self)
        self.header.grid(row=0, column=0, sticky="ew")
        self.header.columnconfigure(0, weight=1)
        self.caption = ttk.Label(self.header, text=f"{self.first_label} / {self.second_label}")
        self.caption.grid(row=0, column=0, sticky="w")
        self.buttons = {}
        for index, side in enumerate(("first", "second"), start=1):
            button = ttk.Button(self.header, command=lambda side=side:self.toggle(side))
            button.grid(row=0, column=index, padx=(4,0))
            self.buttons[side] = button
        self.reset_button = ttk.Button(self.header, text="Reset split", command=self.reset)
        self.reset_button.grid(row=0, column=3, padx=(4,0))
        background = ttk.Style(self).lookup("TFrame", "background") or "#11151B"
        self.paned = tk.PanedWindow(self, orient=orient, sashwidth=12, borderwidth=0,
                                    background=background, showhandle=False, opaqueresize=True)
        self.paned.grid(row=1, column=0, sticky="nsew")
        self.first = ttk.Frame(self.paned)
        self.second = ttk.Frame(self.paned)
        self.paned.add(self.first, minsize=1, stretch="always")
        self.paned.add(self.second, minsize=1, stretch="always")
        self.separator = ttk.Separator(self.paned, orient="vertical" if orient == "horizontal" else "horizontal",
                                        takefocus=True, cursor="sb_h_double_arrow" if orient == "horizontal" else "sb_v_double_arrow")
        self.separator.accessible_name = f"Resize {self.first_label} and {self.second_label}"
        self.separator.bind("<ButtonPress-1>", self._drag_start)
        self.separator.bind("<B1-Motion>", self._drag_move)
        self.separator.bind("<ButtonRelease-1>", self._drag_end)
        self.separator.bind("<Double-Button-1>", lambda event:self._reset_event())
        self.separator.bind("<Escape>", self._cancel_drag)
        self.separator.bind("<FocusOut>", self._cancel_drag)
        self.separator.bind("<Unmap>", self._cancel_drag)
        self.separator.bind("<KeyPress>", self._key)
        self.separator.bind("<FocusIn>", lambda event:self._focus_changed(True))
        self.separator.bind("<FocusOut>", lambda event:self._focus_changed(False), add=True)
        self.paned.bind("<Configure>", self._schedule_layout)
        self.first.bind("<Configure>", self._schedule_sync, add=True)
        self.second.bind("<Configure>", self._schedule_sync, add=True)
        self.bind("<Destroy>", self._destroyed, add=True)
        self._sync_visibility()
        self._schedule_layout()

    @staticmethod
    def _minimum(value):
        if type(value) is not int or value < 1:
            raise ValueError("Pane minimum sizes must be positive integers")
        return value

    @property
    def desired_fraction(self):
        return self._ratios[self.orient]

    @property
    def fraction(self):
        return self.desired_fraction

    @property
    def actual_fraction(self):
        return self._position/self._available if self._available else self.desired_fraction

    @property
    def collapsed(self):
        return tuple(side for side in ("first", "second") if side in self._collapsed)

    def _effective_collapsed(self):
        collapsed = self._collapsed | self._hidden
        return {"second"} if len(collapsed) == 2 else collapsed

    def snapshot(self):
        return {"key":self.key, "orientation":self.orient,
                "desired_fraction":self.desired_fraction, "actual_fraction":self.actual_fraction,
                "position":self._position, "minimum":self._low, "maximum":self._high,
                "available":self._available, "collapsed":list(self.collapsed),
                "effective_collapsed":sorted(self._effective_collapsed()),
                "first_visible":"first" not in self._effective_collapsed(),
                "second_visible":"second" not in self._effective_collapsed(),
                "separator_label":self.separator.accessible_name}

    def _focus_changed(self, focused):
        if self._closed:
            return
        self._focused = focused
        style = ttk.Style(self)
        style.configure("WorkspaceFocused.TSeparator", background=style.lookup("Accent.TButton", "background") or "#A7D7C8")
        self.separator.configure(style="WorkspaceFocused.TSeparator" if focused else "TSeparator")
        self._caption()

    def _caption(self):
        text = f"{self.first_label} / {self.second_label}"
        if self._focused:
            text += f" · {100*self.actual_fraction:.0f}%"
        self.caption.configure(text=text)

    def _schedule_layout(self, event=None):
        if not self._closed and self._layout_job is None:
            self._layout_job = self.after_idle(self._apply_layout)

    def _apply_layout(self):
        self._layout_job = None
        if self._closed or not self.winfo_exists():
            return
        collapsed = self._effective_collapsed()
        if collapsed:
            self.separator.place_forget()
            if self._focus_target is not None and self._focus_target.winfo_viewable():
                self._focus_target.focus_set()
                self._focus_target = None
            return
        extent = self.paned.winfo_width() if self.orient == "horizontal" else self.paned.winfo_height()
        available = max(1, extent-int(self.paned.cget("sashwidth")))
        total_min = self.first_min+self.second_min
        factor = min(1., available/total_min)
        low = max(1, min(available, round(self.first_min*factor)))
        high = max(low, available-max(1, round(self.second_min*factor)))
        high = min(available, high)
        position = min(high, max(low, round(available*self.desired_fraction)))
        self._available, self._low, self._high, self._position = available, low, high, position
        for frame, minimum in ((self.first, low), (self.second, max(1, available-high))):
            if str(self.paned.panecget(frame, "minsize")) != str(minimum):
                self.paned.paneconfigure(frame, minsize=minimum)
        if self.orient == "horizontal":
            self.paned.sash_place(0, position, 0)
        else:
            self.paned.sash_place(0, 0, position)
        if self._sync_job is not None:
            self.after_cancel(self._sync_job)
            self._sync_job = None
        self._sync_separator()
        self._schedule_sync()

    def _schedule_sync(self, event=None):
        if not self._closed and self._sync_job is None:
            self._sync_job = self.after_idle(self._sync_separator)

    def _sync_separator(self):
        self._sync_job = None
        if self._closed or self._effective_collapsed() or len(self.paned.panes()) != 2:
            return
        x, y = self.paned.sash_coord(0)
        # Child notebook/label requests can run Tk's native geometry allocator
        # after our layout callback. Apply the user's split once that settles;
        # pane width/height requests themselves must remain content-owned.
        position = min(self._high, max(self._low, round(self._available*self.desired_fraction)))
        if (x if self.orient == "horizontal" else y) != position:
            self.paned.sash_place(0, position if self.orient == "horizontal" else 0,
                                  0 if self.orient == "horizontal" else position)
            x, y = self.paned.sash_coord(0)
        self._position = x if self.orient == "horizontal" else y
        width = int(self.paned.cget("sashwidth"))
        # Clear both opposite relative dimensions on every direction change.
        # place() retains omitted options, so leaving relwidth=1 from a former
        # horizontal separator would cover the right pane after stacking.
        if self.orient == "horizontal":
            self.separator.place(x=x, y=0, width=width, relwidth=0, height=0, relheight=1)
        else:
            self.separator.place(x=0, y=y, width=0, relwidth=1, height=width, relheight=0)
        self.separator.value_now, self.separator.value_min, self.separator.value_max = self._position, self._low, self._high
        self.separator.orientation = "vertical" if self.orient == "horizontal" else "horizontal"
        self._caption()
        self.separator.lift()
        if self._focus_target is self.separator:
            self.separator.focus_set()
            self._focus_target = None

    def set_minimums(self, *, first=None, second=None):
        previous = self.first_min, self.second_min
        if first is not None:
            self.first_min = self._minimum(first)
        if second is not None:
            self.second_min = self._minimum(second)
        if previous != (self.first_min, self.second_min):
            self._schedule_layout()

    def set_fraction(self, value, *, persist=True):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Pane fraction must be finite and between zero and one")
        self._ratios[self.orient] = float(value)
        self._schedule_layout()
        if persist:
            self._persist()

    def set_orientation(self, value, *, persist=False):
        if value not in ("horizontal", "vertical"):
            raise ValueError("Pane orientation must be horizontal or vertical")
        if value != self.orient:
            self._cancel_drag()
            self.orient = value
            self.paned.configure(orient=value)
            self.separator.configure(orient="vertical" if value == "horizontal" else "horizontal",
                                     cursor="sb_h_double_arrow" if value == "horizontal" else "sb_v_double_arrow")
            self._schedule_layout()
        if persist:
            self._persist()

    def _sync_visibility(self):
        if self._closed:
            return
        collapsed = self._effective_collapsed()
        present = {str(pane) for pane in self.paned.panes()}
        for side, frame in (("first", self.first), ("second", self.second)):
            if side in collapsed:
                focused = self.focus_get()
                if focused is not None and (focused is frame or str(focused).startswith(str(frame)+".")):
                    self._focus_target = (self.second if side == "first" else self.first) if side in self._hidden else self.buttons[side]
                if str(frame) in present:
                    self.paned.forget(frame)
            elif str(frame) not in present:
                if side == "first" and str(self.second) in present:
                    self.paned.add(frame, before=self.second, minsize=1, stretch="always")
                else:
                    self.paned.add(frame, minsize=1, stretch="always")
            button = self.buttons[side]
            if self._collapsible[side] and side not in self._hidden:
                button.configure(text=f"Show {getattr(self, side+'_label')}" if side in self._collapsed else f"Hide {getattr(self, side+'_label')}")
                button.grid()
            else:
                button.grid_remove()
        if self._show_header or self._collapsed:
            self.header.grid()
        else:
            self.header.grid_remove()
        if collapsed:
            self.separator.place_forget()
            if not self._show_header:
                self.caption.grid_remove()
                self.reset_button.grid_remove()
        else:
            self.caption.grid()
            self.reset_button.grid()
        self._schedule_layout()

    def collapse(self, side="second", *, persist=True):
        if side not in ("first", "second"):
            raise ValueError("Unknown pane")
        if persist and not self._collapsible[side]:
            return False
        target = self._collapsed if persist else self._hidden
        if persist and (target | {side}) == {"first", "second"}:
            return False
        target.add(side)
        self._sync_visibility()
        if persist:
            self._persist()
        return True

    def expand(self, side="second", *, persist=True):
        if side not in ("first", "second"):
            raise ValueError("Unknown pane")
        target = self._collapsed if persist else self._hidden
        target.discard(side)
        self._sync_visibility()
        if persist:
            self._focus_target = self.separator
        if persist:
            self._persist()
        return True

    def set_visible(self, side, visible, *, persist=False):
        return self.expand(side, persist=persist) if visible else self.collapse(side, persist=persist)

    def toggle(self, side="second"):
        return self.expand(side) if side in self._collapsed else self.collapse(side)

    def reset(self):
        self._ratios = {"horizontal":self.default_fraction, "vertical":self.default_fraction}
        self._collapsed.clear()
        self._sync_visibility()
        self._persist()

    def _reset_event(self):
        self.reset()
        return "break"

    def _key(self, event):
        horizontal = self.orient == "horizontal"
        negative, positive = (("Left", "Right") if horizontal else ("Up", "Down"))
        step = 64 if event.state & 0x0001 else 16
        if event.keysym in (negative, positive, "Home", "End"):
            if self._effective_collapsed() or not self._available:
                return "break"
            position = self._position+(-step if event.keysym == negative else step)
            if event.keysym in ("Home", "End"):
                position = self._low if event.keysym == "Home" else self._high
            position = min(self._high, max(self._low, position))
            self.set_fraction(min(.999, max(.001, position/self._available)))
            return "break"
        if event.keysym in ("Return", "KP_Enter"):
            return self._reset_event()
        if event.keysym == "space" and self._collapsible["second"]:
            self.toggle("second")
            return "break"

    def _drag_start(self, event):
        if self._effective_collapsed():
            return "break"
        self.separator.focus_set()
        self._drag = self.desired_fraction
        self.separator.grab_set()
        self._watch_drag()
        self._drag_move(event)
        return "break"

    def _drag_move(self, event):
        if self._drag is None or not self._available:
            return "break"
        coordinate = event.x_root-self.paned.winfo_rootx() if self.orient == "horizontal" else event.y_root-self.paned.winfo_rooty()
        coordinate = min(self._high, max(self._low, coordinate))
        self.set_fraction(min(.999, max(.001, coordinate/self._available)))
        return "break"

    def _drag_end(self, event=None):
        if self._drag_watch_job is not None:
            self.after_cancel(self._drag_watch_job)
            self._drag_watch_job = None
        if self._drag is not None:
            self._drag = None
            try:
                self.separator.grab_release()
            except tk.TclError:
                pass
            self._persist()
        return "break"

    def _watch_drag(self):
        self._drag_watch_job = None
        if self._closed or self._drag is None:
            return
        if self.separator.grab_current() is not self.separator:
            self._cancel_drag()
            return
        self._drag_watch_job = self.after(50, self._watch_drag)

    def _cancel_drag(self, event=None):
        original = self._drag
        if original is not None:
            self._drag_end()
            self.set_fraction(original)
        return "break" if original is not None and event is not None and getattr(event, "keysym", None) == "Escape" else None

    def _persist(self):
        self.preferences.set_pane(self.key, {"fractions":self._ratios, "collapsed":list(self.collapsed),
                                             "orientation":self.orient})
        if self._save_job is not None:
            self.after_cancel(self._save_job)
        self._save_job = self.after(300, self._flush)

    def _flush(self):
        self._save_job = None
        self.preferences.flush()

    def _destroyed(self, event):
        if event.widget is self:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._drag is not None:
            self._ratios[self.orient] = self._drag
            self._drag = None
            try:
                self.separator.grab_release()
            except tk.TclError:
                pass
            self.preferences.set_pane(self.key, {"fractions":self._ratios, "collapsed":list(self.collapsed),
                                                 "orientation":self.orient})
        for job in (self._layout_job, self._save_job, self._drag_watch_job, self._sync_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        self._layout_job = self._save_job = self._drag_watch_job = self._sync_job = None
        self._focus_target = None
        self.preferences.flush()
