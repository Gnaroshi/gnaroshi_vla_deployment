"""Resolve the current user's desktop for SSH/tmux without changing global config."""
from __future__ import annotations
import os
from pathlib import Path
import subprocess


def desktop_environment(environment=None):
    env=dict(os.environ if environment is None else environment)
    candidates=[env]
    allowed={"DISPLAY","XAUTHORITY","WAYLAND_DISPLAY","XDG_RUNTIME_DIR","DBUS_SESSION_BUS_ADDRESS"}
    try:
        found=subprocess.run(["pgrep","-u",str(os.getuid()),"-f","gnome-shell|plasmashell|xfce4-session|Xorg"],capture_output=True,text=True,timeout=3)
        for pid in found.stdout.split():
            try:
                raw=Path(f"/proc/{int(pid)}/environ").read_bytes().split(b"\0")
                candidate=dict(env)
                for field in raw:
                    key,separator,value=field.partition(b"=")
                    if separator and key.decode(errors="ignore") in allowed:
                        candidate[key.decode()]=value.decode()
                candidates.append(candidate)
            except (OSError,ValueError):
                continue
    except (OSError,subprocess.TimeoutExpired):
        pass
    import sys
    for candidate in candidates:
        if not candidate.get("DISPLAY"):
            continue
        try:
            probe=subprocess.run([sys.executable,"-c","import tkinter as t; r=t.Tk(); r.withdraw(); r.update_idletasks(); r.destroy()"],env=candidate,capture_output=True,text=True,timeout=5)
        except subprocess.TimeoutExpired:
            continue
        if probe.returncode==0:
            return candidate
    raise RuntimeError("No usable desktop for this SSH user. Log into the target desktop, or connect using trusted X11 forwarding (ssh -Y <host>).")
