"""Resolve the current user's desktop for SSH/tmux without changing global config."""
from __future__ import annotations
import os
import hashlib
import math
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid


VIDEO_SUFFIXES = (".webm", ".mp4", ".avi", ".mkv")


def external_application_environment(environment=None):
    """Keep explicitly marked GUI libraries out of another application's host."""
    result=dict(os.environ if environment is None else environment)
    libraries=result.pop("DEPLOYMENT_GUI_TK_PRELOAD",None)
    if libraries:
        owned=set(re.split(r"[:\s]+",libraries))
        remaining=[item for item in re.split(r"[:\s]+",result.get("LD_PRELOAD",""))
                   if item and item not in owned]
        if remaining:
            result["LD_PRELOAD"]=":".join(remaining)
        else:
            result.pop("LD_PRELOAD",None)
    for marker,key in (("DEPLOYMENT_GUI_TK_SCRIPTS","TK_LIBRARY"),
                       ("DEPLOYMENT_GUI_TCL_SCRIPTS","TCL_LIBRARY")):
        scripts=result.pop(marker,None)
        if scripts and result.get(key)==scripts:
            result.pop(key,None)
    return result


def _check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise RuntimeError("Opening cancelled")


def _run_owned(command, *, timeout, cancel=None, environment=None):
    """Drain and reap this action's child, including cancellation and timeouts."""
    timeout = float(timeout)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Desktop action timeout must be finite and positive")
    _check_cancel(cancel)
    process = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                               start_new_session=os.name == "posix")
    deadline = time.monotonic() + timeout
    try:
        while True:
            _check_cancel(cancel)
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                stdout, stderr = process.communicate(timeout=min(.1, remaining))
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        # The parent is still owned and unreaped here, so its session ID cannot
        # be reused. Kill only this action's group, never a user's player/bus.
        if process.returncode is None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
            except ProcessLookupError:
                pass
        try:
            process.communicate(timeout=1)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.communicate(timeout=2)
        raise


def desktop_environment(environment=None, *, cancel=None):
    env=dict(os.environ if environment is None else environment)
    candidates=[env]
    allowed={"DISPLAY","XAUTHORITY","WAYLAND_DISPLAY","XDG_RUNTIME_DIR","DBUS_SESSION_BUS_ADDRESS",
             "XDG_CURRENT_DESKTOP","XDG_SESSION_TYPE","DESKTOP_SESSION"}
    try:
        found=_run_owned(["pgrep","-u",str(os.getuid()),"-f","gnome-shell|plasmashell|xfce4-session|Xorg"],timeout=3,cancel=cancel)
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
    for candidate in candidates:
        _check_cancel(cancel)
        if not candidate.get("DISPLAY"):
            continue
        try:
            probe=_run_owned([sys.executable,"-c","import tkinter as t; r=t.Tk(); r.withdraw(); r.update_idletasks(); r.destroy()"],environment=candidate,timeout=5,cancel=cancel)
        except subprocess.TimeoutExpired:
            continue
        if probe.returncode==0:
            return candidate
    raise RuntimeError("No usable desktop for this SSH user. Log into the target desktop, or connect using trusted X11 forwarding (ssh -Y <host>).")


def recorded_video(folder, role):
    """Find new and older recordings without allowing a camera role to escape."""
    folder = Path(folder).resolve(strict=True)
    if not role or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role):
        raise ValueError("Camera role must be a simple filename identifier")
    for suffix in VIDEO_SUFFIXES:
        candidate = folder / f"{role}{suffix}"
        if candidate.exists():
            path = candidate.resolve(strict=True)
            if path.parent != folder:
                raise ValueError("Video path is outside the saved rollout")
            if path.is_file() and path.stat().st_size:
                return path
    raise ValueError("The selected camera has no saved video")


def _convert_video(source, destination):
    """CPU conversion preserving the source frame order, count and encoded FPS."""
    import cv2
    cv2.setNumThreads(2)
    capture = cv2.VideoCapture(str(source))
    writer = None
    count = 0
    try:
        if not capture.isOpened():
            raise RuntimeError("The recorded video cannot be decoded")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0:
            raise RuntimeError("The recorded video has no usable frame rate")
        while True:
            available, frame = capture.read()
            if not available:
                break
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError("The installed OpenCV cannot encode VP8/WebM")
            if writer.write(frame) is False:
                raise RuntimeError("The compatibility video writer rejected a frame")
            count += 1
        if not count or (expected > 0 and count != expected):
            raise RuntimeError(f"The recorded video ended early ({count}/{expected} frames decoded)")
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def compatible_video(path, cache_root, *, timeout=120, cancel=None):
    """Cache an Ubuntu-compatible copy of an older MPEG-4 recording on demand."""
    _check_cancel(cancel)
    path = Path(path).resolve(strict=True)
    cache = Path(cache_root).expanduser().resolve()
    if not path.is_file() or not path.stat().st_size:
        raise ValueError("The selected video is empty or unavailable")
    if ".runtime" not in cache.parts:
        raise ValueError("Playback copies must use the project's .runtime cache")
    identity = f"{path}:{path.stat().st_size}:{path.stat().st_mtime_ns}"
    destination = cache / f"{path.stem}-{hashlib.sha256(identity.encode()).hexdigest()[:20]}.webm"
    if destination.is_file() and destination.stat().st_size:
        return destination
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / f".{destination.stem}-{uuid.uuid4().hex}.webm"
    try:
        result = _run_owned([sys.executable, str(Path(__file__).resolve()), "--convert-video", str(path), str(temporary)],
                            timeout=timeout, cancel=cancel, environment=external_application_environment())
        if result.returncode or not temporary.is_file() or not temporary.stat().st_size:
            detail = (result.stderr or result.stdout).strip()[-1500:]
            raise RuntimeError(f"Cannot prepare a playable copy of this video: {detail or 'conversion produced no video'}")
        _check_cancel(cancel)
        temporary.replace(destination)
        return destination
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"Preparing this video exceeded {timeout} seconds; the original is unchanged") from error
    finally:
        temporary.unlink(missing_ok=True)


def open_local_path(path, *, cache_root=None, environment=None, timeout=8, conversion_timeout=120, cancel=None):
    """Open one existing local record; callers run this outside the GUI thread."""
    _check_cancel(cancel)
    path = Path(path).expanduser().resolve(strict=True)
    if not (path.is_file() or path.is_dir()):
        raise ValueError("Select a local file or folder")
    if path.is_file() and not path.stat().st_size:
        raise ValueError("The selected file is empty")
    program = shutil.which("open" if sys.platform == "darwin" else "xdg-open")
    if program is None:
        raise RuntimeError("No desktop opener is installed. Use Copy path to open this record manually.")
    env = desktop_environment(environment,cancel=cancel) if sys.platform.startswith("linux") else dict(os.environ if environment is None else environment)
    if sys.platform.startswith("linux") and path.suffix.lower() == ".mp4" and cache_root is not None:
        path = compatible_video(path, cache_root, timeout=conversion_timeout, cancel=cancel)
    try:
        result = _run_owned([program, str(path)], environment=external_application_environment(env), timeout=timeout, cancel=cancel)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"The desktop opener did not finish within {timeout} seconds. Use Copy path to open this record manually.") from error
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[-1500:]
        raise RuntimeError(f"The desktop could not open this record (exit {result.returncode}): {detail or 'no diagnostic was returned'}")
    return path


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--convert-video":
        _convert_video(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit("This module only accepts --convert-video SOURCE DESTINATION")
