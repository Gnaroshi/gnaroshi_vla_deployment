"""Atomic session records and videos from the exact observations used by policy."""
from __future__ import annotations
import json
import math
from pathlib import Path
import subprocess
import time
import numpy as np


def write_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    temporary.replace(path)


def provenance(repo: Path):
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": git("rev-parse", "HEAD"), "status": git("status", "--short"),
            "repo": str(repo), "created_ns": time.time_ns()}


class RolloutRecorder:
    def __init__(self, folder: Path, fps: float, video: bool):
        self.folder = folder
        folder.mkdir()
        self.log = (folder / "steps.jsonl").open("x")
        self.timestamps = (folder / "frames.jsonl").open("x")
        self.writers = {}
        self.video = video
        self.target_hz = fps
        # Resample by capture timestamps; target 60 Hz does not imply 60 images/s.
        # A 14 Hz rollout must not turn into a four-times-faster MP4.
        self.fps = 30.0
        self.first_capture_ns = None
        self.last_capture_ns = None
        self.video_frames = 0
        self.count = 0
        self.pending_images = None
        self.pending_source = None

    def _emit_pending(self,count):
        if count<=0 or self.pending_images is None:
            return
        start=self.video_frames
        if self.video:
            import cv2
            for role,frame in self.pending_images.items():
                bgr=cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                for _ in range(count):
                    self.writers[role].write(bgr)
            self.timestamps.write(json.dumps({"kind":"video_frames",**self.pending_source,
                                              "video_frame_start":start,"video_frame_end_exclusive":start+count,
                                              "video_fps":self.fps})+"\n")
        self.video_frames+=count

    def prepare(self, images):
        """Check codec/role/shape before issuing the first policy command."""
        for role,frame in images.items():
            if not role or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in role):
                raise ValueError("Camera role must be a simple filename identifier")
            if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:
                raise ValueError("Recorded images must be RGB uint8 HWC")
            if self.video and role not in self.writers:
                import cv2
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(self.folder / f"{role}.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
                if not writer.isOpened():
                    writer.release()
                    raise RuntimeError(f"Cannot create {role} video")
                self.writers[role] = writer

    def write(self, row, observation):
        started = time.perf_counter()
        self.prepare(observation.images)
        capture_clock=observation.captured_monotonic_ns if observation.captured_monotonic_ns is not None else observation.captured_ns
        if self.first_capture_ns is None:
            self.first_capture_ns = capture_clock
        if self.last_capture_ns is not None and capture_clock<self.last_capture_ns:
            raise ValueError("Capture timestamp moved backwards")
        self.last_capture_ns=capture_clock
        elapsed = (capture_clock-self.first_capture_ns)/1e9
        # Encode ticks strictly before this observation with the PREVIOUS image.
        # The current image remains pending until the next capture or close.
        # This avoids displaying a newly arrived image over the preceding gap.
        before_current=math.ceil(elapsed*self.fps-1e-9)
        self._emit_pending(max(0,before_current-self.video_frames))
        self.pending_images={role:frame.copy() for role,frame in observation.images.items()}
        self.pending_source={"source_step":row["step"],"captured_ns":observation.captured_ns,
                             "captured_monotonic_ns":observation.captured_monotonic_ns}
        self.timestamps.write(json.dumps({"kind":"observation","frame": self.count, "step": row["step"],
                                          "captured_ns": observation.captured_ns,
                                          "captured_monotonic_ns":observation.captured_monotonic_ns,
                                          "video_frame_start":None,"video_frame_end_exclusive":None,
                                          "video_fps": self.fps if self.video else None,
                                          "camera_metadata": observation.camera_metadata})+"\n")
        self.timestamps.flush()
        row["media_write_ms"] = 1000*(time.perf_counter()-started)
        if "work_ms" in row:
            row["loop_work_ms"] = row["work_ms"]+row["media_write_ms"]
            row["deadline_miss"] = row["loop_work_ms"]>1000/self.target_hz
        self.log.write(json.dumps(row, allow_nan=False)+"\n")
        self.log.flush()
        self.count += 1

    def close(self):
        self._emit_pending(1)
        self.pending_images=None
        self.log.close()
        self.timestamps.close()
        for writer in self.writers.values():
            writer.release()


def summary(rows):
    times = [r["command_monotonic_s"] for r in rows]
    actual = (len(times)-1)/(times[-1]-times[0]) if len(times)>1 and times[-1]>times[0] else None
    def stats(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return {"count": len(values), "mean_ms": float(np.mean(values)),
                "p50_ms": float(np.percentile(values,50)), "p95_ms": float(np.percentile(values,95))} if values else None
    return {"steps": len(rows), "achieved_command_hz": actual,
            "active_duration_s": times[-1]-times[0] if len(times)>1 else 0.0,
            "latency": {key: stats(key) for key in ("capture_ms", "policy_ms", "rpc_ms", "server_infer_ms", "model_gpu_ms", "command_ms", "work_ms", "media_write_ms", "loop_work_ms")},
            "deadline_misses": sum(r["deadline_miss"] for r in rows),
            "full_calls": sum(r.get("backbone_call_delta") or 0 for r in rows) if any(r.get("backbone_calls") is not None or (r.get("backbone_call_delta") or 0)>0 for r in rows) else None,
            "buffered_steps": sum(bool(r.get("buffered")) for r in rows)}
