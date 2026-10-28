"""Worker-owned rollout state machine shared by GUI and hardware-free tests."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from .contracts import SessionConfig
import numpy as np
from .records import RolloutRecorder, provenance, summary, write_json


class DeploymentSession:
    def __init__(self, config: SessionConfig, policy, embodiment, repo: Path):
        config.validate()
        self.config, self.policy, self.embodiment = config, policy, embodiment
        self.repo = Path(repo).resolve()
        self.folder = Path(config.output_root).expanduser().resolve() / f"{config.task}_{config.method}_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        self.folder.mkdir(parents=True)
        self.events = queue.Queue()
        self.commands = queue.Queue()
        self.cancel = threading.Event()
        self.closing = threading.Event()
        self.lock = threading.RLock()
        self.state = "CONNECTING"
        self.request = None
        self.results, self.rows = [], []
        self.latest = None
        self.error = None
        self.error_detail = None
        self._stop_thread = None
        self._stop_threads = []
        self._stop_results = {}
        self._stop_generation = 0
        self._stop_failed = False
        self._stop_confirmed = True
        self._started = False
        self._cleanup_complete = False
        self._cleanup_started = False
        self._cleanup_failed = False
        self._policy_load_error = False
        self._startup_error = False
        self._preview_failed = False
        self._policy_loader = None
        self._notes = ""
        self._sequence = 0
        self.manifest = {"config": config.asdict(), "source": provenance(self.repo),
                         "timing": {"achieved_command_hz": "(commands-1)/(last-first command monotonic time)",
                                    "policy_ms": "client preprocessing/inference/postprocessing including RPC; buffered steps remain distinct",
                                    "video": "policy observations resampled at 30 fps using capture timestamps; duplicate/drop mapping in frames.jsonl"}}
        write_json(self.folder / "manifest.json", self.manifest)
        self.worker = threading.Thread(target=self._run, name="deployment-worker", daemon=True)

    def start(self):
        with self.lock:
            if self._started:
                raise RuntimeError("This session has already started")
            self._started = True
            self.worker.start()

    @property
    def shutdown_complete(self):
        with self.lock:
            return (self._started and not self.worker.is_alive() and self._cleanup_complete
                    and (self._policy_loader is None or not self._policy_loader.is_alive())
                    and not any(thread.is_alive() for thread in self._stop_threads))

    @property
    def replacement_closed(self):
        """No worker/device cleanup remains and every stop was acknowledged."""
        with self.lock:
            return (self.shutdown_complete and not self._cleanup_failed
                    and self._stop_confirmed and not self._stop_failed)

    @property
    def can_replace(self):
        """READY is request eligibility; replacement_closed is required before construction."""
        with self.lock:
            if self._stop_failed or self._cleanup_failed or not self._stop_confirmed:
                return False
            if self.state in {"READY", "SELECTING"} and not self.closing.is_set():
                return True
            return self.replacement_closed and (self.state == "CLOSED" or
                                                self.state == "ERROR" and self._policy_load_error)

    def _state(self, state, detail=""):
        with self.lock:
            self.state = state
        self.events.put({"kind": "state", "state": state, "detail": detail})

    def _journal(self, event, **fields):
        with self.lock:
            with (self.folder / "events.jsonl").open("a") as log:
                log.write(json.dumps({"time_ns": time.time_ns(), "event": event, **fields}, allow_nan=False)+"\n")

    def record_event(self, event, **fields):
        self._journal(event, **fields)

    def start_rollout(self):
        with self.lock:
            if self.state != "READY":
                return False
            # Reserve the state before queueing: double-click cannot queue two homes.
            self.state = "STARTING"
        self.commands.put("start")
        return True

    def signal(self, action):
        if action not in {"success", "failure", "stop", "retry", "close"}:
            raise ValueError(action)
        with self.lock:
            if action != "close" and self.state not in {"STARTING", "HOMING", "RUNNING", "WAITING_OUTCOME", "STOPPING"}:
                return False
            if action in {"success", "failure"} and self.state not in {"RUNNING", "WAITING_OUTCOME"}:
                return False
            # Cancellation wins over a previous outcome, including its pending home.
            if self.request in {"close", "stop", "retry"} and action in {"success", "failure"}:
                return False
            if self.state=="STOPPING" and self.request==action:
                return False
            self.request = action
            self.cancel.set()
            if action == "close":
                self.closing.set()
            self.state = "STOPPING"
            self._stop_confirmed = False
            self._stop_generation += 1
            generation=self._stop_generation
            self._stop_thread = threading.Thread(target=self._stop,args=(generation,), name="operator-stop", daemon=True)
            self._stop_threads.append(self._stop_thread)
            self._stop_thread.start()
        self._journal("operator", action=action)
        return True

    def _stop(self,generation):
        try:
            confirmed = bool(self.embodiment.stop())
        except Exception as error:
            confirmed = False
            self.events.put({"kind": "warning", "detail": f"Stop failed: {error}"})
        with self.lock:
            self._stop_results[generation]=confirmed
            if not confirmed:
                self._stop_failed=True

    def _wait_for_stops(self,timeout=3):
        """Wait for every cancellation generation, including a newer home stop."""
        deadline=time.monotonic()+timeout
        while True:
            with self.lock:
                generation=self._stop_generation
                threads=list(self._stop_threads)
            for thread in threads:
                thread.join(timeout=max(0,deadline-time.monotonic()))
            with self.lock:
                alive=any(thread.is_alive() for thread in self._stop_threads)
                if self._stop_failed:
                    self._stop_failed=True
                    self._stop_confirmed=False
                    return False
                if generation!=self._stop_generation:
                    continue
                if alive:
                    self._stop_failed=True
                    self._stop_confirmed=False
                    return False
                if generation==self._stop_generation:
                    self._stop_confirmed=True
                    return True
            if time.monotonic()>=deadline:
                self._stop_failed=True
                self._stop_confirmed=False
                return False

    def _ready_after_stop(self):
        while True:
            if not self._wait_for_stops():
                raise RuntimeError("Stop failed or timed out; restart this session after checking the robot")
            with self.lock:
                if any(thread.is_alive() for thread in self._stop_threads):
                    continue
                if self._stop_failed:
                    raise RuntimeError("A stop was not confirmed; automatic rearming is disabled")
                if not self.closing.is_set():
                    self.request=None
                    self.cancel.clear()
                    self._state("READY")
                return

    def apply_settings(self, target_hz, max_steps):
        with self.lock:
            if self.state != "READY":
                raise RuntimeError("Runtime settings can only change while ready")
            old = self.config.target_hz, self.config.max_steps
            try:
                self.config.target_hz, self.config.max_steps = float(target_hz), int(max_steps)
                self.config.validate()
            except Exception:
                self.config.target_hz, self.config.max_steps = old
                raise
            if "reload_selection" in self.config.metadata:
                self.config.metadata["reload_selection"].update(target_hz=self.config.target_hz, max_steps=self.config.max_steps)
            self.manifest["config"] = self.config.asdict()
            write_json(self.folder / "manifest.json", self.manifest)

    def save(self, notes=None):
        with self.lock:
            if notes is not None:
                self._notes = notes
            successes = sum(r["outcome"] == "success" for r in self.results)
            payload = {"results": list(self.results), "successes": successes,
                       "failures": sum(r["outcome"] == "failure" for r in self.results),
                       "success_rate": successes/len(self.results) if self.results else None,
                       "notes": self._notes, "state": self.state, "error": self.error}
            write_json(self.folder / "results.json", payload)
            (self.folder / "notes.txt").write_text(self._notes)

    def delete_previous(self):
        with self.lock:
            if self.state != "READY" or not self.results:
                return False
            deleted = self.results.pop()
            folder = (self.folder / deleted["directory"]).resolve()
            if folder.parent != self.folder:
                raise RuntimeError("Result path escaped session")
            shutil.rmtree(folder)
            self._journal("delete_result", rollout_id=deleted["id"])
            self.save()
            return True

    def _preview(self, preview_only=False):
        started = time.perf_counter()
        observation = self.embodiment.observe_preview() if preview_only else self.embodiment.observe()
        if preview_only:
            if np.asarray(observation.state).size:
                raise ValueError("Camera preview must have an empty robot state")
            observation.preview_only = True
            observation.camera_metadata = {**observation.camera_metadata, "preview_only": True}
        elif getattr(observation, "preview_only", False) or observation.camera_metadata.get("preview_only", False):
            raise ValueError("Camera-only preview cannot be used as a policy observation")
        if any(role not in observation.images for role in self.config.camera_roles):
            raise ValueError("Observation is missing a configured camera role")
        for role,image in observation.images.items():
            if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
                raise ValueError(f"{role} must be an RGB HWC uint8 image")
        if not np.isfinite(observation.state).all():
            raise ValueError("Robot state contains nonfinite values")
        with self.lock:
            self.latest = observation
        return observation, 1000*(time.perf_counter()-started)

    def _check_policy_ready(self):
        check = getattr(self.policy, "check_ready", None)
        if callable(check):
            check()

    def _connect_policy_with_preview(self):
        completed, outcome = threading.Event(), {}
        def load():
            try:
                self.policy.connect()
            except BaseException as error:
                outcome["error"] = error if isinstance(error, Exception) else RuntimeError(str(error))
            finally:
                completed.set()
        self._policy_loader = threading.Thread(target=load, name="policy-loader", daemon=True)
        started, last_status = time.monotonic(), None
        self._state("LOADING_POLICY", "Loading policy; camera preview is available")
        self._policy_loader.start()
        try:
            while not completed.is_set():
                if not self.closing.is_set():
                    try:
                        self._preview(preview_only=True)
                    except Exception:
                        self._preview_failed = True
                        raise
                    status = getattr(self.policy, "loading_status", "")
                    status = status() if callable(status) else status
                    elapsed = time.monotonic() - started
                    detail = str(status or "Loading policy")
                    rendered = f"{detail} ({elapsed:.0f}s) · camera preview only"
                    if rendered != last_status:
                        self.events.put({"kind": "policy_loading", "detail": rendered})
                        self._journal("policy_loading", detail=detail, elapsed_s=round(elapsed, 1))
                        last_status = rendered
                completed.wait(.04)
        finally:
            if self._policy_loader.is_alive() and not completed.is_set():
                self.closing.set()
                self.cancel.set()
            self._policy_loader.join()
        if "error" in outcome:
            raise outcome["error"]

    def _run(self):
        stage = "dependencies"
        try:
            if self.closing.is_set():
                return
            if self.config.record_video:
                import cv2  # Fail dependency validation before any device connects.
            preview = (self.config.metadata.get("preview_before_policy", False)
                       and callable(getattr(self.embodiment, "connect_preview", None))
                       and callable(getattr(self.embodiment, "observe_preview", None)))
            if preview:
                stage = "preview"
                self._state("CONNECTING_PREVIEW", "Opening cameras without robot control")
                self.embodiment.connect_preview()
                if self.closing.is_set():
                    return
            if self.config.metadata.get("defer_policy_load", False):
                stage = "selection"
                self._state("SELECTING", "Select a task and checkpoint, then Load checkpoint · policy and robot controls are not connected")
                while not self.closing.is_set():
                    if preview:
                        self._preview(preview_only=True)
                    self.closing.wait(.04)
                return
            # Metadata validation belongs in policy.connect, before robot.connect.
            stage = "policy"
            if preview:
                self._connect_policy_with_preview()
            else:
                self._state("CONNECTING", "Loading and validating the selected policy")
                self.policy.connect()
            if self.closing.is_set():
                return
            stage = "embodiment"
            self.embodiment.connect()
            self.manifest.update(policy=self.policy.metadata, embodiment=self.embodiment.metadata)
            write_json(self.folder / "manifest.json", self.manifest)
            if not self.closing.is_set():
                self._state("READY")
            stage = "running"
            while not self.closing.is_set():
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if command == "start":
                    self._rollout()
                elif self.state == "READY":
                    self._check_policy_ready()
                    self._preview()
                    self.closing.wait(0.04)
                else:
                    self.closing.wait(0.02)
        except Exception as error:
            if self.closing.is_set() and isinstance(error, InterruptedError):
                self._journal("cancelled_startup", stage=stage)
            else:
                self._policy_load_error = stage == "policy" and not self._preview_failed
                self._startup_error = stage != "running"
                self.error = f"{type(error).__name__}: {error}"
                self.error_detail = traceback.format_exc()
                self._journal("error", detail=self.error, traceback=self.error_detail,
                              selection=self.config.metadata.get("reload_selection", {}))
                self._state("ERROR", self.error)
        finally:
            with self.lock:
                self._cleanup_started = True
            try:
                try:
                    if self._stop_threads:
                        if not self._wait_for_stops():
                            self.error=self.error or "Final stop failed or timed out"
                    else:
                        self._stop_confirmed = bool(self.embodiment.stop())
                        if not self._stop_confirmed:
                            self._stop_failed = True
                            self.error=self.error or "Final stop was not confirmed"
                except Exception as error:
                    self._stop_failed = True
                    self._stop_confirmed = False
                    self.error=self.error or f"Final stop: {error}"
                    self.events.put({"kind":"warning","detail":f"Final stop: {error}"})
                finally:
                    self.embodiment.close()
            except Exception as error:
                self._cleanup_failed = True
                self.error=self.error or f"Hardware close: {error}"
                self.events.put({"kind": "warning", "detail": f"Hardware close: {error}"})
            try:
                self.policy.close()
            except Exception as error:
                self._cleanup_failed = True
                self.error = self.error or f"Policy close: {error}"
                self.events.put({"kind": "warning", "detail": f"Policy close: {error}"})
            finally:
                self._state("CLOSED" if self.closing.is_set() and not self.error else "ERROR", self.error or "")
                try:
                    self.save()
                except Exception as error:
                    self._cleanup_failed = True
                    self.error = self.error or f"Final save: {error}"
                    self._state("ERROR", self.error)
                finally:
                    self._cleanup_complete = True

    def _rollout(self):
        recorder = None
        folder = None
        self.rows = []
        # Preserve a stop issued after Start but before worker dispatch.
        with self.lock:
            already_cancelled=self.cancel.is_set() or self.closing.is_set()
            if not already_cancelled:
                self.request = None
        if already_cancelled:
            self._ready_after_stop()
            return
        try:
            self._state("HOMING")
            self._check_policy_ready()
            self.embodiment.home(self.cancel)
            if self.cancel.is_set():
                raise InterruptedError("home cancelled")
            self._sequence += 1
            episode = f"rollout_{self._sequence:04d}_{uuid.uuid4().hex[:6]}"
            folder = self.folder / episode
            recorder = RolloutRecorder(folder, self.config.target_hz, self.config.record_video)
            self.policy.reset(episode)
            if self.cancel.is_set():
                raise InterruptedError("reset cancelled")
            self.embodiment.begin_episode()
            self._state("RUNNING")
            for step in range(self.config.max_steps):
                if self.cancel.is_set():
                    break
                started = time.perf_counter()
                observation, capture_ms = self._preview()
                if step==0:
                    recorder.prepare(observation.images)
                prediction_started = time.perf_counter()
                prediction = self.policy.infer(observation, self.config.instruction)
                policy_ms = 1000*(time.perf_counter()-prediction_started)
                if self.cancel.is_set():
                    self._journal("cancelled_prediction", step=step,
                                  raw_action=prediction.action.tolist(), telemetry=prediction.telemetry)
                    break
                command_started = time.perf_counter()
                command = self.embodiment.apply(prediction)
                command_end = time.perf_counter()
                work_ms = 1000*(command_end-started)
                row = {**prediction.telemetry, **command, "step": step,
                       "captured_ns": observation.captured_ns, "state": observation.state.tolist(),
                       "captured_monotonic_ns":observation.captured_monotonic_ns,
                       "camera_metadata": observation.camera_metadata,
                       "command_monotonic_s": command_end, "capture_ms": capture_ms,
                       "policy_ms": policy_ms, "command_ms": 1000*(command_end-command_started),
                       "work_ms": work_ms, "deadline_miss": work_ms>1000/self.config.target_hz}
                recorder.write(row, observation)
                with self.lock:
                    self.rows.append(row)
                remaining = 1/self.config.target_hz-(time.perf_counter()-started)
                if remaining > 0:
                    self.cancel.wait(remaining)
            if not self.cancel.is_set():
                self._stop_confirmed = bool(self.embodiment.stop())
                if not self._stop_confirmed:
                    self._stop_failed=True
                    raise RuntimeError("Robot did not acknowledge stop at max_steps")
                self._state("WAITING_OUTCOME", "Maximum steps reached; select Success, Failure, Stop or Retry")
                while not self.cancel.wait(0.05):
                    self._preview()
        except InterruptedError:
            pass
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.request = "error"
            try:
                self._stop_confirmed = bool(self.embodiment.stop())
            except Exception:
                self._stop_confirmed=False
            if not self._stop_confirmed:
                self._stop_failed=True
            self._journal("rollout_error", detail=self.error)
        finally:
            if recorder:
                recorder.close()
            if not self._wait_for_stops():
                self.error=self.error or "Stop failed or timed out; automatic rearming disabled"
            outcome = self.request
            if folder and outcome in {"success", "failure", "error"}:
                result = {"id": self._sequence, "outcome": outcome, "directory": folder.name,
                          "target_hz": self.config.target_hz, "max_steps": self.config.max_steps,
                          "outcome_home": "pending" if outcome in {"success","failure"} else "not_requested",
                          "stop_confirmed": self._stop_confirmed, "error": self.error, **summary(self.rows)}
                write_json(folder / "result.json", result)
                with self.lock:
                    if outcome != "error":
                        self.results.append(result)
                self.save()
            elif folder:
                shutil.rmtree(folder)
                self._journal("discard_rollout", directory=folder.name, reason=outcome)
            if self.closing.is_set():
                return
            if not self._stop_confirmed or self.error:
                raise RuntimeError(self.error or "Arm stop was not confirmed; no home command issued")
            if outcome in {"success", "failure"}:
                # Clear cancellation only while reserving the outcome-home state.
                # A later Stop sets it again and the cancellable home remains stopped.
                with self.lock:
                    if self.request in {"success", "failure"}:
                        self.cancel.clear()
                        self._state("HOMING", "Outcome saved; returning home")
                if not self.cancel.is_set():
                    try:
                        self.embodiment.home(self.cancel)
                        result["outcome_home"] = "completed"
                    except InterruptedError:
                        result["outcome_home"] = "cancelled"
                    except Exception as error:
                        result["outcome_home"] = "error"
                        result["outcome_home_error"] = str(error)
                        write_json(folder / "result.json", result)
                        self.save()
                        raise
                else:
                    result["outcome_home"] = "cancelled"
                write_json(folder / "result.json", result)
            self._ready_after_stop()
            self.save()

    def close(self):
        with self.lock:
            if self.closing.is_set():
                return
            if self._cleanup_started:
                # Final cleanup already owns stopping/disconnection; do not race it.
                self.closing.set()
                self.cancel.set()
                return
            if self._started and not self.worker.is_alive() and self._cleanup_complete:
                self.closing.set()
                return
            self.signal("close")
            if not self._started:
                # A factory may return after Exit. Clean it up without connecting.
                self._started = True
                self.worker.start()
