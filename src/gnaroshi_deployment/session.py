"""Worker-owned rollout state machine shared by GUI and hardware-free tests."""
from __future__ import annotations
from collections import deque
import copy
from pathlib import Path
import json
import queue
import shutil
import threading
import time
import traceback
import uuid
from .contracts import CameraError, RobotRecoveryRequired, SessionConfig
import numpy as np
from .records import RolloutRecorder, provenance, summary, write_json, write_text


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
        self._policy_runtime_error = False
        self._policy_connected = False
        self._startup_error = False
        self._preview_failed = False
        self._policy_loader = None
        self._recovery_fault = False
        self._recovery_status = {}
        self._rollout_active = False
        self._home_active = False
        self._embodiment_connected = False
        self._notes = ""
        self._automatic_notes = []
        self._sequence = 0
        self._camera_error = None
        self._camera_return_state = None
        self._camera_stats = {}
        self._recorder = None
        self._recording_status = {"enabled": config.record_video, "healthy": True,
                                  "frames": 0, "writers": [], "error": None}
        selected = dict(getattr(embodiment, "metadata", {}).get("camera_serials", {}))
        self._device_status = {"scan_busy": False, "busy": False, "devices": [],
                               "selected": selected, "configuration": {}, "warnings": [],
                               "error": None}
        self._storage_health = {}
        self._last_storage_check = 0.0
        self.manifest = {"config": config.asdict(), "source": provenance(self.repo),
                         "timing": {"achieved_command_hz": "(commands-1)/(last-first command monotonic time)",
                                    "policy_ms": "client preprocessing/inference/postprocessing including RPC; buffered steps remain distinct",
                                    "video": "policy observations resampled at 30 fps using capture timestamps; duplicate/drop mapping in frames.jsonl"}}
        write_json(self.folder / "manifest.json", self.manifest)
        self.worker = threading.Thread(target=self._run, name="deployment-worker", daemon=True)
        self._update_storage_health(force=True)

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
                    and not bool(getattr(self.embodiment, "recovery_pending", False))
                    and not bool(getattr(self.embodiment, "camera_operation_pending", False))
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
                                                self.state == "ERROR" and (self._policy_load_error or self._policy_runtime_error))

    def _devices_allowed(self):
        return (self.state in {"SELECTING", "READY", "CAMERA_ERROR", "RECOVERY_REQUIRED", "LOADING_POLICY"}
                and not self._rollout_active and not self._home_active
                and not self.closing.is_set() and not self._cleanup_started
                and not self._cleanup_failed and self.worker.is_alive()
                and not bool(getattr(self.embodiment, "camera_operation_pending", False))
                and not bool(getattr(self.embodiment, "recovery_pending", False))
                and not any(thread.is_alive() for thread in self._stop_threads))

    @property
    def device_status(self):
        with self.lock:
            result = copy.deepcopy(self._device_status)
            allowed = self._devices_allowed()
            result.update(supported=callable(getattr(self.embodiment, "list_camera_devices", None)),
                          can_scan=allowed and not result["scan_busy"] and not result["busy"],
                          can_apply=allowed and not result["scan_busy"] and not result["busy"]
                          and (not self._embodiment_connected or self._recovery_fault or self._stop_confirmed and not self._stop_failed)
                          and callable(getattr(self.embodiment, "configure_cameras", None)))
            return result

    def request_device_scan(self):
        with self.lock:
            if not self.device_status["can_scan"]:
                return False
            self._device_status.update(scan_busy=True, error=None)
        self.commands.put(("device_scan",))
        return True

    def request_camera_update(self, serials):
        if not isinstance(serials, dict) or set(serials) != set(self.config.camera_roles):
            raise ValueError("Assign one camera serial to each configured camera role")
        serials = {role: str(value).strip() for role, value in serials.items()}
        if any(not value for value in serials.values()) or len(set(serials.values())) != len(serials):
            raise ValueError("Camera roles must use nonempty, distinct serials")
        with self.lock:
            if not self.device_status["can_apply"]:
                return False
            previous = self.state
            self._device_status.update(busy=True, error=None)
            self.state = "RECONFIGURING"
            self.cancel.clear()
            generation = self._stop_generation
        self.commands.put(("camera_update", serials, previous, generation))
        return True

    def request_camera_reconnect(self):
        return self.request_camera_update(self.device_status["selected"])

    def _update_storage_health(self, force=False):
        now = time.monotonic()
        if not force and now-self._last_storage_check < 2:
            return
        self._last_storage_check = now
        minimum = self.config.metadata.get("min_free_disk_bytes", 256*1024**2)
        warning = self.config.metadata.get("warn_free_disk_bytes", 2*1024**3)
        try:
            if type(minimum) is not int or minimum < 0 or type(warning) is not int or warning < minimum:
                raise ValueError("Storage thresholds must be nonnegative integer byte counts; warning >= minimum")
            usage = shutil.disk_usage(self.folder)
            error = f"Output storage has {usage.free/1024**2:.0f} MiB free; needs at least {minimum/1024**2:.0f} MiB" if usage.free < minimum else None
            status = {"disk_free_bytes": usage.free, "disk_total_bytes": usage.total,
                      "disk_warning": usage.free < warning, "disk_error": error,
                      "minimum_free_bytes": minimum}
        except (OSError, ValueError) as error:
            status = {"disk_free_bytes": None, "disk_total_bytes": None,
                      "disk_warning": True, "disk_error": str(error), "minimum_free_bytes": minimum}
        with self.lock:
            self._storage_health = status

    @property
    def operator_health(self):
        with self.lock:
            health = copy.deepcopy(self._storage_health)
            latest = self.latest
            cameras = copy.deepcopy(self._camera_stats)
            now_ns = time.monotonic_ns()
            for role in self.config.camera_roles:
                camera = cameras.setdefault(role, {})
                camera.update(error=self._camera_error, serial=self._device_status["selected"].get(role))
                camera["age_ms"] = max(0, (now_ns-latest.captured_monotonic_ns)/1e6) if latest is not None and latest.captured_monotonic_ns is not None else None
            recording = copy.deepcopy(self._recording_status)
            if self._recorder is not None:
                recording.update(self._recorder.health)
            health.update(cameras=cameras, recording=recording,
                          can_start=self.state == "READY" and not self._camera_error and not health.get("disk_error")
                          and not self._device_status["busy"] and not self._device_status["scan_busy"]
                          and not bool(getattr(self.embodiment, "camera_operation_pending", False))
                          and not self._recovery_fault and self._stop_confirmed and not self._stop_failed,
                          reason=self._camera_error or health.get("disk_error") or self.error,
                          policy_connected=self._policy_connected,
                          embodiment_connected=self._embodiment_connected)
            return health

    def _camera_failure(self, error, *, return_state=None):
        with self.lock:
            if self._camera_error is None:
                self._note_failure(self._failure_details(error, "camera"), event="camera_fault")
            self._camera_error = f"{type(error).__name__}: {error}"
            self._camera_return_state = return_state or ("READY" if self._embodiment_connected else "SELECTING")
            self._device_status["error"] = self._camera_error
            self.latest = None
            self.error = self._camera_error
            if not self.closing.is_set():
                self._state("RECOVERY_REQUIRED" if self._recovery_fault else "CAMERA_ERROR", self._camera_error)
        self.save()

    def _sync_camera_selection(self, selected):
        self.config.metadata["camera_serials"] = dict(selected)
        robot = self.config.metadata.get("robot")
        if isinstance(robot, dict):
            robot["camera_serials"] = dict(selected)
        selection = self.config.metadata.get("reload_selection")
        if isinstance(selection, dict):
            selection["camera_serials"] = dict(selected)
            for role, serial in selected.items():
                selection[f"{role}_serial"] = serial
            resolved = selection.get("resolved_config", {})
            if isinstance(resolved.get("robot"), dict):
                resolved["robot"]["camera_serials"] = dict(selected)
        self.manifest.update(config=self.config.asdict(), embodiment=self.embodiment.metadata)
        write_json(self.folder / "manifest.json", self.manifest)

    def _device_command(self, command):
        if not isinstance(command, tuple) or command[0] not in {"device_scan", "camera_update"}:
            return False
        if command[0] == "device_scan":
            try:
                report = self.embodiment.list_camera_devices()
                if not isinstance(report, dict) or not isinstance(report.get("devices"), list):
                    raise ValueError("Device enumeration must return a devices list")
                with self.lock:
                    self._device_status.update(copy.deepcopy(report), error=None)
                self._journal("camera_devices_scanned", devices=report["devices"])
            except Exception as error:
                with self.lock:
                    self._device_status["error"] = f"{type(error).__name__}: {error}"
                self._journal("camera_scan_failed", detail=self._device_status["error"])
            finally:
                with self.lock:
                    self._device_status["scan_busy"] = False
                self.events.put({"kind": "device_scan", "status": self.device_status})
            return True
        _, selected, previous, generation = command
        origin = self._camera_return_state or previous
        try:
            if self.closing.is_set() or self.cancel.is_set() or generation != self._stop_generation:
                raise InterruptedError("Camera change cancelled")
            if self._embodiment_connected and not self._recovery_fault:
                if not self._wait_for_stops() or not self.embodiment.stop():
                    self._stop_confirmed = False
                    self._stop_failed = True
                    raise RuntimeError("Stop was not confirmed before changing cameras")
                self._stop_confirmed = True
            report = self.embodiment.configure_cameras(selected, self.cancel)
            if self.closing.is_set() or self.cancel.is_set() or generation != self._stop_generation:
                raise InterruptedError("Camera change cancelled")
            actual = dict(getattr(self.embodiment, "metadata", {}).get("camera_serials", {}))
            if not isinstance(report, dict) or report.get("selected") != selected or actual != selected:
                raise RuntimeError("Camera adapter did not confirm the requested assignments")
            with self.lock:
                self._device_status.update(copy.deepcopy(report), selected=actual)
                self._sync_camera_selection(actual)
            if self._policy_connected:
                try:
                    self._policy_operation(self.policy.reset, f"camera_change_{uuid.uuid4().hex}")
                except Exception:
                    self._policy_runtime_error = True
                    raise
            if self.closing.is_set() or self.cancel.is_set() or generation != self._stop_generation:
                raise InterruptedError("Camera change cancelled during policy reset")
            with self.lock:
                self._device_status.update(copy.deepcopy(report), selected=actual, error=None)
                self._camera_error = None
                self._camera_return_state = None
                self._camera_stats.clear()
                self.latest = None
                if not self._recovery_fault and self._stop_confirmed and not self._stop_failed:
                    self.error = self.error_detail = None
            self._journal("camera_configuration_changed", selected=actual, report=report,
                          policy_reloaded=False, motion_started=False)
            self.events.put({"kind": "cameras_changed", "selected": actual})
        except CameraError as error:
            with self.lock:
                self._device_status["selected"] = dict(getattr(self.embodiment, "metadata", {}).get("camera_serials", self._device_status["selected"]))
            self._camera_failure(error, return_state=origin)
        except RobotRecoveryRequired as error:
            self._note_failure(self._failure_details(error, "camera_configuration"), event="camera_change_robot_fault")
            self._require_recovery(error)
        except InterruptedError:
            self._journal("camera_change_cancelled")
            if not self.closing.is_set():
                self._wait_for_stops()
                actual = dict(getattr(self.embodiment, "metadata", {}).get("camera_serials", {}))
                self._device_status.update(selected=actual, error="Camera change cancelled; reconnect cameras before starting")
                self._camera_failure(CameraError("Camera change was cancelled; reconnect cameras"), return_state=previous)
        except Exception as error:
            if self._policy_runtime_error or self._stop_failed or not self._stop_confirmed:
                raise
            self._camera_failure(CameraError(str(error), category="camera_configuration"), return_state=origin)
        finally:
            with self.lock:
                self._device_status["busy"] = False
                if not self.closing.is_set() and self.state == "RECONFIGURING":
                    if self._recovery_fault:
                        self._state("RECOVERY_REQUIRED", "Robot recovery is still required")
                    elif self._camera_error:
                        self._state("CAMERA_ERROR", self._camera_error)
                    elif origin == "LOADING_POLICY" or (self._policy_loader is not None and self._policy_loader.is_alive()) or self._policy_connected and not self._embodiment_connected:
                        self._state("LOADING_POLICY")
                    else:
                        self.cancel.clear()
                        self.request = None
                        self._state("READY" if self._embodiment_connected else "SELECTING")
            self.save()
        return True

    def _next_device_command(self):
        try:
            command = self.commands.get_nowait()
        except queue.Empty:
            return False
        if not self._device_command(command):
            self.commands.put(command)
            return False
        return True

    def _wait_for_camera_repair(self):
        while self._camera_error and not self.closing.is_set():
            self._update_storage_health()
            self._next_device_command()
            self.closing.wait(.04)

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

    @staticmethod
    def _json_safe(value):
        """Keep diagnostic evidence even when a failed policy emits NaN/Inf."""
        if isinstance(value, dict):
            return {str(key): DeploymentSession._json_safe(item) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            return DeploymentSession._json_safe(value.item() if value.ndim == 0 else value.tolist())
        if isinstance(value, (list, tuple)):
            return [DeploymentSession._json_safe(item) for item in value]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and not np.isfinite(value):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)

    def _failure_details(self, error, phase, step=None):
        default = ("robot_fault" if isinstance(error, RobotRecoveryRequired) else
                   "policy_error" if phase in {"policy_ready", "reset", "infer"} else
                   "invalid_action" if phase == "apply" and isinstance(error, ValueError) else
                   "recording_error" if phase in {"record", "record_close"} else f"{phase}_error")
        status = self._json_safe(getattr(error, "status", {}))
        category = str(getattr(error, "category", default))
        note = f"[{category}] {error}"
        if isinstance(status, dict) and status.get("action_translation_norm_m") is not None:
            value = status["action_translation_norm_m"]
            if isinstance(value, (int, float)):
                note += f" · translation action {1000*value:.2f} mm"
        return {"category": category, "reason": str(error), "note": note, "phase": phase,
                "exception_type": type(error).__name__, "status": status, "step": step,
                "time_ns": time.time_ns()}

    def _note_failure(self, failure, rollout_id=None, *, event="rollout_failed"):
        note = {"rollout_id": rollout_id, **failure}
        with self.lock:
            self._automatic_notes.append(note)
        self._journal(event, **note)
        self.events.put({"kind": "failure_note", **note})

    @property
    def automatic_notes(self):
        with self.lock:
            return list(self._automatic_notes)

    @property
    def can_home(self):
        with self.lock:
            return (self.state == "READY" and self._embodiment_connected
                    and not self._rollout_active and not self._home_active
                    and not self._recovery_fault and not self.closing.is_set()
                    and not self._cleanup_started and not self._cleanup_failed
                    and self._stop_confirmed and not self._stop_failed
                    and not bool(getattr(self.embodiment, "recovery_pending", False))
                    and not bool(getattr(self.embodiment, "camera_operation_pending", False))
                    and not any(thread.is_alive() for thread in self._stop_threads)
                    and self.worker.is_alive())

    def home_robot(self):
        """Queue an idle pose-only home; never creates a rollout or resets policy."""
        with self.lock:
            if not self.can_home:
                return False
            self.request = "home"
            self.cancel.clear()
            self.state = "HOMING"
            generation = self._stop_generation
        self._journal("idle_home_requested", stop_generation=generation)
        self.commands.put(("home", generation))
        return True

    def _home_robot(self, generation):
        if (self.closing.is_set() or self.cancel.is_set() or self.request != "home"
                or generation != self._stop_generation):
            if self._recovery_fault:
                # The fault/recovery operation owns the current state. Dropping
                # an older queued Home must not cancel a newer Recover request.
                return
            self._ready_after_stop()
            return
        try:
            if not self._wait_for_stops():
                raise RuntimeError("Stop was not acknowledged; home remains blocked")
            self._check_policy_ready()
            if self.cancel.is_set() or self.closing.is_set() or generation != self._stop_generation:
                raise InterruptedError("Idle home cancelled")
            self._state("HOMING", "Moving to task home; no rollout or policy reset")
            home = getattr(self.embodiment, "home_idle", None)
            if not callable(home):
                home = self.embodiment.home
            home(self.cancel)
            if self.cancel.is_set() or self.closing.is_set():
                raise InterruptedError("Idle home cancelled")
            self._stop_confirmed = bool(self.embodiment.stop())
            if not self._stop_confirmed:
                self._stop_failed = True
                fault = getattr(self.embodiment, "recovery_error", None)
                if isinstance(fault, RobotRecoveryRequired):
                    raise fault
                raise RuntimeError("Robot did not acknowledge stop after idle home")
            self._journal("idle_home_completed")
            self._ready_after_stop()
        except InterruptedError:
            self._journal("idle_home_cancelled")
            self._ready_after_stop()
        except RobotRecoveryRequired as error:
            self._note_failure(self._failure_details(error, "idle_home"), event="idle_home_failed")
            self._require_recovery(error)
        except Exception as error:
            self._note_failure(self._failure_details(error, "idle_home"), event="idle_home_failed")
            raise
        finally:
            self.save()

    def start_rollout(self):
        with self.lock:
            if not self.operator_health["can_start"]:
                if self.state == "READY":
                    self.events.put({"kind": "warning", "detail": self.operator_health["reason"] or "Session is not ready to start"})
                return False
            # Reserve the state before queueing: double-click cannot queue two homes.
            self.state = "STARTING"
        self.commands.put("start")
        return True

    def signal(self, action):
        if action not in {"success", "failure", "stop", "retry", "close"}:
            raise ValueError(action)
        with self.lock:
            if self._cleanup_started:
                if action == "close":
                    self.closing.set()
                    self.cancel.set()
                return False
            if action != "close" and self.state not in {"STARTING", "HOMING", "RUNNING", "WAITING_OUTCOME", "STOPPING", "RECOVERY_REQUIRED", "RECOVERING", "RECONFIGURING", "CAMERA_ERROR"}:
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

    @property
    def can_recover(self):
        with self.lock:
            return (self.state == "RECOVERY_REQUIRED" and self._recovery_fault and not self.closing.is_set()
                    and not self._rollout_active
                    and not self._home_active
                    and self.worker.is_alive() and not self._cleanup_failed
                    and callable(getattr(self.embodiment, "recover", None))
                    and not any(thread.is_alive() for thread in self._stop_threads))

    def recover_robot(self):
        with self.lock:
            if not self.can_recover:
                return False
            self.request = "recover"
            self.cancel.clear()
            self.state = "RECOVERING"
            generation = self._stop_generation
        self._journal("robot_recovery_requested", stop_generation=generation)
        self.commands.put(("recover", generation))
        return True

    def _require_recovery(self, error):
        with self.lock:
            self.error = f"{type(error).__name__}: {error}"
            self.error_detail = traceback.format_exc()
            self._recovery_fault = True
            self._recovery_status = self._json_safe(error.status)
            self.cancel.set()
            if not self.closing.is_set():
                self.request = "recovery_required"
                self.state = "RECOVERY_REQUIRED"
            self._stop_confirmed = False
            self._stop_generation += 1
            thread = threading.Thread(target=self._stop, args=(self._stop_generation,), name="recovery-stop", daemon=True)
            self._stop_threads.append(thread)
            thread.start()
        self._journal("robot_recovery_required", detail=self.error, status=self._recovery_status,
                      traceback=self.error_detail)
        if not self.closing.is_set():
            self._state("RECOVERY_REQUIRED", self.error+" · Release the fault on the teach pendant, then Recover robot")

    def _recovery_cancelled(self, generation):
        return (self.closing.is_set() or self.cancel.is_set() or self.request != "recover"
                or generation != self._stop_generation)

    def _recover_robot(self, generation):
        if self._recovery_cancelled(generation):
            return
        # Model transport/reset errors retain the existing fatal-error behavior.
        self._check_policy_ready()
        try:
            if any(thread.is_alive() for thread in self._stop_threads):
                raise RuntimeError("Wait for all previous stop requests to finish")
            report = self.embodiment.recover(self.cancel)
            if self._recovery_cancelled(generation):
                raise InterruptedError("Robot recovery cancelled")
            if not isinstance(report, dict) or report.get("healthy") is not True or report.get("stop_confirmed") is not True:
                with self.lock:
                    self._stop_failed = True
                    self._stop_confirmed = False
                raise RuntimeError("Recovery did not verify fresh healthy state and acknowledged stop")
        except Exception as error:
            self.error_detail = traceback.format_exc()
            self._journal("robot_recovery_failed", detail=f"{type(error).__name__}: {error}",
                          status=getattr(error, "status", {}), traceback=self.error_detail)
            if not self.closing.is_set():
                self.error = f"Recovery failed: {type(error).__name__}: {error}"
                self.cancel.set()
                self._state("RECOVERY_REQUIRED", self.error)
                self.save()
            return
        # Reset only after adapter verification. Never reuse the old action buffer.
        self._policy_operation(self.policy.reset, f"recovery_{uuid.uuid4().hex}")
        with self.lock:
            if self._recovery_cancelled(generation) or any(thread.is_alive() for thread in self._stop_threads):
                self._journal("robot_recovery_cancelled", stop_generation=generation, stage="policy_reset")
                if not self.closing.is_set():
                    self.cancel.set()
                    self._state("RECOVERY_REQUIRED", "Recovery cancelled; robot remains stopped")
                return
            self._stop_failed = False
            self._stop_confirmed = True
            self._stop_threads.clear()
            self._stop_results.clear()
            self._recovery_fault = False
            self._recovery_status = {}
            self.error = self._camera_error
            self.error_detail = None
            self.request = None
            self.cancel.clear()
            self._state("CAMERA_ERROR" if self._camera_error else "READY",
                        self._camera_error or "Robot recovery verified; press Start separately to move home and begin a new rollout")
        self._journal("robot_recovery_succeeded", report=report)
        self.save()

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
                fault = getattr(self.embodiment, "recovery_error", None)
                if isinstance(fault, RobotRecoveryRequired):
                    if not self._recovery_fault:
                        self._note_failure(self._failure_details(fault, "stop"), event="robot_stop_fault")
                        self._require_recovery(fault)
                    self.save()
                    return
                raise RuntimeError("Stop failed or timed out; restart this session after checking the robot")
            with self.lock:
                if any(thread.is_alive() for thread in self._stop_threads):
                    continue
                if self._stop_failed:
                    raise RuntimeError("A stop was not confirmed; automatic rearming is disabled")
                if not self.closing.is_set():
                    if self._recovery_fault:
                        # A queued Home may become stale while an idle observation
                        # discovers a device fault. Acknowledging Stop alone does
                        # not complete the explicit robot recovery contract.
                        self.request = "recovery_required"
                        self.cancel.set()
                        self._state("RECOVERY_REQUIRED", "Robot remains stopped; an explicit recovery is required")
                        return
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

    def apply_recording(self, enabled):
        if type(enabled) is not bool:
            raise ValueError("recording must be enabled or disabled explicitly")
        with self.lock:
            if self.state != "READY":
                return False
            self.config.record_video = enabled
            self._recording_status.update(enabled=enabled)
            selection = self.config.metadata.get("reload_selection", {})
            selection["record_video"] = enabled
            if isinstance(selection.get("resolved_config"), dict):
                selection["resolved_config"]["record_video"] = enabled
            self.manifest["config"] = self.config.asdict()
            write_json(self.folder / "manifest.json", self.manifest)
            self._journal("recording_setting_changed", enabled=enabled)
        return True

    def save(self, notes=None):
        with self.lock:
            if notes is not None:
                self._notes = notes
            successes = sum(r["outcome"] == "success" for r in self.results)
            payload = {"results": list(self.results), "successes": successes,
                       "failures": sum(r["outcome"] == "failure" for r in self.results),
                       "success_rate": successes/len(self.results) if self.results else None,
                       "notes": self._notes, "automatic_notes": list(self._automatic_notes),
                       "state": self.state, "error": self.error}
            write_json(self.folder / "results.json", payload)
            automatic = "\n".join(
                f"Rollout {item['rollout_id']}: {item['note']}" if item.get("rollout_id") is not None
                else item['note'] for item in self._automatic_notes)
            write_text(self.folder / "notes.txt", "\n\n".join(part for part in (self._notes, automatic) if part))

    def annotate_result(self, rollout_id, note):
        """Annotate an accepted result without changing its outcome or media."""
        if not isinstance(note, str) or len(note) > 100000:
            raise ValueError("A rollout note must be text shorter than 100000 characters")
        with self.lock:
            if self.closing.is_set() or self._cleanup_started and not self._cleanup_complete:
                return False
            result = next((row for row in self.results if str(row["id"]) == str(rollout_id)), None)
            if result is None or result.get("outcome_home") == "pending":
                return False
            target = (self.folder/result["directory"]).resolve()
            if target.parent != self.folder or not target.is_dir():
                raise ValueError("Saved result path is outside this session or is unavailable")
            result.update(operator_note=note, operator_note_updated_ns=time.time_ns())
            write_json(target/"result.json", result)
            self._journal("result_annotated", rollout_id=result["id"], note=note)
            self.save()
        self.events.put({"kind": "result_annotated", "rollout_id": result["id"]})
        return True

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
            now = time.monotonic()
            for role in self.config.camera_roles:
                stats = self._camera_stats.setdefault(role, {})
                metadata = observation.camera_metadata.get(role, {})
                metadata = metadata if isinstance(metadata, dict) else {"serial": metadata}
                previous = stats.get("last_observed_s")
                interval = now-previous if previous is not None else None
                if interval is not None and interval > 0:
                    measured = 1/interval
                    stats["fps"] = .8*stats.get("fps", measured)+.2*measured
                stats.update(last_observed_s=now, frame_number=metadata.get("frame_number"))
                timestamp = metadata.get("device_timestamp_ms")
                domain = str(metadata.get("timestamp_domain", ""))
                stats["source_age_ms"] = max(0, time.time()*1000-timestamp) if isinstance(timestamp, (float, int)) and ("global_time" in domain or "system_time" in domain) else None
        return observation, 1000*(time.perf_counter()-started)

    def _check_policy_ready(self):
        check = getattr(self.policy, "check_ready", None)
        if callable(check):
            try:
                self._policy_operation(check)
            except Exception:
                self._policy_runtime_error = True
                raise

    @staticmethod
    def _policy_operation(operation, *args):
        try:
            return operation(*args)
        except RobotRecoveryRequired as error:
            raise RuntimeError(f"Policy operation failed: {error}") from error

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
                    self._update_storage_health()
                    self._next_device_command()
                    try:
                        if not self._camera_error:
                            self._preview(preview_only=True)
                    except CameraError as error:
                        self._camera_failure(error, return_state="LOADING_POLICY")
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
                try:
                    self.embodiment.connect_preview()
                except CameraError as error:
                    self._camera_failure(error, return_state="SELECTING" if self.config.metadata.get("defer_policy_load", False) else "CONNECTING_PREVIEW")
                    self._wait_for_camera_repair()
                if self.closing.is_set():
                    return
            if self.config.metadata.get("defer_policy_load", False):
                stage = "selection"
                self._state("SELECTING", "Select a task and checkpoint, then Load checkpoint · policy and robot controls are not connected")
                while not self.closing.is_set():
                    self._update_storage_health()
                    self._next_device_command()
                    if preview and not self._camera_error:
                        try:
                            self._preview(preview_only=True)
                        except CameraError as error:
                            self._camera_failure(error, return_state="SELECTING")
                    self.closing.wait(.04)
                return
            # Metadata validation belongs in policy.connect, before robot.connect.
            stage = "policy"
            if preview:
                self._connect_policy_with_preview()
            else:
                self._state("CONNECTING", "Loading and validating the selected policy")
                self.policy.connect()
            self._policy_connected = True
            if self._camera_error:
                self._wait_for_camera_repair()
            if self.closing.is_set():
                return
            stage = "embodiment"
            self.embodiment.connect()
            self._embodiment_connected = True
            self.manifest.update(policy=self.policy.metadata, embodiment=self.embodiment.metadata)
            write_json(self.folder / "manifest.json", self.manifest)
            if not self.closing.is_set():
                self._state("READY")
            stage = "running"
            while not self.closing.is_set():
                self._update_storage_health()
                try:
                    command = self.commands.get_nowait()
                except queue.Empty:
                    command = None
                if self._device_command(command):
                    pass
                elif command == "start" and not self._recovery_fault and not self._camera_error:
                    self._rollout_active = True
                    try:
                        self._rollout()
                    finally:
                        self._rollout_active = False
                elif isinstance(command, tuple) and command[0] == "recover":
                    self._recover_robot(command[1])
                elif isinstance(command, tuple) and command[0] == "home":
                    self._home_active = True
                    try:
                        self._home_robot(command[1])
                    finally:
                        self._home_active = False
                elif self.state == "READY":
                    self._check_policy_ready()
                    try:
                        self._preview()
                    except RobotRecoveryRequired as error:
                        self._note_failure(self._failure_details(error, "idle_observe"), event="idle_robot_fault")
                        self._require_recovery(error)
                        self.save()
                    except CameraError as error:
                        self._camera_failure(error)
                    self.closing.wait(0.04)
                elif self._recovery_fault:
                    if self.state == "STOPPING" and not any(thread.is_alive() for thread in self._stop_threads):
                        self._state("RECOVERY_REQUIRED", "Robot remains stopped; an explicit recovery is required")
                    self._check_policy_ready()
                    if not self._camera_error and callable(getattr(self.embodiment, "observe_preview", None)):
                        try:
                            self._preview(preview_only=True)
                        except CameraError as error:
                            self._camera_failure(error, return_state="RECOVERY_REQUIRED")
                    self.closing.wait(.04)
                elif self.state == "STOPPING" and self._camera_error:
                    self._wait_for_stops()
                    self._state("CAMERA_ERROR", self._camera_error)
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
            if (bool(getattr(self.embodiment, "recovery_pending", False))
                    or bool(getattr(self.embodiment, "camera_operation_pending", False))
                    or any(thread.is_alive() for thread in self._stop_threads)):
                self.events.put({"kind": "warning", "detail": "Waiting for the pending robot operation before closing devices"})
                self._journal("waiting_for_native_robot_cleanup")
                while (bool(getattr(self.embodiment, "recovery_pending", False))
                       or bool(getattr(self.embodiment, "camera_operation_pending", False))
                       or any(thread.is_alive() for thread in self._stop_threads)):
                    # Native SDK calls cannot be interrupted safely. Keep the GUI
                    # responsive and let this owner finish cleanup after the call.
                    time.sleep(.05)
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
                    self._embodiment_connected = False
            except Exception as error:
                self._cleanup_failed = True
                self.error=self.error or f"Hardware close: {error}"
                self.events.put({"kind": "warning", "detail": f"Hardware close: {error}"})
            try:
                self.policy.close()
                self._policy_connected = False
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
        recoverable_error = None
        camera_error = None
        fatal_rollout_error = False
        failure = None
        recording_errors = []
        phase, step = "start", None
        observation = prediction = None
        self.rows = []
        # Preserve a stop issued after Start but before worker dispatch.
        with self.lock:
            if self._recovery_fault:
                return
            already_cancelled=self.cancel.is_set() or self.closing.is_set()
            if not already_cancelled:
                self.request = None
        if already_cancelled:
            self._ready_after_stop()
            return
        self._update_storage_health(force=True)
        if self._storage_health.get("disk_error"):
            self._journal("start_blocked_storage", detail=self._storage_health["disk_error"])
            self.events.put({"kind": "warning", "detail": self._storage_health["disk_error"]})
            self._state("READY", self._storage_health["disk_error"])
            return
        camera_serials = dict(self._device_status["selected"])
        try:
            self._state("HOMING")
            self._sequence += 1
            episode = f"rollout_{self._sequence:04d}_{uuid.uuid4().hex[:6]}"
            folder = self.folder / episode
            phase = "record_open"
            recorder = RolloutRecorder(folder, self.config.target_hz, self.config.record_video)
            self._recorder = recorder
            phase = "policy_ready"
            self._check_policy_ready()
            phase = "home"
            self.embodiment.home(self.cancel)
            if self.cancel.is_set():
                raise InterruptedError("home cancelled")
            phase = "reset"
            self._policy_operation(self.policy.reset, episode)
            if self.cancel.is_set():
                raise InterruptedError("reset cancelled")
            phase = "begin_episode"
            self.embodiment.begin_episode()
            self._state("RUNNING")
            for step in range(self.config.max_steps):
                if self.cancel.is_set():
                    break
                self._update_storage_health()
                if self._storage_health.get("disk_error"):
                    phase = "record"
                    raise OSError(self._storage_health["disk_error"])
                started = time.perf_counter()
                observation = prediction = None
                phase = "observe"
                observation, capture_ms = self._preview()
                if step==0:
                    phase = "record"
                    recorder.prepare(observation.images)
                prediction_started = time.perf_counter()
                phase = "infer"
                prediction = self._policy_operation(self.policy.infer, observation, self.config.instruction)
                policy_ms = 1000*(time.perf_counter()-prediction_started)
                if self.cancel.is_set():
                    self._journal("cancelled_prediction", step=step,
                                  raw_action=prediction.action.tolist(), telemetry=prediction.telemetry)
                    break
                command_started = time.perf_counter()
                phase = "apply"
                command = self.embodiment.apply(prediction)
                command_end = time.perf_counter()
                work_ms = 1000*(command_end-started)
                row = {**prediction.telemetry, **command, "step": step, "step_status": "executed",
                       "captured_ns": observation.captured_ns, "state": observation.state.tolist(),
                       "captured_monotonic_ns":observation.captured_monotonic_ns,
                       "camera_metadata": observation.camera_metadata,
                       "command_monotonic_s": command_end, "capture_ms": capture_ms,
                       "policy_ms": policy_ms, "command_ms": 1000*(command_end-command_started),
                       "work_ms": work_ms, "deadline_miss": work_ms>1000/self.config.target_hz}
                with self.lock:
                    self.rows.append(row)
                phase = "record"
                recorder.write(row, observation)
                remaining = 1/self.config.target_hz-(time.perf_counter()-started)
                if remaining > 0:
                    self.cancel.wait(remaining)
            if not self.cancel.is_set():
                phase = "stop"
                observation = prediction = None
                self._stop_confirmed = bool(self.embodiment.stop())
                if not self._stop_confirmed:
                    self._stop_failed=True
                    fault = getattr(self.embodiment, "recovery_error", None)
                    if isinstance(fault, RobotRecoveryRequired):
                        raise fault
                    raise RuntimeError("Robot did not acknowledge stop at max_steps")
                self._state("WAITING_OUTCOME", "Maximum steps reached; select Success, Failure, Stop or Retry")
                while not self.cancel.wait(0.05):
                    phase = "observe"
                    self._preview()
        except InterruptedError:
            pass
        except RobotRecoveryRequired as error:
            recoverable_error = error
            failure = self._failure_details(error, phase, step)
            if folder is None:
                self._sequence += 1
                folder = self.folder / f"rollout_{self._sequence:04d}_{uuid.uuid4().hex[:6]}"
                recorder = RolloutRecorder(folder, self.config.target_hz, self.config.record_video)
            self._require_recovery(error)
        except CameraError as error:
            camera_error = error
            failure = self._failure_details(error, phase, step)
            self.request = "camera_error"
            try:
                self._stop_confirmed = bool(self.embodiment.stop())
            except Exception:
                self._stop_confirmed = False
            if not self._stop_confirmed:
                self._stop_failed = True
            self._camera_failure(error)
        except Exception as error:
            fatal_rollout_error = True
            self._policy_runtime_error = phase in {"policy_ready", "reset", "infer"}
            if phase in {"record", "record_open", "record_close"}:
                self._recording_status.update(healthy=False, error=f"{type(error).__name__}: {error}")
                if recorder is not None:
                    recorder.error = self._recording_status["error"]
            failure = self._failure_details(error, phase, step)
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
            accepted_stop_outcome = self.request if failure is None and self.request in {"success", "failure"} else None
            if not self._wait_for_stops():
                fault = getattr(self.embodiment, "recovery_error", None)
                if not fatal_rollout_error and recoverable_error is None and isinstance(fault, RobotRecoveryRequired):
                    accepted_stop_outcome = self.request if self.request in {"success", "failure"} else None
                    recoverable_error = fault
                    failure = self._failure_details(fault, "stop", step)
                    observation = prediction = None
                    if not self._recovery_fault:
                        self._require_recovery(fault)
                else:
                    self.error=self.error or "Stop failed or timed out; automatic rearming disabled"
                    if failure is None:
                        accepted_stop_outcome = self.request if self.request in {"success", "failure"} else None
                        failure = self._failure_details(RuntimeError(self.error), "stop", step)
                        observation = prediction = None
            if failure is not None and folder:
                status = failure["status"]
                command_outcome = ("unconfirmed" if failure["phase"] == "apply" else
                                   "acknowledged" if failure["phase"] == "record" and self.rows else
                                   "not_attempted")
                if failure["phase"] == "apply" and isinstance(status, dict):
                    if status.get("arm_command_acknowledged") is True:
                        command_outcome = "acknowledged"
                    elif status.get("arm_command_attempted") is False or status.get("arm_command_sent") is False:
                        command_outcome = "not_attempted"
                attempted = {"step": step, "step_status": "failed", "failure": failure,
                             "command_outcome": command_outcome}
                if prediction is not None:
                    attempted.update(raw_action=self._json_safe(getattr(prediction, "action", prediction)),
                                     telemetry=self._json_safe(getattr(prediction, "telemetry", {})))
                if observation is not None:
                    attempted.update(state=self._json_safe(observation.state),
                                     captured_ns=observation.captured_ns,
                                     captured_monotonic_ns=observation.captured_monotonic_ns,
                                     camera_metadata=self._json_safe(observation.camera_metadata))
                try:
                    write_json(folder / "failed_step.json", attempted)
                    if recorder and observation is not None and phase != "record":
                        # The last policy input belongs in video even when infer/apply failed.
                        # This row has no command timestamp and is excluded from self.rows.
                        recorder.write(attempted, observation)
                except Exception as error:
                    recording_errors.append(f"Failure evidence: {type(error).__name__}: {error}")
            if recorder:
                try:
                    recorder.close()
                except Exception as error:
                    recording_errors.append(f"Recorder close: {type(error).__name__}: {error}")
                    if failure is None:
                        failure = self._failure_details(error, "record_close", step)
                        self.error = f"{type(error).__name__}: {error}"
                        fatal_rollout_error = True
                finally:
                    with self.lock:
                        self._recording_status = recorder.health
                        if recording_errors:
                            self._recording_status.update(healthy=False, error="; ".join(recording_errors))
                        self._recorder = None
            if failure is not None:
                self._note_failure(failure, self._sequence)
            outcome = accepted_stop_outcome or ("failure" if failure is not None else self.request)
            if folder and outcome in {"success", "failure"}:
                result = {"id": self._sequence, "outcome": outcome, "directory": folder.name,
                          "target_hz": self.config.target_hz, "max_steps": self.config.max_steps,
                          "outcome_source": "automatic_fault" if failure is not None and not accepted_stop_outcome else "manual",
                          "outcome_home": ("blocked_recovery" if recoverable_error is not None and accepted_stop_outcome else
                                           "blocked_error" if failure is not None and accepted_stop_outcome else
                                           "not_requested" if failure is not None else "pending"),
                          "stop_confirmed": self._stop_confirmed, "error": self.error,
                          "recording_errors": recording_errors,
                          "camera_serials": camera_serials,
                          "recording": dict(self._recording_status),
                          "failure": failure,
                          "failure_category": failure["category"] if failure else None,
                          "failure_reason": failure["reason"] if failure else None,
                          "failure_note": failure["note"] if failure else None,
                          "failure_status": failure["status"] if failure else {},
                          "failure_phase": failure["phase"] if failure else None,
                          "recovery_required": recoverable_error is not None, **summary(self.rows)}
                write_json(folder / "result.json", result)
                with self.lock:
                    self.results.append(result)
                self.save()
            elif folder:
                shutil.rmtree(folder)
                self._journal("discard_rollout", directory=folder.name, reason=outcome)
            if self.closing.is_set():
                return
            if recoverable_error is not None:
                self.save()
                return
            if camera_error is not None and self._stop_confirmed and not self._stop_failed and not fatal_rollout_error:
                self._state("CAMERA_ERROR", self._camera_error)
                self.save()
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
                    except RobotRecoveryRequired as error:
                        self._require_recovery(error)
                        failure = self._failure_details(error, "outcome_home")
                        self._note_failure(failure, self._sequence, event="outcome_home_failed")
                        result.update(outcome_home="error", outcome_home_error=str(error), error=self.error,
                                      recovery_required=True, stop_confirmed=False, outcome_home_failure=failure,
                                      failure_category=failure["category"], failure_reason=failure["reason"],
                                      failure_note=failure["note"], failure_status=failure["status"], failure_phase=failure["phase"])
                        write_json(folder / "result.json", result)
                        self.save()
                        return
                    except Exception as error:
                        failure = self._failure_details(error, "outcome_home")
                        self._note_failure(failure, self._sequence, event="outcome_home_failed")
                        result["outcome_home"] = "error"
                        result["outcome_home_error"] = str(error)
                        result.update(outcome_home_failure=failure, failure_category=failure["category"],
                                      failure_reason=failure["reason"], failure_note=failure["note"],
                                      failure_status=failure["status"], failure_phase=failure["phase"])
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
