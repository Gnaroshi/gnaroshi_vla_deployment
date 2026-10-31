"""Explicit two-RealSense UR5e adapter for tool-local relative EEF actions.

The owning project supplies its robot address, task home and camera serial roles.
Camera-only preview is separate from policy-gated control connections.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import socket
import threading
import time
import importlib.metadata
import numpy as np
from ..contracts import CameraError, Observation, RobotActionError, RobotRecoveryRequired, action_vector


def _timed_call(timings, name, operation, *args):
    """Measure an existing driver call without adding reads, retries or motion."""
    started=time.perf_counter()
    try:
        return operation(*args)
    finally:
        timings[name]=1000*(time.perf_counter()-started)


@dataclass
class UR5eConfig:
    robot_ip: str = ""
    camera_serials: dict = field(default_factory=lambda:{"primary":"","wrist":""})
    camera_fps: int = 30
    width: int = 640
    height: int = 480
    camera_wait_timeout_ms: int = 1000
    camera_connect_timeout_s: float = 10.0
    camera_scan_timeout_s: float = 5.0
    max_camera_frame_age_s: float = 1.0
    home_joints: tuple = ()
    home_duration_s: float = 3.0
    home_hz: int = 10
    home_joint_tolerance_rad: float = 0.01
    home_settle_timeout_s: float = 3.0
    close_threshold: float = 0.25
    release_threshold: float = 0.10
    gripper_speed: int = 255
    gripper_force: int = 10
    gripper_period_s: float = 0.5
    gripper_open_tolerance: int = 8
    gripper_open_timeout_s: float = 3.0
    initial_open_steps: int = 5
    servo_time_s: float = 1/500
    servo_lookahead_s: float = 0.2
    servo_gain: float = 100
    velocity: float = 0.5
    acceleration: float = 0.5
    recovery_timeout_s: float = 10.0
    recovery_freshness_timeout_s: float = 1.0
    # None preserves the accepted baseline's raw action convention.
    max_rel_position: float | None = None
    max_rel_rotation: float | None = None

    def validate(self):
        if not isinstance(self.robot_ip,str) or not self.robot_ip.strip():
            raise ValueError("Robot address must be supplied by the owning project")
        if set(self.camera_serials)!={"primary","wrist"} or any(not isinstance(v,str) or not v.strip() or v!=v.strip() for v in self.camera_serials.values()):
            raise ValueError("Both primary and wrist camera serials must be explicit")
        if len(set(self.camera_serials.values()))!=2:
            raise ValueError("Primary and wrist must be different cameras")
        if any(type(value) is not int or value<1 for value in (self.width,self.height,self.camera_fps,self.camera_wait_timeout_ms)):
            raise ValueError("Camera resolution, FPS and frame deadline must be positive integers")
        if self.camera_wait_timeout_ms>10000 or any(not np.isfinite(value) or value<=0 for value in (self.camera_connect_timeout_s,self.camera_scan_timeout_s,self.max_camera_frame_age_s)):
            raise ValueError("Camera deadlines must be finite and positive; frame deadline is at most10000ms")
        if not 0<=self.release_threshold<self.close_threshold<=1:
            raise ValueError("Invalid gripper hysteresis thresholds")
        if len(self.home_joints)!=6 or not np.isfinite(self.home_joints).all():
            raise ValueError("Task home must contain six finite joint angles")
        if (type(self.home_hz) is not int or not 1<=self.home_hz<=500
                or any(not np.isfinite(value) or value<=0 for value in
                       (self.home_duration_s,self.home_joint_tolerance_rad,
                        self.home_settle_timeout_s,self.gripper_open_timeout_s))):
            raise ValueError("Home timing/tolerance and gripper verification deadline must be finite and positive")
        if type(self.gripper_open_tolerance) is not int or not 0<=self.gripper_open_tolerance<=25:
            raise ValueError("Gripper open tolerance must be an integer register position in [0,25]")
        if not 0<self.recovery_freshness_timeout_s<=self.recovery_timeout_s or not np.isfinite(self.recovery_timeout_s):
            raise ValueError("Recovery deadlines must be finite and positive")


class Gripper:
    """Bounded Robotiq requests; firmware activation belongs to operator Start."""
    def __init__(self,host):
        self.socket=socket.create_connection((host,63352),timeout=2)
        self.socket.settimeout(2)
        self.lock=threading.Lock()

    def request(self,text):
        with self.lock:
            self.socket.sendall((text+"\n").encode())
            reply=self.socket.recv(1024)
        if not reply:
            raise ConnectionError("Robotiq connection closed")
        return reply.decode().strip()

    def get(self,name):
        response=self.request(f"GET {name}").split()
        if len(response)!=2 or response[0]!=name:
            raise RuntimeError(f"Invalid Robotiq {name} response")
        return int(response[1])

    def set(self,**values):
        reply=self.request("SET "+" ".join(f"{name} {int(value)}" for name,value in values.items()))
        if reply!="ack":
            raise RuntimeError("Robotiq did not acknowledge command")

    def activate(self,cancel,command_lock):
        def set_if_active(**values):
            with command_lock:
                if cancel.is_set():
                    raise InterruptedError("Gripper activation cancelled")
                self.set(**values)
        def wait():
            if cancel.wait(.02):
                raise InterruptedError("Gripper activation cancelled")
        if self.get("STA")==3:
            return
        set_if_active(ACT=0,ATR=0)
        deadline=time.monotonic()+10
        while self.get("STA")!=0:
            if time.monotonic()>deadline:
                raise TimeoutError("Robotiq reset timed out")
            wait()
        set_if_active(ACT=1)
        while self.get("STA")!=3:
            if time.monotonic()>deadline:
                raise TimeoutError("Robotiq activation timed out")
            wait()

    def halt(self):
        # Activation ignores GTO, so abort a pending activation using ACT=0.
        if self.get("STA")==3:
            self.set(GTO=0)
        else:
            self.set(ACT=0,GTO=0)

    def move(self,position,speed,force):
        self.set(POS=position,SPE=speed,FOR=force,GTO=1)

    def close(self):
        self.socket.close()


class UR5eEmbodiment:
    def __init__(self,config:UR5eConfig):
        config.validate()
        self.config=config
        self.metadata={"kind":"ur5e","robot_ip":config.robot_ip,
                       "camera_serials":dict(config.camera_serials),
                       "state":"TCP xyz meters + Euler xyz radians + measured gripper/255",
                       "action":"tool-local relative xyz meters + Euler xyz radians + absolute gripper position",
                       "pose_integration":"previous commanded transform @ local delta; seed actual TCP each episode",
                       "home_joints":list(config.home_joints),"close_threshold":config.close_threshold,
                       "release_threshold":config.release_threshold,"camera_fps":config.camera_fps,
                       "initial_open_steps":config.initial_open_steps,"gripper_period_s":config.gripper_period_s}
        self.metadata["gripper_activation"]="operator Start only; inactive activation may trigger firmware calibration"
        self.metadata["home_verification"]={"joint_tolerance_rad":config.home_joint_tolerance_rad,
            "settle_timeout_s":config.home_settle_timeout_s,"servo_period_s":1/config.home_hz,
            "gripper_open_tolerance":config.gripper_open_tolerance,
            "gripper_open_timeout_s":config.gripper_open_timeout_s}
        self.armed=threading.Event()
        self.command_lock=threading.Lock()
        self.control=self.receive=self.gripper=None
        self.cameras={}
        self._camera_frames={}
        self._camera_open_job=None
        self._camera_scan_job=None
        self._camera_open_lock=threading.RLock()
        self.metadata["camera_configuration"]={"width":config.width,"height":config.height,"fps":config.camera_fps,
            "format":"rgb8","wait_timeout_ms":config.camera_wait_timeout_ms,
            "max_frame_age_s":config.max_camera_frame_age_s,"scan_timeout_s":config.camera_scan_timeout_s}
        self.commanded=None
        self.steps=0
        self.gripper_status="open"
        self.last_gripper_position=None
        self.last_gripper_time=0.0
        self._fault_lock=threading.RLock()
        self._fault=None
        self._recovery_guard=threading.RLock()
        self._recovery_thread=None
        self._recovery_call_active=False
        self._sdk_version="unknown"
        self._control_no_upload=False

    @property
    def recovery_pending(self):
        with self._recovery_guard:
            return self._recovery_call_active or (self._recovery_thread is not None and self._recovery_thread.is_alive())

    @property
    def recovery_required(self):
        with self._fault_lock:
            faulted=self._fault is not None
        return faulted or self.recovery_pending

    @property
    def recovery_error(self):
        """Expose a latched device fault when stop() can only return False."""
        with self._fault_lock:
            if self._fault is None:
                return None
            return RobotRecoveryRequired(self._fault["reason"], status=dict(self._fault["status"]),
                                         category=self._fault["category"])

    def _fault_error(self,reason,status=None,*,category="robot_fault"):
        self.armed.clear()
        payload=RobotRecoveryRequired(str(reason),status=status,category=category).status
        with self._fault_lock:
            if self._fault is None:
                self._fault={"reason":str(reason),"category":category,"status":dict(payload)}
            payload["original_fault"]={**self._fault,"status":dict(self._fault["status"])}
            self.metadata["robot_fault"]={"reason":str(reason),"category":category,"status":dict(payload)}
        return RobotRecoveryRequired(str(reason),status=payload,category=category)

    def _failure_context(self,error,details):
        """Keep the attempted command with a typed fault, without another device call."""
        details=RobotRecoveryRequired(str(error),status=details,category=error.category).status
        error.status={**details,**error.status}
        with self._fault_lock:
            if self._fault is not None:
                # Do not put original_fault back into its own nested status.
                # The inner servo evidence wins over outer apply defaults (for
                # example an acknowledged command followed by a disconnect).
                self._fault["status"].update({key:value for key,value in error.status.items()
                                              if key!="original_fault"})
            self.metadata["robot_fault"]={"reason":str(error),"category":error.category,
                                          "status":dict(error.status)}
        return error

    def _driver_failure(self,reason,details,*,category,error=None):
        """Prefer observed robot status to guesses about why a command failed."""
        context=dict(details)
        if error is not None:
            context.update(driver_error=str(error),driver_error_type=type(error).__name__)
        context.update(operation_failure=str(reason),operation_failure_category=category)
        try:
            self._require_status(self._read_status())
        except RobotRecoveryRequired as fault:
            return self._failure_context(fault,context)
        return self._fault_error(reason,context,category=category)

    def _read_status(self):
        if self.receive is None or self.control is None:
            raise self._fault_error("Robot control/receive connection is unavailable",category="robot_disconnected")
        status={}
        try:
            status={"receive_connected":bool(self.receive.isConnected()),
                    "control_connected":bool(self.control.isConnected())}
            if not status["receive_connected"] or not status["control_connected"]:
                # Disconnected SDK feedback can be stale or raise. Preserve the
                # direct connection result before requesting any cached state.
                return status
            status.update({"safety_mode":int(self.receive.getSafetyMode()),
                    "robot_mode":int(self.receive.getRobotMode()),
                    "program_running":bool(self.control.isProgramRunning())})
            for name,key in (("isProtectiveStopped","protective_stop"),("isEmergencyStopped","emergency_stop")):
                method=getattr(self.receive,name,None)
                status[key]=bool(method()) if callable(method) else status["safety_mode"] in ({3} if key=="protective_stop" else {6,7})
            timestamp=getattr(self.receive,"getTimestamp",None)
            if callable(timestamp):
                status["timestamp"]=float(timestamp())
            runtime=getattr(self.receive,"getRuntimeState",None)
            if callable(runtime):
                status["runtime_state"]=int(runtime())
            return status
        except RobotRecoveryRequired:
            raise
        except Exception as error:
            category="robot_disconnected" if isinstance(error,ConnectionError) else "status_feedback_error"
            raise self._fault_error("Robot safety/status query failed",{**status,"driver_error":str(error),
                                      "driver_error_type":type(error).__name__},category=category) from error

    def _require_status(self,status,*,program=True):
        if not status["receive_connected"] or not status["control_connected"]:
            raise self._fault_error("RTDE connection is disconnected; restore the robot connection before retrying",status,
                                    category="robot_disconnected")
        if status["emergency_stop"]:
            raise self._fault_error("Robot emergency stop is active; clear the reported condition on the teach pendant before Recover",status,
                                    category="emergency_stop")
        if status["protective_stop"]:
            raise self._fault_error("Robot protective stop is active; clear the reported condition on the teach pendant before Recover",status,
                                    category="protective_stop")
        if status["safety_mode"] not in {1,2}:
            raise self._fault_error("Robot is safety stopped; clear the reported condition on the teach pendant before Recover",status,
                                    category="robot_safety_stop")
        if status["robot_mode"]!=7:
            raise self._fault_error("Robot is not in RUNNING mode; restore it on the teach pendant before Recover",status,
                                    category="robot_not_ready")
        if program and not status["program_running"]:
            raise self._fault_error("RTDE control program is not running",status,category="control_program_stopped")
        return status

    def _assert_robot_ready(self,context):
        if self.recovery_pending:
            raise self._fault_error("A native RTDE recovery operation is still pending",{"context":context,"native_pending":True})
        with self._fault_lock:
            fault=self._fault
        if fault is not None:
            raise RobotRecoveryRequired(fault["reason"],status={**fault["status"],"context":context,"recovery_required":True},
                                         category=fault["category"])
        return self._require_status(self._read_status())

    def _recover_guard(self,cancel,abandoned,deadline):
        self.armed.clear()
        if cancel.is_set() or abandoned.is_set():
            raise InterruptedError("Robot recovery cancelled; policy commands remain disarmed")
        if time.monotonic()>=deadline:
            raise self._fault_error("Robot recovery deadline expired")

    def _fresh_status(self,cancel,abandoned,deadline):
        self._recover_guard(cancel,abandoned,deadline)
        self._require_status(self._read_status(),program=False)
        timestamp=getattr(self.receive,"getTimestamp",None)
        if not callable(timestamp):
            raise self._fault_error("Fresh RTDE timestamp feedback is unavailable")
        initial=float(timestamp())
        if not np.isfinite(initial):
            raise self._fault_error("Invalid RTDE timestamp feedback")
        freshness_deadline=min(deadline,time.monotonic()+self.config.recovery_freshness_timeout_s)
        while True:
            self._recover_guard(cancel,abandoned,deadline)
            current=float(timestamp())
            if np.isfinite(current) and current>initial:
                return self._require_status(self._read_status(),program=False)
            if time.monotonic()>=freshness_deadline:
                raise self._fault_error("No fresh RTDE feedback arrived; cached NORMAL status is insufficient")
            cancel.wait(min(.01,max(0,freshness_deadline-time.monotonic())))

    def _recover_connected(self,cancel,abandoned,deadline):
        """Restore only an already-connected control channel using stop/idle script.

        Audited ur_rtde 1.6.3 sendCommand() returns False only after sending
        NO_CMD (input integer register 0). reuploadScript() itself DOES NOT clear
        that register and may kill an existing program, so it is never used
        without the preceding clear handshake and a stopped-program check.
        Disconnected clients are deliberately not recreated/reconnected here:
        those SDK operations have unbounded first-state waits and upload paths.
        """
        if not self._control_no_upload:
            raise self._fault_error("Control connection was not created in no-upload mode; reopen the hardware connection before recovery")
        before=self._fresh_status(cancel,abandoned,deadline)
        restored=False
        self._recover_guard(cancel,abandoned,deadline)
        if not before["program_running"]:
            if self._sdk_version!="1.6.3":
                raise self._fault_error("Idle-script recovery requires the audited ur_rtde 1.6.3 implementation",{"ur_rtde_version":self._sdk_version})
            # STOPPED/script-not-running False paths in 1.6.3 clear NO_CMD;
            # an exception never counts as a successful clear handshake.
            cleared=self.control.servoStop()
            self._recover_guard(cancel,abandoned,deadline)
            if cleared is not True and cleared is not False:
                raise self._fault_error("RTDE command-clear handshake returned an invalid acknowledgement")
            current=self._fresh_status(cancel,abandoned,deadline)
            self._recover_guard(cancel,abandoned,deadline)
            if not current["program_running"]:
                # Do not intentionally replace or stop a running program.
                if not self.control.isProgramRunning():
                    self._recover_guard(cancel,abandoned,deadline)
                    if self.control.reuploadScript() is not True:
                        raise self._fault_error("RTDE idle control-script upload was not acknowledged",current)
                    restored=True
                    while not self.control.isProgramRunning():
                        self._recover_guard(cancel,abandoned,deadline)
                        cancel.wait(.01)
        self._fresh_status(cancel,abandoned,deadline)
        self._recover_guard(cancel,abandoned,deadline)
        if self.control.servoStop() is not True:
            raise self._fault_error("Fresh RTDE servoStop acknowledgement was not confirmed",self._read_status())
        self._recover_guard(cancel,abandoned,deadline)
        after=self._fresh_status(cancel,abandoned,deadline)
        self._require_status(after)
        self._recover_guard(cancel,abandoned,deadline)
        pose=np.asarray(self.receive.getActualTCPPose(),dtype=np.float64)
        if pose.shape!=(6,) or not np.isfinite(pose).all():
            raise self._fault_error("Fresh TCP feedback is invalid after recovery",after)
        return {"healthy":True,"stop_confirmed":True,"script_restored":restored,
                "ur_rtde_version":self._sdk_version,"before":before,"after":after,
                "actual_tcp_rotvec":pose.tolist(),"armed":False}

    def recover(self,cancel):
        """Operator-requested control recovery; never homes, moves or activates grip.

        Native SDK calls cannot be interrupted safely. A bounded caller wait
        therefore leaves a persistent pending fence if an SDK call outlives its
        deadline. No Start, second Recover, stop command or RTDE disconnect may
        race that call. Even a late successful completion never clears the fault.
        """
        self.armed.clear()
        with self._recovery_guard:
            if self._recovery_call_active or self.recovery_pending:
                raise self._fault_error("A native RTDE recovery operation is still pending",{"native_pending":True})
            self._recovery_call_active=True
            deadline=time.monotonic()+self.config.recovery_timeout_s
            job={"done":threading.Event(),"abandoned":threading.Event()}
            def work():
                acquired=False
                try:
                    acquired=self.command_lock.acquire(timeout=min(2,max(0,deadline-time.monotonic())))
                    if not acquired:
                        raise self._fault_error("Robot command channel is busy during recovery")
                    self._recover_guard(cancel,job["abandoned"],deadline)
                    job["result"]=self._recover_connected(cancel,job["abandoned"],deadline)
                except BaseException as error:
                    job["error"]=error
                finally:
                    self.armed.clear()
                    if acquired:
                        self.command_lock.release()
                    job["done"].set()
            self._recovery_thread=threading.Thread(target=work,name="ur5e-idle-recovery",daemon=True)
            self._recovery_thread.start()
        try:
            while not job["done"].wait(.01):
                if cancel.is_set() or time.monotonic()>=deadline:
                    job["abandoned"].set()
                    reason="Robot recovery cancelled" if cancel.is_set() else "Native RTDE recovery timed out"
                    error=self._fault_error(reason,{"native_pending":self.recovery_pending})
                    if cancel.is_set():
                        raise InterruptedError(str(error))
                    raise error
            self._recovery_thread.join(timeout=max(0,deadline-time.monotonic()))
            if self._recovery_thread.is_alive():
                raise self._fault_error("Native RTDE recovery completion is still pending",{"native_pending":True})
            self._recover_guard(cancel,job["abandoned"],deadline)
            if "error" in job:
                error=job["error"]
                if isinstance(error,(RobotRecoveryRequired,InterruptedError)):
                    raise error
                raise self._fault_error("RTDE recovery failed",{"driver_error":str(error)}) from error
            acquired=self.command_lock.acquire(timeout=min(2,max(0,deadline-time.monotonic())))
            if not acquired:
                raise self._fault_error("Robot command channel remained busy at final recovery validation")
            try:
                self._recover_guard(cancel,job["abandoned"],deadline)
                self._require_status(self._read_status())
                self._recover_guard(cancel,job["abandoned"],deadline)
                self.commanded=None
                self.steps=0
                self.last_gripper_position=None
                self.last_gripper_time=0.0
                self.armed.clear()
                with self._fault_lock:
                    self._fault=None
                    self.metadata.pop("robot_fault",None)
                report=job["result"]
                self.metadata["last_robot_recovery"]=report
                return report
            finally:
                self.command_lock.release()
        except InterruptedError:
            self._fault_error("Robot recovery cancelled; a fresh Recover request is required",{"native_pending":self.recovery_pending})
            raise
        finally:
            self.armed.clear()
            with self._recovery_guard:
                self._recovery_call_active=False

    @property
    def camera_operation_pending(self):
        with self._camera_open_lock:
            return any(job is not None and not job["done"].is_set()
                       for job in (self._camera_open_job,self._camera_scan_job))

    def list_camera_devices(self):
        """Explicit identity/profile scan; never start a stream or robot control."""
        with self._camera_open_lock:
            if self.camera_operation_pending:
                raise CameraError("A camera driver operation is still finishing",category="camera_operation_pending")
            job={"done":threading.Event(),"abandoned":threading.Event(),"result":None,"error":None}
            self._camera_scan_job=job
        def scan():
            report,error=None,None
            try:
                report=self._camera_device_report()
            except Exception as caught:
                # Do not retain SDK objects through a native exception traceback.
                error=CameraError(str(caught),category=getattr(caught,"category","camera_discovery_failed"),
                                  status=getattr(caught,"status",{}))
            finally:
                with self._camera_open_lock:
                    if not job["abandoned"].is_set():
                        job.update(result=report,error=error)
                    job["done"].set()
        threading.Thread(target=scan,name="camera-discovery",daemon=True).start()
        if not job["done"].wait(self.config.camera_scan_timeout_s):
            with self._camera_open_lock:
                job["abandoned"].set()
            raise CameraError("Camera discovery timed out; wait for the driver before retrying",
                              category="camera_scan_timeout",status={"native_pending":self.camera_operation_pending})
        if job["error"] is not None:
            raise job["error"]
        if job["result"] is None:
            raise CameraError("Camera discovery was cancelled",category="camera_scan_cancelled")
        return job["result"]

    def _camera_device_report(self):
        import pyrealsense2 as rs
        devices, warnings = [], []
        try:
            connected = list(rs.context().query_devices())
        except Exception as error:
            raise CameraError(f"Camera discovery failed: {error}", category="camera_discovery_failed") from error
        for device in connected:
            def info(key):
                try:
                    return str(device.get_info(getattr(rs.camera_info, key)))
                except Exception:
                    return ""
            serial, name = info("serial_number"), info("name")
            if not serial:
                warnings.append("A connected camera did not expose a serial number")
                continue
            profiles = set()
            try:
                for sensor in device.query_sensors():
                    for profile in sensor.get_stream_profiles():
                        if profile.stream_type() != rs.stream.color or profile.format() != rs.format.rgb8:
                            continue
                        video = profile.as_video_stream_profile()
                        profiles.add((video.width(), video.height(), profile.fps(), "rgb8"))
            except Exception as error:
                warnings.append(f"Could not inspect color profiles for {serial}: {error}")
            devices.append({"serial":serial,"name":name,"label":f"{name or 'Camera'} [{serial}]",
                            "firmware":info("firmware_version"),"usb_type":info("usb_type_descriptor"),
                            "color_profiles":[dict(zip(("width","height","fps","format"), values))
                                              for values in sorted(profiles)]})
        return {"devices":sorted(devices,key=lambda item:item["serial"]),
                "selected":dict(self.config.camera_serials),
                "configuration":dict(self.metadata["camera_configuration"]),"warnings":warnings}

    @staticmethod
    def _stop_camera_pipelines(pipelines):
        errors = []
        for role, pipeline in pipelines.items():
            try:
                pipeline.stop()
            except Exception as error:
                errors.append(f"{role}: {error}")
        return errors

    def _abandon_camera_operations(self):
        """A late native start retains ownership and closes its own pipelines."""
        cleanup = {}
        with self._camera_open_lock:
            scan = self._camera_scan_job
            if scan is not None:
                scan["abandoned"].set()
            job = self._camera_open_job
            if job is None:
                return scan is not None and not scan["done"].is_set()
            job["abandoned"].set()
            if job["done"].is_set():
                cleanup, job["result"] = job["result"], {}
            pending = not job["done"].is_set() or (scan is not None and not scan["done"].is_set())
        self._stop_camera_pipelines(cleanup)
        return pending

    def _close_camera_streams(self):
        pending = self._abandon_camera_operations()
        if pending:
            raise CameraError("Camera driver operation is still finishing; retry after it closes",
                              category="camera_operation_pending",status={"native_pending":True})
        previous, self.cameras = self.cameras, {}
        self._camera_frames.clear()
        errors = self._stop_camera_pipelines(previous)
        if errors:
            raise CameraError("Camera stop failed",category="camera_stop_failed",status={"errors":errors})

    def connect_preview(self, cancel=None):
        """Open RGB only, with finite caller wait and ownership of late starts."""
        if self.camera_operation_pending:
            raise CameraError("A camera driver operation is still finishing",category="camera_operation_pending")
        if set(self.cameras) == set(self.config.camera_serials):
            return
        if self.cameras:
            self._close_camera_streams()
        import pyrealsense2 as rs
        cancel = cancel or threading.Event()
        with self._camera_open_lock:
            if self._camera_open_job is not None and not self._camera_open_job["done"].is_set():
                raise CameraError("A previous camera driver operation is still finishing",
                                  category="camera_operation_pending",status={"native_pending":True})
            job = {"done":threading.Event(),"abandoned":threading.Event(),"result":{},"error":None}
            self._camera_open_job = job
        serials = dict(self.config.camera_serials)
        cfg_values = (self.config.width,self.config.height,self.config.camera_fps)
        def open_streams():
            opened, error = {}, None
            try:
                for role, serial in serials.items():
                    if job["abandoned"].is_set() or cancel.is_set():
                        raise InterruptedError("Camera connection cancelled")
                    pipeline = rs.pipeline()
                    # Even a start that raises can have allocated USB resources.
                    opened[role] = pipeline
                    cfg = rs.config()
                    cfg.enable_device(serial)
                    cfg.enable_stream(rs.stream.color,*cfg_values[:2],rs.format.rgb8,cfg_values[2])
                    pipeline.start(cfg)
            except BaseException as caught:
                error = caught
            with self._camera_open_lock:
                job["error"] = error
                cleanup = opened if error is not None or job["abandoned"].is_set() or cancel.is_set() else {}
                job["result"] = {} if cleanup else opened
            self._stop_camera_pipelines(cleanup)
            # Mark completion atomically with the final abandonment check. No
            # stop may race a still-running pipeline.start on its native handle.
            with self._camera_open_lock:
                cleanup = job["result"] if job["abandoned"].is_set() or cancel.is_set() else {}
                if cleanup:
                    job["result"] = {}
                else:
                    job["done"].set()
            if cleanup:
                self._stop_camera_pipelines(cleanup)
                job["done"].set()
        threading.Thread(target=open_streams,name="camera-connect",daemon=True).start()
        deadline = time.monotonic()+self.config.camera_connect_timeout_s
        while not job["done"].wait(.02):
            if cancel.is_set() or time.monotonic()>=deadline:
                self._abandon_camera_operations()
                if cancel.is_set():
                    raise InterruptedError("Camera connection cancelled")
                raise CameraError("Camera connection timed out; the driver will close late streams",
                                  category="camera_connect_timeout",status={"native_pending":self.camera_operation_pending})
        if cancel.is_set():
            self._abandon_camera_operations()
            raise InterruptedError("Camera connection cancelled")
        with self._camera_open_lock:
            if job["error"] is not None:
                raise CameraError(f"Camera connection failed: {job['error']}",category="camera_connect_failed",
                                  status={"selected":serials}) from job["error"]
            self.cameras, job["result"] = job["result"], {}
        self._camera_frames.clear()
        if set(self.cameras)!=set(serials):
            raise CameraError("Camera connection was cancelled before ownership transfer",category="camera_connect_cancelled")

    def configure_cameras(self, serials, cancel):
        """Change sensor roles while idle, preserving controls and gripper state."""
        if self.armed.is_set():
            raise ValueError("Stop the active rollout before changing cameras")
        if cancel.is_set():
            raise InterruptedError("Camera change cancelled")
        if not isinstance(serials,dict) or set(serials)!=set(self.config.camera_serials):
            raise ValueError("Camera mapping must supply every configured role exactly once")
        if any(not isinstance(value,str) or not value.strip() or value!=value.strip() for value in serials.values()):
            raise ValueError("Each camera role requires an explicit serial number")
        if len(set(serials.values()))!=len(serials):
            raise ValueError("Camera roles must use distinct devices")
        if self.camera_operation_pending:
            raise CameraError("A camera driver operation is still finishing",category="camera_operation_pending")
        scan = self.list_camera_devices()
        available = {item["serial"]:item for item in scan["devices"]}
        for role, serial in serials.items():
            if serial not in available:
                raise CameraError(f"Selected {role} camera is not connected: {serial}",
                                  category="camera_not_found",status={"role":role,"serial":serial})
            profiles = available[serial]["color_profiles"]
            if profiles and not any(p["width"]==self.config.width and p["height"]==self.config.height
                                    and p["fps"]==self.config.camera_fps and p["format"]=="rgb8" for p in profiles):
                raise CameraError(f"Selected {role} camera does not support the configured RGB profile",
                                  category="camera_profile_unsupported",status={"role":role,"serial":serial,
                                  "requested":self.metadata["camera_configuration"]})
        previous = dict(self.config.camera_serials)
        self._close_camera_streams()
        self.config.camera_serials = dict(serials)
        try:
            self.connect_preview(cancel)
            _, camera_metadata = self._images()
            if cancel.is_set():
                raise InterruptedError("Camera change cancelled")
        except BaseException as error:
            rollback = False
            rollback_error = None
            try:
                self._close_camera_streams()
                self.config.camera_serials = previous
                if not cancel.is_set():
                    self.connect_preview(cancel)
                    self._images()
                    rollback = True
            except Exception as caught:
                rollback_error = str(caught)
            self.config.camera_serials = previous
            self.metadata["camera_serials"] = previous
            if isinstance(error,InterruptedError):
                raise
            raise CameraError(f"Camera change failed: {error}",category=getattr(error,"category","camera_change_failed"),
                              status={"requested":dict(serials),"previous":previous,"rollback_restored":rollback,
                                      "rollback_error":rollback_error,"native_pending":self.camera_operation_pending}) from error
        self.metadata["camera_serials"] = dict(serials)
        self.metadata["camera_metadata"] = camera_metadata
        return {"selected":dict(serials),"configuration":dict(self.metadata["camera_configuration"]),
                "camera_metadata":camera_metadata}

    def connect(self):
        self.connect_preview()
        import rtde_control
        import rtde_receive
        from scipy.spatial.transform import Rotation
        self.Rotation=Rotation
        try:
            self._sdk_version=importlib.metadata.version("ur-rtde")
        except importlib.metadata.PackageNotFoundError:
            self._sdk_version="unknown"
        self.metadata["ur_rtde_version"]=self._sdk_version
        try:
            self.receive=rtde_receive.RTDEReceiveInterface(self.config.robot_ip)
            # The SDK automatically reconnects on some command/receiver errors.
            # NO_WAIT without UPLOAD_SCRIPT makes those internal reconnects
            # unable to restart/replace a program behind the operator's back.
            flags=int(rtde_control.RTDEControlInterface.FLAG_NO_WAIT)
            if flags&0x01:
                raise RuntimeError("RTDE control flags unexpectedly enable automatic script upload")
            self.control=rtde_control.RTDEControlInterface(self.config.robot_ip,-1.0,flags=flags)
            self._control_no_upload=True
            self.metadata["control_connection_flags"]={"value":flags,"upload_script":False,"no_wait":True}
            self.gripper=Gripper(self.config.robot_ip)
            self.metadata["gripper_status_at_connect"]=self.gripper.get("STA")
        except BaseException:
            self.close()
            raise

    def _transform(self,xyz,euler):
        transform=np.eye(4)
        transform[:3,3]=xyz
        transform[:3,:3]=self.Rotation.from_euler("xyz",euler).as_matrix()
        return transform

    def _state(self):
        try:
            tcp=np.asarray(self.receive.getActualTCPPose(),dtype=np.float64)
            if tcp.shape!=(6,) or not np.isfinite(tcp).all():
                raise ValueError("TCP pose must contain six finite values")
            euler=self.Rotation.from_rotvec(tcp[3:]).as_euler("xyz")
            return np.r_[tcp[:3],euler,self.gripper.get("POS")/255].astype(np.float32)
        except RobotRecoveryRequired:
            raise
        except Exception as error:
            raise self._driver_failure("Robot state feedback failed",{},category="state_feedback_error",error=error) from error

    def _images(self):
        if set(self.cameras)!=set(self.config.camera_serials):
            raise CameraError("Camera streams are not connected for every configured role",
                              category="camera_not_connected",status={"selected":dict(self.config.camera_serials),
                              "connected_roles":sorted(self.cameras),"native_pending":self.camera_operation_pending})
        images,metadata={},{}
        for role,pipeline in self.cameras.items():
            deadline=time.monotonic()+self.config.camera_wait_timeout_ms/1000
            while True:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise CameraError(f"No fresh RGB frame from {role} camera before its deadline",
                                      category="camera_frame_timeout",status={"role":role,
                                      "serial":self.config.camera_serials[role],"previous_frame":self._camera_frames.get(role)})
                try:
                    frame=pipeline.wait_for_frames(timeout_ms=max(1,int(remaining*1000))).get_color_frame()
                except Exception as error:
                    raise CameraError(f"RGB capture failed for {role}: {error}",category="camera_capture_failed",
                                      status={"role":role,"serial":self.config.camera_serials[role]}) from error
                if not frame:
                    continue
                try:
                    number=int(frame.get_frame_number())
                    timestamp=float(frame.get_timestamp())
                    domain=str(frame.get_frame_timestamp_domain())
                except Exception as error:
                    raise CameraError(f"Invalid {role} camera frame metadata: {error}",
                                      category="camera_frame_invalid",status={"role":role}) from error
                if number<0 or not np.isfinite(timestamp):
                    raise CameraError(f"Invalid {role} camera timestamp or frame number",
                                      category="camera_frame_invalid",status={"role":role})
                if number<=self._camera_frames.get(role,-1):
                    continue
                received_ns=time.time_ns()
                age_s=None
                if domain.rsplit(".",1)[-1] in {"global_time","system_time"}:
                    age_s=received_ns/1e9-timestamp/1000
                    if age_s>self.config.max_camera_frame_age_s or age_s < -self.config.max_camera_frame_age_s:
                        raise CameraError(f"Stale or clock-inconsistent {role} camera frame",
                                          category="camera_stale_frame",status={"role":role,"frame_age_s":age_s,
                                          "timestamp_domain":domain,"frame_number":number})
                try:
                    image=np.asanyarray(frame.get_data())
                except Exception as error:
                    raise CameraError(f"Cannot read {role} RGB pixels: {error}",category="camera_frame_invalid",
                                      status={"role":role}) from error
                expected=(self.config.height,self.config.width,3)
                if image.dtype!=np.uint8 or image.shape!=expected:
                    raise CameraError(f"Unexpected {role} RGB shape/type: {image.shape}, {image.dtype}",
                                      category="camera_profile_mismatch",status={"role":role,"expected_shape":list(expected),
                                      "actual_shape":list(image.shape),"actual_dtype":str(image.dtype)})
                images[role]=image.copy()
                self._camera_frames[role]=number
                metadata[role]={"serial":self.config.camera_serials[role],"frame_number":number,
                                "device_timestamp_ms":timestamp,"timestamp_domain":domain,
                                "host_received_ns":received_ns,"host_received_monotonic_ns":time.monotonic_ns(),
                                "frame_age_s":age_s}
                break
        self.metadata["camera_metadata"]=metadata
        return images, metadata

    def observe_preview(self):
        images, metadata = self._images()
        metadata["preview_only"] = True
        observation = Observation(images, np.empty(0, dtype=np.float32), time.time_ns(), metadata, time.monotonic_ns())
        observation.preview_only = True
        return observation

    def observe(self):
        self._assert_robot_ready("observation")
        images, metadata = self._images()
        state=self._state()
        self._assert_robot_ready("observation feedback")
        return Observation(images,state,time.time_ns(),metadata,time.monotonic_ns())

    def _servo(self,joints,*,duration_s=None,cancel=None):
        cfg=self.config
        duration=cfg.servo_time_s if duration_s is None else duration_s
        timings={}
        details={"target_joints":list(joints),"arm_command_attempted":False,"arm_command_sent":False,
                 "arm_command_acknowledged":False,"servo_timing_ms":timings}
        try:
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Home cancelled")
            before=_timed_call(timings,"status_before_ms",self._assert_robot_ready,"servo command")
            details["status_before"]=before
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Home cancelled")
            token=_timed_call(timings,"init_period_ms",self.control.initPeriod)
            if cancel is not None and cancel.is_set():
                raise InterruptedError("Home cancelled")
            details.update(arm_command_attempted=True,arm_command_sent=None)
            if not _timed_call(timings,"servo_j_call_ms",self.control.servoJ,list(joints),cfg.velocity,
                               cfg.acceleration,duration,cfg.servo_lookahead_s,cfg.servo_gain):
                raise self._driver_failure("UR5e servoJ rejected command",details,category="servo_rejected")
            details.update(arm_command_sent=True,arm_command_acknowledged=True)
            _timed_call(timings,"wait_period_ms",self.control.waitPeriod,token)
            _timed_call(timings,"status_after_ms",self._assert_robot_ready,"servo feedback")
            return timings
        except InterruptedError:
            raise
        except RobotRecoveryRequired as error:
            raise self._failure_context(error,details)
        except Exception as error:
            raise self._driver_failure("UR5e servo command failed",details,category="servo_command_failed",error=error) from error

    def home(self,cancel):
        return self._home(cancel,prepare_gripper=True)

    def home_idle(self,cancel):
        """Explicit idle Home moves the arm and preserves anything in the gripper."""
        self.armed.clear()
        return self._home(cancel,prepare_gripper=False)

    def _home(self,cancel,*,prepare_gripper):
        cfg=self.config
        self.armed.clear()
        self.metadata.pop("last_home_report",None)
        if cancel.is_set():
            raise InterruptedError("Home cancelled")
        self._assert_robot_ready("home")
        gripper_position=None
        if prepare_gripper:
            # Idle Home must never open a held object or activate the fingers.
            try:
                self.gripper.activate(cancel,self.command_lock)
                with self.command_lock:
                    if cancel.is_set():
                        raise InterruptedError("Gripper preparation cancelled")
                    # A previous SET target is not measured position. In particular
                    # stop() halts GTO while an opening command may still be in flight.
                    self.gripper.move(0,cfg.gripper_speed,cfg.gripper_force)
                    self.last_gripper_position=0
                    self.last_gripper_time=time.monotonic()
                deadline=time.monotonic()+cfg.gripper_open_timeout_s
                while True:
                    if cancel.is_set():
                        raise InterruptedError("Gripper preparation cancelled")
                    gripper_position=self.gripper.get("POS")
                    if not 0<=gripper_position<=255:
                        raise ValueError(f"Invalid measured gripper position: {gripper_position}")
                    if gripper_position<=cfg.gripper_open_tolerance:
                        self.gripper_status="open"
                        break
                    if time.monotonic()>=deadline:
                        raise self._fault_error("Gripper opening was not confirmed before home",
                            {"requested_position":0,"measured_position":gripper_position,
                             "open_tolerance":cfg.gripper_open_tolerance,"timeout_s":cfg.gripper_open_timeout_s},
                            category="gripper_open_unconfirmed")
                    if cancel.wait(.02):
                        raise InterruptedError("Gripper preparation cancelled")
            except InterruptedError:
                raise
            except RobotRecoveryRequired:
                raise
            except Exception as error:
                raise self._driver_failure("Gripper preparation failed before home",{},
                                           category="gripper_command_failed",error=error) from error
        try:
            initial=np.asarray(self.receive.getActualQ(),dtype=np.float64)
            if initial.shape!=(6,) or not np.isfinite(initial).all():
                raise ValueError("Home requires six finite measured joint positions")
        except Exception as error:
            raise self._driver_failure("Joint feedback failed before home",{},
                                       category="state_feedback_error",error=error) from error
        steps=max(1,round(cfg.home_duration_s*cfg.home_hz))
        period=1/cfg.home_hz
        target=np.asarray(cfg.home_joints,dtype=np.float64)
        for index in range(steps):
            started=time.monotonic()
            with self.command_lock:
                if cancel.is_set():
                    raise InterruptedError("Home cancelled")
                joints=initial+(target-initial)*(index+1)/steps
                # The home loop supplies setpoints at home_hz, so the command
                # duration must span that period rather than the rollout's 2ms.
                self._servo(joints,duration_s=period,cancel=cancel)
            if cancel.wait(max(0,period-(time.monotonic()-started))):
                raise InterruptedError("Home cancelled")
        deadline=time.monotonic()+cfg.home_settle_timeout_s
        stable=0
        while True:
            started=time.monotonic()
            with self.command_lock:
                if cancel.is_set():
                    raise InterruptedError("Home cancelled")
                self._assert_robot_ready("home position verification")
                try:
                    actual=np.asarray(self.receive.getActualQ(),dtype=np.float64)
                    if actual.shape!=(6,) or not np.isfinite(actual).all():
                        raise ValueError("Home requires six finite measured joint positions")
                except Exception as error:
                    raise self._driver_failure("Joint feedback failed after home",{},
                                               category="state_feedback_error",error=error) from error
                error=float(np.max(np.abs(actual-target)))
                stable=stable+1 if error<=cfg.home_joint_tolerance_rad else 0
                if stable>=3:
                    if prepare_gripper:
                        try:
                            gripper_position=self.gripper.get("POS")
                        except Exception as error:
                            raise self._driver_failure("Gripper feedback failed after home",{},
                                                       category="gripper_command_failed",error=error) from error
                        if not 0<=gripper_position<=cfg.gripper_open_tolerance:
                            raise self._fault_error("Open gripper position was lost during home",
                                {"measured_position":gripper_position,"open_tolerance":cfg.gripper_open_tolerance},
                                category="gripper_open_unconfirmed")
                    if cancel.is_set():
                        raise InterruptedError("Home cancelled")
                    report={"verified":True,"target_joints":target.tolist(),"actual_joints":actual.tolist(),
                            "max_joint_error_rad":error,"joint_tolerance_rad":cfg.home_joint_tolerance_rad,
                            "consecutive_position_checks":stable,"servo_period_s":period,
                            "gripper_prepared":prepare_gripper,"gripper_position":gripper_position}
                    self.metadata["last_home_report"]=report
                    return report
                if time.monotonic()>=deadline:
                    raise self._fault_error("Home joint target was not reached before the verification deadline",
                        {"target_joints":target.tolist(),"actual_joints":actual.tolist(),
                         "max_joint_error_rad":error,"joint_tolerance_rad":cfg.home_joint_tolerance_rad,
                         "timeout_s":cfg.home_settle_timeout_s},category="home_position_unconfirmed")
                self._servo(target,duration_s=period,cancel=cancel)
            if cancel.wait(max(0,period-(time.monotonic()-started))):
                raise InterruptedError("Home cancelled")

    def begin_episode(self):
        self._assert_robot_ready("episode reset")
        state=self._state()
        with self.command_lock:
            self._assert_robot_ready("episode arming")
            position=float(state[6])*255
            valid_position=np.isfinite(position) and 0<=position<=255
            if valid_position:
                # _state stores POS/255 as float32; recover the integer register
                # before comparing a boundary such as POS8 (8.00000047 here).
                position=round(position)
            if not valid_position or position>self.config.gripper_open_tolerance:
                raise self._fault_error("A fresh rollout requires a confirmed open gripper",
                    {"measured_position":position,"open_tolerance":self.config.gripper_open_tolerance},
                    category="gripper_open_unconfirmed")
            self.commanded=self._transform(state[:3],state[3:6])
            self.steps=0
            self.armed.set()

    def apply(self,prediction):
        timings={}
        started=time.perf_counter()
        try:
            raw=action_vector(prediction.action)
        except (TypeError,ValueError,OverflowError) as error:
            self.armed.clear()
            raise RobotActionError(str(error),status={"received_action":prediction.action,
                                   "arm_command_sent":False}) from error
        action=raw.copy()
        cfg=self.config
        if cfg.max_rel_position is not None:
            action[:3]=np.clip(action[:3],-cfg.max_rel_position,cfg.max_rel_position)
        if cfg.max_rel_rotation is not None:
            action[3:6]=np.clip(action[3:6],-cfg.max_rel_rotation,cfg.max_rel_rotation)
        action[6]=np.clip(action[6],0,1)
        with np.errstate(over="ignore",invalid="ignore"):
            translation=float(np.hypot.reduce(raw[:3]))
            rotation=float(np.hypot.reduce(raw[3:6]))
        details={"raw_action":raw.tolist(),"clipped_action":action.tolist(),
                 "action_translation_norm_m":translation if np.isfinite(translation) else repr(translation),
                 "action_rotation_norm_rad":rotation if np.isfinite(rotation) else repr(rotation),
                 "arm_command_attempted":False,"arm_command_sent":False,"arm_command_acknowledged":False,
                 "gripper_command_attempted":False,"gripper_command_sent":False,
                 "gripper_command_acknowledged":False,"command_timing_ms":timings}
        timings["action_validation_ms"]=1000*(time.perf_counter()-started)
        if self.recovery_required:
            try:
                self._assert_robot_ready("policy command")
            except RobotRecoveryRequired as error:
                raise self._failure_context(error,details)
        lock_started=time.perf_counter()
        with self.command_lock:
            timings["command_lock_wait_ms"]=1000*(time.perf_counter()-lock_started)
            if not self.armed.is_set():
                raise InterruptedError("Policy command cancelled")
            try:
                _timed_call(timings,"status_check_ms",self._assert_robot_ready,"policy command")
                details["previous_commanded_transform"]=self.commanded.tolist()
                transform_started=time.perf_counter()
                try:
                    with np.errstate(over="ignore",invalid="ignore"):
                        target=self.commanded @ self._transform(action[:3],action[3:6])
                    if not np.isfinite(target).all():
                        raise ValueError("Predicted action produced a nonfinite target transform")
                    tcp=np.r_[target[:3,3],self.Rotation.from_matrix(target[:3,:3]).as_rotvec()]
                except (TypeError,ValueError,OverflowError) as error:
                    self.armed.clear()
                    raise RobotActionError(str(error),status=details,category="invalid_action_target") from error
                finally:
                    timings["target_transform_ms"]=1000*(time.perf_counter()-transform_started)
                details["target_tcp_rotvec"]=tcp.tolist()
                try:
                    current=np.asarray(_timed_call(timings,"joint_feedback_ms",self.receive.getActualQ),dtype=np.float64)
                    if current.shape!=(6,) or not np.isfinite(current).all():
                        raise ValueError("IK requires six finite measured joint positions")
                except Exception as error:
                    raise self._driver_failure("Joint feedback failed before inverse kinematics",details,
                                               category="state_feedback_error",error=error) from error
                details["measured_joints"]=current.tolist()
                try:
                    joints=_timed_call(timings,"inverse_kinematics_ms",self.control.getInverseKinematics,tcp.tolist(),current.tolist())
                except Exception as error:
                    raise self._driver_failure("Inverse kinematics request failed",details,
                                               category="ik_request_failed",error=error) from error
                try:
                    valid_joints=np.asarray(joints,dtype=np.float64)
                    valid_ik=valid_joints.shape==(6,) and np.isfinite(valid_joints).all()
                except (TypeError,ValueError,OverflowError):
                    valid_ik=False
                if not valid_ik:
                    reason=("No inverse kinematics solution for predicted action "
                            f"(translation={1000*translation:.3f} mm, rotation={rotation:.6f} rad)")
                    raise self._driver_failure(reason,details,category="ik_no_solution")
                details["target_joints"]=valid_joints.tolist()
                # Stop latches without waiting for this lock, so recheck after IK.
                if not self.armed.is_set():
                    raise InterruptedError("Policy command cancelled during IK")
                servo_timings=_timed_call(timings,"servo_total_ms",self._servo,joints)
                if isinstance(servo_timings,dict):
                    details["servo_timing_ms"]=servo_timings
                details.update(arm_command_attempted=True,arm_command_sent=True,arm_command_acknowledged=True)
                self.commanded=target
                self.steps+=1
                desired="open" if self.steps<=cfg.initial_open_steps else self.gripper_status
                if self.steps>cfg.initial_open_steps:
                    if self.gripper_status=="open" and action[6]>=cfg.close_threshold:
                        desired="close"
                    elif self.gripper_status=="close" and action[6]<=cfg.release_threshold:
                        desired="open"
                position=255 if desired=="close" else 0
                now=time.monotonic()
                sent=False
                if position!=self.last_gripper_position and now-self.last_gripper_time>=cfg.gripper_period_s:
                    # A cancellation after arm command must not issue another gripper move.
                    if self.armed.is_set():
                        # A lost SET reply cannot establish whether the gripper
                        # received the request. Never record it as "not sent".
                        details.update(gripper_command_attempted=True,gripper_command_sent=None,
                                       gripper_command_acknowledged=False)
                        try:
                            _timed_call(timings,"gripper_move_ms",self.gripper.move,position,cfg.gripper_speed,cfg.gripper_force)
                        except Exception as error:
                            raise self._driver_failure("Gripper move command failed",details,
                                                       category="gripper_command_failed",error=error) from error
                        details.update(gripper_command_sent=True,gripper_command_acknowledged=True)
                        self.last_gripper_position=position
                        self.last_gripper_time=now
                        self.gripper_status=desired
                        sent=True
                return {"raw_action":raw.tolist(),"clipped_action":action.tolist(),"target_tcp_rotvec":tcp.tolist(),
                        "target_joints":list(joints),"gripper_desired":desired,"gripper_command":self.last_gripper_position,
                        "gripper_command_attempted":details["gripper_command_attempted"],
                        "gripper_command_sent":sent,"gripper_command_acknowledged":details["gripper_command_acknowledged"],
                        "gripper_state":self.gripper_status,"command_timing_ms":timings,
                        "servo_timing_ms":details.get("servo_timing_ms",{}),
                        "command_timing_schema":"ur5e_driver_wall_v1"}
            except RobotRecoveryRequired as error:
                raise self._failure_context(error,details)

    def stop(self):
        self.armed.clear()
        if self.recovery_pending:
            self._fault_error("A native RTDE recovery operation is still pending",{"native_pending":True})
            return False
        if not self.command_lock.acquire(timeout=2):
            return False
        try:
            confirmed=True
            error=None
            if self.control is not None:
                try:
                    confirmed=self.control.servoStop() is not False
                except Exception as exc:
                    error=self._driver_failure("UR5e stop command failed",{},category="stop_command_failed",error=exc)
            if self.gripper is not None:
                try:
                    self.gripper.halt()
                    self.last_gripper_position=None
                    self.last_gripper_time=0.0
                except Exception as exc:
                    error=error or self._driver_failure("Gripper halt command failed",{},
                                                       category="gripper_command_failed",error=exc)
            if error:
                raise error
            if not confirmed:
                self._driver_failure("UR5e stop was not acknowledged",{},category="stop_unconfirmed")
            return confirmed
        finally:
            self.command_lock.release()

    def close(self):
        camera_pending=self._abandon_camera_operations()
        if self.recovery_pending:
            # Native C++ calls are not thread-safe against disconnect. Preserve
            # RTDE handles and report incomplete cleanup instead of racing them.
            for pipeline in self.cameras.values():
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.cameras.clear()
            if self.gripper is not None:
                self.gripper.close()
                self.gripper=None
            raise self._fault_error("Native RTDE recovery is still pending; control cleanup is incomplete",{"native_pending":True})
        try:
            if not self.stop():
                raise RuntimeError("UR5e stop was not confirmed during close")
        finally:
            for pipeline in self.cameras.values():
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.cameras.clear()
            for device in (self.control,self.receive):
                if device is not None:
                    try:
                        device.disconnect()
                    except Exception:
                        pass
            if self.gripper is not None:
                try:
                    self.gripper.close()
                except Exception:
                    pass
            self.control=self.receive=self.gripper=None
            self._control_no_upload=False
            self._camera_frames.clear()
            if camera_pending:
                raise CameraError("Camera connection cleanup is still pending in its native driver",
                                  category="camera_operation_pending",status={"native_pending":True})
