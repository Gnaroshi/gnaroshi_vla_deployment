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
from ..contracts import Observation, RobotRecoveryRequired, action_vector


@dataclass
class UR5eConfig:
    robot_ip: str = ""
    camera_serials: dict = field(default_factory=lambda:{"primary":"","wrist":""})
    camera_fps: int = 30
    width: int = 640
    height: int = 480
    home_joints: tuple = ()
    home_duration_s: float = 3.0
    home_hz: int = 10
    close_threshold: float = 0.25
    release_threshold: float = 0.10
    gripper_speed: int = 255
    gripper_force: int = 10
    gripper_period_s: float = 0.5
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
        if set(self.camera_serials)!={"primary","wrist"} or any(not v for v in self.camera_serials.values()):
            raise ValueError("Both primary and wrist camera serials must be explicit")
        if len(set(self.camera_serials.values()))!=2:
            raise ValueError("Primary and wrist must be different cameras")
        if not 0<=self.release_threshold<self.close_threshold<=1:
            raise ValueError("Invalid gripper hysteresis thresholds")
        if len(self.home_joints)!=6 or not np.isfinite(self.home_joints).all():
            raise ValueError("Task home must contain six finite joint angles")
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
        self.armed=threading.Event()
        self.command_lock=threading.Lock()
        self.control=self.receive=self.gripper=None
        self.cameras={}
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
            return RobotRecoveryRequired(self._fault["reason"], status=dict(self._fault["status"]))

    def _fault_error(self,reason,status=None):
        self.armed.clear()
        payload=dict(status or {})
        with self._fault_lock:
            if self._fault is None:
                self._fault={"reason":str(reason),"status":dict(payload)}
            payload["original_fault"]=self._fault
            self.metadata["robot_fault"]={"reason":str(reason),"status":dict(payload)}
        return RobotRecoveryRequired(str(reason),status=payload)

    def _read_status(self):
        if self.receive is None or self.control is None:
            raise self._fault_error("Robot control/receive connection is unavailable")
        try:
            status={"receive_connected":bool(self.receive.isConnected()),
                    "control_connected":bool(self.control.isConnected()),
                    "safety_mode":int(self.receive.getSafetyMode()),
                    "robot_mode":int(self.receive.getRobotMode()),
                    "program_running":bool(self.control.isProgramRunning())}
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
            raise self._fault_error("Robot safety/status query failed",{"driver_error":str(error)}) from error

    def _require_status(self,status,*,program=True):
        if not status["receive_connected"] or not status["control_connected"]:
            raise self._fault_error("RTDE connection is disconnected; restore the robot connection before retrying",status)
        if status["protective_stop"] or status["emergency_stop"] or status["safety_mode"] not in {1,2}:
            raise self._fault_error("Robot is safety stopped; clear the reported condition on the teach pendant before Recover",status)
        if status["robot_mode"]!=7:
            raise self._fault_error("Robot is not in RUNNING mode; restore it on the teach pendant before Recover",status)
        if program and not status["program_running"]:
            raise self._fault_error("RTDE control program is not running",status)
        return status

    def _assert_robot_ready(self,context):
        if self.recovery_pending:
            raise self._fault_error("A native RTDE recovery operation is still pending",{"context":context,"native_pending":True})
        with self._fault_lock:
            fault=self._fault
        if fault is not None:
            raise RobotRecoveryRequired(fault["reason"],status={**fault["status"],"context":context,"recovery_required":True})
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

    def connect_preview(self):
        """Open RGB cameras only; never import/connect robot or gripper drivers."""
        import pyrealsense2 as rs
        self.rs=rs
        try:
            for role,serial in self.config.camera_serials.items():
                if role in self.cameras:
                    continue
                pipeline=rs.pipeline()
                cfg=rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color,self.config.width,self.config.height,rs.format.rgb8,self.config.camera_fps)
                self.cameras[role]=pipeline
                pipeline.start(cfg)
        except BaseException:
            self.close()
            raise

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
            raise self._fault_error("Robot state feedback failed",{"driver_error":str(error)}) from error

    def _images(self):
        images,metadata={},{}
        for role,pipeline in self.cameras.items():
            frame=pipeline.wait_for_frames(timeout_ms=1000).get_color_frame()
            if not frame:
                raise RuntimeError(f"Missing {role} camera RGB frame")
            images[role]=np.asanyarray(frame.get_data()).copy()
            metadata[role]={"serial":self.config.camera_serials[role],"frame_number":frame.get_frame_number(),
                            "device_timestamp_ms":frame.get_timestamp(),"timestamp_domain":str(frame.get_frame_timestamp_domain())}
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

    def _servo(self,joints):
        cfg=self.config
        before=self._assert_robot_ready("servo command")
        try:
            token=self.control.initPeriod()
            if not self.control.servoJ(list(joints),cfg.velocity,cfg.acceleration,cfg.servo_time_s,cfg.servo_lookahead_s,cfg.servo_gain):
                self._fault_error("UR5e servoJ rejected command",{"status_before":before})
                try:
                    after=self._read_status()
                except RobotRecoveryRequired as error:
                    after=error.status
                raise self._fault_error("UR5e servoJ rejected command",{"status_before":before,"status_after":after})
            self.control.waitPeriod(token)
            self._assert_robot_ready("servo feedback")
        except RobotRecoveryRequired:
            raise
        except Exception as error:
            raise self._fault_error("UR5e servo command failed",{"driver_error":str(error)}) from error

    def home(self,cancel):
        cfg=self.config
        if cancel.is_set():
            raise InterruptedError("Home cancelled")
        self._assert_robot_ready("home")
        # rACT=1 can move the fingers. Activation belongs to operator Start.
        self.gripper.activate(cancel,self.command_lock)
        initial=np.asarray(self.receive.getActualQ())
        steps=max(1,round(cfg.home_duration_s*cfg.home_hz))
        for index in range(steps):
            started=time.monotonic()
            with self.command_lock:
                if cancel.is_set():
                    raise InterruptedError("Home cancelled")
                joints=initial+(np.asarray(cfg.home_joints)-initial)*(index+1)/steps
                self._servo(joints)
                if self.last_gripper_position!=0:
                    self.gripper.move(0,cfg.gripper_speed,cfg.gripper_force)
                    self.last_gripper_position=0
                    self.last_gripper_time=time.monotonic()
                self.gripper_status="open"
            if cancel.wait(max(0,1/cfg.home_hz-(time.monotonic()-started))):
                raise InterruptedError("Home cancelled")
        if cancel.wait(0.5):
            raise InterruptedError("Home cancelled")

    def begin_episode(self):
        self._assert_robot_ready("episode reset")
        state=self._state()
        with self.command_lock:
            self._assert_robot_ready("episode arming")
            self.commanded=self._transform(state[:3],state[3:6])
            self.steps=0
            self.armed.set()

    def apply(self,prediction):
        raw=action_vector(prediction.action)
        if self.recovery_required:
            self._assert_robot_ready("policy command")
        action=raw.copy()
        cfg=self.config
        if cfg.max_rel_position is not None:
            action[:3]=np.clip(action[:3],-cfg.max_rel_position,cfg.max_rel_position)
        if cfg.max_rel_rotation is not None:
            action[3:6]=np.clip(action[3:6],-cfg.max_rel_rotation,cfg.max_rel_rotation)
        action[6]=np.clip(action[6],0,1)
        with self.command_lock:
            if not self.armed.is_set():
                raise InterruptedError("Policy command cancelled")
            self._assert_robot_ready("policy command")
            target=self.commanded @ self._transform(action[:3],action[3:6])
            tcp=np.r_[target[:3,3],self.Rotation.from_matrix(target[:3,:3]).as_rotvec()]
            try:
                current=list(self.receive.getActualQ())
                joints=self.control.getInverseKinematics(tcp.tolist(),current)
            except Exception as error:
                raise self._fault_error("Inverse kinematics request failed",{"driver_error":str(error),"target_tcp_rotvec":tcp.tolist()}) from error
            if joints is None or len(joints)!=6 or not np.isfinite(joints).all():
                raise self._fault_error("No inverse kinematics solution for predicted action",{"target_tcp_rotvec":tcp.tolist(),"raw_action":raw.tolist()})
            # Stop latches without waiting for this lock, so recheck after IK.
            if not self.armed.is_set():
                raise InterruptedError("Policy command cancelled during IK")
            self._servo(joints)
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
                    self.gripper.move(position,cfg.gripper_speed,cfg.gripper_force)
                    self.last_gripper_position=position
                    self.last_gripper_time=now
                    self.gripper_status=desired
                    sent=True
            return {"raw_action":raw.tolist(),"clipped_action":action.tolist(),"target_tcp_rotvec":tcp.tolist(),
                    "target_joints":list(joints),"gripper_desired":desired,"gripper_command":self.last_gripper_position,
                    "gripper_command_sent":sent,"gripper_state":self.gripper_status}

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
                    error=exc
            if self.gripper is not None:
                try:
                    self.gripper.halt()
                except Exception as exc:
                    error=error or exc
            if error:
                raise error
            if not confirmed:
                self._fault_error("UR5e stop was not acknowledged")
            return confirmed
        finally:
            self.command_lock.release()

    def close(self):
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
