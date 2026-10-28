"""Explicit two-RealSense UR5e adapter for tool-local relative EEF actions.

The owning project supplies its robot address, task home and camera serial roles.
Camera-only preview is separate from policy-gated control connections.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import socket
import threading
import time
import numpy as np
from ..contracts import Observation, action_vector


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
            self.receive=rtde_receive.RTDEReceiveInterface(self.config.robot_ip)
            self.control=rtde_control.RTDEControlInterface(self.config.robot_ip)
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
        tcp=np.asarray(self.receive.getActualTCPPose(),dtype=np.float64)
        euler=self.Rotation.from_rotvec(tcp[3:]).as_euler("xyz")
        return np.r_[tcp[:3],euler,self.gripper.get("POS")/255].astype(np.float32)

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
        images, metadata = self._images()
        return Observation(images,self._state(),time.time_ns(),metadata,time.monotonic_ns())

    def _servo(self,joints):
        cfg=self.config
        token=self.control.initPeriod()
        if not self.control.servoJ(list(joints),cfg.velocity,cfg.acceleration,cfg.servo_time_s,cfg.servo_lookahead_s,cfg.servo_gain):
            raise RuntimeError("UR5e servoJ rejected command")
        self.control.waitPeriod(token)

    def home(self,cancel):
        cfg=self.config
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
        state=self._state()
        with self.command_lock:
            self.commanded=self._transform(state[:3],state[3:6])
            self.steps=0
            self.armed.set()

    def apply(self,prediction):
        raw=action_vector(prediction.action)
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
            target=self.commanded @ self._transform(action[:3],action[3:6])
            tcp=np.r_[target[:3,3],self.Rotation.from_matrix(target[:3,:3]).as_rotvec()]
            current=list(self.receive.getActualQ())
            joints=self.control.getInverseKinematics(tcp.tolist(),current)
            if joints is None or len(joints)!=6 or not np.isfinite(joints).all():
                raise RuntimeError("No inverse kinematics solution for predicted action")
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
            return confirmed
        finally:
            self.command_lock.release()

    def close(self):
        try:
            self.stop()
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
