"""Operator cancellation and outcomes must remain correct during slow inference."""
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from gnaroshi_deployment.adapters.mock import MockEmbodiment,MockPolicy
from gnaroshi_deployment.adapters.ur5e import UR5eConfig,UR5eEmbodiment
from gnaroshi_deployment.contracts import SessionConfig,Observation
from gnaroshi_deployment.records import RolloutRecorder,summary
from gnaroshi_deployment.session import DeploymentSession


def ur5e_config(**overrides):
    values={"robot_ip":"127.0.0.1","camera_serials":{"primary":"mock-primary","wrist":"mock-wrist"},
            "home_joints":(0.0,)*6}
    values.update(overrides)
    return UR5eConfig(**values)


class HealthyRobot:
    def isConnected(self): return True
    def isProgramRunning(self): return True
    def getSafetyMode(self): return 1
    def getRobotMode(self): return 7
    def isProtectiveStopped(self): return False
    def isEmergencyStopped(self): return False


class Sessions(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.sessions=[]

    def tearDown(self):
        for session in self.sessions:
            session.close()
            session.worker.join(3)
            self.assertFalse(session.worker.is_alive())
        self.temporary.cleanup()

    def wait(self,predicate):
        deadline=time.monotonic()+3
        while not predicate():
            if time.monotonic()>deadline:
                self.fail("State transition timed out")
            time.sleep(.005)

    def make(self,policy=None,embodiment=None,steps=200):
        cfg=SessionConfig(output_root=self.temporary.name,record_video=False,max_steps=steps,target_hz=200)
        session=DeploymentSession(cfg,policy or MockPolicy(.002),embodiment or MockEmbodiment(),Path(__file__).parents[1])
        self.sessions.append(session)
        session.start()
        self.wait(lambda:session.state=="READY")
        return session

    def test_success_and_failure_are_manual_save_then_home(self):
        session=self.make()
        for outcome in ("success","failure"):
            self.assertTrue(session.start_rollout())
            self.assertFalse(session.start_rollout())
            self.wait(lambda:session.state=="RUNNING" and len(session.rows)>=3)
            session.signal(outcome)
            self.wait(lambda:session.state=="READY")
        self.assertEqual(session.embodiment.home_count,4)
        self.assertEqual([r["outcome"] for r in session.results],["success","failure"])
        self.assertTrue(all(r["outcome_home"]=="completed" for r in session.results))
        records=json.loads((session.folder/"results.json").read_text())
        self.assertEqual(records["success_rate"],.5)
        session.delete_previous()
        self.assertEqual(len(session.results),1)
        self.assertEqual(len(list(session.folder.glob("rollout_*"))),1)

    def test_stop_during_inference_never_applies_late_action_or_homes(self):
        entered,release=threading.Event(),threading.Event()
        class SlowPolicy(MockPolicy):
            def infer(self,observation,instruction):
                entered.set()
                release.wait(2)
                return super().infer(observation,instruction)
        session=self.make(SlowPolicy())
        session.start_rollout()
        self.assertTrue(entered.wait(2))
        session.signal("stop")
        self.wait(lambda:session.embodiment.cancel.is_set())
        release.set()
        self.wait(lambda:session.state=="READY")
        self.assertEqual(len(session.embodiment.applied),0)
        self.assertEqual(session.embodiment.home_count,1)
        self.assertEqual(session.results,[])
        self.assertEqual(list(session.folder.glob("rollout_*")),[])

    def test_max_steps_waits_for_label_instead_of_claiming_success(self):
        session=self.make(steps=3)
        session.start_rollout()
        self.wait(lambda:session.state=="WAITING_OUTCOME")
        self.assertEqual(len(session.embodiment.applied),3)
        self.assertEqual(session.results,[])
        with self.assertRaises(RuntimeError):
            session.apply_settings(60,100)
        session.signal("failure")
        self.wait(lambda:session.state=="READY")
        self.assertEqual(session.results[0]["outcome"],"failure")

    def test_invalid_model_reply_stops_and_preserves_automatic_failure(self):
        class BadPolicy(MockPolicy):
            def infer(self,observation,instruction):
                result=super().infer(observation,instruction)
                result.action[:]=np.nan
                return result
        session=self.make(BadPolicy())
        session.start_rollout()
        self.wait(lambda:session.state=="ERROR")
        self.assertEqual(session.embodiment.applied,[])
        self.assertEqual([row["outcome"] for row in session.results],["failure"])
        files=list(session.folder.glob("rollout_*/result.json"))
        self.assertEqual(len(files),1)
        self.assertEqual(json.loads(files[0].read_text())["outcome"],"failure")

    def test_policy_health_failure_prevents_home_and_rearming(self):
        class HealthPolicy(MockPolicy):
            failed = False
            def check_ready(self):
                if self.failed:
                    raise RuntimeError("Policy process is unavailable")
        policy = HealthPolicy()
        session = self.make(policy)
        policy.failed = True
        session.start_rollout()
        self.wait(lambda:session.state == "ERROR")
        session.worker.join(1)
        self.assertEqual(session.embodiment.home_count, 0)
        self.assertEqual(session.embodiment.applied, [])
        self.assertFalse(session.can_replace)

    def test_policy_health_is_checked_while_ready(self):
        class HealthPolicy(MockPolicy):
            failed = False
            def check_ready(self):
                if self.failed:
                    raise RuntimeError("Policy process is unavailable")
        policy = HealthPolicy()
        session = self.make(policy)
        policy.failed = True
        self.wait(lambda:session.state == "ERROR")
        self.assertEqual(session.embodiment.home_count, 0)

    def test_stop_before_worker_dispatch_does_not_move_home(self):
        session=self.make()
        with session.lock:
            session.start_rollout()
            session.signal("stop")
        self.wait(lambda:session.state=="READY")
        self.assertEqual(session.embodiment.home_count,0)

    def test_close_during_inference_is_bounded_and_motion_free(self):
        entered=threading.Event()
        class SlowPolicy(MockPolicy):
            def infer(self,observation,instruction):
                entered.set()
                time.sleep(.1)
                return super().infer(observation,instruction)
        session=self.make(SlowPolicy())
        session.start_rollout()
        self.assertTrue(entered.wait(2))
        session.close()
        session.worker.join(1)
        self.assertFalse(session.worker.is_alive())
        self.assertEqual(session.embodiment.applied,[])
        self.assertEqual(session.state,"CLOSED")

    def test_close_cancels_policy_startup_without_reporting_an_error(self):
        entered = threading.Event()
        class LoadingPolicy(MockPolicy):
            def connect(self):
                entered.set()
                session.closing.wait(2)
                raise InterruptedError("Policy startup cancelled")
        class UnconnectedEmbodiment(MockEmbodiment):
            connect_calls = 0
            def connect(self):
                self.connect_calls += 1
        embodiment = UnconnectedEmbodiment()
        config = SessionConfig(output_root=self.temporary.name, record_video=False)
        session = DeploymentSession(config, LoadingPolicy(), embodiment, Path(__file__).parents[1])
        self.sessions.append(session)
        session.start()
        self.assertTrue(entered.wait(1))
        session.close()
        session.worker.join(1)
        self.assertEqual(session.state, "CLOSED")
        self.assertIsNone(session.error)
        self.assertEqual(embodiment.connect_calls, 0)
        self.assertTrue(session.replacement_closed)

    def test_close_during_cleanup_does_not_issue_another_stop(self):
        entered, release = threading.Event(), threading.Event()
        class HealthPolicy(MockPolicy):
            failed = False
            def check_ready(self):
                if self.failed:
                    raise RuntimeError("Policy process is unavailable")
        class ClosingEmbodiment(MockEmbodiment):
            stop_calls = 0
            def stop(self):
                self.stop_calls += 1
                return super().stop()
            def close(self):
                entered.set()
                release.wait(2)
        policy, embodiment = HealthPolicy(), ClosingEmbodiment()
        session = self.make(policy, embodiment)
        policy.failed = True
        try:
            self.assertTrue(entered.wait(1))
            self.assertEqual(embodiment.stop_calls, 1)
            session.close()
            self.assertEqual(embodiment.stop_calls, 1)
        finally:
            release.set()
        session.worker.join(1)
        self.assertTrue(session.shutdown_complete)
        self.assertEqual(session.state, "ERROR")

    def test_predispatch_stop_blocks_new_start_until_stop_acknowledged(self):
        observed,observe_release=threading.Event(),threading.Event()
        stop_entered,stop_release=threading.Event(),threading.Event()
        class BlockedEmbodiment(MockEmbodiment):
            def observe(self):
                observed.set()
                observe_release.wait(2)
                return super().observe()
            def stop(self):
                stop_entered.set()
                stop_release.wait(2)
                return super().stop()
        session=self.make(embodiment=BlockedEmbodiment())
        self.assertTrue(observed.wait(1))
        session.start_rollout()
        session.signal("stop")
        self.assertTrue(stop_entered.wait(1))
        observe_release.set()
        time.sleep(.05)
        self.assertEqual(session.state,"STOPPING")
        self.assertFalse(session.start_rollout())
        stop_release.set()
        self.wait(lambda:session.state=="READY")
        self.assertEqual(session.embodiment.home_count,0)

    def test_stop_during_outcome_home_blocks_new_start_until_second_stop_ack(self):
        home_entered=threading.Event()
        stop_entered,stop_release=threading.Event(),threading.Event()
        class BlockedEmbodiment(MockEmbodiment):
            stop_count=0
            def home(self,cancel):
                if self.home_count==1:
                    home_entered.set()
                    cancel.wait(2)
                    raise InterruptedError("outcome home cancelled")
                return super().home(cancel)
            def stop(self):
                self.stop_count+=1
                if self.stop_count==2:
                    stop_entered.set()
                    stop_release.wait(2)
                return super().stop()
        session=self.make(embodiment=BlockedEmbodiment())
        session.start_rollout()
        self.wait(lambda:len(session.rows)>=3)
        session.signal("success")
        self.assertTrue(home_entered.wait(1))
        session.signal("stop")
        self.assertTrue(stop_entered.wait(1))
        time.sleep(.05)
        self.assertEqual(session.state,"STOPPING")
        self.assertFalse(session.start_rollout())
        stop_release.set()
        self.wait(lambda:session.state=="READY")
        self.assertEqual(session.results[0]["outcome_home"],"cancelled")

    def test_unacknowledged_stop_is_sticky_error_and_closes_resources(self):
        class RejectStop(MockEmbodiment):
            closed=False
            def stop(self):
                super().stop()
                return False
            def close(self):
                self.closed=True
        embodiment=RejectStop()
        session=self.make(embodiment=embodiment)
        session.start_rollout()
        self.wait(lambda:len(session.rows)>=3)
        session.signal("success")
        self.wait(lambda:session.state=="ERROR")
        session.worker.join(1)
        self.assertTrue(embodiment.closed)
        self.assertFalse(session.start_rollout())
        self.assertEqual(embodiment.home_count,1)

    def test_stop_exception_does_not_skip_close(self):
        class FailedStop(MockEmbodiment):
            closed=False
            def stop(self):
                raise RuntimeError("stop transport error")
            def close(self):
                self.closed=True
        embodiment=FailedStop()
        session=self.make(embodiment=embodiment)
        session.close()
        session.worker.join(1)
        self.assertTrue(embodiment.closed)
        self.assertEqual(session.state,"ERROR")


class HardwareContract(unittest.TestCase):
    def test_robot_identity_and_home_are_supplied_by_owning_project(self):
        blank=UR5eConfig()
        self.assertEqual(blank.robot_ip,"")
        self.assertEqual(blank.home_joints,())
        self.assertEqual(blank.camera_serials,{"primary":"","wrist":""})
        with self.assertRaisesRegex(ValueError,"Robot address"):
            UR5eEmbodiment(blank)
        self.assertIsNone(UR5eEmbodiment(ur5e_config()).control)

    def test_connect_reads_gripper_status_without_activation_or_motion(self):
        from types import SimpleNamespace
        import gnaroshi_deployment.adapters.ur5e as driver
        class Pipeline:
            def start(self,config):
                pass
            def stop(self):
                pass
        class CameraConfig:
            def enable_device(self,serial):
                pass
            def enable_stream(self,*args):
                pass
        class Device(HealthyRobot):
            FLAG_NO_WAIT=16
            def __init__(self,host,frequency=-1.0,*,flags=1):
                self.frequency,self.flags=frequency,flags
            def servoStop(self):
                return True
            def disconnect(self):
                pass
        class FakeGripper:
            activation_calls=0
            def __init__(self,host):
                pass
            def get(self,name):
                return 0
            def activate(self,*args):
                self.activation_calls+=1
            def halt(self):
                pass
            def close(self):
                pass
        fake_rs=SimpleNamespace(pipeline=Pipeline,config=CameraConfig,
                                stream=SimpleNamespace(color=1),format=SimpleNamespace(rgb8=1))
        modules={"pyrealsense2":fake_rs,
                 "rtde_control":SimpleNamespace(RTDEControlInterface=Device),
                 "rtde_receive":SimpleNamespace(RTDEReceiveInterface=Device),
                 "scipy.spatial.transform":SimpleNamespace(Rotation=object)}
        env=UR5eEmbodiment(ur5e_config())
        with patch.dict("sys.modules",modules),patch.object(driver,"Gripper",FakeGripper):
            env.connect()
            self.assertEqual(env.gripper.activation_calls,0)
            self.assertEqual(env.metadata["gripper_status_at_connect"],0)
            self.assertEqual((env.control.frequency,env.control.flags),(-1.0,16))
            self.assertEqual(env.control.flags&1,0)
            env.close()

    def test_cancelled_gripper_activation_never_sends_activation_register(self):
        from gnaroshi_deployment.adapters.ur5e import Gripper
        gripper=Gripper.__new__(Gripper)
        commands=[]
        gripper.get=lambda name:0
        gripper.set=lambda **values:commands.append(values)
        cancel=threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            gripper.activate(cancel,threading.Lock())
        self.assertEqual(commands,[])

    def test_false_servo_stop_ack_is_not_reported_as_success(self):
        env=UR5eEmbodiment(ur5e_config())
        class Control(HealthyRobot):
            def servoStop(self):
                return False
        env.control=Control()
        self.assertFalse(env.stop())

    def test_device_close_runs_even_if_stop_raises(self):
        env=UR5eEmbodiment(ur5e_config())
        class Control:
            closed=False
            def servoStop(self):
                raise RuntimeError("stop failed")
            def disconnect(self):
                self.closed=True
        class Device:
            closed=False
            def stop(self):
                self.closed=True
            def disconnect(self):
                self.closed=True
            def close(self):
                self.closed=True
        control,receive,camera,gripper=Control(),Device(),Device(),Device()
        env.control,env.receive,env.gripper=control,receive,gripper
        env.cameras={"primary":camera}
        with self.assertRaises(RuntimeError):
            env.close()
        self.assertTrue(all(device.closed for device in (control,receive,camera,gripper)))

    def test_gripper_hysteresis_and_commanded_accumulator(self):
        env=UR5eEmbodiment(ur5e_config())
        class Rotation:
            @staticmethod
            def from_matrix(matrix):
                return Rotation()
            def as_rotvec(self):
                return np.zeros(3)
        class Control(HealthyRobot):
            def getInverseKinematics(self,tcp,current):
                return [0.0]*6
            def initPeriod(self):
                return None
            def waitPeriod(self,token):
                pass
            def servoJ(self,*args):
                return True
        class Receive(HealthyRobot):
            def getActualQ(self):
                return [0.0]*6
        class Gripper:
            commands=[]
            def move(self,position,speed,force):
                self.commands.append(position)
        env.Rotation=Rotation
        env.control,env.receive,env.gripper=Control(),Receive(),Gripper()
        env.commanded=np.eye(4)
        env.armed.set()
        def transform(xyz,euler):
            result=np.eye(4)
            result[:3,3]=xyz
            return result
        env._transform=transform
        from gnaroshi_deployment.contracts import Prediction
        env.last_gripper_position=0
        with patch("gnaroshi_deployment.adapters.ur5e.time.monotonic",return_value=10):
            for _ in range(5):
                env.apply(Prediction(np.array([.001,0,0,0,0,0,.32])))
            self.assertEqual(env.gripper.commands,[])
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.32])))
            self.assertEqual(env.gripper.commands,[255])
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.05])))
            self.assertEqual(env.gripper.commands,[255])
        with patch("gnaroshi_deployment.adapters.ur5e.time.monotonic",return_value=10.6):
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.05])))
        self.assertEqual(env.gripper.commands,[255,0])
        self.assertAlmostEqual(env.commanded[0,3],.008)

    def test_camera_roles_are_explicit_and_distinct_without_importing_drivers(self):
        with self.assertRaises(ValueError):
            ur5e_config(camera_serials={"primary":"same","wrist":"same"}).validate()
        env=UR5eEmbodiment(ur5e_config())
        self.assertIsNone(env.control)
        self.assertEqual(env.metadata["camera_serials"]["primary"],"mock-primary")

    def test_stop_latches_before_command_lock_and_late_apply_is_rejected(self):
        env=UR5eEmbodiment(ur5e_config())
        env.armed.set()
        with env.command_lock:
            thread=threading.Thread(target=env.stop)
            thread.start()
            deadline=time.monotonic()+1
            while env.armed.is_set() and time.monotonic()<deadline:
                time.sleep(.001)
            self.assertFalse(env.armed.is_set())
        thread.join(1)
        from gnaroshi_deployment.contracts import Prediction
        with self.assertRaises(InterruptedError):
            env.apply(Prediction(np.zeros(7)))

    def test_actual_command_hz_is_not_requested_hz_or_chunk_latency(self):
        rows=[{"command_monotonic_s":x,"policy_ms":50,"deadline_miss":True} for x in (10.,10.1,10.2)]
        self.assertAlmostEqual(summary(rows)["achieved_command_hz"],10)

    def test_video_duration_follows_wall_clock_instead_of_target_rate(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder=RolloutRecorder(Path(temporary)/"rollout",60,False)
            for step in range(15):
                observation=Observation({"primary":np.zeros((2,2,3),np.uint8)},np.zeros(7),step*100000000)
                recorder.write({"step":step},observation)
            recorder.close()
            frames=[json.loads(line) for line in (recorder.folder/"frames.jsonl").read_text().splitlines()]
            self.assertIsNone(frames[-1]["video_frame_end_exclusive"])
            self.assertAlmostEqual((recorder.video_frames-1)/recorder.fps,1.4)

    def test_backward_capture_timestamp_is_rejected_even_within_one_video_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            recorder=RolloutRecorder(Path(temporary)/"rollout",60,False)
            try:
                for step,timestamp in enumerate((1000,2000)):
                    recorder.write({"step":step},Observation({},np.zeros(7),timestamp))
                with self.assertRaises(ValueError):
                    recorder.write({"step":2},Observation({},np.zeros(7),1999))
            finally:
                recorder.close()

    def test_video_holds_previous_image_until_new_capture_and_preserves_source_mapping(self):
        from types import SimpleNamespace
        encoded=[]
        class Writer:
            def isOpened(self):
                return True
            def write(self,frame):
                encoded.append(frame.copy())
            def release(self):
                pass
        fakecv=SimpleNamespace(VideoWriter=lambda *args:Writer(),VideoWriter_fourcc=lambda *args:0,
                               cvtColor=lambda frame,code:frame,COLOR_RGB2BGR=0)
        with tempfile.TemporaryDirectory() as temporary,patch.dict("sys.modules",{"cv2":fakecv}):
            recorder=RolloutRecorder(Path(temporary)/"rollout",60,True)
            red=np.zeros((2,2,3),np.uint8)
            red[:,:,0]=255
            blue=np.zeros((2,2,3),np.uint8)
            blue[:,:,2]=255
            recorder.write({"step":0},Observation({"primary":red},np.zeros(7),0))
            recorder.write({"step":1},Observation({"primary":blue},np.zeros(7),1000000000))
            recorder.close()
            self.assertEqual(len(encoded),31)
            self.assertTrue(all(np.array_equal(frame,red) for frame in encoded[:30]))
            self.assertTrue(np.array_equal(encoded[30],blue))
            segments=[json.loads(line) for line in (recorder.folder/"frames.jsonl").read_text().splitlines() if json.loads(line)["kind"]=="video_frames"]
            self.assertEqual([(s["source_step"],s["video_frame_start"],s["video_frame_end_exclusive"]) for s in segments],[(0,0,30),(1,30,31)])


if __name__=="__main__":
    unittest.main()
