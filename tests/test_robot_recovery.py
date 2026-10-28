"""UR5e recovery uses fake RTDE objects only: no SDK/device constructors."""
import json
import threading
import time
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from gnaroshi_deployment.adapters.ur5e import UR5eConfig,UR5eEmbodiment
from gnaroshi_deployment.contracts import Prediction,RobotActionError,RobotRecoveryRequired


class Receive:
    def __init__(self):
        self.connected=True
        self.safety=1
        self.robot_mode=7
        self.timestamp=10.0
        self.advance=True
        self.disconnects=0

    def isConnected(self):
        return self.connected

    def getSafetyMode(self):
        return self.safety

    def getRobotMode(self):
        return self.robot_mode

    def isProtectiveStopped(self):
        return self.safety==3

    def isEmergencyStopped(self):
        return self.safety in {6,7}

    def getTimestamp(self):
        if self.advance:
            self.timestamp+=.002
        return self.timestamp

    def getRuntimeState(self):
        return 2

    def getActualTCPPose(self):
        return [.3,.1,.4,0,0,0]

    def getActualQ(self):
        return [0.0]*6

    def reconnect(self):
        raise AssertionError("Recovery must not reconnect this SDK object")

    def disconnect(self):
        self.disconnects+=1
        self.connected=False


class Control:
    def __init__(self):
        self.connected=True
        self.program=True
        self.calls=[]
        self.ik=[0.0]*6
        self.servo_result=True
        self.stop_hook=None
        self.upload_hook=None
        self.upload_result=True
        self.command_register=11  # Old SERVOJ-like command must be cleared.
        self.disconnects=0

    def isConnected(self):
        return self.connected

    def isProgramRunning(self):
        return self.program

    def servoStop(self):
        self.calls.append("servoStop")
        if self.stop_hook:
            self.stop_hook()
        # This models the audited 1.6.3 sendCommand False/True clear paths.
        self.command_register=0
        return self.program

    def reuploadScript(self):
        self.calls.append("reuploadScript")
        if self.command_register!=0 or self.program:
            raise AssertionError("Upload attempted without an idle command register and stopped program")
        if self.upload_hook:
            self.upload_hook()
        if self.upload_result:
            self.program=True
        return self.upload_result

    def getInverseKinematics(self,tcp,current):
        self.calls.append("IK")
        return self.ik

    def initPeriod(self):
        return None

    def waitPeriod(self,token):
        pass

    def servoJ(self,*args):
        self.calls.append("servoJ")
        return self.servo_result

    def reconnect(self):
        raise AssertionError("Recovery must not call the implicit uploading reconnect API")

    def disconnect(self):
        self.disconnects+=1
        self.connected=False


class Gripper:
    def __init__(self):
        self.calls=[]

    def get(self,name):
        return 0

    def activate(self,*args):
        self.calls.append("activate")

    def move(self,*args):
        self.calls.append("move")

    def halt(self):
        self.calls.append("halt")

    def close(self):
        self.calls.append("close")


class UR5eRecovery(unittest.TestCase):
    def setUp(self):
        self.environments=[]
        self.releases=[]

    def tearDown(self):
        for release in self.releases:
            release.set()
        for env in self.environments:
            thread=env._recovery_thread
            if thread:
                thread.join(2)
                self.assertFalse(thread.is_alive(),"Fake SDK recovery worker leaked")

    def make(self,*,timeout=.5,freshness=.05):
        cfg=UR5eConfig(robot_ip="MOCK",camera_serials={"primary":"MOCK_P","wrist":"MOCK_W"},
                       home_joints=(0,-1,1,-1,-1,0),recovery_timeout_s=timeout,
                       recovery_freshness_timeout_s=freshness)
        env=UR5eEmbodiment(cfg)
        env.control,env.receive,env.gripper=Control(),Receive(),Gripper()
        env._sdk_version="1.6.3"
        env._control_no_upload=True
        env.commanded=np.eye(4)
        env.steps=99
        env.armed.set()
        env._images=lambda:({}, {})
        env.Rotation=SimpleNamespace(
            from_euler=lambda *args:SimpleNamespace(as_matrix=lambda:np.eye(3)),
            from_matrix=lambda *args:SimpleNamespace(as_rotvec=lambda:np.zeros(3)),
            from_rotvec=lambda *args:SimpleNamespace(as_euler=lambda *args:np.zeros(3)))
        camera=SimpleNamespace(stop_calls=0)
        def stop_camera():
            camera.stop_calls+=1
        camera.stop=stop_camera
        env.cameras={"primary":camera}
        self.environments.append(env)
        return env

    def block(self):
        entered,release=threading.Event(),threading.Event()
        self.releases.append(release)
        def wait():
            entered.set()
            release.wait(2)
        return entered,release,wait

    def test_safety_faults_block_every_motion_entry_and_disarm(self):
        for method in ("observe","home","begin_episode","apply","_servo"):
            with self.subTest(method=method):
                env=self.make()
                env.receive.safety=3
                args={"observe":(),"home":(threading.Event(),),"begin_episode":(),
                      "apply":(Prediction(np.zeros(7)),),"_servo":([0.0]*6,)}[method]
                with self.assertRaises(RobotRecoveryRequired) as raised:
                    getattr(env,method)(*args)
                self.assertEqual(raised.exception.status["safety_mode"],3)
                self.assertFalse(env.armed.is_set())
                self.assertEqual(env.control.calls,[])
                self.assertEqual(env.gripper.calls,[])

    def test_fault_categories_distinguish_safety_connection_program_and_ik(self):
        for category,change in (
            ("protective_stop",lambda env:setattr(env.receive,"safety",3)),
            ("emergency_stop",lambda env:setattr(env.receive,"safety",6)),
            ("robot_safety_stop",lambda env:setattr(env.receive,"safety",9)),
            ("robot_disconnected",lambda env:setattr(env.receive,"connected",False)),
            ("robot_disconnected",lambda env:setattr(env.control,"connected",False)),
            ("robot_not_ready",lambda env:setattr(env.receive,"robot_mode",5)),
            ("control_program_stopped",lambda env:setattr(env.control,"program",False)),
        ):
            with self.subTest(category=category):
                env=self.make()
                change(env)
                with self.assertRaises(RobotRecoveryRequired) as raised:
                    env.observe()
                self.assertEqual(raised.exception.category,category)
                self.assertEqual(env.recovery_error.category,category)
                json.dumps(raised.exception.asdict(),allow_nan=False)
                self.assertEqual(env.control.calls,[])
                self.assertEqual(env.gripper.calls,[])

    def test_disconnection_is_recorded_before_querying_stale_feedback(self):
        env=self.make()
        env.receive.connected=False
        env.receive.getSafetyMode=lambda:(_ for _ in ()).throw(AssertionError("Stale feedback was queried"))
        with self.assertRaises(RobotRecoveryRequired) as raised:
            env.observe()
        self.assertEqual(raised.exception.category,"robot_disconnected")
        self.assertFalse(raised.exception.status["receive_connected"])
        self.assertEqual(env.control.calls,[])

    def test_no_ik_records_action_size_and_target_without_claiming_a_cause(self):
        env=self.make()
        env.control.ik=None
        action=np.array([-.039492,.004053,-.074909,.01,.02,-.003,.7])
        previous=env.commanded.copy()
        with self.assertRaises(RobotRecoveryRequired) as raised:
            env.apply(Prediction(action))
        error=raised.exception
        self.assertEqual(error.category,"ik_no_solution")
        self.assertIn("84.779 mm",str(error))
        self.assertAlmostEqual(error.status["action_translation_norm_m"],np.linalg.norm(action[:3]))
        self.assertAlmostEqual(error.status["action_rotation_norm_rad"],np.linalg.norm(action[3:6]))
        np.testing.assert_allclose(error.status["raw_action"],action)
        np.testing.assert_allclose(error.status["previous_commanded_transform"],previous)
        self.assertIn("target_tcp_rotvec",error.status)
        self.assertIn("measured_joints",error.status)
        self.assertFalse(error.status["arm_command_attempted"])
        np.testing.assert_array_equal(env.commanded,previous)
        self.assertEqual(env.control.calls,["IK"])
        self.assertEqual(env.gripper.calls,[])
        self.assertNotIn("too large",str(error))
        json.dumps(error.asdict(),allow_nan=False)
        json.dumps(env.metadata,allow_nan=False)

    def test_invalid_action_is_json_safe_fatal_and_never_sends_a_command(self):
        for action in (np.array([np.nan]*7),np.zeros(6),["bad"]*7):
            with self.subTest(action=repr(action)):
                env=self.make()
                with self.assertRaises(RobotActionError) as raised:
                    env.apply(Prediction(action))
                self.assertEqual(raised.exception.category,"invalid_action")
                self.assertFalse(env.armed.is_set())
                self.assertFalse(env.recovery_required)
                self.assertEqual(env.control.calls,[])
                self.assertEqual(env.gripper.calls,[])
                json.dumps(raised.exception.asdict(),allow_nan=False)

    def test_servo_rejection_reports_observed_protective_stop_and_action(self):
        env=self.make()
        def rejected(*args):
            env.control.calls.append("servoJ")
            env.receive.safety=3
            return False
        env.control.servoJ=rejected
        with self.assertRaises(RobotRecoveryRequired) as raised:
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.2])))
        self.assertEqual(raised.exception.category,"protective_stop")
        self.assertEqual(raised.exception.status["operation_failure_category"],"servo_rejected")
        self.assertEqual(raised.exception.status["safety_mode"],3)
        self.assertEqual(raised.exception.status["raw_action"][0],.001)
        self.assertTrue(raised.exception.status["arm_command_attempted"])
        self.assertIsNone(raised.exception.status["arm_command_sent"])
        self.assertFalse(raised.exception.status["arm_command_acknowledged"])
        self.assertEqual(env.control.calls,["IK","servoJ"])
        self.assertEqual(env.gripper.calls,[])
        json.dumps(raised.exception.asdict(),allow_nan=False)

    def test_disconnect_after_servo_ack_keeps_the_sent_command_evidence(self):
        env=self.make()
        def lost(token):
            env.receive.connected=False
            raise ConnectionError("Mock RTDE connection lost after acknowledgement")
        env.control.waitPeriod=lost
        with self.assertRaises(RobotRecoveryRequired) as raised:
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.2])))
        error=raised.exception
        self.assertEqual(error.category,"robot_disconnected")
        self.assertTrue(error.status["arm_command_attempted"])
        self.assertTrue(error.status["arm_command_sent"])
        self.assertTrue(error.status["arm_command_acknowledged"])
        self.assertEqual(env.recovery_error.category,error.category)
        for field in ("arm_command_attempted","arm_command_sent","arm_command_acknowledged"):
            self.assertEqual(env.recovery_error.status[field],error.status[field])
        self.assertEqual(error.status["operation_failure_category"],"servo_command_failed")
        self.assertIn("after acknowledgement",error.status["driver_error"])
        self.assertEqual(env.control.calls,["IK","servoJ"])
        self.assertEqual(env.gripper.calls,[])
        self.assertFalse(env.armed.is_set())
        json.dumps(error.asdict(),allow_nan=False)

    def test_overflowing_action_target_is_recordable_and_never_reaches_ik(self):
        env=self.make()
        env.commanded[0,3]=1e308
        with self.assertRaises(RobotActionError) as raised:
            env.apply(Prediction(np.array([1e308,0,0,0,0,0,.2])))
        self.assertEqual(raised.exception.category,"invalid_action_target")
        self.assertFalse(env.armed.is_set())
        self.assertFalse(env.recovery_required)
        self.assertEqual(env.control.calls,[])
        self.assertEqual(env.gripper.calls,[])
        json.dumps(raised.exception.asdict(),allow_nan=False)

    def test_lost_gripper_reply_keeps_sent_unknown_and_preserves_arm_ack(self):
        env=self.make()
        def send_then_lose_reply(*args):
            env.gripper.calls.append("move_sent")
            raise ConnectionError("Mock gripper SET reply lost after send")
        env.gripper.move=send_then_lose_reply
        with self.assertRaises(RobotRecoveryRequired) as raised:
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.2])))
        error=raised.exception
        self.assertEqual(error.category,"gripper_command_failed")
        self.assertTrue(error.status["arm_command_sent"])
        self.assertTrue(error.status["arm_command_acknowledged"])
        self.assertTrue(error.status["gripper_command_attempted"])
        self.assertIsNone(error.status["gripper_command_sent"])
        self.assertFalse(error.status["gripper_command_acknowledged"])
        for field in ("arm_command_attempted","arm_command_sent","arm_command_acknowledged",
                      "gripper_command_attempted","gripper_command_sent","gripper_command_acknowledged"):
            self.assertEqual(env.recovery_error.status[field],error.status[field])
        self.assertEqual(env.control.calls,["IK","servoJ"])
        self.assertEqual(env.gripper.calls,["move_sent"])
        self.assertFalse(env.armed.is_set())
        json.dumps(error.asdict(),allow_nan=False)
        json.dumps(env.metadata,allow_nan=False)

    def test_idle_home_preserves_gripper_while_start_home_still_prepares_it(self):
        class Running:
            def is_set(self):return False
            def wait(self,timeout):return False
        for idle in (True,False):
            with self.subTest(idle=idle):
                env=self.make()
                env.config.home_duration_s=.001
                env.config.home_hz=1000
                env.last_gripper_position=255
                env.gripper_status="close"
                before=env.commanded.copy()
                (env.home_idle if idle else env.home)(Running())
                self.assertEqual(env.control.calls,["servoJ"])
                self.assertEqual(env.gripper.calls,[] if idle else ["activate","move"])
                self.assertEqual(env.last_gripper_position,255 if idle else 0)
                self.assertEqual(env.gripper_status,"close" if idle else "open")
                self.assertEqual(env.steps,99)
                np.testing.assert_array_equal(env.commanded,before)
                if idle:self.assertFalse(env.armed.is_set())

    def test_unsafe_modes_and_emergency_stop_are_not_unlocked_by_recover(self):
        for safety,robot_mode in ((3,7),(6,7),(7,7),(9,7),(1,5),(1,3)):
            with self.subTest(safety=safety,robot_mode=robot_mode):
                env=self.make()
                env.receive.safety,env.receive.robot_mode=safety,robot_mode
                with self.assertRaises(RobotRecoveryRequired):
                    env.recover(threading.Event())
                self.assertEqual(env.control.calls,[])
                self.assertEqual(env.gripper.calls,[])
                self.assertTrue(env.recovery_required)

    def test_ik_failure_preserves_target_and_requires_an_explicit_recover(self):
        env=self.make()
        env.control.ik=None
        with self.assertRaisesRegex(RobotRecoveryRequired,"inverse kinematics") as raised:
            env.apply(Prediction(np.array([.001,0,0,0,0,0,.2])))
        self.assertIn("target_tcp_rotvec",raised.exception.status)
        self.assertFalse(env.armed.is_set())
        self.assertEqual(env.control.calls,["IK"])
        with self.assertRaises(RobotRecoveryRequired):
            env.begin_episode()
        report=env.recover(threading.Event())
        self.assertTrue(report["healthy"])
        self.assertTrue(report["stop_confirmed"])
        self.assertIsNone(env.commanded)
        self.assertEqual(env.steps,0)
        self.assertFalse(env.armed.is_set())
        self.assertFalse(env.recovery_required)
        self.assertEqual(env.gripper.calls,[])
        self.assertEqual(env.cameras["primary"].stop_calls,0)
        json.dumps(report)
        json.dumps(env.metadata)

    def test_servo_rejection_is_typed_and_latched(self):
        env=self.make()
        env.control.servo_result=False
        with self.assertRaisesRegex(RobotRecoveryRequired,"servoJ rejected"):
            env._servo([0.0]*6)
        self.assertFalse(env.armed.is_set())
        self.assertTrue(env.recovery_required)
        self.assertEqual(env.control.calls,["servoJ"])

    def test_stopped_script_has_clear_then_idle_upload_then_acknowledged_stop(self):
        env=self.make()
        env.control.program=False
        report=env.recover(threading.Event())
        self.assertTrue(report["script_restored"])
        self.assertEqual(env.control.calls,["servoStop","reuploadScript","servoStop"])
        self.assertEqual(env.control.command_register,0)
        self.assertEqual(env.gripper.calls,[])
        self.assertFalse(env.armed.is_set())

    def test_existing_running_script_is_not_reuploaded(self):
        env=self.make()
        report=env.recover(threading.Event())
        self.assertFalse(report["script_restored"])
        self.assertEqual(env.control.calls,["servoStop"])
        self.assertEqual(env.gripper.calls,[])

    def test_unaudited_version_does_not_upload_a_stopped_script(self):
        env=self.make()
        env.control.program=False
        env._sdk_version="unreviewed"
        with self.assertRaisesRegex(RobotRecoveryRequired,"audited"):
            env.recover(threading.Event())
        self.assertEqual(env.control.calls,[])

    def test_clear_exception_never_permits_an_upload(self):
        env=self.make()
        env.control.program=False
        env.control.stop_hook=lambda:(_ for _ in ()).throw(ConnectionError("mock connection loss"))
        with self.assertRaisesRegex(RobotRecoveryRequired,"recovery failed"):
            env.recover(threading.Event())
        self.assertEqual(env.control.calls,["servoStop"])
        self.assertTrue(env.recovery_required)

    def test_failed_upload_does_not_report_ready(self):
        env=self.make()
        env.control.program=False
        env.control.upload_result=False
        with self.assertRaisesRegex(RobotRecoveryRequired,"upload was not acknowledged"):
            env.recover(threading.Event())
        self.assertTrue(env.recovery_required)
        self.assertFalse(env.armed.is_set())
        self.assertEqual(env.control.calls,["servoStop","reuploadScript"])

    def test_stale_normal_feedback_and_disconnected_channels_issue_no_sdk_commands(self):
        for fault in ("stale","receive_disconnected","control_disconnected"):
            with self.subTest(fault=fault):
                env=self.make(timeout=.15,freshness=.025)
                if fault=="stale":
                    env.receive.advance=False
                elif fault=="receive_disconnected":
                    env.receive.connected=False
                else:
                    env.control.connected=False
                with self.assertRaises(RobotRecoveryRequired):
                    env.recover(threading.Event())
                self.assertEqual(env.control.calls,[])
                self.assertEqual(env.gripper.calls,[])

    def test_cancel_before_recovery_never_sends_a_command(self):
        env=self.make()
        cancel=threading.Event()
        cancel.set()
        with self.assertRaises(InterruptedError):
            env.recover(cancel)
        self.assertEqual(env.control.calls,[])
        self.assertTrue(env.recovery_required)
        self.assertFalse(env.armed.is_set())

    def test_cancelled_native_stop_cannot_late_upload_or_rearm(self):
        env=self.make()
        env.control.program=False
        entered,release,wait=self.block()
        env.control.stop_hook=wait
        cancel=threading.Event()
        errors=[]
        def recover():
            try:
                env.recover(cancel)
            except BaseException as error:
                errors.append(error)
        caller=threading.Thread(target=recover)
        caller.start()
        self.assertTrue(entered.wait(1))
        cancel.set()
        caller.join(1)
        self.assertFalse(caller.is_alive())
        self.assertIsInstance(errors[0],InterruptedError)
        self.assertTrue(env.recovery_pending)
        self.assertFalse(env.stop())
        with self.assertRaisesRegex(RobotRecoveryRequired,"pending"):
            env.recover(threading.Event())
        with self.assertRaises(RobotRecoveryRequired):
            env.begin_episode()
        self.assertEqual(env.control.calls,["servoStop"])
        release.set()
        env._recovery_thread.join(1)
        self.assertFalse(env.recovery_pending)
        self.assertTrue(env.recovery_required)
        self.assertFalse(env.armed.is_set())
        self.assertEqual(env.control.calls,["servoStop"])
        self.assertEqual(env.gripper.calls,[])
        self.assertEqual(env.cameras["primary"].stop_calls,0)

    def test_timed_out_native_upload_is_fenced_until_it_finishes_and_never_auto_ready(self):
        env=self.make(timeout=.04,freshness=.02)
        env.control.program=False
        entered,release,wait=self.block()
        env.control.upload_hook=wait
        with self.assertRaisesRegex(RobotRecoveryRequired,"timed out"):
            env.recover(threading.Event())
        self.assertTrue(entered.is_set())
        self.assertTrue(env.recovery_pending)
        self.assertFalse(env.stop())
        with self.assertRaisesRegex(RobotRecoveryRequired,"pending"):
            env.recover(threading.Event())
        with self.assertRaises(RobotRecoveryRequired):
            env.apply(Prediction(np.zeros(7)))
        self.assertEqual(env.control.calls,["servoStop","reuploadScript"])
        release.set()
        env._recovery_thread.join(1)
        self.assertFalse(env.recovery_pending)
        self.assertTrue(env.recovery_required)
        self.assertFalse(env.armed.is_set())
        # The late upload is idle; no late follow-up servoStop, home or servoJ.
        self.assertEqual(env.control.calls,["servoStop","reuploadScript"])
        self.assertEqual(env.control.command_register,0)
        env.control.upload_hook=None
        report=env.recover(threading.Event())
        self.assertTrue(report["healthy"])
        self.assertFalse(env.recovery_required)
        self.assertFalse(env.armed.is_set())

    def test_pending_native_operation_is_not_disconnected_by_adapter_close(self):
        env=self.make(timeout=.04,freshness=.02)
        env.control.program=False
        entered,release,wait=self.block()
        env.control.stop_hook=wait
        with self.assertRaises(RobotRecoveryRequired):
            env.recover(threading.Event())
        with self.assertRaisesRegex(RobotRecoveryRequired,"cleanup is incomplete"):
            env.close()
        self.assertEqual(env.control.disconnects,0)
        self.assertEqual(env.receive.disconnects,0)
        release.set()
        env._recovery_thread.join(1)
        self.assertFalse(env.recovery_pending)
        self.assertTrue(env.recovery_required)

    def test_connect_disables_automatic_upload_even_during_sdk_internal_reconnect(self):
        import gnaroshi_deployment.adapters.ur5e as driver
        env=self.make()
        receive,gripper=env.receive,env.gripper
        made=[]
        class SafeControl(Control):
            FLAG_NO_WAIT=16
            FLAG_UPLOAD_SCRIPT=1
            def __init__(self,host,frequency=-1.0,flags=1):
                super().__init__()
                self.host,self.frequency,self.flags=host,frequency,flags
                self.program=False
                self.implicit_uploads=0
                self.dashboard_stops=0
                made.append(self)
            def native_reconnect(self):
                # SDK1.6.3 reconnect's script upload branch is flag-controlled.
                if self.flags&self.FLAG_UPLOAD_SCRIPT:
                    self.implicit_uploads+=1
                    self.dashboard_stops+=int(self.program)
                    self.program=True
        env.connect_preview=lambda:None
        modules={"rtde_control":SimpleNamespace(RTDEControlInterface=SafeControl),
                 "rtde_receive":SimpleNamespace(RTDEReceiveInterface=lambda host:receive),
                 "scipy.spatial.transform":SimpleNamespace(Rotation=env.Rotation)}
        with patch.dict("sys.modules",modules),patch.object(driver,"Gripper",lambda host:gripper), \
             patch.object(driver.importlib.metadata,"version",return_value="1.6.3"):
            env.connect()
        control=made[0]
        self.assertEqual((control.frequency,control.flags),(-1.0,16))
        self.assertTrue(env._control_no_upload)
        self.assertFalse(env.metadata["control_connection_flags"]["upload_script"])
        self.assertEqual(control.calls,[])
        def internal_reconnect_once():
            control.stop_hook=None
            control.native_reconnect()
        control.stop_hook=internal_reconnect_once
        report=env.recover(threading.Event())
        self.assertTrue(report["healthy"])
        self.assertEqual(control.implicit_uploads,0)
        self.assertEqual(control.dashboard_stops,0)
        self.assertEqual(control.calls,["servoStop","reuploadScript","servoStop"])

    def test_upload_enabled_legacy_connection_cannot_run_recovery(self):
        env=self.make()
        env._control_no_upload=False
        with self.assertRaisesRegex(RobotRecoveryRequired,"no-upload"):
            env.recover(threading.Event())
        self.assertEqual(env.control.calls,[])
        self.assertTrue(env.recovery_required)

    def test_pending_includes_final_validation_after_native_worker_finishes(self):
        env=self.make()
        entered,release,wait=self.block()
        original=env._read_status
        def status():
            if threading.current_thread().name=="recover-caller":
                wait()
            return original()
        env._read_status=status
        cancel=threading.Event()
        errors=[]
        def recover():
            try:
                env.recover(cancel)
            except BaseException as error:
                errors.append(error)
        caller=threading.Thread(target=recover,name="recover-caller")
        caller.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(env._recovery_thread.is_alive())
        self.assertTrue(env.recovery_pending)
        self.assertFalse(env.stop())
        with self.assertRaisesRegex(RobotRecoveryRequired,"pending"):
            env.recover(threading.Event())
        cancel.set()
        release.set()
        caller.join(1)
        self.assertFalse(caller.is_alive())
        self.assertIsInstance(errors[0],InterruptedError)
        self.assertTrue(env.recovery_required)
        self.assertFalse(env.recovery_pending)
        self.assertFalse(env.armed.is_set())

    def test_final_command_lock_uses_a_deadline_and_cannot_mark_ready_when_busy(self):
        env=self.make()
        class FinalLock:
            def __init__(self):
                self.calls=[]
            def acquire(self,*,timeout):
                self.calls.append(timeout)
                return len(self.calls)==1
            def release(self):
                pass
        lock=FinalLock()
        env.command_lock=lock
        with self.assertRaisesRegex(RobotRecoveryRequired,"final recovery validation"):
            env.recover(threading.Event())
        self.assertEqual(len(lock.calls),2)
        self.assertTrue(all(0<=timeout<=env.config.recovery_timeout_s for timeout in lock.calls))
        self.assertTrue(env.recovery_required)
        self.assertFalse(env.armed.is_set())

    def test_session_exit_waits_for_native_completion_before_sdk_cleanup(self):
        from gnaroshi_deployment.adapters.mock import MockPolicy
        from gnaroshi_deployment.contracts import Observation,SessionConfig
        from gnaroshi_deployment.session import DeploymentSession
        env=self.make(timeout=.04,freshness=.02)
        env.connect_preview=lambda:None
        env.connect=lambda:None
        env.home=lambda cancel:None
        images={role:np.zeros((2,2,3),np.uint8) for role in ("primary","wrist")}
        env._images=lambda:(images,{})
        env.observe_preview=lambda:Observation(images,np.empty(0,np.float32),time.time_ns())
        class KeptPolicy(MockPolicy):
            closed=0
            def close(self):
                self.closed+=1
        policy=KeptPolicy(.001)
        entered,release,wait=self.block()
        def until(predicate):
            deadline=time.monotonic()+2
            while not predicate():
                if time.monotonic()>deadline:
                    self.fail("Session recovery/cleanup transition timed out")
                time.sleep(.005)
        with tempfile.TemporaryDirectory() as directory:
            config=SessionConfig(output_root=directory,record_video=False,target_hz=100)
            session=DeploymentSession(config,policy,env,Path(__file__).parents[1])
            try:
                session.start()
                until(lambda:session.state=="READY")
                env.control.ik=None
                session.start_rollout()
                until(lambda:session.state=="RECOVERY_REQUIRED" and session.can_recover)
                env.control.stop_hook=wait
                self.assertTrue(session.recover_robot())
                self.assertTrue(entered.wait(1))
                session.close()
                until(lambda:env.recovery_pending)
                time.sleep(.06)
                self.assertTrue(session.worker.is_alive())
                self.assertFalse(session.shutdown_complete)
                self.assertEqual(env.control.disconnects,0)
                self.assertEqual(env.receive.disconnects,0)
                self.assertEqual(env.cameras["primary"].stop_calls,0)
                self.assertEqual(policy.closed,0)
            finally:
                release.set()
                session.close()
                session.worker.join(3)
            self.assertTrue(session.shutdown_complete)
            self.assertEqual(policy.closed,1)
            self.assertFalse(env.recovery_pending)


if __name__=="__main__":
    unittest.main()
