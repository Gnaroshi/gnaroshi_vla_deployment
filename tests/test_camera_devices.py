"""Camera identity, stream transactions and stale input checks with fake SDKs."""
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from gnaroshi_deployment.adapters.mock import MockEmbodiment
from gnaroshi_deployment.adapters.ur5e import UR5eConfig, UR5eEmbodiment
from gnaroshi_deployment.contracts import CameraError, RobotRecoveryRequired


class SDK:
    def __init__(self):
        self.started=[]
        self.pipelines=[]
        self.fail_serial=None
        self.block_serial=None
        self.release=threading.Event()
        self.entered=threading.Event()
        self.devices=['device-a','device-b','device-c']
        self.context_calls=0
        self.query_block=False
        self.rs=SimpleNamespace(context=self.context,pipeline=self.pipeline,config=self.config,
            camera_info=SimpleNamespace(serial_number='serial',name='name',firmware_version='firmware',usb_type_descriptor='usb'),
            stream=SimpleNamespace(color='color'),format=SimpleNamespace(rgb8='rgb8'))

    def context(self):
        self.context_calls+=1
        class Profile:
            def stream_type(self):return 'color'
            def format(self):return 'rgb8'
            def fps(self):return 30
            def width(self):return 2
            def height(self):return 2
            def as_video_stream_profile(self):return self
        def device(serial):
            return SimpleNamespace(get_info=lambda key:{'serial':serial,'name':'Mock RGB camera','firmware':'test','usb':'mock'}[key],
                query_sensors=lambda:[SimpleNamespace(get_stream_profiles=lambda:[Profile(),Profile()])])
        def query_devices():
            if self.query_block:
                self.entered.set()
                self.release.wait(2)
            return [device(serial) for serial in reversed(self.devices)]
        return SimpleNamespace(query_devices=query_devices)

    @staticmethod
    def config():
        class Config:
            serial=None
            def enable_device(self,serial):self.serial=serial
            def enable_stream(self,*args):self.profile=args
        return Config()

    def pipeline(self):
        sdk=self
        class Pipeline:
            serial=None
            stopped=0
            number=0
            freeze=False
            pixels=None
            timestamp=None
            domain='mock'
            def start(self,config):
                self.serial=config.serial
                sdk.started.append(self.serial)
                if self.serial==sdk.block_serial:
                    sdk.entered.set()
                    sdk.release.wait(2)
                if self.serial==sdk.fail_serial:
                    raise RuntimeError('mock USB stream failure')
            def stop(self):self.stopped+=1
            def wait_for_frames(self,timeout_ms):
                if not self.freeze:self.number+=1
                number=self.number
                pixels=self.pixels if self.pixels is not None else np.full((2,2,3),sdk.devices.index(self.serial),dtype=np.uint8)
                return SimpleNamespace(get_color_frame=lambda:SimpleNamespace(
                    get_frame_number=lambda:number,
                    get_timestamp=lambda:self.timestamp if self.timestamp is not None else time.time()*1000,
                    get_frame_timestamp_domain=lambda:self.domain,
                    get_data=lambda:pixels))
        instance=Pipeline();self.pipelines.append(instance);return instance


def embodiment(**changes):
    cfg=dict(robot_ip='mock-control',camera_serials={'primary':'device-a','wrist':'device-b'},home_joints=(0.,)*6,
             width=2,height=2,camera_wait_timeout_ms=20,camera_connect_timeout_s=.15)
    cfg.update(changes)
    return UR5eEmbodiment(UR5eConfig(**cfg))


class CameraDevices(unittest.TestCase):
    def setUp(self):
        self.sdk=SDK()
        self.modules=patch.dict('sys.modules',{'pyrealsense2':self.sdk.rs})
        self.modules.start();self.addCleanup(self.modules.stop)
        self.env=embodiment()
        self.addCleanup(self.sdk.release.set)
        self.addCleanup(self.close)

    def close(self):
        self.sdk.release.set()
        deadline=time.monotonic()+1
        while self.env.camera_operation_pending and time.monotonic()<deadline:time.sleep(.005)
        self.env.close()

    def test_explicit_discovery_lists_deduplicated_profiles_without_stream_or_control(self):
        report=self.env.list_camera_devices()
        self.assertEqual([item['serial'] for item in report['devices']],['device-a','device-b','device-c'])
        self.assertEqual(len(report['devices'][0]['color_profiles']),1)
        self.assertIn('device-a',report['devices'][0]['label'])
        self.assertEqual(self.sdk.started,[])
        self.assertIsNone(self.env.control)
        self.assertIsNone(self.env.receive)
        self.assertIsNone(self.env.gripper)
        self.assertEqual(report['selected'],self.env.config.camera_serials)

    def test_role_change_preserves_control_gripper_and_state_and_uses_requested_camera(self):
        self.env.connect_preview()
        old=list(self.env.cameras.values())
        sentinel=object()
        self.env.control=self.env.receive=self.env.gripper=sentinel
        try:
            self.env.commanded=np.eye(4)
            self.env.gripper_status='close'
            report=self.env.configure_cameras({'primary':'device-c','wrist':'device-a'},threading.Event())
            self.assertEqual(report['selected'],{'primary':'device-c','wrist':'device-a'})
            self.assertEqual(report['camera_metadata']['primary']['serial'],'device-c')
            self.assertTrue(all(p.stopped==1 for p in old))
            self.assertIs(self.env.control,sentinel)
            self.assertIs(self.env.gripper,sentinel)
            self.assertEqual(self.env.gripper_status,'close')
            np.testing.assert_array_equal(self.env.commanded,np.eye(4))
            self.assertFalse(self.env.armed.is_set())
        finally:self.env.control=self.env.receive=self.env.gripper=None

    def test_invalid_selection_is_rejected_before_closing_existing_streams(self):
        self.env.connect_preview();old=dict(self.env.cameras)
        for mapping in ({'primary':'device-a','wrist':'device-a'},{'primary':'device-c'},
                        {'primary':'device-c','wrist':'missing-device'}):
            with self.subTest(mapping=mapping),self.assertRaises((ValueError,CameraError)):
                self.env.configure_cameras(mapping,threading.Event())
            self.assertEqual(self.env.cameras,old)
            self.assertTrue(all(p.stopped==0 for p in old.values()))

    def test_stream_failure_closes_partial_open_and_restores_old_mapping(self):
        self.env.connect_preview();self.sdk.fail_serial='device-c'
        with self.assertRaises(CameraError) as caught:
            self.env.configure_cameras({'primary':'device-a','wrist':'device-c'},threading.Event())
        self.assertTrue(caught.exception.status['rollback_restored'])
        self.assertEqual(self.env.config.camera_serials,{'primary':'device-a','wrist':'device-b'})
        self.assertEqual(set(self.env.cameras),{'primary','wrist'})
        failed=[p for p in self.sdk.pipelines if p.serial=='device-c'][0]
        self.assertEqual(failed.stopped,1)
        self.assertNotIsInstance(caught.exception,RobotRecoveryRequired)

    def test_cancellation_never_changes_mapping_and_active_arm_rejects_changes(self):
        cancel=threading.Event();cancel.set()
        with self.assertRaises(InterruptedError):
            self.env.configure_cameras({'primary':'device-c','wrist':'device-a'},cancel)
        self.env.armed.set()
        with self.assertRaises(ValueError):
            self.env.configure_cameras({'primary':'device-c','wrist':'device-a'},threading.Event())
        self.env.armed.clear()
        self.assertEqual(self.sdk.context_calls,0)
        self.assertEqual(self.sdk.started,[])

    def test_duplicate_frames_timeout_and_never_reuse_stale_image(self):
        self.env.connect_preview();self.env._images()
        self.env.cameras['primary'].freeze=True
        started=time.monotonic()
        with self.assertRaises(CameraError) as caught:self.env._images()
        self.assertLess(time.monotonic()-started,.2)
        self.assertEqual(caught.exception.category,'camera_frame_timeout')
        self.assertEqual(caught.exception.status['role'],'primary')

    def test_wrong_rgb_profile_and_old_global_timestamp_are_rejected(self):
        self.env.connect_preview()
        camera=self.env.cameras['primary'];camera.pixels=np.zeros((2,2),np.uint8)
        with self.assertRaises(CameraError) as caught:self.env._images()
        self.assertEqual(caught.exception.category,'camera_profile_mismatch')
        camera.pixels=None;camera.domain='timestamp_domain.global_time';camera.timestamp=(time.time()-5)*1000
        with self.assertRaises(CameraError) as caught:self.env._images()
        self.assertEqual(caught.exception.category,'camera_stale_frame')

    def test_hardware_clock_is_not_compared_with_host_epoch(self):
        self.env.connect_preview()
        for camera in self.env.cameras.values():camera.domain='timestamp_domain.hardware_clock';camera.timestamp=10.
        _,metadata=self.env._images()
        self.assertIsNone(metadata['primary']['frame_age_s'])

    def test_timed_out_native_start_closes_its_late_pipelines_and_blocks_overlapping_open(self):
        self.sdk.block_serial='device-a';self.env.config.camera_connect_timeout_s=.03
        with self.assertRaises(CameraError) as caught:self.env.connect_preview()
        self.assertEqual(caught.exception.category,'camera_connect_timeout')
        self.assertTrue(self.env.camera_operation_pending)
        self.assertEqual(self.env.cameras,{})
        self.assertEqual(self.sdk.pipelines[0].stopped,0)
        with self.assertRaises(CameraError) as caught:self.env.connect_preview()
        self.assertEqual(caught.exception.category,'camera_operation_pending')
        self.sdk.release.set()
        deadline=time.monotonic()+1
        while self.env.camera_operation_pending and time.monotonic()<deadline:time.sleep(.005)
        self.assertFalse(self.env.camera_operation_pending)
        self.assertEqual(self.sdk.pipelines[0].stopped,1)
        self.assertEqual(self.env.cameras,{})
        self.sdk.block_serial=None;self.env.connect_preview()
        self.assertEqual(set(self.env.cameras),{'primary','wrist'})

    def test_discovery_timeout_blocks_overlap_until_late_scan_finishes_without_opening_streams(self):
        self.sdk.query_block=True;self.env.config.camera_scan_timeout_s=.03
        started=time.monotonic()
        with self.assertRaises(CameraError) as caught:self.env.list_camera_devices()
        self.assertEqual(caught.exception.category,'camera_scan_timeout')
        self.assertLess(time.monotonic()-started,.2)
        self.assertTrue(self.env.camera_operation_pending)
        self.assertEqual(self.sdk.started,[])
        with self.assertRaises(CameraError) as caught:self.env.connect_preview()
        self.assertEqual(caught.exception.category,'camera_operation_pending')
        with self.assertRaises(CameraError):self.env.list_camera_devices()
        self.sdk.release.set()
        deadline=time.monotonic()+1
        while self.env.camera_operation_pending and time.monotonic()<deadline:time.sleep(.005)
        self.assertFalse(self.env.camera_operation_pending)
        self.assertEqual(self.sdk.started,[])
        self.sdk.query_block=False
        self.assertEqual(len(self.env.list_camera_devices()['devices']),3)

    def test_partial_start_failure_does_not_stop_or_close_robot(self):
        self.sdk.fail_serial='device-b'
        self.env.stop=lambda:(_ for _ in ()).throw(AssertionError('camera failure touched robot stop'))
        with self.assertRaises(CameraError):self.env.connect_preview()
        self.assertEqual(self.env.cameras,{})
        self.assertEqual([p.stopped for p in self.sdk.pipelines],[1,1])
        self.env.stop=lambda:True


class MockCameraDevices(unittest.TestCase):
    def test_switch_is_instance_local_and_supports_arbitrary_camera_roles(self):
        first=MockEmbodiment(camera_serials={'front':'mock-primary','hand':'mock-wrist'})
        second=MockEmbodiment()
        state=first.state.copy();first.gripper='held'
        report=first.configure_cameras({'front':'mock-extra','hand':'mock-primary'},threading.Event())
        self.assertEqual(report['camera_metadata']['front']['serial'],'mock-extra')
        self.assertEqual(second.metadata['camera_serials'],{'primary':'mock-primary','wrist':'mock-wrist'})
        self.assertEqual(set(first.observe().images),{'front','hand'})
        self.assertEqual(first.home_count,0)
        self.assertEqual(first.gripper,'held')
        np.testing.assert_array_equal(first.state,state)

    def test_mock_requires_idle_valid_mapping_and_cancellable_change(self):
        env=MockEmbodiment();env.begin_episode()
        with self.assertRaises(ValueError):env.configure_cameras({'primary':'mock-extra','wrist':'mock-wrist'},threading.Event())
        env.stop();cancel=threading.Event();cancel.set()
        with self.assertRaises(InterruptedError):env.configure_cameras({'primary':'mock-extra','wrist':'mock-wrist'},cancel)
        with self.assertRaises(ValueError):env.configure_cameras({'primary':'mock-extra','wrist':'mock-extra'},threading.Event())


if __name__=='__main__':unittest.main()
