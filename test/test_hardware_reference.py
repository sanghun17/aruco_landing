import unittest
import numpy as np
from aruco_landing.hardware_reference import HardwareReference, HardwareReferenceTracker
from aruco_landing.landing_trial import Trial


class HardwareReferenceTest(unittest.TestCase):
    def make(self, **overrides):
        c=np.zeros((1,3,6));c[0,2,:2]=[2.,-.55]
        ref=HardwareReference(dict(schema_version=1,frame_id='landing_pad',reference_point='camera_center',
            provenance={'bank_sha256':'a'*64},coefficients=c.tolist(),durations_s=[1.8/.55]))
        args=dict(kp=1.,kd=.15,kp_z=2.,speed_limit=2.,vertical_limit=1.6,
                  derivative_tau=.1,derivative_limit=2.,max_pose_age=.3,prediction_limit=.2)
        args.update(overrides)
        tracker=HardwareReferenceTracker(ref,**args);tracker.start(10.)
        return tracker

    def command(self,tr,t,height,age=0.,**kwargs):
        args=dict(now=10+t,position=[0.,0.,height],stamp=10+t-age,age=age,frame_id='landing_pad',visible=True)
        args.update(kwargs)
        return tr.command(**args)

    def test_gt_resume_preserves_elapsed_reference_and_clears_cross_source_derivative(self):
        tr=self.make();self.command(tr,1.,1.45)
        tr.derivative=(1.,-1.)
        tr.resume_from_last_command(25.)
        self.assertEqual(tr.started,24.)
        self.assertEqual(tr.derivative,(0.,0.));self.assertIsNone(tr.previous_stamp)
        np.testing.assert_allclose(tr.command(25.,[0,0,1.45],25.,0.,'landing_pad',True),[0,0,-.55])
        self.assertEqual(tr.state,'tracking')

    def test_gt_resume_unstarted_and_terminal_references(self):
        tr=self.make();tr.reset();tr.resume_from_last_command(20.)
        self.assertEqual(tr.state,'waiting');self.assertIsNone(tr.started)
        tr.start(20.)
        self.assertIsNone(tr.command(25.,[0,0,.4],25.,0.,'landing_pad',True))
        tr.resume_from_last_command(30.)
        self.assertEqual(tr.state,'terminal_landing')
        self.assertIsNone(tr.command(30.,[0,0,.4],30.,0.,'landing_pad',True))

    def test_vertical_error_sign_and_user_gain(self):
        tr=self.make()
        self.assertAlmostEqual(self.command(tr,1.,1.55)[2],-.75)
        self.assertAlmostEqual(self.command(tr,1.1,2-.55*1.1-.1)[2],-.35)

    def test_downward_and_upward_saturation(self):
        self.assertAlmostEqual(self.command(self.make(),1.,9.)[2],-1.6)
        self.assertAlmostEqual(self.command(self.make(),1.,-9.)[2],1.6)

    def test_capture_timestamp_alignment(self):
        # A delayed measurement exactly on the reference must not cause a
        # spurious vertical correction based on current reference height.
        tr=self.make()
        self.assertAlmostEqual(self.command(tr,1.,2-.55*.9,age=.1)[2],-.55)

    def test_reference_end_hands_off_instead_of_zero_or_completion(self):
        tr=self.make();T=tr.reference.duration
        self.assertIsNotNone(self.command(tr,T-.001,.4))
        self.assertIsNone(self.command(tr,T,.4))
        self.assertEqual(tr.state,'terminal_landing')
        self.assertIsNone(self.command(tr,T+1.,.3))
        self.assertEqual(tr.state,'terminal_landing')

    def test_invalid_input_does_not_handoff(self):
        tr=self.make();T=tr.reference.duration
        np.testing.assert_array_equal(self.command(tr,T,.4,age=.4),np.zeros(3))
        self.assertEqual(tr.state,'failed_measurement')
        tr=self.make()
        np.testing.assert_array_equal(self.command(tr,T,.4,visible=False),np.zeros(3))
        self.assertEqual(tr.state,'tracking')

    def test_clock_reversal_and_reset(self):
        tr=self.make();self.command(tr,1.,1.45)
        np.testing.assert_array_equal(self.command(tr,.9,1.5),np.zeros(3))
        self.assertEqual(tr.state,'failed_clock')
        tr.reset();self.assertEqual(tr.state,'waiting');tr.start(20.)
        np.testing.assert_allclose(tr.command(20.,[0,0,2],20.,0.,'landing_pad',True),[0,0,-.55])

    def test_correction_authority_required(self):
        with self.assertRaises(ValueError):self.make(vertical_limit=.55)
        with self.assertRaises(ValueError):self.make(kp_z=0.)

    def test_existing_dual_height_finish_owns_completion(self):
        tr=self.make();T=tr.reference.duration
        self.assertIsNone(self.command(tr,T,.4))
        trial=Trial(finish_mode='force_disarm',phase='DESCEND')
        args=dict(offboard=True,armed=True,healthy=True,marker_good=True)
        self.assertEqual(trial.step(T,marker_height=.4,mocap_height=.4,**args),'DESCEND')
        self.assertEqual(trial.step(T+.01,marker_height=.19,mocap_height=.3,**args),'DESCEND')
        self.assertEqual(trial.step(T+.02,marker_height=.19,mocap_height=.24,**args),'CUT_WAIT')
        self.assertEqual(trial.step(T+.03,offboard=True,armed=False,healthy=True,marker_good=True),'COMPLETE')

    def test_existing_marker_loss_after_reference_end(self):
        tr=self.make();T=tr.reference.duration
        self.assertIsNone(self.command(tr,T,.4))
        trial=Trial(finish_mode='force_disarm',phase='DESCEND',last_good=T)
        self.assertEqual(trial.step(T+.6,offboard=True,armed=True,healthy=True,marker_good=False),'FAILED_HOLD')

    def test_real_controller_update_uses_legacy_command_after_end(self):
        import importlib.util
        import sys
        import threading
        from pathlib import Path
        from types import SimpleNamespace as NS
        from unittest.mock import MagicMock, patch
        def twist():
            return NS(header=NS(stamp=None,frame_id=''),twist=NS(
                linear=NS(x=0.,y=0.,z=0.),angular=NS(x=0.,y=0.,z=0.)))
        ros=MagicMock();geometry=MagicMock();geometry.TwistStamped=twist
        modules={'rospy':ros,'geometry_msgs':MagicMock(),'geometry_msgs.msg':geometry,
                 'std_msgs':MagicMock(),'std_msgs.msg':MagicMock(),
                 'std_srvs':MagicMock(),'std_srvs.srv':MagicMock()}
        script=Path(__file__).resolve().parents[1]/'scripts/landing_controller_node.py'
        with patch.dict(sys.modules,modules):
            spec=importlib.util.spec_from_file_location('controller_under_test',script)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            controller=module.LandingController.__new__(module.LandingController)
            controller.__dict__.update(lock=threading.Lock(),state=controller.DESCENDING,
                visible=True,error_derivative=(0.,0.),enabled=True,preview_only=False,
                loss_timeout=.3,latency_compensation_enabled=False,kp=1.,kd=.15,speed_limit=2.,
                touchdown_height_reference='camera',h_min=0.,descent_speed=.3,yaw_enabled=False,
                reference_tracker=self.make(),trajectory_set=None)
            for key in ['yaw_error','command','yaw','active','abort','touchdown','state']:
                setattr(controller,key+'_publisher',MagicMock())
            controller.yaw_reference=0.
            for now,height,expected in [(11.,1.55,-.75),(14.,.4,-.3),(15.,.3,-.3),(15.1,0.,0.)]:
                ros.get_time.return_value=now
                stamp=NS(to_sec=lambda:now)
                pose=NS(header=NS(stamp=stamp,frame_id='landing_pad'),pose=NS(pose=NS(position=NS(x=0.,y=0.,z=height))))
                controller.pose=controller.camera_pose=controller.vehicle_pose=pose
                controller.last_valid_receipt=controller.previous_pose_stamp=now
                controller.update(NS(current_real=stamp))
                command=controller.command_publisher.publish.call_args[0][0]
                self.assertAlmostEqual(command.twist.linear.z,expected)
                self.assertEqual(controller.state,controller.TOUCHDOWN if height==0. else controller.DESCENDING)

if __name__=='__main__':unittest.main()
