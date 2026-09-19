import unittest
from aruco_landing.landing_trial import Trial, limit_horizontal_velocity


class TrialTest(unittest.TestCase):
    def setUp(self):self.t=Trial()
    def step(self,now,**kw):
        d=dict(offboard=True,armed=True,healthy=True,marker_good=False)
        d.update(kw);return self.t.step(now,**d)
    def descend(self):
        self.assertTrue(self.t.prepare(0));self.step(1)
        for i in range(62):self.step(1.02+i/60,marker_good=True,marker_time=1.02+i/60,marker_height=1.)
        self.assertEqual(self.t.phase,'DESCEND')
    def test_requires_explicit_start_and_pilot_offboard(self):
        self.step(1);self.assertEqual(self.t.phase,'IDLE');self.t.prepare(2)
        self.step(3,offboard=False);self.assertEqual(self.t.phase,'PRESTREAM')
        self.step(3.1);self.assertEqual(self.t.phase,'APPROACH')
    def test_qualification_resets_and_counts_measurement_time(self):
        self.t.prepare(0);self.step(1)
        for i in range(30):self.step(1.1+i/60,marker_good=True,marker_time=1.1+i/60)
        self.step(1.7);self.assertEqual(self.t.phase,'APPROACH')
        for i in range(50):self.step(1.8+i/60,marker_good=True,marker_time=1.8)
        self.assertEqual(self.t.phase,'APPROACH')
    def test_loss_timeout_uses_last_observation_not_freshness_expiry(self):
        self.descend();last=self.t.last_good
        self.step(last+.19,marker_good=True,marker_time=last)
        self.step(last+.49);self.assertEqual(self.t.phase,'DESCEND')
        self.step(last+.501);self.assertEqual(self.t.phase,'FAILED_HOLD')
        self.step(last+2,marker_good=True,marker_height=.1);self.assertEqual(self.t.phase,'FAILED_HOLD')
    def test_height_handoff_is_not_motor_kill_or_success(self):
        self.descend();t=self.t.last_good+.01
        self.step(t,marker_good=True,marker_height=.19);self.assertEqual(self.t.phase,'AUTO_LAND')
        self.step(t+.1,offboard=False,auto_land=True,landed=True);self.assertEqual(self.t.phase,'AUTO_LAND')
        self.step(t+.2,offboard=False,auto_land=True,landed=True,armed=False);self.assertEqual(self.t.phase,'COMPLETE')
    def test_auto_land_airborne_loss_fails(self):
        self.descend();t=self.t.last_good+.01;self.step(t,marker_good=True,marker_height=.19)
        self.step(t+.51,offboard=False,auto_land=True);self.assertEqual(self.t.phase,'FAILED_HOLD')
    def test_takeover_and_input_failure_latch(self):
        self.descend();self.step(3,offboard=False);self.assertEqual(self.t.phase,'CANCELLED')
        self.step(4);self.assertEqual(self.t.phase,'CANCELLED')
        self.t=Trial();self.t.prepare(0);self.step(1);self.step(1.1,healthy=False);self.assertEqual(self.t.phase,'FAILED_HOLD')

class VelocityLimitTest(unittest.TestCase):
    def test_final_diagonal_command_caps_norm_preserves_z_and_direction(self):
        x,y,z=limit_horizontal_velocity((.6,.8,-.3))
        self.assertAlmostEqual((x*x+y*y)**.5,.5)
        self.assertAlmostEqual(x/y,.75);self.assertEqual(z,-.3)
    def test_transformed_descent_component_is_also_limited(self):
        # Tilt can project descent into XY after the pad-frame cap.
        x,y,z=limit_horizontal_velocity((.58,0.,-.2))
        self.assertEqual((x,y,z),(.5,0.,-.2))
        self.assertEqual(limit_horizontal_velocity((.1,-.2,0.)),(.1,-.2,0.))
    def test_nonfinite_command_rejected(self):
        with self.assertRaises(ValueError):limit_horizontal_velocity((float('nan'),0.,0.))

if __name__=='__main__':unittest.main()

class ForceDisarmPolicyTest(unittest.TestCase):
    def trial(self):return Trial(phase='DESCEND',last_good=10.,finish_mode='force_disarm')
    def step(self,t,**kw):
        args=dict(offboard=True,armed=True,healthy=True,marker_good=True,marker_height=.19,mocap_height=.24,marker_time=10.1)
        args.update(kw);return t.step(10.1,**args)
    def test_both_heights_required(self):
        for extra in [dict(marker_height=.201),dict(mocap_height=.251),dict(mocap_height=None),dict(mocap_height=float('nan')),dict(marker_good=False)]:
            t=self.trial();self.assertEqual(self.step(t,**extra),'DESCEND')
        t=self.trial();self.assertEqual(self.step(t),'CUT_WAIT')
    def test_no_cut_outside_armed_healthy_offboard_descent(self):
        for extra in [dict(armed=False),dict(offboard=False),dict(healthy=False)]:
            self.assertNotEqual(self.step(self.trial(),**extra),'CUT_WAIT')
        t=Trial(phase='APPROACH',finish_mode='force_disarm');self.assertEqual(self.step(t),'APPROACH')
    def test_only_confirmed_disarm_completes(self):
        t=self.trial();self.step(t)
        self.assertEqual(self.step(t,landed=True),'CUT_WAIT')
        self.assertEqual(self.step(t,armed=False,healthy=False),'COMPLETE')
