"""Simulation-time isolation and numerical compatibility with the ROS policy."""
import math
import unittest
import numpy as np
from aruco_landing.batch_landing import LandingPolicy, initial_condition
from aruco_landing.landing_math import filtered_derivative, horizontal_feedback


CFG=dict(kp_xy=.8,kd_xy=.15,speed_limit_m_s=2.,descent_speed_m_s=.5,h_min_m=.2,
    derivative_filter_tau_s=.1,derivative_speed_limit_m_s=2.,acquisition_timeout_s=2.,
    loss_timeout_s=.5,latency_compensation_enabled=True,max_pose_prediction_s=.2,
    touchdown_stop_lead_s=.08,yaw_kp=1.,yaw_rate_limit_rad_s=.35,yaw_deadband_deg=1.)


class BatchLandingTest(unittest.TestCase):
    def test_shared_math_matches_previous_ros_equations(self):
        rng=np.random.default_rng(42)
        derivative=(0.,0.)
        for _ in range(100):
            error,previous,position=rng.normal(size=(3,2))
            dt=float(rng.uniform(.001,.24))
            alpha=dt/(.1+dt)
            expected=(1-alpha)*np.array(derivative)+alpha*(error-previous)/dt
            expected*=min(1.,2./np.linalg.norm(expected))
            derivative=filtered_derivative(error,previous,derivative,dt,.1,2.)
            np.testing.assert_allclose(derivative,expected,atol=1e-14)
            expected=.8*(-position+np.array(derivative)*.1)+.15*np.array(derivative)
            expected*=min(1.,2./np.linalg.norm(expected))
            np.testing.assert_allclose(horizontal_feedback(position,derivative,.1,.8,.15,2.),expected,atol=1e-14)
        self.assertEqual(filtered_derivative((1,1),(0,0),(1,2),.3,.1,2.),(1,2))

    def test_delay_and_loss_use_capture_time(self):
        policy=LandingPolicy(CFG)
        pose=np.eye(4);pose[:3,3]=[.2,-.1,1.]
        policy.submit(0.,pose,.1)
        self.assertEqual(policy.command(.05).tolist(),[0.,0.,0.,0.])
        self.assertEqual(policy.state,'waiting')
        command=policy.command(.1)
        np.testing.assert_allclose(command[:3],[-.16,.08,-.5],atol=1e-7)
        policy.submit(.2,None,0.)
        self.assertEqual(policy.command(.2).tolist(),[0.,0.,0.,0.])
        self.assertEqual(policy.state,'descending')
        policy.submit(.4,None,.1)
        policy.command(.51)
        self.assertEqual(policy.state,'aborted')

    def test_zero_timestamp_is_valid_and_reset_has_no_pose_history(self):
        pose=np.eye(4);pose[2,3]=.23
        policy=LandingPolicy(CFG);policy.submit(0.,pose,0.)
        self.assertEqual(policy.command(0.).tolist(),[0.,0.,0.,0.])
        self.assertEqual(policy.state,'touchdown')
        fresh=LandingPolicy(CFG)
        fresh.command(.01)
        self.assertEqual(fresh.state,'waiting')
        self.assertIsNone(fresh.stamp)

    def test_inputs_independent_of_batch_order(self):
        bounds=dict(x=(-.3,.3),y=(-.3,.3),z=(1.,2.),yaw_deg=(-25.,25.))
        ordered=[initial_condition(1701,i,bounds) for i in range(25)]
        for i in [24,0,7,3,15]: self.assertEqual(ordered[i],initial_condition(1701,i,bounds))
        self.assertNotEqual(ordered[0],ordered[1])


if __name__=='__main__': unittest.main()
