import math
import unittest
from aruco_landing.yaw_control import yaw_feedback


def q(degrees):
    a=math.radians(degrees)/2
    return [0,0,math.sin(a),math.cos(a)]


class YawFeedbackTest(unittest.TestCase):
    def test_direction_and_saturation(self):
        for start in [-179,-30,30,179]:
            error,rate=yaw_feedback(q(start))
            self.assertLess(rate*start,0)
            self.assertLessEqual(abs(rate),.35)

    def test_wrap_chooses_short_path(self):
        error,rate=yaw_feedback(q(179),target=math.radians(-179))
        self.assertAlmostEqual(math.degrees(error),2)
        self.assertGreater(rate,0)

    def test_deadband(self):
        self.assertEqual(yaw_feedback(q(.5))[1],0)

    def test_convergence_of_heading_integrator(self):
        for initial in [-179,-30,30,179]:
            yaw=math.radians(initial)
            previous=abs(yaw)
            for _ in range(1200):
                _,rate=yaw_feedback(q(math.degrees(yaw)))
                yaw+=rate/60
                self.assertLessEqual(abs(yaw),previous+1e-10)
                previous=abs(yaw)
            self.assertLessEqual(abs(yaw),math.radians(1.01))

    def test_invalid_quaternion(self):
        with self.assertRaises(ValueError):yaw_feedback([0,0,0,0])
        with self.assertRaises(ValueError):yaw_feedback([float('nan'),0,0,1])
