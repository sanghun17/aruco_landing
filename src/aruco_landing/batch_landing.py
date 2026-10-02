"""ROS-free landing state and reproducible independent trial inputs.

All timestamps are supplied by the simulator. Image processing wall time never
changes controller dt or measurement age. Ground truth is not an input here.
"""
from collections import deque
import math
import numpy as np
from aruco_landing.landing_math import filtered_derivative, horizontal_feedback, predicted_height
from aruco_landing.yaw_control import yaw_feedback


def initial_condition(seed, trial_id, bounds):
    rng = np.random.default_rng(np.random.SeedSequence([seed, trial_id]))
    return dict(zip(('x', 'y', 'z', 'yaw_deg'),
                    [float(rng.uniform(*bounds[k])) for k in ('x', 'y', 'z', 'yaw_deg')]))


class LandingPolicy:
    def __init__(self, config):
        self.cfg = config
        self.queue = deque()
        self.pose = None
        self.stamp = None
        self.previous_error = None
        self.derivative = (0., 0.)
        self.state = 'waiting'
        self.visible_count = 0
        self.visible = False

    def submit(self, capture_time, pose, latency):
        if latency < 0 or capture_time < 0:
            raise ValueError('negative simulation timestamp/latency')
        self.queue.append((capture_time+latency, capture_time, pose))

    def command(self, now):
        cfg = self.cfg
        while self.queue and self.queue[0][0] <= now + 1e-9:
            _, stamp, pose = self.queue.popleft()
            self.visible = pose is not None
            # A failed detection cannot refresh the last valid pose's age.
            if pose is not None:
                error = tuple(-float(p) for p in pose[:2, 3])
                dt = 0. if self.stamp is None else stamp-self.stamp
                self.derivative = filtered_derivative(error, self.previous_error,
                    self.derivative, dt, cfg['derivative_filter_tau_s'], cfg['derivative_speed_limit_m_s'])
                self.previous_error, self.pose, self.stamp = error, pose, stamp
                self.visible_count += 1
                if self.state == 'waiting':
                    self.state = 'descending'
        cmd = np.zeros(4, np.float32)  # pad/world velocity xyz, yaw rate
        if self.state in ('touchdown', 'aborted'):
            return cmd
        if self.stamp is None:
            if now >= cfg['acquisition_timeout_s']:
                self.state = 'aborted'
            return cmd
        age = max(0., now-self.stamp)
        if age > cfg['loss_timeout_s']:
            self.state = 'aborted'
            return cmd
        if not self.visible:
            return cmd  # match ROS behavior: withhold motion during marker loss
        prediction = min(age, cfg['max_pose_prediction_s']) if cfg['latency_compensation_enabled'] else 0.
        lead = cfg['touchdown_stop_lead_s'] if cfg['latency_compensation_enabled'] else 0.
        if predicted_height(self.pose[2, 3], prediction, cfg['descent_speed_m_s'], lead) <= cfg['h_min_m']:
            self.state = 'touchdown'
            return cmd
        cmd[:2] = horizontal_feedback(self.pose[:2, 3], self.derivative, prediction,
                                     cfg['kp_xy'], cfg['kd_xy'], cfg['speed_limit_m_s'])
        cmd[2] = -cfg['descent_speed_m_s']
        # Only image-derived orientation is used for the upper-level yaw policy.
        R = self.pose[:3, :3]
        yaw = math.atan2(R[1, 0], R[0, 0])
        cmd[3] = yaw_feedback([0.,0.,math.sin(yaw/2),math.cos(yaw/2)],
            0., cfg['yaw_kp'], cfg['yaw_rate_limit_rad_s'],
            math.radians(cfg['yaw_deadband_deg']))[1]
        return cmd
