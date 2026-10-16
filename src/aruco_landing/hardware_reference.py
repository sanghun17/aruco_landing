"""Exact camera-centre quintic references and ROS-independent hardware tracking.

No arming, mode switching, coordinate conversion or motor commands live here.
Reference time is supplied by the trial's monotonic clock; measurement age comes
from the ROS timestamp. A complete reference is not a ground-contact event.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from aruco_landing.landing_math import filtered_derivative, horizontal_feedback, limit_xy


class HardwareReference:
    def __init__(self, document):
        if document.get('schema_version') != 1 or document.get('frame_id') != 'landing_pad':
            raise ValueError('expected version-1 landing_pad reference')
        if document.get('reference_point') != 'camera_center':
            raise ValueError('reference point must be camera_center')
        self.provenance = document['provenance']
        if not isinstance(self.provenance.get('bank_sha256'), str) or len(self.provenance['bank_sha256']) != 64:
            raise ValueError('bank SHA-256 provenance required')
        self.coefficients = np.asarray(document['coefficients'], dtype=float)
        self.durations = np.asarray(document['durations_s'], dtype=float)
        n = len(self.durations)
        if n < 1 or self.durations.shape != (n,) or self.coefficients.shape != (n, 3, 6):
            raise ValueError('expected ascending local-time quintics [piece,3,6]')
        if not np.isfinite(self.coefficients).all() or not np.isfinite(self.durations).all() or np.any(self.durations <= 0):
            raise ValueError('finite coefficients and positive durations required')
        self.knots = np.r_[0., np.cumsum(self.durations)]
        self.duration = float(self.knots[-1])
        if not math.isfinite(self.duration):
            raise ValueError('invalid duration')
        for i in range(n-1):
            for order in range(3):
                if not np.allclose(self._piece(i, self.durations[i], order), self._piece(i+1, 0., order), atol=1e-7, rtol=0):
                    raise ValueError('reference must be position/velocity/acceleration continuous')
        self.sha256 = None

    @classmethod
    def load(cls, path, expected_sha256):
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError('explicit reference SHA-256 required')
        raw = Path(path).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected_sha256:
            raise ValueError('hardware reference SHA-256 mismatch')
        result = cls(json.loads(raw))
        result.sha256 = digest
        return result

    def _piece(self, piece, t, order=0):
        return np.array([np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(c, order))
                         for c in self.coefficients[piece]])

    def sample(self, t):
        if not math.isfinite(t) or t < 0 or t > self.duration + 1e-9:
            raise ValueError('reference time outside trajectory')
        t = min(t, self.duration)
        i = min(int(np.searchsorted(self.knots, t, side='right')-1), len(self.durations)-1)
        return self._piece(i, t-self.knots[i]), self._piece(i, t-self.knots[i], 1)

    def validate_limits(self, horizontal_speed, vertical_speed):
        """Check polynomial extrema, not just sampled trajectory points."""
        if not all(math.isfinite(x) and x > 0 for x in (horizontal_speed, vertical_speed)):
            raise ValueError('positive reference speed limits required')
        for c, duration in zip(self.coefficients, self.durations):
            v = [np.polynomial.polynomial.polyder(x) for x in c]
            xy2 = np.polynomial.polynomial.polyadd(
                np.polynomial.polynomial.polymul(v[0], v[0]),
                np.polynomial.polynomial.polymul(v[1], v[1]))
            for polynomial, limit in ((xy2, horizontal_speed**2),
                                      (np.polynomial.polynomial.polymul(v[2], v[2]), vertical_speed**2)):
                roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(polynomial))
                times = [0., duration] + [float(z.real) for z in roots if abs(z.imag) < 1e-8 and 0 < z.real < duration]
                if max(np.polynomial.polynomial.polyval(t, polynomial) for t in times) > limit + 1e-9:
                    raise ValueError('reference exceeds configured hardware speed limit')


class HardwareReferenceTracker:
    def __init__(self, reference, *, kp, kd, kp_z, speed_limit, vertical_limit,
                 derivative_tau, derivative_limit, max_pose_age, prediction_limit):
        self.reference = reference
        values = (kp, kd, kp_z, speed_limit, vertical_limit, derivative_tau,
                  derivative_limit, max_pose_age, prediction_limit)
        if not all(math.isfinite(v) for v in values) or min(kp, kd, derivative_tau, prediction_limit) < 0:
            raise ValueError('invalid tracking parameters')
        if min(speed_limit, vertical_limit, derivative_limit, max_pose_age) <= 0:
            raise ValueError('positive tracking limits required')
        if kp_z <= 0:
            raise ValueError('positive vertical position gain required')
        reference.validate_limits(speed_limit, vertical_limit)
        self.kp, self.kd, self.speed_limit = kp, kd, speed_limit
        self.kp_z, self.vertical_limit = kp_z, vertical_limit
        # Reserve downward correction authority beyond the reference speed.
        for c, duration in zip(reference.coefficients, reference.durations):
            v = np.polynomial.polynomial.polyder(c[2])
            roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(v))
            times = [0., duration] + [float(z.real) for z in roots if abs(z.imag) < 1e-8 and 0 < z.real < duration]
            if max(-np.polynomial.polynomial.polyval(t, v) for t in times) >= vertical_limit - 1e-9:
                raise ValueError('vertical limit must exceed reference descent speed for lag correction')
        self.derivative_tau, self.derivative_limit = derivative_tau, derivative_limit
        self.max_pose_age, self.prediction_limit = max_pose_age, prediction_limit
        self.reset()

    def reset(self):
        self.started = None
        self.last_now = None
        self.previous_stamp = None
        self.previous_error = None
        self.derivative = (0., 0.)
        self.state = 'waiting'

    def start(self, now):
        if not math.isfinite(now) or self.state != 'waiting':
            raise ValueError('reference may start once per explicitly reset trial')
        self.started = self.last_now = now
        self.state = 'tracking'

    def resume_from_last_command(self, now):
        """Preserve path progress while changing the feedback pose source.

        Time spent holding for GT admission does not advance the reference.
        Derivatives are relearned from GT samples instead of crossing sources.
        """
        if not math.isfinite(now):raise ValueError('finite resume time required')
        if self.started is None:
            if self.state!='waiting':raise ValueError('invalid unstarted tracker')
            return
        if self.last_now is None or now<self.last_now or self.state=='failed_clock':
            raise ValueError('cannot resume after clock reversal')
        elapsed=self.last_now-self.started
        self.started=now-elapsed;self.last_now=now
        self.previous_stamp=self.previous_error=None;self.derivative=(0.,0.)
        self.state='terminal_landing' if elapsed>=self.reference.duration else 'tracking'

    def command(self, now, position, stamp, age, frame_id, visible):
        """Return pad-frame velocity, or None to use the legacy landing command.

        None is a handoff, not a stop/disarm/completion command. The caller's
        existing height, marker-loss and flight-safety checks retain authority.
        """
        zero = np.zeros(3)
        if self.state not in ('tracking', 'terminal_landing'):
            return zero
        if not math.isfinite(now) or now < self.last_now:
            self.state = 'failed_clock'
            return zero
        self.last_now = now
        elapsed = now-self.started
        if not visible:
            return zero
        position = np.asarray(position, dtype=float)
        if (position.shape != (3,) or not np.isfinite(position).all() or frame_id != 'landing_pad'
                or not math.isfinite(age) or not 0 <= age <= self.max_pose_age or not math.isfinite(stamp)):
            self.state = 'failed_measurement'
            return zero
        # A pre-start sample cannot establish post-start tracking feedback.
        if elapsed-age < -1e-6:
            return zero
        if self.previous_stamp is not None and stamp < self.previous_stamp:
            self.state = 'failed_measurement'
            return zero
        if elapsed >= self.reference.duration:
            self.state = 'terminal_landing'
            self.previous_stamp = stamp
            return None
        reference_at_capture, _ = self.reference.sample(max(0., elapsed-age))
        error = reference_at_capture[:2]-position[:2]
        if self.previous_stamp is None or stamp > self.previous_stamp:
            dt = 0. if self.previous_stamp is None else stamp-self.previous_stamp
            self.derivative = filtered_derivative(error, self.previous_error, self.derivative,
                                                  dt, self.derivative_tau, self.derivative_limit)
            self.previous_stamp, self.previous_error = stamp, error
        elif stamp < self.previous_stamp:
            self.state = 'failed_measurement'
            return zero
        feedback = horizontal_feedback(-error, self.derivative, min(age, self.prediction_limit),
                                       self.kp, self.kd, self.speed_limit)
        _, feedforward = self.reference.sample(elapsed)
        xy = limit_xy(*(feedforward[:2]+feedback), self.speed_limit)
        # Compare reference and measurement at the same capture timestamp.
        # In the pad frame +z is up: a vehicle above reference gets more -vz.
        vz = feedforward[2] + self.kp_z*(reference_at_capture[2]-position[2])
        vz = float(np.clip(vz, -self.vertical_limit, self.vertical_limit))
        return np.array([xy[0], xy[1], vz])
