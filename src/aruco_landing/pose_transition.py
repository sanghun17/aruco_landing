"""Source selection only: consume the physical estimator's already aligned pose."""
from collections import deque
import numpy as np
from aruco_landing.pose_alignment import matrix_pose, quaternion_distance_deg


class PoseTransition:
    def __init__(self, allow=False, max_age=.2, pair_dt=.035, max_position=.15,
                 max_angle=12., stable_duration=1., min_samples=30, fallback_timeout=None, reject_inconsistent_marker=False, latch_fallback=False):
        self.allow, self.max_age, self.pair_dt = allow, max_age, pair_dt
        self.max_position, self.max_angle = max_position, max_angle
        self.stable_duration, self.min_samples = stable_duration, min_samples
        self.fallback_timeout = fallback_timeout
        self.reject_inconsistent_marker = reject_inconsistent_marker
        self.latch_fallback = latch_fallback
        self.fallback_latched = False
        self.enabled = True
        self.last_valid_marker_receipt = None
        self.source = 'optitrack'
        self.samples = {}
        self.mocap = deque(maxlen=300)
        self.quality = {}
        self.good_since = None
        self.good_count = 0
        self.last_output_stamp = -float('inf')
        self.last_agreement = None

    def set_quality(self, key, value, now):
        self.quality[key] = (value, now)
        if not value:
            self.good_since, self.good_count = None, 0

    def quality_ok(self, now):
        return (self.quality.get('aligned', (False, 0))[0]
                and all(self.quality.get(k, (False, 0))[0]
                        and 0 <= now-self.quality[k][1] <= self.max_age
                        for k in ('visible', 'inliers')))

    def fresh(self, sample, now):
        return sample is not None and -.02 <= now-sample[0] <= self.max_age and 0 <= now-sample[2] <= self.max_age

    def ingest(self, source, stamp, transform, now):
        sample = (stamp, transform, now)
        if not np.isfinite(transform).all() or not self.fresh(sample, now):
            return False
        previous = self.samples.get(source)
        if previous is not None and stamp <= previous[0]:
            return False
        self.samples[source] = sample
        if source == 'optitrack':
            self.mocap.append(sample)
        elif source == 'marker':
            agreement = self.agreement(now)
            if self.quality_ok(now) and (not self.reject_inconsistent_marker or agreement['consistent']):
                self.last_valid_marker_receipt = now
            if agreement['consistent']:
                # A gap must restart the qualification interval.
                if previous is None or stamp-previous[0] > self.max_age:
                    self.good_since, self.good_count = None, 0
                if self.good_since is None:
                    self.good_since = stamp
                self.good_count += 1
            else:
                self.good_since, self.good_count = None, 0
        if source != self.source or stamp <= self.last_output_stamp:
            return False
        if source == 'marker' and (not self.enabled or self.fallback_latched or not self.quality_ok(now) or
                (self.reject_inconsistent_marker and not agreement['consistent'])):
            return False
        self.last_output_stamp = stamp
        return True

    def agreement(self, now):
        marker, mocap = self.samples.get('marker'), self.samples.get('optitrack')
        result = dict(consistent=False, translation_jump_m=None, rotation_jump_deg=None, pair_dt_s=None)
        if not self.quality_ok(now) or not self.fresh(marker, now) or not self.fresh(mocap, now) or not self.mocap:
            return result
        pair = min(self.mocap, key=lambda p: abs(p[0]-marker[0]))
        result['pair_dt_s'] = abs(pair[0]-marker[0])
        if result['pair_dt_s'] > self.pair_dt:
            return result
        result['translation_jump_m'] = float(np.linalg.norm(pair[1][:3,3]-marker[1][:3,3]))
        result['rotation_jump_deg'] = quaternion_distance_deg(matrix_pose(pair[1])[1], matrix_pose(marker[1])[1])
        result['consistent'] = (result['translation_jump_m'] <= self.max_position
                                and result['rotation_jump_deg'] <= self.max_angle)
        self.last_agreement = result
        return result

    def check(self, now):
        result = self.agreement(now)
        marker = self.samples.get('marker')
        duration = marker[0]-self.good_since if marker is not None and self.good_since is not None else 0.
        result.update(allowed=self.allow, alignment_ready=self.quality.get('aligned',(False,0))[0],
                      marker_fresh=self.fresh(marker,now), mocap_fresh=self.fresh(self.samples.get('optitrack'),now),
                      quality_valid=self.quality_ok(now), stable_samples=self.good_count, stable_duration_s=duration)
        result['ready'] = bool(self.allow and self.enabled and not self.fallback_latched and result['consistent'] and self.good_count >= self.min_samples and duration >= self.stable_duration)
        return result

    def fallback(self, now):
        """A timed, explicit fallback to a fresh, trusted global OptiTrack source."""
        if self.fallback_timeout is None or self.source != 'marker':
            return None
        if self.last_valid_marker_receipt is None:
            return None
        age = now-self.last_valid_marker_receipt
        if age < self.fallback_timeout or not self.fresh(self.samples.get('optitrack'),now):
            return None
        self.source = 'optitrack'
        self.fallback_latched = self.latch_fallback
        self.good_since, self.good_count = None, 0
        return {'reason':'marker_timeout', 'marker_absence_s':age,
                'fallback_timeout_s':self.fallback_timeout}

    def set_enabled(self, enabled):
        if enabled != self.enabled or not enabled:
            self.good_since, self.good_count = None, 0
        self.enabled = enabled

    def return_to_mocap(self, now, latch=False):
        if latch:
            self.fallback_latched = True
        if not self.fresh(self.samples.get('optitrack'), now):
            return False
        self.source = 'optitrack'
        self.good_since, self.good_count = None, 0
        return True

    def select(self, marker, now):
        if marker:
            check = self.check(now)
            if not check['ready']:
                return False, check
            self.source = 'marker'
            return True, check
        if not self.fresh(self.samples.get('optitrack'), now):
            return False, {'reason':'optitrack_stale'}
        if self.source == 'marker' and not self.agreement(now)['consistent']:
            return False, {'reason':'return_to_optitrack_requires_fresh_consistent_overlap'}
        self.source = 'optitrack'
        self.good_since, self.good_count = None, 0
        return True, {'source':self.source}
