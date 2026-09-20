"""ArUco trial lifecycle. No ROS, arming, mode changes, or estimator side effects."""
from dataclasses import dataclass


@dataclass
class Trial:
    marker_confirm_s: float = 1.
    marker_loss_s: float = .5
    contact_height_m: float = .2
    prestream_s: float = 1.
    phase: str = 'IDLE'
    reason: str = ''
    good_since: object = None
    last_good: object = None
    prepared_at: object = None
    finish_mode: str = 'auto_land'

    def __post_init__(self):
        import math
        if self.finish_mode not in ('auto_land','force_disarm'):raise ValueError('invalid finish mode')
        if not all(math.isfinite(x) and x>0 for x in
                   (self.marker_confirm_s,self.marker_loss_s,self.contact_height_m,self.prestream_s)):
            raise ValueError('trial thresholds must be finite and positive')

    def prepare(self, now):
        if self.phase not in ('IDLE','COMPLETE','CANCELLED'):
            return False
        self.phase='PRESTREAM';self.prepared_at=now
        self.reason='waiting_for_pilot_offboard';self.good_since=self.last_good=None
        return True

    def cancel(self, reason='pilot_cancel'):
        self.phase='CANCELLED';self.reason=reason;self.good_since=None

    def fail(self, reason):
        self.phase='FAILED_HOLD';self.reason=reason;self.good_since=None

    def step(self, now, *, offboard, armed, healthy, marker_good, marker_height=None,
             landed=False, auto_land=False, marker_time=None, mocap_height=None, estimation_ready=True):
        old=self.phase
        if old in ('IDLE','COMPLETE','CANCELLED','FAILED_HOLD'):
            return self.phase
        if old=='CUT_WAIT':
            if not armed:self.phase='COMPLETE';self.reason='force_disarm_confirmed'
            return self.phase
        if old=='AUTO_LAND' and landed and not armed:
            self.phase='COMPLETE';self.reason='px4_ground_contact_and_disarmed';return self.phase
        if not armed:
            self.cancel('disarmed_before_landing_confirmation');return self.phase
        if not healthy:
            self.fail('required_input_or_estimation_route_invalid');return self.phase
        if old=='PRESTREAM':
            if offboard and now-self.prepared_at>=self.prestream_s:
                self.phase='APPROACH';self.reason='hold_entry_altitude_and_approach'
            return self.phase
        if not offboard and not (old=='AUTO_LAND' and auto_land):
            self.cancel('pilot_or_safety_left_offboard');return self.phase
        if marker_good:
            stamp=now if marker_time is None else marker_time
            if self.last_good is None or stamp-self.last_good>.2:self.good_since=None
            if self.good_since is None:self.good_since=stamp
            self.last_good=stamp
        else:self.good_since=None
        if old=='APPROACH':
            if estimation_ready and self.good_since is not None and self.last_good-self.good_since>=self.marker_confirm_s:
                self.phase='DESCEND';self.reason='marker_qualified'
        elif old in ('DESCEND','AUTO_LAND'):
            if not landed and (self.last_good is None or now-self.last_good>=self.marker_loss_s):
                self.fail('marker_lost_trial_failed')
            elif old=='DESCEND' and marker_good and marker_height is not None and 0<=marker_height<=self.contact_height_m:
                if self.finish_mode=='auto_land':
                    self.phase='AUTO_LAND';self.reason='marker_height_reached_handoff_to_px4'
                elif mocap_height is not None and 0<=mocap_height<=.25:
                    self.phase='CUT_WAIT';self.reason='dual_height_force_disarm_requested'
        return self.phase


def limit_horizontal_velocity(velocity, limit=.5):
    """Cap the final command's XY norm after all coordinate transforms."""
    import math
    x,y,z=map(float,velocity)
    if not all(math.isfinite(v) for v in (x,y,z,limit)) or limit<=0:
        raise ValueError('finite velocity and positive limit required')
    scale=min(1.,limit/max(math.hypot(x,y),1e-12))
    return x*scale,y*scale,z
