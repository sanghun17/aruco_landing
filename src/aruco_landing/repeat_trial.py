"""Ground-start experiment stages; no ROS services or actuator side effects."""
import math
import random


class RepeatTrial:
    PHASES = ('ARMING', 'TAKEOFF', 'CENTER', 'RANDOM_POSITION')

    def __init__(self, config, geofence):
        self.height = float(config['height_m'])
        self.speed = float(config['speed_mps'])
        self.vertical_speed = float(config['vertical_speed_mps'])
        self.tolerance = float(config['position_tolerance_m'])
        self.velocity_tolerance = float(config['velocity_tolerance_mps'])
        self.dwell = float(config['settle_s'])
        self.timeout = float(config['stage_timeout_s'])
        self.arm_timeout = float(config['arm_timeout_s'])
        self.ground_max = float(config['ground_max_z_m'])
        self.min_radius = float(config['random_min_radius_m'])
        self.extra_margin = float(config['extra_geofence_margin_m'])
        values = (self.height, self.speed, self.vertical_speed, self.tolerance,
                  self.velocity_tolerance, self.dwell, self.timeout,
                  self.arm_timeout, self.ground_max, self.min_radius, self.extra_margin)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('repeat trial values must be finite and positive')
        if self.height <= self.ground_max or self.speed > .5:
            raise ValueError('repeat takeoff height / transfer speed invalid')
        if not {'x', 'y'}.issubset(geofence['enabled_axes']):
            raise ValueError('repeat trial requires an XY geofence')
        margin = float(geofence['margin_m']) + self.extra_margin
        self.bounds = tuple((float(geofence['box'][axis][0]) + margin,
                             float(geofence['box'][axis][1]) - margin)
                            for axis in ('x', 'y'))
        if not math.isfinite(margin) or any(not (math.isfinite(lo) and math.isfinite(hi) and lo < 0 < hi)
                                          for lo, hi in self.bounds):
            raise ValueError('repeat bounds must contain the origin')
        if max(math.hypot(x, y) for x in self.bounds[0] for y in self.bounds[1]) <= self.min_radius:
            raise ValueError('random minimum radius leaves no valid targets')
        self.seed = config.get('random_seed')
        self.rng = random.Random(self.seed)
        self.number = 0
        self.phase = None
        self.goal = None
        self.random_goal = None
        self.settled_since = None
        self.started = None
        self.origin = None

    def begin(self, now, position):
        x, y, z = map(float, position)
        if not all(math.isfinite(v) for v in (x, y, z)) or not 0 <= z <= self.ground_max:
            raise ValueError('ground start requires body Z within configured ground range')
        if not all(lo <= v <= hi for v, (lo, hi) in zip((x, y), self.bounds)):
            raise ValueError('ground start outside inset experiment box')
        for _ in range(10000):
            target = [self.rng.uniform(*b) for b in self.bounds]
            if math.hypot(*target) >= self.min_radius:
                break
        else:
            raise ValueError('unable to sample random target')
        self.number += 1
        self.origin = (x, y, z)
        self.random_goal = tuple(target) + (self.height,)
        self.phase = 'ARMING'
        self.goal = self.origin
        self.started = now
        self.settled_since = None

    def step(self, now, *, armed, airborne, offboard, healthy, position, speed):
        if self.phase not in self.PHASES:
            return self.phase, None
        if not offboard:
            self.phase = 'CANCELLED'
            return self.phase, 'pilot_left_autonomous_mode'
        if not healthy:
            self.phase = 'FAILED_HOLD'
            return self.phase, 'repeat_required_input_invalid'
        if now - self.started > (self.arm_timeout if self.phase == 'ARMING' else self.timeout):
            reason = self.phase.lower() + '_timeout'
            self.phase = 'FAILED_HOLD'
            return self.phase, reason
        if self.phase == 'ARMING':
            if armed:
                self.phase = 'TAKEOFF'
                self.goal = self.origin[:2] + (self.height,)
                self.started = now
            return self.phase, None
        if not armed:
            self.phase = 'FAILED_HOLD'
            return self.phase, 'unexpected_disarm_during_repeat'
        distance = math.sqrt(sum((float(p)-g)**2 for p, g in zip(position, self.goal)))
        settled = airborne and math.isfinite(speed) and speed <= self.velocity_tolerance and distance <= self.tolerance
        if not settled:
            self.settled_since = None
        elif self.settled_since is None:
            self.settled_since = now
        elif now - self.settled_since >= self.dwell:
            if self.phase == 'TAKEOFF':
                self.phase, self.goal = 'CENTER', (0., 0., self.height)
            elif self.phase == 'CENTER':
                self.phase, self.goal = 'RANDOM_POSITION', self.random_goal
            else:
                self.phase, self.goal = 'APPROACH', (0., 0., self.height)
            self.settled_since = None
            self.started = now
        return self.phase, None

    def status(self):
        return dict(trial=self.number, phase=self.phase, goal_global_m=self.goal,
                    random_goal_global_m=self.random_goal, height_m=self.height,
                    bounds_xy_m=self.bounds, random_seed=self.seed)
