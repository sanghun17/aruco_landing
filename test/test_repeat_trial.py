import math

import pytest

from aruco_landing.repeat_trial import RepeatTrial


def cycle():
    return RepeatTrial(dict(height_m=2., speed_mps=.5, vertical_speed_mps=.5,
        position_tolerance_m=.1, velocity_tolerance_mps=.1, settle_s=1.,
        stage_timeout_s=30., arm_timeout_s=5., ground_max_z_m=.5,
        random_min_radius_m=.75, extra_geofence_margin_m=.3, random_seed=73),
        dict(enabled_axes=['x','y'], margin_m=.4, box=dict(x=[-2.5,2.5],y=[-2.5,2.5])))


def step(c, now, **extra):
    values=dict(armed=True, airborne=True, offboard=True, healthy=True,
                position=c.goal, speed=0.)
    values.update(extra)
    return c.step(now, **values)


def test_takeoff_center_random_then_existing_approach_and_new_random_target():
    c=cycle(); c.begin(0.,(.03,.04,.2))
    assert c.phase=='ARMING'
    step(c,.1,armed=False,airborne=False);assert c.phase=='ARMING'
    step(c,.2);assert c.phase=='TAKEOFF' and c.goal==(.03,.04,2.)
    for t, expected in [(1.,'TAKEOFF'),(2.01,'CENTER'),(3.,'CENTER'),
                        (4.01,'RANDOM_POSITION'),(5.,'RANDOM_POSITION'),(6.01,'APPROACH')]:
        step(c,t);assert c.phase==expected
    first=c.random_goal
    assert all(-1.8<=v<=1.8 for v in first[:2]) and math.hypot(*first[:2])>=.75
    c.begin(8.,(.02,.01,.2));assert c.random_goal!=first and c.number==2


def test_motion_resets_convergence_dwell():
    c=cycle();c.begin(0.,(0.,0.,.2));step(c,.1)
    step(c,1.);step(c,1.9,speed=.2);step(c,2.01)
    assert c.phase=='TAKEOFF'
    step(c,3.02);assert c.phase=='CENTER'


@pytest.mark.parametrize('extra,phase', [({'healthy':False},'FAILED_HOLD'),
    ({'offboard':False},'CANCELLED'),({'armed':False},'FAILED_HOLD')])
def test_failure_and_takeover_do_not_automatically_resume(extra,phase):
    c=cycle();c.begin(0.,(0.,0.,.2));step(c,.1);step(c,.2,**extra)
    assert c.phase==phase
    step(c,.3);assert c.phase==phase


def test_arm_timeout_and_rejected_ground_start():
    c=cycle();c.begin(0.,(0.,0.,.2));step(c,5.1,armed=False)
    assert c.phase=='FAILED_HOLD'
    with pytest.raises(ValueError):c.begin(6.,(0.,0.,1.))
    with pytest.raises(ValueError):c.begin(6.,(2.4,0.,.2))
