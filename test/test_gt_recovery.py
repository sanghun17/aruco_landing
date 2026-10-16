"""Failed visual trials recover separately from successful marker landings."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from aruco_landing.landing_trial import Trial


def recovery(**kw):
    t=Trial(finish_mode='force_disarm');t.fail('trajectory_initial_alignment_timeout')
    args=dict(enabled=True,offboard=True,armed=True,healthy=True);args.update(kw)
    return t,t.recover_failed(**args)


@pytest.mark.parametrize('key',['enabled','offboard','armed','healthy'])
def test_recovery_admission_requires_all_conditions(key):
    t,ok=recovery(**{key:False});assert not ok and t.phase=='FAILED_HOLD'


def test_unrelated_failures_do_not_autoland():
    for reason in ['required_input_or_estimation_route_invalid','pad_alignment_changed','force_disarm_unconfirmed']:
        t=Trial(finish_mode='force_disarm');t.fail(reason)
        assert not t.recover_failed(enabled=True,offboard=True,armed=True,healthy=True)


def test_gt_reference_landing_uses_same_height_cut_without_px4_mode_change():
    t,ok=recovery();assert ok
    args=dict(offboard=True,armed=True,healthy=True,marker_good=False,estimation_ready=False)
    assert t.step(1,**args)=='GT_RECOVERY'
    args['estimation_ready']=True
    assert t.step(2,**args)=='GT_DESCEND'
    assert t.step(3,mocap_height=.21,**args)=='GT_DESCEND'
    assert t.step(4,mocap_height=.19,**args)=='GT_CUT_WAIT'
    args.update(armed=False,landed=False,healthy=False)
    assert t.step(5,**args)=='GT_CUT_WAIT'
    args['landed']=True
    assert t.step(6,**args)=='FAILED_COMPLETE'
    assert t.failure_reason=='trajectory_initial_alignment_timeout'


def test_recovery_never_accepts_px4_autoland_or_auto_land_finish():
    t,_=recovery()
    assert t.step(1,offboard=False,auto_land=True,armed=True,healthy=True,marker_good=False)=='CANCELLED'
    t=Trial();t.fail('trajectory_initial_alignment_timeout')
    assert not t.recover_failed(enabled=True,offboard=True,armed=True,healthy=True)


@pytest.mark.parametrize('kwargs,phase',[
    ({'offboard':False},'CANCELLED'),({'healthy':False},'FAILED_HOLD'),
    ({'armed':False},'CANCELLED')])
def test_recovery_respects_takeover_and_input_loss(kwargs,phase):
    t,_=recovery();args=dict(offboard=True,armed=True,healthy=True,marker_good=False);args.update(kwargs)
    assert t.step(1,**args)==phase
    assert t.failure_reason=='trajectory_initial_alignment_timeout'


def test_node_requires_fresh_verified_gt_route_and_dwell():
    # Compile the actual ROS node gate independently of installed ROS messages.
    tree=ast.parse((Path(__file__).parents[1]/'scripts/landing_trial_node.py').read_text())
    cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='LandingTrial')
    method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='recovery_gt_ready')
    module=ast.Module(body=[method],type_ignores=[]);ns={};exec(compile(ast.fix_missing_locations(module),'gate','exec'),ns)
    node=SimpleNamespace(health=lambda now:True,fresh=lambda *a:True,
        router_status=dict(source='optitrack',last_output_source='optitrack'),recovery_gt_since=1.)
    gate=ns['recovery_gt_ready'];assert gate(node,1.5) and not gate(node,1.49)
    for key in ['source','last_output_source']:
        node.router_status[key]='marker';assert not gate(node,2);node.router_status[key]='optitrack'
    node.fresh=lambda *a:False;assert not gate(node,2)
    node.fresh=lambda *a:True;node.health=lambda now:False;assert not gate(node,2)
