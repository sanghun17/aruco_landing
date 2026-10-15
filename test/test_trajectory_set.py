import hashlib
import json
import numpy as np
import pytest
from aruco_landing.trajectory_set import TrajectorySet,TrajectoryProgress
from test_repeat_trial import cycle,step


def fixture_set(tmp_path,selections=None):
    c=np.zeros((2,1,3,6));c[:,0,2,:2]=[2.,-.55];c[:,0,0,0]=[.2,-.2]
    c[:,0,0,1]=-c[:,0,0,0]/(1.8/.55)
    bank=tmp_path/'bank.npz'
    np.savez(bank,coefficients=c,durations=np.full((2,1),1.8/.55),candidate_ids=[10,20])
    doc=dict(schema_version=1,reference_id_convention='zero_based_bank_index',
             banks={'inside':dict(path='bank.npz',sha256=hashlib.sha256(bank.read_bytes()).hexdigest())},
             selections=selections if selections is not None else [dict(type='inside',reference_id=i) for i in [0,1]])
    path=tmp_path/'set.json';path.write_text(json.dumps(doc))
    return path


def test_number_identity_and_camera_mount_conversion(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));assert s.entries[1]['raw_candidate_id']==20
    pad=np.eye(4);pad[:3,3]=[.1,.2,.05]
    goal=s.body_start(0,pad,np.eye(3),[.03,0.,.1],[(-2,2),(-2,2)])
    np.testing.assert_allclose(goal,[.27,.2,1.95])
    with pytest.raises(ValueError):s.body_start(0,pad,np.eye(3),[0,0,0],[(-.1,.1),(-2,2)])


@pytest.mark.parametrize('selections',[[],[{'type':'inside','reference_id':2}],
    [{'type':'inside','reference_id':True}],[{'type':'inside','reference_id':0}]*2])
def test_invalid_selection_rejected(tmp_path,selections):
    with pytest.raises(ValueError):TrajectorySet(fixture_set(tmp_path,selections))


def test_hash_mismatch_rejected(tmp_path):
    p=fixture_set(tmp_path);doc=json.loads(p.read_text());doc['banks']['inside']['sha256']='0'*64;p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):TrajectorySet(p)


def test_once_each_and_restart_resume(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));path=tmp_path/'progress.json';q=TrajectoryProgress(s,path)
    q.begin();assert q.active==0
    with pytest.raises(ValueError):q.complete(grounded=False,disarmed=True)
    with pytest.raises(ValueError):TrajectoryProgress(s,path)
    q.complete(grounded=True,disarmed=True)
    q=TrajectoryProgress(s,path);assert q.index==1 and q.available
    q.begin();q.complete(grounded=True,disarmed=True)
    assert not q.available
    with pytest.raises(ValueError):q.begin()


def test_failed_flight_does_not_advance_or_resume(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));path=tmp_path/'progress.json';q=TrajectoryProgress(s,path)
    q.begin();q.fail();assert q.index==0 and not q.available
    with pytest.raises(ValueError):TrajectoryProgress(s,path)


def test_progress_write_failure_blocks_next_flight(tmp_path,monkeypatch):
    s=TrajectorySet(fixture_set(tmp_path));q=TrajectoryProgress(s,tmp_path/'progress.json');q.begin()
    def fail(*args):raise OSError('disk unavailable')
    monkeypatch.setattr('aruco_landing.trajectory_set.os.replace',fail)
    with pytest.raises(OSError):q.complete(grounded=True,disarmed=True)
    assert q.halted and not q.available
    with pytest.raises(ValueError):q.begin()


def test_selected_start_replaces_random_stage_and_stays_for_alignment():
    c=cycle();c.begin(0.,(0.,0.,.2),target=(.4,-.3,1.9));step(c,.1)
    assert c.goal==(0.,0.,1.9)
    step(c,1.);step(c,2.01);assert c.phase=='INITIAL_POSITION' and c.goal==(.4,-.3,1.9)
    step(c,3.);step(c,4.01);assert c.phase=='APPROACH' and c.goal==(.4,-.3,1.9)
    assert c.random_goal is None


def test_selected_start_invalid_before_arming():
    c=cycle()
    for goal in [(99.,0.,2.),(0.,0.,float('nan')),(0.,0.,.1)]:
        with pytest.raises(ValueError):c.begin(0.,(0.,0.,.2),target=goal)
    assert c.phase is None and c.number==0
