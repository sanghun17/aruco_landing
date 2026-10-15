import json
from pathlib import Path
import pytest
from aruco_landing.campaign import pad_identity,progress_path,recover_progress,acquire_campaign_lock
from aruco_landing.trajectory_set import TrajectorySet,TrajectoryProgress
from test_trajectory_set import fixture_set


def pad(name='proposed_pad_1_layout_physical_20260920_runtime.yaml',raw=b'metric layout'):
    return pad_identity(name,raw)


def test_perception_named_pad_and_custom_layouts_are_separate(tmp_path):
    a=pad();b=pad('paper_pad_layout_physical_20260920_runtime.yaml')
    assert a['key']=='proposed1' and b['key']=='baseline'
    assert progress_path(tmp_path,'a'*64,a)!=progress_path(tmp_path,'a'*64,b)
    assert progress_path(tmp_path,'a'*64,a)!=progress_path(tmp_path,'a'*64,pad(raw=b'new layout'))


def test_resume_failed_attempt_preserves_successes_and_retries_missing(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));path=tmp_path/'progress.json'
    q=TrajectoryProgress(s,path,pad=pad());q.begin();q.complete(grounded=True,disarmed=True);q.begin();q.fail()
    original=path.read_bytes()
    q=TrajectoryProgress(s,path,pad=pad(),resume_interrupted=True)
    assert q.completed==[0] and q.index==1 and q.available
    assert next(tmp_path.glob('progress.previous.*.json')).read_bytes()==original
    with pytest.raises(ValueError):TrajectoryProgress(s,path,pad=pad(raw=b'different'))


def test_sparse_recovery_skips_only_confirmed_successes_and_never_counts_duplicates(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));p=pad();root=tmp_path/'checkpoints'
    rec=lambda i,result:dict(set_sha256=s.sha256,pad=p,selection=s.selection(i),outcome=result)
    result=recover_progress(s,root,[rec(1,'success'),rec(1,'success'),rec(0,'failed')])
    q=TrajectoryProgress(s,result[0]['path'],pad=p)
    assert q.completed==[1] and q.index==0 and q.available
    q.begin();q.complete(grounded=True,disarmed=True)
    q=TrajectoryProgress(s,q.path,pad=p,resume_interrupted=True)
    assert q.completed==[0,1] and q.index==2 and not q.available and q.status()['all_complete']
    assert q.status()['current'] is None
    with pytest.raises(ValueError):q.begin()


def test_pads_have_independent_completion_and_recovery_is_idempotent(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));root=tmp_path/'checkpoints'
    records=[dict(set_sha256=s.sha256,pad=p,selection=s.selection(0),outcome='success')
             for p in [pad(),pad('paper_pad_layout_physical_20260920_runtime.yaml')]]
    result=recover_progress(s,root,records)
    assert len(result)==2 and all(r['completed']==1 for r in result)
    old={r['path']:Path(r['path']).read_bytes() for r in result}
    recover_progress(s,root,records)
    assert all(Path(p).read_bytes()==raw for p,raw in old.items())
    assert not list(root.rglob('*.previous.*'))


def test_only_one_planner_or_recovery_can_own_campaign(tmp_path):
    with acquire_campaign_lock(tmp_path):
        with pytest.raises(BlockingIOError):acquire_campaign_lock(tmp_path)
    with acquire_campaign_lock(tmp_path):pass


def test_wrong_reference_or_set_does_not_seed_progress(tmp_path):
    s=TrajectorySet(fixture_set(tmp_path));record=dict(set_sha256=s.sha256,pad=pad(),selection=s.selection(0),outcome='success')
    record['selection']['reference_id']=99
    with pytest.raises(ValueError):recover_progress(s,tmp_path/'checkpoints',[record])
