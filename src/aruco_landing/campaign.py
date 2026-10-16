"""Pad-scoped hardware campaign identities and read-only flight evidence parsing."""
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
import yaml

NAMED_PADS={
    'baseline_b2_700mm_runtime.yaml':'b2',
    'paper_pad_layout_physical_20260920_runtime.yaml':'baseline',
    'proposed_pad_1_layout_physical_20260920_runtime.yaml':'proposed1',
    'physical_pad.yaml':'legacy',
}


def pad_identity(configuration,raw=None):
    path=Path(configuration)
    raw=path.read_bytes() if raw is None else raw
    key=NAMED_PADS.get(path.name,re.sub(r'[^a-zA-Z0-9_-]+','_',path.stem).strip('_') or 'custom')
    return dict(key=key,layout_sha256=hashlib.sha256(raw).hexdigest(),configuration=path.name)


def progress_path(root,set_sha,pad):
    if not re.fullmatch('[a-f0-9]{64}',set_sha) or not re.fullmatch('[a-f0-9]{64}',pad['layout_sha256']):
        raise ValueError('campaign requires full set and pad layout hashes')
    if not re.fullmatch('[a-zA-Z0-9_-]+',pad['key']):raise ValueError('invalid pad key')
    return Path(root)/set_sha/(pad['key']+'-'+pad['layout_sha256'])/'progress.json'


def manifest_pad(document,owner_checkout):
    snapshot=document['snapshots']['start'];params=snapshot['ros']['parameters']
    configuration=params['physical_pad_estimator']['pad_configuration']
    for item in snapshot.get('config_files',{}).values():
        if item.get('path')==configuration and item.get('sha256'):
            result=pad_identity(configuration,b'')
            result['layout_sha256']=item['sha256'];return result
    # Old manifests captured the active path/revision but not the chosen YAML.
    # Recover the exact committed blob; never substitute today's calibration.
    revision=snapshot['repositories']['aruco_landing']['head']
    if not re.fullmatch('[a-f0-9]{40}',revision):raise ValueError('invalid saved owner revision')
    prefix='/work/ws/aruco-landing/src/aruco_landing/'
    if not configuration.startswith(prefix):raise ValueError('custom pad content not preserved in manifest')
    relative=configuration[len(prefix):]
    command=['git','-C',str(owner_checkout),'show',revision+':'+relative]
    owner=Path(owner_checkout).stat()
    if os.geteuid()==0 and owner.st_uid!=0:
        # ROS runs as root in Docker; read Git as the checkout owner instead of
        # relaxing Git ownership checks or changing global trust configuration.
        command=['setpriv','--reuid='+str(owner.st_uid),'--regid='+str(owner.st_gid),'--clear-groups']+command
    raw=subprocess.check_output(command,stderr=subprocess.DEVNULL)
    return pad_identity(configuration,raw)


def flight_record(bag_path,owner_checkout):
    """Parse one finalized arm recording; never publish or invoke ROS services."""
    import rosbag
    bag_path=Path(bag_path);manifest=bag_path.with_suffix('.runtime_manifest.yaml')
    document=yaml.safe_load(manifest.read_text())
    if not document.get('session',{}).get('disarm_wall_time_utc'):
        raise ValueError('recording manifest is not finalized')
    if not document.get('snapshots',{}).get('start',{}).get('ros',{}).get('parameters',{}).get('landing_trial',{}).get('trajectory_set_file'):
        raise ValueError('not a selected-trajectory flight')
    pad=manifest_pad(document,owner_checkout)
    selection=None;set_sha=None;complete=False;failed=False;final=None;faults=set()
    armed=None;ground=None;pad_seen=None
    with rosbag.Bag(str(bag_path)) as bag:
        for topic,m,_ in bag.read_messages(topics=['/landing/trial/status','/mavros/state',
                '/mavros/extended_state','/flight_safety/fault','/landing/estimator/status']):
            if topic=='/mavros/state':armed=m.armed
            elif topic=='/mavros/extended_state':ground=m.landed_state==1
            elif topic=='/flight_safety/fault':
                if m.level:faults.add((int(m.level),tuple(m.names),tuple(m.messages)))
            elif topic=='/landing/estimator/status':
                identity=json.loads(m.data).get('pad_identity')
                if identity:
                    if pad_seen is not None and identity!=pad_seen:raise ValueError('pad changed during flight')
                    pad_seen=identity
            else:
                d=json.loads(m.data);q=d.get('trajectory_set') or {}
                if q.get('active') is not None:
                    item=dict(q['current'])
                    if selection is not None and item!=selection:raise ValueError('multiple selected references in one arm recording')
                    selection=item;set_sha=q['set_sha256']
                if selection is not None:
                    final=d
                    if d.get('failure_reason') or d['phase'] in ('FAILED_HOLD','CANCELLED','GT_RECOVERY','FAILED_COMPLETE'):failed=True
                    if d['phase']=='COMPLETE' and d['reason'] in ('force_disarm_confirmed','pad_trajectory_set_complete'):
                        complete=True
    if selection is None:raise ValueError('no active selected trajectory in recording')
    if pad_seen is not None and pad_seen!=pad:raise ValueError('perception pad differs from recorded configuration')
    success=bool(complete and not failed and armed is False and ground is True)
    repos=document['snapshots']['start'].get('repositories',{})
    return dict(session=bag_path.stem,pad=pad,set_sha256=set_sha,selection=selection,
        outcome='success' if success else 'failed' if failed else 'incomplete',
        final_phase=final['phase'],reason=final['reason'],disarmed=armed is False,grounded=ground is True,
        safety_faults=[dict(level=x,names=list(y),messages=list(z)) for x,y,z in sorted(faults)],
        revisions={k:v.get('head') for k,v in repos.items()},
        bag_path=str(bag_path.resolve()),manifest_path=str(manifest.resolve()),
        bag_sha256=hashlib.sha256(bag_path.read_bytes()).hexdigest())


def recover_progress(trajectory_set,root,records):
    """Merge only confirmed successes for the exact pad/layout/set/reference."""
    from aruco_landing.trajectory_set import TrajectoryProgress
    grouped={}
    for record in records:
        if record['set_sha256']!=trajectory_set.sha256:continue
        item=record['selection'];index=item['index']
        if type(index) is not int or not 0<=index<len(trajectory_set.entries):raise ValueError('recorded reference index out of range')
        if item!=trajectory_set.selection(index):raise ValueError('recorded reference identity differs from selected bank')
        path=progress_path(root,trajectory_set.sha256,record['pad'])
        grouped.setdefault(path,dict(pad=record['pad'],completed=set()))
        if record['outcome']=='success':grouped[path]['completed'].add(index)
    result=[]
    for path,data in grouped.items():
        queue=TrajectoryProgress(trajectory_set,path,pad=data['pad'],resume_interrupted=True)
        completed=sorted(set(queue.completed)|data['completed'])
        if completed!=queue.completed or not path.exists():
            if path.exists():queue.archive()
            queue.completed=completed;queue.save()
        result.append(dict(path=str(path),**queue.status()))
    return result


def acquire_campaign_lock(root):
    import fcntl
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    stream=(root/'.campaign.lock').open('a')
    try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:stream.close();raise
    return stream
