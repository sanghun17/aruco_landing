"""Selected immutable bank references and a durable, once-per-entry flight queue."""
import hashlib
import json
import math
import os
import time
from pathlib import Path
import numpy as np
from aruco_landing.hardware_reference import HardwareReference


class TrajectorySet:
    def __init__(self, path):
        path=Path(path).resolve();raw=path.read_bytes()
        self.sha256=hashlib.sha256(raw).hexdigest();doc=json.loads(raw)
        if doc.get('schema_version')!=1 or doc.get('reference_id_convention')!='zero_based_bank_index':
            raise ValueError('trajectory set requires version 1 and explicit zero_based_bank_index')
        selections=doc.get('selections')
        if not isinstance(selections,list) or not selections:
            raise ValueError('select at least one trajectory before enabling trajectory-set mode')
        self.entries=[];self.references=[];seen=set();banks={}
        for entry in selections:
            kind=entry.get('type');rid=entry.get('reference_id')
            if kind not in ('inside','mixed','outside') or type(rid) is not int or rid<0:
                raise ValueError('each selection needs a valid type and nonnegative integer reference_id')
            key=(kind,rid)
            if key in seen:raise ValueError('duplicate trajectory in once-per-entry set')
            seen.add(key)
            if kind not in banks:
                record=doc['banks'][kind];bank_path=(path.parent/record['path']).resolve()
                digest=hashlib.sha256(bank_path.read_bytes()).hexdigest()
                if digest!=record['sha256']:raise ValueError('trajectory bank hash mismatch')
                with np.load(bank_path,allow_pickle=False) as b:
                    banks[kind]=(b['coefficients'],b['durations'],b['candidate_ids'],digest)
            c,d,ids,digest=banks[kind]
            if rid>=len(c):raise ValueError('trajectory reference_id outside bank')
            active=d[rid]>0
            ref=HardwareReference(dict(schema_version=1,frame_id='landing_pad',reference_point='camera_center',
                provenance=dict(bank_sha256=digest,type=kind,reference_id=rid,raw_candidate_id=int(ids[rid])),
                coefficients=c[rid,active].tolist(),durations_s=d[rid,active].tolist()))
            self.entries.append(dict(type=kind,reference_id=rid,raw_candidate_id=int(ids[rid]),bank_sha256=digest))
            self.references.append(ref)

    def selection(self,index):
        return dict(set_sha256=self.sha256,index=index,**self.entries[index])

    def body_start(self,index,pad,body_rotation,mount_translation,bounds,z_bounds=None):
        """Validate the full reference's Bernstein hull in the global frame.

        Camera path -> body origin using the held entry attitude and measured
        mounting translation. No trajectory or pad coordinates are rewritten.
        """
        pad=np.asarray(pad);rotation=np.asarray(body_rotation);mount=np.asarray(mount_translation)
        if pad.shape!=(4,4) or rotation.shape!=(3,3) or mount.shape!=(3,) or not all(np.isfinite(x).all() for x in (pad,rotation,mount)):
            raise ValueError('finite pad transform, body rotation and camera mount required')
        offset=pad[:3,3]-rotation@mount;ref=self.references[index]
        controls=[]
        for c,d in zip(ref.coefficients,ref.durations):
            q=pad[:3,:3]@c;q[:,0]+=offset
            b=np.zeros((3,6))
            for k in range(6):
                for j in range(k+1):b[:,k]+=q[:,j]*d**j*math.comb(k,j)/math.comb(5,j)
            controls.append(b)
        hull=np.concatenate(controls,axis=1)
        for axis,(lo,hi) in enumerate(bounds):
            if hull[axis].min()<lo or hull[axis].max()>hi:raise ValueError('reference leaves inset experiment XY box')
        if z_bounds is not None:
            lo,hi=z_bounds
            if hull[2].min()<lo or hull[2].max()>hi:raise ValueError('reference leaves configured Z geofence')
        p,_=ref.sample(0.)
        return pad[:3,:3]@p+offset


class TrajectoryProgress:
    def __init__(self,trajectory_set,path,*,restart_on_start=False,resume_interrupted=False,preserve_interrupted=False,pad=None):
        self.set=trajectory_set;self.path=Path(path)
        self.completed=[];self.failed=[];self.active=None;self.halted=False;self.pad=pad
        if restart_on_start:
            if self.path.exists():
                # Preserve exact previous bytes before atomically replacing progress.
                backup=self.path.with_name(self.path.stem+'.previous.'+str(time.time_ns())+'.json')
                with backup.open('xb') as stream:
                    stream.write(self.path.read_bytes());stream.flush();os.fsync(stream.fileno())
            self.save()
        elif self.path.exists():
            data=json.loads(self.path.read_text())
            if data.get('set_sha256')!=self.set.sha256:raise ValueError('progress belongs to another trajectory set')
            self.failed=data.get('failed',[])
            self.completed=data['completed'];self.active=data['active'];self.halted=data['halted']
            if (not isinstance(self.completed,list) or any(type(i) is not int or not 0<=i<len(self.set.entries) for i in self.completed)
                    or self.completed!=sorted(set(self.completed))):
                raise ValueError('invalid trajectory progress')
            if (not isinstance(self.failed,list) or any(not isinstance(f,dict) or type(f.get('index')) is not int
                    or not 0<=f['index']<len(self.set.entries) or not isinstance(f.get('reason'),str) or not f['reason']
                    for f in self.failed) or len({f['index'] for f in self.failed})!=len(self.failed)
                    or any(f['index'] in self.completed for f in self.failed)):
                raise ValueError('invalid failed trajectory records')
            if self.pad is not None and data.get('pad')!=self.pad:raise ValueError('progress belongs to another pad')
            if self.active is not None and (type(self.active) is not int or not 0<=self.active<len(self.set.entries) or self.active in self.finished):
                raise ValueError('invalid active trajectory')
            if (self.active is not None or self.halted) and not preserve_interrupted:
                if not resume_interrupted:
                    raise ValueError('unfinished/failed flight in progress file; explicit operator review required')
                self.archive()
                self.active=None;self.halted=False;self.save()

    def archive(self):
        backup=self.path.with_name(self.path.stem+'.previous.'+str(time.time_ns())+'.json')
        with backup.open('xb') as stream:
            stream.write(self.path.read_bytes());stream.flush();os.fsync(stream.fileno())

    @property
    def finished(self):return set(self.completed)|{f['index'] for f in self.failed}

    @property
    def index(self):return next((i for i in range(len(self.set.entries)) if i not in self.finished),len(self.set.entries))

    @property
    def available(self):return not self.halted and self.active is None and self.index<len(self.set.entries)

    def save(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_name(self.path.name+'.tmp')
            with temporary.open('w') as stream:
                json.dump(dict(set_sha256=self.set.sha256,completed=self.completed,active=self.active,halted=self.halted,**({'failed':self.failed} if self.failed else {}),**({'pad':self.pad} if self.pad is not None else {})),stream)
                stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        except OSError:
            self.halted=True
            raise

    def begin(self):
        if not self.available:raise ValueError('trajectory queue exhausted, active or halted')
        self.active=self.index;self.save()

    def complete(self,*,grounded,disarmed):
        if self.halted or self.active is None or not grounded or not disarmed:raise ValueError('confirmed grounded disarm and healthy progress required')
        self.completed=sorted(set(self.completed+[self.active]));self.active=None;self.save()

    def finish_failed(self,*,grounded,disarmed,reason):
        if self.active is None or not grounded or not disarmed or not isinstance(reason,str) or not reason:
            raise ValueError('failed flight requires confirmed grounded disarm and reason')
        self.archive()
        self.failed.append(dict(index=self.active,reason=reason,selection=self.set.selection(self.active)))
        self.active=None;self.halted=False;self.save()

    def fail(self):
        self.halted=True;self.save()

    def status(self):
        return dict(set_sha256=self.set.sha256,next_index=self.index,total=len(self.set.entries),
                    completed=len(self.completed),failed=len(self.failed),finished=len(self.finished),failure_records=self.failed,active=self.active,halted=self.halted,pad=self.pad,all_complete=len(self.finished)==len(self.set.entries),
                    current=self.set.selection(self.active if self.active is not None else self.index)
                    if self.index<len(self.set.entries) else None)
