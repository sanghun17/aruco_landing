#!/usr/bin/env python3
"""Hardware landing sequence through the existing flight-safety NORMAL lane.

Only the opt-in ground repeat experiment may arm; never enters OFFBOARD or publishes vision. Force-disarm is an explicit
stack policy, requested through the common safety authority. Pilot starts
OFFBOARD. Default dry-run writes shadow setpoints only. Current experiment
supports OptiTrack-only and the explicitly enabled, trial-gated pose router.
"""
import json
import math
import os
from pathlib import Path
import threading
import time
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState, PositionTarget
from mavros_msgs.srv import SetMode, ParamGet, CommandBool
from std_msgs.msg import Bool, String, Int32MultiArray
from std_srvs.srv import Trigger, TriggerResponse
from topic_tools.srv import MuxSelect
from flight_safety.msg import FlightState
from aruco_landing.landing_trial import Trial, limit_horizontal_velocity
from aruco_landing.physical_pad import SessionAlignment, valid_transform
from aruco_landing.pose_alignment import pose_matrix, matrix_pose, quaternion_distance_deg
from aruco_landing.yaw_control import yaw_feedback
from aruco_landing.repeat_trial import RepeatTrial
from aruco_landing.campaign import pad_identity,progress_path,acquire_campaign_lock
from aruco_landing.trajectory_set import TrajectorySet, TrajectoryProgress


def transform(msg):
    p=msg.pose.pose if hasattr(msg.pose,'pose') else msg.pose
    return pose_matrix([p.position.x,p.position.y,p.position.z],[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w])


def yaw(T):
    return math.atan2(T[1,0],T[0,0])


class LandingTrial:
    def __init__(self):
        rospy.init_node('landing_trial');self.lock=threading.RLock()
        self.dry=bool(rospy.get_param('~dry_run',True));self.inputs={};self.receipts={}
        self.auto_start=bool(rospy.get_param('~auto_start_on_offboard',False))
        self.repeat=None;self.arm_pending=False;self.session_reset_pending=False;self.session_reset_time=-1e30
        if rospy.get_param('~repeat_test',False):
            if not self.auto_start:raise ValueError('repeat test requires pilot OFFBOARD edge start')
            with open(rospy.get_param('~repeat_config')) as stream:repeat_config=yaml.safe_load(stream)
            with open(rospy.get_param('~geofence_config')) as stream:geofence=yaml.safe_load(stream)
            self.repeat=RepeatTrial(repeat_config,geofence)
            self.sync_status_path=Path(repeat_config['sync_status_file'])
            self.sync_reason='not_checked'
        self.transition=bool(rospy.get_param('~estimation_transition',False))
        self.gt_recovery=bool(rospy.get_param('~failed_trial_gt_recovery',False))
        self.recovery_started=None;self.recovery_gt_since=None
        self.gt_resume_pending=False;self.gt_resume_ack=False;self.gt_resume_failed=False
        self.alignment_timeout=float(rospy.get_param('~trajectory_initial_alignment_timeout_s',30.))
        if not math.isfinite(self.alignment_timeout) or self.alignment_timeout<=0:raise ValueError('positive initial alignment timeout required')
        self.routed_topic=rospy.get_param('~routed_pose_topic','/landing/vision_pose_selected')
        self.router_status={}
        self.routed_by_stamp={}
        self.transition_enable_pub=rospy.Publisher('/landing/pose_transition/enable',Bool,queue_size=1,latch=True)
        self.transition_enable_pub.publish(False)
        self.entry_ready=False;self.prepare_pending=False;self.prepare_retry=-1e30
        self.standby_since=None;self.last_fcu_mode=None;self.entry_time=None
        self.global_frame=rospy.get_param('~global_frame','odom')
        self.mocap_topic=rospy.get_param('~mocap_topic','/vrpn_client_node/pure/pose')
        self.normal_topic=rospy.get_param('~normal_topic','/local_controller/setpoint_raw/local')
        if self.normal_topic=='/mavros/setpoint_raw/local':raise ValueError('flight-safety must own MAVROS setpoints')
        self.trial=Trial(float(rospy.get_param('~marker_confirm_s',1.)),float(rospy.get_param('~marker_loss_s',.5)),
            termination_height_source=rospy.get_param('~termination_height_source','dual_height'))
        self.trial.finish_mode=rospy.get_param('~landing_finish_mode','auto_land')
        if self.trial.finish_mode not in ('auto_land','force_disarm'):raise ValueError('invalid landing_finish_mode')
        self.cut_pending=False;self.cut_started=None;self.cut_result=None
        self.termination_service=rospy.ServiceProxy('/flight_safety_response/request_termination',Trigger)
        self.termination_reset=rospy.ServiceProxy('/flight_safety_response/reset_external_termination',Trigger)
        self.speed=float(rospy.get_param('~approach_speed_mps',.5));self.kp=float(rospy.get_param('~approach_kp',.8))
        self.landing_speed=float(rospy.get_param('~landing_horizontal_speed_mps',.5))
        self.yaw_rate=float(rospy.get_param('~yaw_rate_limit_rad_s',.35))
        self.target=np.array(rospy.get_param('~initial_target_global_xy',[0.,0.]),float)
        if self.target.shape!=(2,)or not np.isfinite(self.target).all():raise ValueError('invalid initial target')
        if not all(math.isfinite(v)and v>0 for v in (self.speed,self.landing_speed,self.kp,self.yaw_rate)):raise ValueError('invalid trial limits')
        with open(rospy.get_param('~extrinsic_file'))as f:mount=yaml.safe_load(f)
        self.X=np.array(mount['matrix_row_major']).reshape(4,4)
        if mount['validation_result']!='PASS'or not valid_transform(self.X):raise ValueError('invalid mount')
        self.campaign_pad=None;self.estimator_status={}
        self.trajectory_set=None;self.trajectory_progress=None;self.trajectory_ack={}
        self.trajectory_goal=None;self.trajectory_pad=None;self.trajectory_settled_since=None
        self.last_selection_publish=-1e30;self.trajectory_z_bounds=None
        set_path=rospy.get_param('~trajectory_set_file','')
        if set_path:
            if self.repeat is None:raise ValueError('trajectory set requires existing ground repeat mode')
            self.trajectory_set=TrajectorySet(set_path)
            campaign_root=rospy.get_param('~trajectory_campaign_root','')
            if campaign_root:
                self.campaign_lease=acquire_campaign_lock(campaign_root)
                self.campaign_pad=pad_identity(rospy.get_param('/physical_pad_estimator/pad_configuration'))
                checkpoint=progress_path(campaign_root,self.trajectory_set.sha256,self.campaign_pad)
                self.trajectory_progress=TrajectoryProgress(self.trajectory_set,checkpoint,
                    resume_interrupted=True,preserve_interrupted=self.gt_recovery,pad=self.campaign_pad)
            else:
                self.trajectory_progress=TrajectoryProgress(self.trajectory_set,rospy.get_param('~trajectory_progress_file'),
                    restart_on_start=bool(rospy.get_param('~trajectory_restart_on_start',False)))
            if self.gt_recovery and self.trajectory_progress.active is not None:
                self.trial.fail('interrupted_trial_requires_review')
            if 'z' in geofence['enabled_axes']:
                self.trajectory_z_bounds=tuple(map(float,geofence['box']['z']))
            self.selection_pub=rospy.Publisher('/landing/trajectory/selection',String,queue_size=1,latch=True)
        self.pad=None;self.aligned=False;self.hold=None;self.altitude=None;self.heading=None
        self.state_phase='IDLE';self.land_requested=False;self.last_mode_call=-1e30;self.mode_pending=False;self.handoff_started=None;self.last_mux_call=-1e30
        self.last_matched=-1e30;self.vision_mismatch=False;self.mocap_by_stamp={};self.marker_cache=None
        self.history=SessionAlignment();self.last_tick=None;self.previous_now=None
        self.pub=rospy.Publisher('/landing/trial/setpoint_preview',PositionTarget,queue_size=1)
        self.live=None
        if not self.dry:
            publishers=rospy.get_master().getSystemState()[2][0]
            if dict(publishers).get(self.normal_topic):raise RuntimeError('NORMAL input already has a publisher')
            self.live=rospy.Publisher(self.normal_topic,PositionTarget,queue_size=1)
        self.mission_pub=rospy.Publisher('/control/mission_status',String,queue_size=1)
        self.status=rospy.Publisher('/landing/trial/status',String,queue_size=1,latch=True)
        self.body_pub=rospy.Publisher('/landing/trial/body_pose_pad',PoseWithCovarianceStamped,queue_size=1)
        self.camera_pub=rospy.Publisher('/landing/trial/camera_pose_pad',PoseWithCovarianceStamped,queue_size=1)
        self.visible_pub=rospy.Publisher('/landing/trial/control_valid',Bool,queue_size=1)
        self.mode_service=rospy.ServiceProxy('/mavros/set_mode',SetMode)
        self.controller_reset=rospy.ServiceProxy('/landing_trial_controller/reset',Trigger)
        self.controller_gt_resume=rospy.ServiceProxy('/landing_trial_controller/resume_with_gt',Trigger)
        self.param_service=rospy.ServiceProxy('/mavros/param/get',ParamGet)
        self.arm_service=rospy.ServiceProxy('/mavros/cmd/arming',CommandBool)
        self.mux_service=rospy.ServiceProxy('/vision_pose_mux/select',MuxSelect)
        topics=[('mocap',self.mocap_topic,PoseStamped),('local','/mavros/local_position/pose',PoseStamped),
            ('state','/mavros/state',State),('extended','/mavros/extended_state',ExtendedState),
            ('velocity','/mavros/local_position/velocity_local',TwistStamped),
            ('safety','/flight_safety/state',FlightState),('vision','/mavros/vision_pose/pose',PoseStamped),
            ('selected','/vision_pose_mux/selected',String),('pad','/landing/pad_pose_global',PoseStamped),
            ('aligned','/landing/alignment/ready',Bool),('visible','/landing/target_visible',Bool),
            ('inliers','/landing/estimator/inlier_ids',Int32MultiArray),
            ('marker_body','/landing/vehicle_pose_pad',PoseWithCovarianceStamped),
            ('marker_camera','/landing/camera_pose_pad',PoseWithCovarianceStamped),
            ('command','/landing/trial/landing_cmd_pad',TwistStamped),
            ('routed',self.routed_topic,PoseStamped),
            ('router','/landing/pose_transition/status',String),
            ('trajectory_ready','/landing/trajectory/ready',String),
            ('estimator','/landing/estimator/status',String)]
        self.subscribers=[rospy.Subscriber(topic,kind,self.receive,key,queue_size=10,tcp_nodelay=True)for key,topic,kind in topics]
        rospy.Service('~accept_failed_trial',Trigger,self.accept_failed_trial);rospy.Service('~start',Trigger,self.start);rospy.Service('~reset',Trigger,self.reset);rospy.Service('~abort',Trigger,self.abort)
        rospy.Timer(rospy.Duration(1/60.),self.tick)

    def accept_failed_trial(self,_):
        with self.lock:
            now=time.monotonic();state=self.inputs.get('state');ext=self.inputs.get('extended')
            if not (self.trajectory_progress and self.trajectory_progress.active is not None
                    and self.trial.phase in ('FAILED_HOLD','CANCELLED','FAILED_COMPLETE')
                    and self.fresh('state',now,2.) and state.connected and not state.armed
                    and self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_ON_GROUND):
                return TriggerResponse(success=False,message='requires a failed active trial, grounded and disarmed')
            reason=rospy.get_param('~reviewed_failure_reason','') or self.trial.failure_reason or self.trial.reason or 'operator_confirmed_failure'
            try:self.trajectory_progress.finish_failed(grounded=True,disarmed=True,reason=reason)
            except (ValueError,OSError) as error:return TriggerResponse(success=False,message=str(error))
            self.trial.phase='FAILED_COMPLETE';self.trial.reason=self.trial.failure_reason=reason;self.entry_ready=False;self.standby_since=None;self.trajectory_ack={}
            return TriggerResponse(success=True,message='recorded failed trial; next trajectory selected')

    def prepare_standby(self,now):
        # Service calls must never block the setpoint timer. Preparation is
        # permitted on the ground; only a later pilot OFFBOARD edge starts motion.
        if self.entry_ready or self.prepare_pending or now-self.prepare_retry<1.:return
        self.prepare_pending=True;self.prepare_retry=now
        def prepare():
            ok=False
            try:
                if not self.dry and self.trial.finish_mode=='force_disarm':
                    if not rospy.get_param('/flight_safety_response/allow_external_termination',False):return
                    rospy.wait_for_service('/flight_safety_response/request_termination',timeout=1.)
                if not self.dry and self.trial.finish_mode=='auto_land':
                    value=self.param_service(param_id='COM_DISARM_LAND')
                    if not value.success or max(value.value.real,value.value.integer)<=0:return
                rospy.wait_for_service('/landing_trial_controller/reset',timeout=1.)
                ok=self.controller_reset().success
            except (rospy.ROSException,rospy.ServiceException):pass
            finally:
                with self.lock:
                    state=self.inputs.get('state')
                    self.entry_ready=bool(ok and state and state.mode!='OFFBOARD')
                    self.prepare_pending=False
        threading.Thread(target=prepare,daemon=True).start()

    def sync_ready(self):
        if self.repeat is None:return True
        try:
            status=json.loads(self.sync_status_path.read_text())
            age=time.time()-float(status['checked_at'])
            logs=self.sync_status_path.parent
            sessions=sorted(p.stem for p in logs.glob('*.bag'))
            expected=sessions[-1] if sessions else None
            ready=(0<=age<=6. and status.get('ready') is True
                   and status.get('latest_session')==expected
                   and not any(logs.glob('*.active')) and not any(logs.glob('*.ready')))
            self.sync_reason='verified' if ready else status.get('reason','stale_or_pending_transfer')
            return ready
        except (OSError,ValueError,KeyError,TypeError):
            self.sync_reason='sync_status_unavailable';return False

    def request_arm(self):
        if self.dry or self.arm_pending:return
        self.arm_pending=True;cycle=self.repeat.number
        def call():
            try:
                with self.lock:
                    now=time.monotonic();state=self.inputs.get('state');ext=self.inputs.get('extended')
                    allowed=(self.repeat.number==cycle and self.trial.phase=='ARMING'
                        and state and state.mode=='OFFBOARD' and not state.armed
                        and self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_ON_GROUND
                        and self.health(now) and self.sync_ready())
                if allowed:
                    result=self.arm_service(value=True)
                    with self.lock:
                        if self.repeat.number==cycle and self.trial.phase=='ARMING' and not result.success:
                            self.repeat.phase='FAILED_HOLD';self.trial.fail('arm_rejected')
            except rospy.ServiceException as error:
                with self.lock:
                    if self.repeat.number==cycle and self.trial.phase=='ARMING':
                        self.repeat.phase='FAILED_HOLD';self.trial.fail('arm_service_failed')
                rospy.logwarn('repeat arm failed: %s',error)
            finally:self.arm_pending=False
        threading.Thread(target=call,daemon=True).start()

    def prepare_external_session(self,now):
        if self.dry or self.session_reset_pending or now-self.session_reset_time<1.:return
        self.session_reset_pending=True;self.session_reset_time=now
        def call():
            try:self.termination_reset()
            except rospy.ServiceException:pass
            finally:self.session_reset_pending=False
        threading.Thread(target=call,daemon=True).start()

    def receive(self,m,key):
        with self.lock:
            now=time.monotonic()
            if key in ('mocap','local','vision','pad','marker_body','marker_camera','routed'):
                try:
                    if not valid_transform(transform(m)):return
                except (ValueError,TypeError):return
            if key=='mocap'and m.header.frame_id!=self.global_frame:return
            self.inputs[key]=m;self.receipts[key]=now
            if key=='estimator':
                try:self.estimator_status=json.loads(m.data)
                except (ValueError,TypeError):self.estimator_status={}
                if self.campaign_pad is not None and self.estimator_status.get('pad_identity')!=self.campaign_pad:
                    self.entry_ready=False;self.standby_since=None
                    if self.trial.phase not in ('IDLE','COMPLETE','CANCELLED'):
                        self.trial.fail('perception_pad_changed');self.capture_hold()
            if key=='trajectory_ready':
                try:self.trajectory_ack=json.loads(m.data)
                except (ValueError,TypeError):self.trajectory_ack={}
            if key=='router':
                try:self.router_status=json.loads(m.data)
                except (ValueError,TypeError):self.router_status={}
            if key=='routed':
                self.routed_by_stamp[m.header.stamp.to_nsec()]=transform(m)
                if len(self.routed_by_stamp)>500:self.routed_by_stamp.pop(next(iter(self.routed_by_stamp)))
            if key=='pad':
                T=transform(m)
                if m.header.frame_id==self.global_frame and valid_transform(T):
                    if self.pad is not None and not np.allclose(self.pad,T,atol=1e-5)and (self.trial.phase in ('APPROACH','DESCEND','AUTO_LAND') or (self.trajectory_progress is not None and self.trajectory_progress.active is not None)):
                        self.trial.fail('session_pad_reference_changed');self.capture_hold()
                    self.pad=T
            elif key=='aligned':
                if self.aligned and not m.data and (self.trial.phase in ('APPROACH','DESCEND','AUTO_LAND') or (self.trajectory_progress is not None and self.trajectory_progress.active is not None)):
                    self.trial.fail('session_alignment_reset');self.capture_hold()
                self.aligned=m.data
                if not self.aligned:self.pad=None
            elif key=='mocap':
                T=transform(m)
                if m.header.frame_id!=self.global_frame or not valid_transform(T):return
                self.history.add_mocap(m.header.stamp.to_sec(),T)
                self.mocap_by_stamp[m.header.stamp.to_nsec()]=T
                if len(self.mocap_by_stamp)>500:self.mocap_by_stamp.pop(next(iter(self.mocap_by_stamp)))
            if key in ('mocap','routed','vision')and 'vision'in self.inputs:
                v=self.inputs['vision'];history=self.routed_by_stamp if self.transition else self.mocap_by_stamp
                ref=history.get(v.header.stamp.to_nsec())
                if ref is not None:
                    self.vision_mismatch=v.header.frame_id!=self.global_frame or not np.allclose(ref,transform(v),atol=1e-7,rtol=0)
                    if not self.vision_mismatch:self.last_matched=now

    def fresh(self,key,now,age=.2):
        if key not in self.inputs or not 0<=now-self.receipts[key]<=age:return False
        m=self.inputs[key]
        return not hasattr(m,'header')or (not m.header.stamp.is_zero()and -.05<=rospy.get_time()-m.header.stamp.to_sec()<=age)

    def health(self,now):
        if self.campaign_pad is not None and (not self.fresh('estimator',now,1.5)
                or self.estimator_status.get('pad_identity')!=self.campaign_pad):return False
        if not self.fresh('mocap',now)or not self.fresh('local',now):return False
        if not self.fresh('state',now,2.)or not self.inputs['state'].connected:return False
        # FlightState stamp/receipt independently prevent acting on a stopped safety node.
        if not self.fresh('safety',now,.5):return False
        safety=self.inputs['safety']
        if safety.level!=0 or safety.kill_switch or safety.control_lane in ('KILL','LAND'):return False
        selected=self.inputs.get('selected')
        expected=self.routed_topic if self.transition else self.mocap_topic
        if self.transition:
            if not self.fresh('router',now,.2) or self.router_status.get('output_topic')!=self.routed_topic:return False
            if self.router_status.get('source') not in ('optitrack','marker'):return False
        # Permit the bounded marker-loss decision window while PX4 local pose and
        # safety remain fresh. Descent still requires current marker/control data.
        grace=self.trial.marker_loss_s+.15 if self.transition and self.router_status.get('source')=='marker' else .3
        return bool(selected and selected.data==expected and now-self.last_matched<grace and not self.vision_mismatch)

    def pending_marker(self,now):
        c=self.marker_cache
        if c and 0<=now-c[1]<=.2 and -.05<=rospy.get_time()-c[2]<=.2:
            return True,c[0],c[1]
        return False,None,None

    def marker(self,now):
        if not self.aligned or self.pad is None:return False,None,None
        if not all(self.fresh(k,now)for k in ('visible','inliers','marker_body','marker_camera')):return False,None,None
        if not self.inputs['visible'].data or len(self.inputs['inliers'].data)<1:return False,None,None
        b=self.inputs['marker_body'];camera=self.inputs['marker_camera']
        if b.header.frame_id!=camera.header.frame_id:return False,None,None
        if abs(b.header.stamp.to_sec()-camera.header.stamp.to_sec())>.01:return self.pending_marker(now)
        ref=self.history.interpolate(b.header.stamp.to_sec())
        if ref is None:return self.pending_marker(now)
        estimate=self.pad@transform(b)
        if np.linalg.norm(estimate[:3,3]-ref[:3,3])>.15:return False,None,None
        if quaternion_distance_deg(matrix_pose(estimate)[1],matrix_pose(ref)[1])>12:return False,None,None
        height=transform(camera)[2,3]
        if np.isfinite(height)and height>=0:
            self.marker_cache=(float(height),self.receipts['marker_body'],b.header.stamp.to_sec())
            return True,float(height),self.receipts['marker_body']
        return False,None,None

    def capture_hold(self):
        if 'local'in self.inputs:self.hold=transform(self.inputs['local']).copy()

    def start(self,_):
        with self.lock:
            now=time.monotonic();state=self.inputs.get('state');ext=self.inputs.get('extended')
            if not self.health(now):return TriggerResponse(False,'fresh OptiTrack, matching vision mux, local pose and safety OK required')
            if not state.armed or not self.fresh('extended',now,2.)or ext.landed_state!=ExtendedState.LANDED_STATE_IN_AIR:
                return TriggerResponse(False,'manual takeoff required: armed and PX4 IN_AIR')
            if state.mode=='OFFBOARD':return TriggerResponse(False,'prepare in POSCTL, then pilot selects OFFBOARD')
            if not self.dry:
                try:
                    value=self.param_service(param_id='COM_DISARM_LAND')
                    if not value.success or max(value.value.real,value.value.integer)<=0:
                        return TriggerResponse(False,'positive COM_DISARM_LAND required for verified automatic disarm')
                except rospy.ServiceException:return TriggerResponse(False,'cannot verify COM_DISARM_LAND')
            try:
                rospy.wait_for_service('/landing_trial_controller/reset',timeout=1.)
                self.controller_reset()
            except (rospy.ROSException,rospy.ServiceException):return TriggerResponse(False,'landing controller reset unavailable')
            if not self.trial.prepare(now):return TriggerResponse(False,'reset completed/failed trial before starting')
            self.capture_hold();self.altitude=self.hold[2,3];self.heading=yaw(self.hold)
            self.land_requested=False
            return TriggerResponse(True,'prestreaming hold; pilot may enter OFFBOARD after 1 second; dry_run='+str(self.dry))

    def reset(self,_):
        with self.lock:
            state=self.inputs.get('state')
            if state and state.mode in ('OFFBOARD','AUTO.LAND'):return TriggerResponse(False,'leave autonomous mode before reset')
            self.trial.cancel('reset');self.hold=None;return TriggerResponse(True,'ready for explicit new start')

    def abort(self,_):
        with self.lock:
            self.capture_hold();self.trial.fail('operator_abort');return TriggerResponse(True,'trial failed; holding current position')

    def mocap_camera_height(self,now):
        if not self.aligned or self.pad is None or not self.fresh('mocap',now):return None
        msg=self.inputs['mocap']
        if msg.header.frame_id!=self.global_frame:return None
        body=transform(msg)
        if not valid_transform(body) or not valid_transform(self.pad):return None
        height=float((np.linalg.inv(self.pad)@body@self.X)[2,3])
        return height if math.isfinite(height) else None

    def request_cut(self):
        if self.dry or self.cut_pending:return
        self.cut_pending=True
        def call():
            try:
                with self.lock:
                    now=time.monotonic();state=self.inputs.get('state')
                    good,height,_=self.marker(now)
                    h=self.mocap_camera_height(now)
                    allowed=bool(self.trial.phase in ('CUT_WAIT','GT_CUT_WAIT') and state and self.trial.can_force_disarm(
                        offboard=state.mode=='OFFBOARD',armed=state.armed,healthy=self.health(now) and (self.trial.phase!='GT_CUT_WAIT' or self.recovery_gt_ready(now)),
                        mocap_height=h,marker_good=good,marker_height=height))
                self.cut_result=bool(allowed and self.termination_service().success)
            except rospy.ServiceException as error:
                self.cut_result=False;rospy.logerr('termination request failed: %s',error)
            finally:self.cut_pending=False
        threading.Thread(target=call,daemon=True).start()

    def request_mode(self,mode,now):
        if self.dry or self.mode_pending or now-self.last_mode_call<1.:return
        self.last_mode_call=now;self.mode_pending=True
        def call():
            try:
                with self.lock:
                    state=self.inputs.get('state')
                    allowed=bool(state and ((mode=='AUTO.LAND'and state.mode=='OFFBOARD'and self.trial.phase=='AUTO_LAND')
                        or (mode=='POSCTL'and self.trial.phase=='FAILED_HOLD'and state.mode=='AUTO.LAND')))
                if allowed:self.mode_service(base_mode=0,custom_mode=mode)
            except rospy.ServiceException as error:rospy.logwarn('mode request failed: %s',error)
            finally:self.mode_pending=False
        threading.Thread(target=call,daemon=True).start()

    def recovery_gt_ready(self,now):
        return bool(self.health(now) and self.fresh('routed',now) and self.fresh('router',now)
            and self.router_status.get('source')=='optitrack'
            and self.router_status.get('last_output_source')=='optitrack'
            and self.recovery_gt_since is not None and now-self.recovery_gt_since>=.5)

    def request_gt_resume(self):
        if self.gt_resume_pending or self.gt_resume_ack or self.gt_resume_failed:return
        self.gt_resume_pending=True
        def call():
            try:
                with self.lock:
                    state=self.inputs.get('state');now=time.monotonic()
                    allowed=bool(state and state.armed and state.mode=='OFFBOARD'
                        and self.trial.phase=='GT_RECOVERY' and self.recovery_gt_ready(now))
                ok=bool(allowed and self.controller_gt_resume().success)
                with self.lock:self.gt_resume_ack=ok;self.gt_resume_failed=not ok
            except rospy.ServiceException as error:
                with self.lock:self.gt_resume_failed=True
                rospy.logerr('GT reference resume failed: %s',error)
            finally:
                with self.lock:self.gt_resume_pending=False
        threading.Thread(target=call,daemon=True).start()

    def output(self,T,stamp):
        m=PoseWithCovarianceStamped();m.header.stamp=stamp;m.header.frame_id='landing_pad'
        p,q=matrix_pose(T);m.pose.pose.position.x,m.pose.pose.position.y,m.pose.pose.position.z=p
        m.pose.pose.orientation.x,m.pose.pose.orientation.y,m.pose.pose.orientation.z,m.pose.pose.orientation.w=q
        return m

    def hold_setpoint(self,T):
        s=PositionTarget();s.coordinate_frame=PositionTarget.FRAME_LOCAL_NED;s.header.frame_id='map'
        s.type_mask=PositionTarget.IGNORE_VX|PositionTarget.IGNORE_VY|PositionTarget.IGNORE_VZ|PositionTarget.IGNORE_AFX|PositionTarget.IGNORE_AFY|PositionTarget.IGNORE_AFZ|PositionTarget.IGNORE_YAW_RATE
        s.position.x,s.position.y,s.position.z=T[:3,3];s.yaw=yaw(T);return s

    def tick(self,_):
        with self.lock:
            now=time.monotonic();ros_now=rospy.get_time();state=self.inputs.get('state')
            if self.previous_now is not None and ros_now<self.previous_now:
                self.trial.cancel('ROS_clock_reset');self.pad=None;self.aligned=False
            self.previous_now=ros_now
            if state is None:return
            healthy=self.health(now)
            data_ready=self.sync_ready() if self.repeat else True
            if self.trajectory_progress is not None:
                progress=self.trajectory_progress
                if progress.active is not None:
                    ext=self.inputs.get('extended')
                    grounded=self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_ON_GROUND
                    try:
                        if self.trial.phase=='COMPLETE' and grounded and not state.armed and not progress.halted:
                            progress.complete(grounded=True,disarmed=True)
                            self.entry_ready=False;self.standby_since=None;self.trajectory_ack={}
                        elif self.trial.phase=='FAILED_COMPLETE' and grounded and not state.armed:
                            progress.finish_failed(grounded=True,disarmed=True,reason=self.trial.failure_reason or self.trial.reason)
                            self.entry_ready=False;self.standby_since=None;self.trajectory_ack={}
                        elif self.trial.phase in ('FAILED_HOLD','CANCELLED') and not progress.halted:
                            progress.fail()
                    except OSError:
                        self.trial.fail('trajectory_progress_write_failed');self.capture_hold()
                if not progress.available and progress.active is None and len(progress.finished)==len(progress.set.entries):
                    self.entry_ready=False;self.standby_since=None
                    if not state.armed:
                        self.trial.phase='COMPLETE';self.trial.reason='pad_trajectory_set_complete'
                data_ready=data_ready and not progress.halted
                if progress.active is None:
                    data_ready=data_ready and progress.available
            if self.auto_start:
                standby=state.mode not in ('OFFBOARD','AUTO.LAND') and self.trial.phase in ('IDLE','CANCELLED','COMPLETE','FAILED_COMPLETE')
                if self.repeat and standby and not state.armed and data_ready and not healthy:
                    self.prepare_external_session(now)
                if standby and healthy and data_ready:
                    if self.standby_since is None:self.standby_since=now
                    self.prepare_standby(now)
                    if self.trajectory_set is not None and now-self.last_selection_publish>=.5:
                        self.selection_pub.publish(json.dumps(self.trajectory_set.selection(self.trajectory_progress.index)))
                        self.last_selection_publish=now
                elif state.mode!='OFFBOARD':self.standby_since=None
                entered=state.mode=='OFFBOARD' and self.last_fcu_mode not in (None,'OFFBOARD')
                if entered and self.trial.phase not in ('AUTO_LAND','CUT_WAIT','GT_RECOVERY','GT_DESCEND','GT_CUT_WAIT') and not (self.trajectory_progress is not None and self.trajectory_progress.index==len(self.trajectory_set.entries)):
                    ext=self.inputs.get('extended')
                    airborne=self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_IN_AIR
                    grounded=self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_ON_GROUND
                    ready=healthy and self.entry_ready and self.standby_since is not None and now-self.standby_since>=1.
                    if self.trajectory_set is not None:
                        ready=bool(ready and self.trajectory_progress.available and self.aligned and self.pad is not None
                            and self.fresh('trajectory_ready',now,1.5)
                            and self.trajectory_ack==self.trajectory_set.selection(self.trajectory_progress.index))
                    allowed=ready and state.armed and airborne and self.repeat is None
                    self.entry_ready=False;self.standby_since=None;self.marker_cache=None
                    if self.repeat and ready and data_ready and grounded and not state.armed and self.fresh('velocity',now):
                        try:
                            target=None
                            if self.trajectory_set is not None:
                                G_B=transform(self.inputs['mocap'])
                                target=self.trajectory_set.body_start(self.trajectory_progress.index,self.pad,
                                    G_B[:3,:3],self.X[:3,3],self.repeat.bounds,self.trajectory_z_bounds)
                                self.trajectory_goal=target;self.trajectory_pad=self.pad.copy()
                                self.trajectory_settled_since=None
                            self.repeat.begin(now,transform(self.inputs['mocap'])[:3,3],target=target)
                            if self.trajectory_progress is not None:self.trajectory_progress.begin()
                            self.trial.phase='ARMING';self.trial.reason='pilot_requested_ground_repeat'
                            self.entry_time=now;self.capture_hold();self.heading=yaw(self.hold)
                            self.cut_result=None;self.trial.failure_reason='';self.request_arm()
                        except (ValueError,OSError) as error:self.trial.fail(str(error))
                    elif allowed and self.trial.prepare(now-1.):
                        self.entry_time=now;self.capture_hold()
                    else:
                        self.trial.fail('OFFBOARD_entry_not_ready');self.capture_hold()
                self.last_fcu_mode=state.mode
            good,height,marker_time=self.marker(now)
            if self.auto_start and (self.entry_time is None or state.mode not in ('OFFBOARD','AUTO.LAND') or marker_time is None or marker_time<self.entry_time):
                good,height,marker_time=False,None,None
            ext=self.inputs.get('extended');landed=bool(self.fresh('state',now,2.)and state.connected and self.fresh('extended',now,2.)and ext.landed_state==ExtendedState.LANDED_STATE_ON_GROUND)
            mocap_height=self.mocap_camera_height(now)
            terminal_ready=self.trial.can_force_disarm(offboard=state.mode=='OFFBOARD',armed=state.armed,
                healthy=healthy,mocap_height=mocap_height,marker_good=good,marker_height=height)
            if self.transition and self.router_status.get('fallback_latched') and self.trial.phase in ('APPROACH','DESCEND','AUTO_LAND') and not terminal_ready:
                self.trial.fail('marker_source_lost_trial_failed');self.capture_hold()
            estimation_ready=not self.transition or (self.router_status.get('source')=='marker' and self.router_status.get('last_output_source')=='marker' and self.router_status.get('switch_check',{}).get('consistent',False) and self.fresh('routed',now) and healthy)
            if self.trajectory_set is not None and self.trial.phase=='APPROACH':
                v=self.inputs.get('velocity')
                speed=np.linalg.norm([v.twist.linear.x,v.twist.linear.y,v.twist.linear.z]) if v else float('inf')
                close=(self.trajectory_goal is not None and self.fresh('mocap',now) and self.fresh('velocity',now)
                    and np.linalg.norm(transform(self.inputs['mocap'])[:3,3]-self.trajectory_goal)<=self.repeat.tolerance
                    and speed<=self.repeat.velocity_tolerance)
                if not close:self.trajectory_settled_since=None
                elif self.trajectory_settled_since is None:self.trajectory_settled_since=now
                estimation_ready=bool(estimation_ready and self.trajectory_settled_since is not None
                    and now-self.trajectory_settled_since>=self.repeat.dwell
                    and self.trajectory_ack==self.trajectory_set.selection(self.trajectory_progress.active))
                if now-self.repeat.started>=self.alignment_timeout:
                    self.trial.fail('trajectory_initial_alignment_timeout');self.capture_hold()
            gt_phase=self.trial.phase in ('GT_RECOVERY','GT_DESCEND')
            if gt_phase:
                gt_ok=(healthy and self.fresh('routed',now) and self.fresh('router',now)
                    and self.router_status.get('source')=='optitrack'
                    and self.router_status.get('last_output_source')=='optitrack')
                if not gt_ok:self.recovery_gt_since=None
                elif self.recovery_gt_since is None:self.recovery_gt_since=now
                estimation_ready=self.recovery_gt_ready(now) and self.gt_resume_ack
                if self.trial.phase=='GT_RECOVERY':
                    if self.recovery_gt_ready(now):self.request_gt_resume()
                    if self.gt_resume_failed or now-self.recovery_started>5.:
                        self.trial.fail('gt_reference_resume_unconfirmed');self.capture_hold()
                elif not gt_ok:
                    self.trial.fail('gt_recovery_route_invalid');self.capture_hold()
            if self.trial.phase=='FAILED_HOLD' and not self.trial.failure_reason:self.trial.failure_reason=self.trial.reason
            previous=self.trial.phase
            if self.repeat and self.trial.phase in self.repeat.PHASES:
                v=self.inputs.get('velocity');speed=math.sqrt(v.twist.linear.x**2+v.twist.linear.y**2+v.twist.linear.z**2) if v else float('inf')
                phase,reason=self.repeat.step(now,armed=state.armed,
                    airborne=self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_IN_AIR,
                    offboard=state.mode=='OFFBOARD',healthy=healthy and self.fresh('velocity',now),
                    position=transform(self.inputs['mocap'])[:3,3],speed=speed)
                self.trial.phase=phase;self.trial.reason=reason or 'repeat_'+phase.lower()
                if phase=='APPROACH':
                    # Fresh marker dwell begins only after the random target has settled.
                    self.entry_time=now;self.marker_cache=None
                    self.trial.good_since=self.trial.last_good=None
            else:
                phase=self.trial.step(now,offboard=state.mode=='OFFBOARD',armed=state.armed,healthy=healthy,marker_good=good,marker_height=height,marker_time=marker_time,mocap_height=mocap_height,estimation_ready=estimation_ready,landed=landed,auto_land=state.mode=='AUTO.LAND')
            if self.trial.recover_failed(enabled=self.gt_recovery and self.transition and self.trajectory_set is not None,
                    offboard=state.mode=='OFFBOARD',armed=state.armed,healthy=healthy):
                phase=self.trial.phase;self.recovery_started=now;self.recovery_gt_since=None
                self.gt_resume_ack=self.gt_resume_failed=False
                self.capture_hold()
            if previous!=phase:
                self.capture_hold()
                if phase=='APPROACH'and self.hold is not None:self.altitude=self.hold[2,3];self.heading=yaw(self.hold)
                if phase=='AUTO_LAND':self.handoff_started=now
                if phase in ('CUT_WAIT','GT_CUT_WAIT'):
                    self.cut_started=now;self.cut_result=None;self.request_cut()
            if phase in ('CUT_WAIT','GT_CUT_WAIT') and state.armed and (self.cut_result is False or now-self.cut_started>2.):
                self.trial.fail('force_disarm_unconfirmed');phase=self.trial.phase;self.capture_hold()
            if phase=='AUTO_LAND'and state.mode=='OFFBOARD'and now-self.handoff_started>3.:
                self.trial.fail('AUTO_LAND_not_acknowledged');self.capture_hold();phase=self.trial.phase
            # Genuine pilot mode change cancels the trial; no automatic OFFBOARD reentry.
            if phase=='FAILED_HOLD' and state.mode not in ('OFFBOARD','AUTO.LAND'):
                self.trial.cancel('pilot_left_autonomous_mode');phase=self.trial.phase
            if self.trial.phase=='FAILED_HOLD' and state.mode=='AUTO.LAND':self.request_mode('POSCTL',now)
            if not self.transition and phase=='FAILED_HOLD'and self.inputs.get('selected')and self.inputs['selected'].data!=self.mocap_topic and not self.dry and now-self.last_mux_call>1.:
                self.last_mux_call=now
                def restore_mocap():
                    try:self.mux_service(topic=self.mocap_topic)
                    except rospy.ServiceException:pass
                threading.Thread(target=restore_mocap,daemon=True).start()
            self.transition_enable_pub.publish(Bool(self.transition and not self.dry and state.armed and state.mode=='OFFBOARD' and phase in ('APPROACH','DESCEND') and healthy))
            control_key='routed' if self.transition else 'mocap'
            control_fresh=self.fresh(control_key,now)
            self.visible_pub.publish(Bool(healthy and estimation_ready and control_fresh
                and ((phase=='DESCEND' and good) or phase=='GT_DESCEND')))
            if self.aligned and self.pad is not None and control_fresh:
                P_B=np.linalg.inv(self.pad)@transform(self.inputs[control_key]);stamp=self.inputs[control_key].header.stamp
                self.body_pub.publish(self.output(P_B,stamp));self.camera_pub.publish(self.output(P_B@self.X,stamp))
            sp=None
            if phase=='PRESTREAM' and self.fresh('local',now):self.capture_hold()
            if self.auto_start and state.mode not in ('OFFBOARD','AUTO.LAND') and healthy:
                self.capture_hold();sp=self.hold_setpoint(self.hold)
            if phase not in ('IDLE','CANCELLED','COMPLETE','FAILED_COMPLETE')and self.hold is not None and self.fresh('local',now):sp=self.hold_setpoint(self.hold)
            if self.repeat and phase in self.repeat.PHASES and healthy and control_fresh:
                L_B=transform(self.inputs['local']);G_B=transform(self.inputs['mocap'])
                L_G=L_B[:3,:3]@G_B[:3,:3].T
                goal=np.array(self.repeat.goal)
                velocity=self.kp*(goal-G_B[:3,3])
                velocity=np.array(limit_horizontal_velocity(velocity,self.repeat.speed))
                velocity[2]=np.clip(velocity[2],-self.repeat.vertical_speed,self.repeat.vertical_speed)
                velocity=L_G@velocity
                velocity=np.array(limit_horizontal_velocity(velocity,self.repeat.speed))
                velocity[2]=np.clip(velocity[2],-self.repeat.vertical_speed,self.repeat.vertical_speed)
                if phase=='ARMING':velocity[:]=0.
                sp=self.hold_setpoint(L_B)
                sp.type_mask=PositionTarget.IGNORE_AFX|PositionTarget.IGNORE_AFY|PositionTarget.IGNORE_AFZ|PositionTarget.IGNORE_YAW
                sp.velocity.x,sp.velocity.y,sp.velocity.z=velocity
                sp.yaw_rate=float(np.clip(math.atan2(math.sin(self.heading-yaw(L_B)),math.cos(self.heading-yaw(L_B))),-self.yaw_rate,self.yaw_rate))
                self.capture_hold()
            if phase in ('APPROACH','DESCEND','GT_DESCEND')and healthy and control_fresh:
                L_B=transform(self.inputs['local']);G_B=transform(self.inputs[control_key]);L_G=L_B[:3,:3]@G_B[:3,:3].T
                velocity=None;rate=0.
                if phase=='APPROACH':
                    if self.trajectory_set is not None:
                        # Stay at the selected camera-path start, not pad center,
                        # while original marker/mocap alignment dwell completes.
                        v=self.kp*(self.trajectory_goal-G_B[:3,3])
                        v=np.array(limit_horizontal_velocity(v,self.repeat.speed))
                        v[2]=np.clip(v[2],-self.repeat.vertical_speed,self.repeat.vertical_speed)
                        velocity=L_G@v
                    else:
                        goal=self.target  # known global pad XY; pre-OFFBOARD detections do not redirect approach
                        v=self.kp*(goal-G_B[:2,3]);v*=min(1.,self.speed/max(np.linalg.norm(v),1e-9));velocity=L_G@np.r_[v,0.];velocity[2]=0.
                        if good or (self.transition and self.router_status.get('source')=='marker'):velocity[:]=0.  # hold altitude/position while validating marker dwell
                    target_yaw=self.heading
                    error=math.atan2(math.sin(target_yaw-yaw(L_B)),math.cos(target_yaw-yaw(L_B)));rate=np.clip(error,-self.yaw_rate,self.yaw_rate)
                elif (good or phase=='GT_DESCEND') and estimation_ready and self.fresh('command',now,.1):
                    c=self.inputs['command'];v=c.twist.linear
                    if c.header.frame_id=='landing_pad':velocity=L_G@self.pad[:3,:3]@np.array([v.x,v.y,v.z]);rate=float(c.twist.angular.z)
                if velocity is not None and np.isfinite(velocity).all()and math.isfinite(rate):
                    # Position=current pose + velocity feed-forward. The shared
                    # safety IdleHold therefore latches the CURRENT position,
                    # never a stale entry point after a velocity-controller crash.
                    sp=self.hold_setpoint(L_B)
                    sp.type_mask=PositionTarget.IGNORE_AFX|PositionTarget.IGNORE_AFY|PositionTarget.IGNORE_AFZ|PositionTarget.IGNORE_YAW
                    speed_limit=self.speed if phase=='APPROACH' else self.landing_speed
                    sp.velocity.x,sp.velocity.y,sp.velocity.z=limit_horizontal_velocity(velocity,speed_limit);sp.yaw_rate=float(np.clip(rate,-self.yaw_rate,self.yaw_rate))
                    if phase=='APPROACH' and self.trajectory_set is None:sp.position.z=self.altitude
                    self.capture_hold()
            if phase=='AUTO_LAND':
                if state.mode=='OFFBOARD':self.request_mode('AUTO.LAND',now)
                # Keep the admission stream alive until PX4 acknowledges LAND.
            if sp is not None:
                sp.header.stamp=rospy.Time.now();self.pub.publish(sp)
                if self.live is not None:self.live.publish(sp)
            mission_state={'FAILED_HOLD':'hold_failed','GT_RECOVERY':'active','GT_DESCEND':'active','GT_CUT_WAIT':'terminating','FAILED_COMPLETE':'complete','AUTO_LAND':'landing','COMPLETE':'complete','CUT_WAIT':'terminating'}.get(self.trial.phase,'active'if self.trial.phase in ('ARMING','TAKEOFF','CENTER','RANDOM_POSITION','INITIAL_POSITION','APPROACH','DESCEND')else 'idle')
            entry_ready=bool(self.entry_ready and healthy and data_ready and self.standby_since is not None and now-self.standby_since>=1.)
            if self.trajectory_set is not None:
                entry_ready=bool(entry_ready and self.trajectory_progress.available and self.aligned and self.pad is not None
                    and self.fresh('trajectory_ready',now,1.5)
                    and self.trajectory_ack==self.trajectory_set.selection(self.trajectory_progress.index))
            mission_hint=dict(state=mission_state,dry_run=self.dry,source='landing_trial')
            if self.repeat:
                # Optional, stack-neutral LED contract. Indicate readiness only
                # on the ground in pilot position mode, before the arm edge.
                position=transform(self.inputs['mocap'])[:3,3] if self.fresh('mocap',now) else None
                valid_origin=position is not None and all(math.isfinite(v) for v in position) and 0<=position[2]<=self.repeat.ground_max and all(lo<=v<=hi for v,(lo,hi) in zip(position[:2],self.repeat.bounds))
                entry_ready=bool(entry_ready and state.mode=='POSCTL' and not state.armed and landed and self.fresh('velocity',now) and valid_origin)
                mission_hint.update(ground_start_enabled=not (self.trajectory_progress is not None and len(self.trajectory_progress.finished)==len(self.trajectory_set.entries)),offboard_entry_ready=entry_ready)
            self.mission_pub.publish(json.dumps(mission_hint))
            self.status.publish(json.dumps(dict(phase=self.trial.phase,reason=self.trial.reason,failure_reason=self.trial.failure_reason,failed_trial_gt_recovery=self.gt_recovery,trajectory_initial_alignment_timeout_s=self.alignment_timeout,gt_reference_resume_ack=self.gt_resume_ack,dry_run=self.dry,
                estimation_source=self.router_status.get('source','unknown') if self.transition else 'optitrack',estimation_transition=self.transition,landing_finish_mode=self.trial.finish_mode,termination_height_source=self.trial.termination_height_source,termination_height_m=self.trial.contact_height_m,mocap_camera_height_m=mocap_height,force_disarm_ack=self.cut_result,offboard_entry_ready=entry_ready,healthy=bool(healthy),marker_qualified=bool(good),marker_height_m=height,
                marker_confirm_s=self.trial.marker_confirm_s,marker_loss_s=self.trial.marker_loss_s,
                repeat_test=self.repeat.status() if self.repeat else None,data_sync_ready=data_ready,
                trajectory_set=self.trajectory_progress.status() if self.trajectory_progress else None,
                trajectory_controller_ready=self.trajectory_ack if self.trajectory_set else None,
                data_sync_reason=self.sync_reason if self.repeat else 'not_required',
                pad_alignment_ready=self.aligned,command_frame='MAVROS local ENU',live_output=self.normal_topic if self.live else None)))


if __name__=='__main__':
    LandingTrial();rospy.spin()
