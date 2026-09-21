#!/usr/bin/env python3
"""Hardware landing sequence through the existing flight-safety NORMAL lane.

Never arms, enters OFFBOARD or publishes vision. Force-disarm is an explicit
stack policy, requested through the common safety authority. Pilot starts
OFFBOARD. Default dry-run writes shadow setpoints only. Current experiment
supports OptiTrack-only and the explicitly enabled, trial-gated pose router.
"""
import json
import math
import threading
import time
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped
from mavros_msgs.msg import State, ExtendedState, PositionTarget
from mavros_msgs.srv import SetMode, ParamGet
from std_msgs.msg import Bool, String, Int32MultiArray
from std_srvs.srv import Trigger, TriggerResponse
from topic_tools.srv import MuxSelect
from flight_safety.msg import FlightState
from aruco_landing.landing_trial import Trial, limit_horizontal_velocity
from aruco_landing.physical_pad import SessionAlignment, valid_transform
from aruco_landing.pose_alignment import pose_matrix, matrix_pose, quaternion_distance_deg
from aruco_landing.yaw_control import yaw_feedback


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
        self.transition=bool(rospy.get_param('~estimation_transition',False))
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
        self.trial=Trial(float(rospy.get_param('~marker_confirm_s',1.)),float(rospy.get_param('~marker_loss_s',.5)))
        self.trial.finish_mode=rospy.get_param('~landing_finish_mode','auto_land')
        if self.trial.finish_mode not in ('auto_land','force_disarm'):raise ValueError('invalid landing_finish_mode')
        self.cut_pending=False;self.cut_started=None;self.cut_result=None
        self.termination_service=rospy.ServiceProxy('/flight_safety_response/request_termination',Trigger)
        self.speed=float(rospy.get_param('~approach_speed_mps',.5));self.kp=float(rospy.get_param('~approach_kp',.8))
        self.landing_speed=float(rospy.get_param('~landing_horizontal_speed_mps',.5))
        self.yaw_rate=float(rospy.get_param('~yaw_rate_limit_rad_s',.35))
        self.target=np.array(rospy.get_param('~initial_target_global_xy',[0.,0.]),float)
        if self.target.shape!=(2,)or not np.isfinite(self.target).all():raise ValueError('invalid initial target')
        if not all(math.isfinite(v)and v>0 for v in (self.speed,self.landing_speed,self.kp,self.yaw_rate)):raise ValueError('invalid trial limits')
        with open(rospy.get_param('~extrinsic_file'))as f:mount=yaml.safe_load(f)
        self.X=np.array(mount['matrix_row_major']).reshape(4,4)
        if mount['validation_result']!='PASS'or not valid_transform(self.X):raise ValueError('invalid mount')
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
        self.param_service=rospy.ServiceProxy('/mavros/param/get',ParamGet)
        self.mux_service=rospy.ServiceProxy('/vision_pose_mux/select',MuxSelect)
        topics=[('mocap',self.mocap_topic,PoseStamped),('local','/mavros/local_position/pose',PoseStamped),
            ('state','/mavros/state',State),('extended','/mavros/extended_state',ExtendedState),
            ('safety','/flight_safety/state',FlightState),('vision','/mavros/vision_pose/pose',PoseStamped),
            ('selected','/vision_pose_mux/selected',String),('pad','/landing/pad_pose_global',PoseStamped),
            ('aligned','/landing/alignment/ready',Bool),('visible','/landing/target_visible',Bool),
            ('inliers','/landing/estimator/inlier_ids',Int32MultiArray),
            ('marker_body','/landing/vehicle_pose_pad',PoseWithCovarianceStamped),
            ('marker_camera','/landing/camera_pose_pad',PoseWithCovarianceStamped),
            ('command','/landing/trial/landing_cmd_pad',TwistStamped),
            ('routed',self.routed_topic,PoseStamped),
            ('router','/landing/pose_transition/status',String)]
        self.subscribers=[rospy.Subscriber(topic,kind,self.receive,key,queue_size=10,tcp_nodelay=True)for key,topic,kind in topics]
        rospy.Service('~start',Trigger,self.start);rospy.Service('~reset',Trigger,self.reset);rospy.Service('~abort',Trigger,self.abort)
        rospy.Timer(rospy.Duration(1/60.),self.tick)

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

    def receive(self,m,key):
        with self.lock:
            now=time.monotonic()
            if key in ('mocap','local','vision','pad','marker_body','marker_camera','routed'):
                try:
                    if not valid_transform(transform(m)):return
                except (ValueError,TypeError):return
            if key=='mocap'and m.header.frame_id!=self.global_frame:return
            self.inputs[key]=m;self.receipts[key]=now
            if key=='router':
                try:self.router_status=json.loads(m.data)
                except (ValueError,TypeError):self.router_status={}
            if key=='routed':
                self.routed_by_stamp[m.header.stamp.to_nsec()]=transform(m)
                if len(self.routed_by_stamp)>500:self.routed_by_stamp.pop(next(iter(self.routed_by_stamp)))
            if key=='pad':
                T=transform(m)
                if m.header.frame_id==self.global_frame and valid_transform(T):
                    if self.pad is not None and not np.allclose(self.pad,T,atol=1e-5)and self.trial.phase in ('APPROACH','DESCEND','AUTO_LAND'):
                        self.trial.fail('session_pad_reference_changed');self.capture_hold()
                    self.pad=T
            elif key=='aligned':
                if self.aligned and not m.data and self.trial.phase in ('APPROACH','DESCEND','AUTO_LAND'):
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

    def request_cut(self):
        if self.dry or self.cut_pending:return
        self.cut_pending=True
        def call():
            try:
                with self.lock:
                    now=time.monotonic();state=self.inputs.get('state')
                    good,height,_=self.marker(now)
                    h=float((np.linalg.inv(self.pad)@transform(self.inputs['mocap'])@self.X)[2,3]) if self.pad is not None and self.fresh('mocap',now) else None
                    allowed=self.trial.phase=='CUT_WAIT' and state and state.armed and state.mode=='OFFBOARD' and self.health(now) and good and height is not None and 0<=height<=.2 and h is not None and 0<=h<=.25
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
                    allowed=bool(state and ((mode=='AUTO.LAND'and self.trial.phase=='AUTO_LAND'and state.mode=='OFFBOARD')
                        or (mode=='POSCTL'and self.trial.phase=='FAILED_HOLD'and state.mode=='AUTO.LAND')))
                if allowed:self.mode_service(base_mode=0,custom_mode=mode)
            except rospy.ServiceException as error:rospy.logwarn('mode request failed: %s',error)
            finally:self.mode_pending=False
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
            if self.auto_start:
                standby=state.mode not in ('OFFBOARD','AUTO.LAND') and self.trial.phase in ('IDLE','CANCELLED','COMPLETE')
                if standby and healthy:
                    if self.standby_since is None:self.standby_since=now
                    self.prepare_standby(now)
                elif state.mode!='OFFBOARD':self.standby_since=None
                entered=state.mode=='OFFBOARD' and self.last_fcu_mode not in (None,'OFFBOARD')
                if entered and self.trial.phase not in ('AUTO_LAND','CUT_WAIT'):
                    ext=self.inputs.get('extended')
                    airborne=self.fresh('extended',now,2.) and ext.landed_state==ExtendedState.LANDED_STATE_IN_AIR
                    allowed=healthy and state.armed and airborne and self.entry_ready and self.standby_since is not None and now-self.standby_since>=1.
                    self.entry_ready=False;self.standby_since=None;self.marker_cache=None
                    if allowed and self.trial.prepare(now-1.):
                        self.entry_time=now;self.capture_hold()
                    else:
                        self.trial.fail('OFFBOARD_entry_not_ready');self.capture_hold()
                self.last_fcu_mode=state.mode
            good,height,marker_time=self.marker(now)
            if self.auto_start and (self.entry_time is None or state.mode not in ('OFFBOARD','AUTO.LAND') or marker_time is None or marker_time<self.entry_time):
                good,height,marker_time=False,None,None
            ext=self.inputs.get('extended');landed=bool(self.fresh('state',now,2.)and state.connected and self.fresh('extended',now,2.)and ext.landed_state==ExtendedState.LANDED_STATE_ON_GROUND)
            mocap_height=None
            if self.aligned and self.pad is not None and self.fresh('mocap',now):
                mocap_height=float((np.linalg.inv(self.pad)@transform(self.inputs['mocap'])@self.X)[2,3])
            if self.transition and self.router_status.get('fallback_latched') and self.trial.phase in ('APPROACH','DESCEND','AUTO_LAND'):
                self.trial.fail('marker_source_lost_trial_failed');self.capture_hold()
            estimation_ready=not self.transition or (self.router_status.get('source')=='marker' and self.router_status.get('last_output_source')=='marker' and self.router_status.get('switch_check',{}).get('consistent',False) and self.fresh('routed',now) and healthy)
            previous=self.trial.phase
            phase=self.trial.step(now,offboard=state.mode=='OFFBOARD',armed=state.armed,healthy=healthy,marker_good=good,marker_height=height,marker_time=marker_time,mocap_height=mocap_height,estimation_ready=estimation_ready,landed=landed,auto_land=state.mode=='AUTO.LAND')
            if previous!=phase:
                self.capture_hold()
                if phase=='APPROACH'and self.hold is not None:self.altitude=self.hold[2,3];self.heading=yaw(self.hold)
                if phase=='AUTO_LAND':self.handoff_started=now
                if phase=='CUT_WAIT':
                    self.cut_started=now;self.cut_result=None;self.request_cut()
            if phase=='CUT_WAIT' and state.armed and (self.cut_result is False or now-self.cut_started>2.):
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
            self.visible_pub.publish(Bool(phase=='DESCEND'and good and healthy and estimation_ready and control_fresh))
            if self.aligned and self.pad is not None and control_fresh:
                P_B=np.linalg.inv(self.pad)@transform(self.inputs[control_key]);stamp=self.inputs[control_key].header.stamp
                self.body_pub.publish(self.output(P_B,stamp));self.camera_pub.publish(self.output(P_B@self.X,stamp))
            sp=None
            if phase=='PRESTREAM' and self.fresh('local',now):self.capture_hold()
            if self.auto_start and state.mode not in ('OFFBOARD','AUTO.LAND') and healthy:
                self.capture_hold();sp=self.hold_setpoint(self.hold)
            if phase not in ('IDLE','CANCELLED','COMPLETE')and self.hold is not None and self.fresh('local',now):sp=self.hold_setpoint(self.hold)
            if phase in ('APPROACH','DESCEND')and healthy and control_fresh:
                L_B=transform(self.inputs['local']);G_B=transform(self.inputs[control_key]);L_G=L_B[:3,:3]@G_B[:3,:3].T
                velocity=None;rate=0.
                if phase=='APPROACH':
                    goal=self.target  # known global pad XY; pre-OFFBOARD detections do not redirect approach
                    v=self.kp*(goal-G_B[:2,3]);v*=min(1.,self.speed/max(np.linalg.norm(v),1e-9));velocity=L_G@np.r_[v,0.];velocity[2]=0.
                    if good or (self.transition and self.router_status.get('source')=='marker'):velocity[:]=0.  # hold altitude/position while validating marker dwell
                    target_yaw=self.heading
                    error=math.atan2(math.sin(target_yaw-yaw(L_B)),math.cos(target_yaw-yaw(L_B)));rate=np.clip(error,-self.yaw_rate,self.yaw_rate)
                elif good and estimation_ready and self.fresh('command',now,.1):
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
                    if phase=='APPROACH':sp.position.z=self.altitude
                    self.capture_hold()
            if phase=='AUTO_LAND':
                if state.mode=='OFFBOARD':self.request_mode('AUTO.LAND',now)
                # Keep the admission stream alive until PX4 acknowledges LAND.
            if sp is not None:
                sp.header.stamp=rospy.Time.now();self.pub.publish(sp)
                if self.live is not None:self.live.publish(sp)
            mission_state={'FAILED_HOLD':'hold_failed','AUTO_LAND':'landing','COMPLETE':'complete','CUT_WAIT':'terminating'}.get(self.trial.phase,'active'if self.trial.phase in ('APPROACH','DESCEND')else 'idle')
            self.mission_pub.publish(json.dumps(dict(state=mission_state,dry_run=self.dry,source='landing_trial')))
            self.status.publish(json.dumps(dict(phase=self.trial.phase,reason=self.trial.reason,dry_run=self.dry,
                estimation_source=self.router_status.get('source','unknown') if self.transition else 'optitrack',estimation_transition=self.transition,landing_finish_mode=self.trial.finish_mode,mocap_camera_height_m=mocap_height,force_disarm_ack=self.cut_result,offboard_entry_ready=bool(self.entry_ready and healthy and self.standby_since is not None and now-self.standby_since>=1.),healthy=bool(healthy),marker_qualified=bool(good),marker_height_m=height,
                marker_confirm_s=self.trial.marker_confirm_s,marker_loss_s=self.trial.marker_loss_s,
                pad_alignment_ready=self.aligned,command_frame='MAVROS local ENU',live_output=self.normal_topic if self.live else None)))


if __name__=='__main__':
    LandingTrial();rospy.spin()
