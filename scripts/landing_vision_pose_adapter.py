#!/usr/bin/env python3
"""Route OptiTrack or already aligned marker body poses to one output."""
import json
import threading
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from std_msgs.msg import Bool, Int32MultiArray, String
from std_srvs.srv import SetBool, SetBoolResponse
from aruco_landing.pose_alignment import pose_matrix
from aruco_landing.pose_transition import PoseTransition


class LandingVisionPoseAdapter:
    def __init__(self):
        rospy.init_node('landing_vision_pose_adapter')
        self.lock = threading.RLock()
        self.frame = rospy.get_param('~global_frame','odom')
        self.auto = rospy.get_param('~auto_switch',False)
        self.require_trial = rospy.get_param('~require_trial_enable',False)
        self.trial_enable = (False, -float('inf'))
        self.fcu = None
        self.fcu_receipt = -float('inf')
        self.output_topic = rospy.get_param('~output_topic','/landing/vision_pose_selected')
        self.router = PoseTransition(
            allow=rospy.get_param('~allow_marker_switch',False),
            reject_inconsistent_marker=rospy.get_param('~reject_inconsistent_marker',False),
            latch_fallback=self.require_trial,
            max_age=rospy.get_param('~max_source_age_s',.2),
            pair_dt=rospy.get_param('~max_pair_dt_s',.035),
            max_position=rospy.get_param('~max_switch_translation_jump_m',.15),
            max_angle=rospy.get_param('~max_switch_rotation_jump_deg',12.),
            stable_duration=rospy.get_param('~stable_duration_s',1.),
            min_samples=rospy.get_param('~stable_min_samples',30),
            fallback_timeout=(float(rospy.get_param('~marker_loss_timeout_s',.5))
                if rospy.get_param('~auto_fallback_to_optitrack',False) else None))
        if self.require_trial:
            self.router.set_enabled(False)
            rospy.Subscriber('/landing/pose_transition/enable',Bool,self.enable,queue_size=1)
            rospy.Subscriber('/mavros/state',State,self.state,queue_size=1)
        self.min_inliers=rospy.get_param('~min_marker_inliers',1)
        # Refuse a second output producer (e.g. flight-safety's vision_pose_mux).
        publishers, _, _ = rospy.get_master().getSystemState()[2]
        existing = dict(publishers).get(rospy.resolve_name(self.output_topic),[])
        if existing:
            raise RuntimeError('output already has publishers: '+str(existing))
        self.output=rospy.Publisher(self.output_topic,PoseStamped,queue_size=2)
        self.source_pub=rospy.Publisher('/landing/vision_pose_source',String,queue_size=1,latch=True)
        self.ready_pub=rospy.Publisher('/landing/pose_transition/ready',Bool,queue_size=1,latch=True)
        self.status_pub=rospy.Publisher('/landing/pose_transition/status',String,queue_size=1,latch=True)
        self.source_pub.publish(self.router.source)
        self.published={'optitrack':0,'marker':0}
        self.last_output_source=None
        self.last_switch=None
        rospy.Subscriber(rospy.get_param('~mocap_topic','/vrpn_client_node/pure/pose'),PoseStamped,self.pose,'optitrack',queue_size=100,tcp_nodelay=True)
        rospy.Subscriber(rospy.get_param('~marker_pose_topic','/landing/vision_pose_marker'),PoseStamped,self.pose,'marker',queue_size=20,tcp_nodelay=True)
        rospy.Subscriber('/landing/alignment/ready',Bool,self.quality,'aligned',queue_size=10)
        rospy.Subscriber('/landing/target_visible',Bool,self.quality,'visible',queue_size=10)
        rospy.Subscriber('/landing/estimator/inlier_ids',Int32MultiArray,self.quality,'inliers',queue_size=10)
        rospy.Service('/landing/pose_transition/select_marker',SetBool,self.select)
        rospy.Timer(rospy.Duration(.02),self.status)
        rospy.loginfo('pose router: aligned marker input; output=%s; switching allowed=%s auto=%s',self.output_topic,self.router.allow,self.auto)

    def enable(self, msg):
        with self.lock:
            self.trial_enable = (bool(msg.data), rospy.get_time())

    def state(self, msg):
        with self.lock:
            self.fcu, self.fcu_receipt = msg, rospy.get_time()

    def update_authorization(self, now):
        if not self.require_trial:
            return
        fresh_state = self.fcu is not None and 0 <= now-self.fcu_receipt <= 2.
        autonomous = fresh_state and self.fcu.connected and self.fcu.armed and self.fcu.mode == 'OFFBOARD'
        if fresh_state and (not self.fcu.armed or self.fcu.mode not in ('OFFBOARD', 'AUTO.LAND')):
            self.router.fallback_latched = False
        enabled = autonomous and self.trial_enable[0] and 0 <= now-self.trial_enable[1] <= .2
        was_enabled = self.router.enabled
        self.router.set_enabled(bool(enabled))
        if not enabled and self.router.source == 'marker':
            # Trial abort / vanished permission must restore the trusted input.
            # Keep the episode latched until the pilot leaves autonomous mode.
            if self.router.return_to_mocap(now, latch=bool(autonomous and was_enabled)):
                self.last_switch = {'time':now,'source':'optitrack','check':{'reason':'trial_disabled'}}
                self.source_pub.publish('optitrack')

    def quality(self,msg,key):
        value=len(msg.data)>=self.min_inliers if key=='inliers' else bool(msg.data)
        with self.lock:
            self.router.set_quality(key,value,rospy.get_time())

    def pose(self,msg,source):
        p=msg.pose; q=[p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w]
        values=[p.position.x,p.position.y,p.position.z]+q
        if msg.header.frame_id!=self.frame or msg.header.stamp.is_zero() or not np.isfinite(values).all() or abs(np.linalg.norm(q)-1)>.01:
            return
        T=pose_matrix(values[:3],q)
        with self.lock:
            self.update_authorization(rospy.get_time())
            if self.router.ingest(source,msg.header.stamp.to_sec(),T,rospy.get_time()):
                self.output.publish(msg) # retain original measurement stamp and pose
                self.published[source]+=1
                self.last_output_source=source

    def select(self,request):
        with self.lock:
            self.update_authorization(rospy.get_time())
            ok,check=self.router.select(request.data,rospy.get_time())
            if ok:
                self.last_switch={'time':rospy.get_time(),'source':self.router.source,'check':check}
                self.source_pub.publish(self.router.source)
            return SetBoolResponse(ok,json.dumps(check))

    def status(self,_event):
        with self.lock:
            self.update_authorization(rospy.get_time())
            fallback=self.router.fallback(rospy.get_time())
            if fallback:
                self.last_switch={'time':rospy.get_time(),'source':self.router.source,'check':fallback}
                self.source_pub.publish(self.router.source)
            check=self.router.check(rospy.get_time())
            if self.auto and self.router.source=='optitrack' and check['ready']:
                self.select(type('Request',(),{'data':True})())
            self.ready_pub.publish(check['ready'])
            self.status_pub.publish(json.dumps({'source':self.router.source,'output_topic':self.output_topic,
                'last_output_source':self.last_output_source,'trial_enabled':self.router.enabled,'fallback_latched':self.router.fallback_latched,
                'switch_check':check,'last_switch':self.last_switch,'published':self.published,
                'marker_loss_policy':('optitrack_after_timeout' if self.router.fallback_timeout is not None else 'withhold_output_no_automatic_fallback')}))


if __name__=='__main__':
    LandingVisionPoseAdapter()
    rospy.spin()
