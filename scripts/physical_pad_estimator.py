#!/usr/bin/env python3
"""Measured-pad detector, body pose and online-only OptiTrack alignment."""
import json
import threading
import time
from collections import deque

import cv2
import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo, Image, CompressedImage
from std_msgs.msg import Bool, String, Int32MultiArray
from std_srvs.srv import Trigger, TriggerResponse

from aruco_landing.physical_pad import PhysicalPadDetector, SessionAlignment, inverse, relative_covariance, valid_transform, center_crop
from aruco_landing.pose_alignment import pose_matrix, matrix_pose


def message_transform(pose):
    return pose_matrix([pose.position.x, pose.position.y, pose.position.z],
                       [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])


def set_pose(message, T):
    p, q = matrix_pose(T)
    message.position.x, message.position.y, message.position.z = map(float, p)
    message.orientation.x, message.orientation.y, message.orientation.z, message.orientation.w = map(float, q)


class PhysicalPadEstimator:
    def __init__(self):
        rospy.init_node('physical_pad_estimator')
        self.lock = threading.RLock()
        self.display_lock = threading.Lock()
        self.image_event = threading.Event()
        self.latest_image = None
        self.latest_display = None
        self.camera_info = None
        self.last_stamp = None
        self.last_valid_receipt = None
        self.last_pad_publish = 0.
        self.last_error = 'waiting_for_image_and_camera_info'
        self.processed = self.accepted = self.received = self.debug_count = 0
        self.durations = deque(maxlen=240)
        self.completed = deque(maxlen=240)
        self.pad_frame = rospy.get_param('~pad_frame', 'physical_landing_pad')
        self.global_frame = rospy.get_param('~global_frame', 'odom')
        self.max_age = float(rospy.get_param('~max_image_age_s', .20))
        self.estimate_hz = float(rospy.get_param('~estimate_rate_hz', 60.))
        self.debug_hz = float(rospy.get_param('~detected_image_rate_hz', 10.))
        self.processing_width = int(rospy.get_param('~processing_width', 720))
        self.processing_height = int(rospy.get_param('~processing_height', 720))
        if not 0 < self.processing_width <= 1280 or not 0 < self.processing_height <= 720:
            raise ValueError('invalid processing crop dimensions')
        self.jpeg_quality = int(rospy.get_param('~jpeg_quality', 80))
        if not 0 < self.debug_hz <= 60 or not 0 < self.estimate_hz <= 60:
            raise ValueError('image/estimate rates must be in (0,60]')
        with open(rospy.get_param('~pad_configuration')) as f:
            manifest = yaml.safe_load(f)
        with open(rospy.get_param('~body_camera_calibration')) as f:
            mount = yaml.safe_load(f)
        if mount['validation_result'] != 'PASS' or mount['parent_frame'] != 'base_link':
            raise ValueError('requires a validated base_link-from-camera calibration')
        self.camera_frame = mount['child_frame']
        self.body_from_camera = np.array(mount['matrix_row_major']).reshape(4,4)
        if not valid_transform(self.body_from_camera):
            raise ValueError('invalid body-camera transform')
        self.camera_from_body = inverse(self.body_from_camera)
        with open(rospy.get_param('~time_alignment')) as f:
            timing = yaml.safe_load(f)
        self.time_offset = float(rospy.get_param('~image_time_offset_s', timing['image_to_body_offset_s']))
        if not np.isfinite(self.time_offset) or abs(self.time_offset) > .2:
            raise ValueError('invalid image time offset')
        cv2.setNumThreads(int(rospy.get_param('~opencv_threads', 2)))
        self.detector = PhysicalPadDetector(manifest, min_markers=int(rospy.get_param('~min_marker_inliers', 1)))
        self.detector.params.adaptiveThreshWinSizeMin = int(rospy.get_param('~adaptive_threshold_min', 7))
        self.detector.params.adaptiveThreshWinSizeMax = int(rospy.get_param('~adaptive_threshold_max', 27))
        self.detector.params.adaptiveThreshWinSizeStep = int(rospy.get_param('~adaptive_threshold_step', 20))
        # Newer OpenCV copies parameters into ArucoDetector.
        if self.detector.detector:
            self.detector.detector.setDetectorParameters(self.detector.params)
        self.alignment = SessionAlignment(
            min_samples=int(rospy.get_param('~alignment_min_samples', 60)),
            min_duration_s=float(rospy.get_param('~alignment_min_duration_s', 2.)),
            window_size=240, max_translation_std_m=.03, max_rotation_std_deg=2.)
        self.target_pub = rospy.Publisher('/landing/target_pose_camera', PoseWithCovarianceStamped, queue_size=1)
        self.body_pub = rospy.Publisher('/landing/vehicle_pose_pad', PoseWithCovarianceStamped, queue_size=1)
        self.camera_pub = rospy.Publisher('/landing/camera_pose_pad', PoseWithCovarianceStamped, queue_size=1)
        self.global_body_pub = rospy.Publisher('/landing/vision_pose_marker', PoseStamped, queue_size=1)
        self.global_pad_pub = rospy.Publisher('/landing/pad_pose_global', PoseStamped, queue_size=1)
        self.ready_pub = rospy.Publisher('/landing/alignment/ready', Bool, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher('/landing/estimator/status', String, queue_size=1, latch=True)
        self.visible_pub = rospy.Publisher('/landing/target_visible', Bool, queue_size=1)
        self.ids_pub = rospy.Publisher('/landing/markers/ids', Int32MultiArray, queue_size=1)
        self.inliers_pub = rospy.Publisher('/landing/estimator/inlier_ids', Int32MultiArray, queue_size=1)
        self.debug_pub = rospy.Publisher('/landing/debug/image', Image, queue_size=1)
        self.jpeg_pub = rospy.Publisher('/landing/debug/image/compressed', CompressedImage, queue_size=1)
        self.ready_pub.publish(False)
        rospy.Service('/landing/alignment/reset', Trigger, self.reset)
        rospy.Subscriber(rospy.get_param('~mocap_topic','/vrpn_client_node/pure/pose'), PoseStamped, self.mocap_callback, queue_size=300, tcp_nodelay=True)
        rospy.Subscriber(rospy.get_param('~camera_info_topic','/landing/camera/camera_info'), CameraInfo, self.info_callback, queue_size=1)
        rospy.Subscriber(rospy.get_param('~image_topic','/landing/camera/image_raw'), Image, self.image_callback, queue_size=1, buff_size=4*1024*1024, tcp_nodelay=True)
        self.worker = threading.Thread(target=self.process_loop, daemon=True)
        self.worker.start()
        self.display_worker = threading.Thread(target=self.display_loop, daemon=True)
        self.display_worker.start()
        self.status_timer = rospy.Timer(rospy.Duration(.5), self.publish_status)
        rospy.loginfo('physical pad estimator: fixed measured mount, session-only alignment, inference %.1f Hz, display %.1f Hz', self.estimate_hz, self.debug_hz)

    def reset(self, _request):
        with self.lock:
            self.alignment.reset()
            self.ready_pub.publish(False)
        return TriggerResponse(True, 'session alignment cleared; learning the current pad placement')

    def mocap_callback(self, msg):
        if msg.header.frame_id != self.global_frame or msg.header.stamp.is_zero():
            return
        stamp = msg.header.stamp.to_sec()
        if abs(rospy.get_time()-stamp) > .25:
            return
        try:
            T = message_transform(msg.pose)
            with self.lock:
                self.alignment.add_mocap(stamp, T)
        except (ValueError, np.linalg.LinAlgError):
            pass

    def info_callback(self, msg):
        if (msg.width, msg.height) != (1280,720) or msg.header.frame_id != self.camera_frame:
            return
        if msg.distortion_model not in ('plumb_bob','rational_polynomial') or msg.K[0] <= 0:
            return
        K, D = np.array(msg.K).reshape(3,3), np.array(msg.D)
        if np.isfinite(K).all() and np.isfinite(D).all():
            self.camera_info = (K, D)

    def image_callback(self, msg):
        self.received += 1
        with self.lock:
            self.latest_image = msg
        self.image_event.set()

    def covariance_pose(self, publisher, T, covariance, stamp, frame):
        output = PoseWithCovarianceStamped()
        output.header.stamp, output.header.frame_id = stamp, frame
        set_pose(output.pose.pose, T)
        output.pose.covariance = covariance.ravel().tolist()
        publisher.publish(output)

    def global_pose(self, publisher, T, stamp):
        output = PoseStamped()
        output.header.stamp, output.header.frame_id = stamp, self.global_frame
        set_pose(output.pose, T)
        publisher.publish(output)

    def process_loop(self):
        next_time = time.monotonic()
        while not rospy.is_shutdown():
            self.image_event.wait(.1)
            if rospy.is_shutdown():
                break
            delay = next_time-time.monotonic()
            if delay > 0:
                time.sleep(delay)
            with self.lock:
                msg, self.latest_image = self.latest_image, None
                self.image_event.clear()
            if msg is None:
                continue
            start = time.monotonic()
            processed_before = self.processed
            try:
                self.process(msg)
            except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
                self.last_error = str(error)
                self.visible_pub.publish(False)
                self.inliers_pub.publish(Int32MultiArray(data=[]))
                rospy.logwarn_throttle(2., 'physical pad pose withheld: %s', error)
            if self.processed > processed_before:
                self.durations.append((time.monotonic()-start)*1000)
                self.completed.append(time.monotonic())
            next_time = max(next_time + 1/self.estimate_hz, start)

    def process(self, msg):
        stamp_s = msg.header.stamp.to_sec()+self.time_offset
        if self.last_stamp is not None and stamp_s < self.last_stamp - 1.:
            self.reset(None)  # bag seek / clock reset cannot reuse session alignment
            self.last_stamp = None
        age = rospy.get_time()-stamp_s
        if msg.header.stamp.is_zero() or age > self.max_age or age < -.02:
            raise ValueError('stale or future image')
        if self.last_stamp is not None and stamp_s <= self.last_stamp:
            return
        self.last_stamp = stamp_s
        if self.camera_info is None:
            raise ValueError('waiting for valid raw CameraInfo')
        if msg.header.frame_id != self.camera_frame or (msg.width,msg.height)!=(1280,720):
            raise ValueError('image differs from calibrated camera frame/resolution')
        channels = 1 if msg.encoding == 'mono8' else 3
        if msg.encoding not in ('rgb8','bgr8','mono8'):
            raise ValueError('unsupported raw image encoding')
        array = np.frombuffer(msg.data, np.uint8).reshape(msg.height,msg.step)[:,:msg.width*channels]
        array = array.reshape(msg.height,msg.width,channels)
        gray = array[:,:,0] if channels==1 else cv2.cvtColor(array,cv2.COLOR_RGB2GRAY if msg.encoding=='rgb8' else cv2.COLOR_BGR2GRAY)
        K, D = self.camera_info
        gray, cropped_K = center_crop(gray, K, self.processing_width, self.processing_height)
        result, corners, ids = self.detector.detect(gray, cropped_K, D)
        if rospy.get_time()-stamp_s > self.max_age:
            raise ValueError('image expired during processing')
        self.processed += 1
        stamp = rospy.Time.from_sec(stamp_s)
        self.ids_pub.publish(Int32MultiArray(data=ids))
        valid = result is not None
        self.visible_pub.publish(valid)
        self.inliers_pub.publish(Int32MultiArray(data=[] if not valid else result['inlier_ids']))
        if self.debug_pub.get_num_connections() or self.jpeg_pub.get_num_connections():
            with self.display_lock:
                self.latest_display = (msg, corners, ids, [] if not valid else result['inlier_ids'])
        if not valid:
            self.last_error = 'insufficient_valid_markers'
            return
        self.accepted += 1
        self.last_valid_receipt = rospy.get_time()
        self.last_error = ''
        C_P, covariance = result['camera_from_pad'], result['covariance']
        P_C = inverse(C_P)
        P_B = P_C @ self.camera_from_body
        if self.target_pub.get_num_connections():
            self.covariance_pose(self.target_pub,C_P,covariance,stamp,self.camera_frame)
        if self.camera_pub.get_num_connections():
            self.covariance_pose(self.camera_pub,P_C,relative_covariance(C_P,np.eye(4),covariance),stamp,self.pad_frame)
        self.covariance_pose(self.body_pub,P_B,relative_covariance(C_P,self.camera_from_body,covariance),stamp,self.pad_frame)
        with self.lock:
            global_body = self.alignment.observe(stamp_s,P_B)
            if global_body is not None:
                self.global_pose(self.global_body_pub,global_body,stamp)
                # Stamp is the last measurement used to learn this now-frozen alignment.
                if time.monotonic()-self.last_pad_publish > .2:
                    self.global_pose(self.global_pad_pub,self.alignment.transform,rospy.Time.from_sec(self.alignment.reference_time))
                    self.last_pad_publish = time.monotonic()
            self.ready_pub.publish(self.alignment.ready)

    def display_loop(self):
        while not rospy.is_shutdown():
            start = time.monotonic()
            if self.debug_pub.get_num_connections() or self.jpeg_pub.get_num_connections():
                with self.display_lock:
                    data, self.latest_display = self.latest_display, None
                if data is not None:
                    try:
                        self.publish_display(*data)
                    except (cv2.error, ValueError) as error:
                        rospy.logwarn_throttle(2., 'detected image error: %s', error)
            time.sleep(max(.001, 1/self.debug_hz-(time.monotonic()-start)))

    def publish_display(self,msg,corners,ids,inliers):
        if rospy.get_time()-msg.header.stamp.to_sec() > self.max_age:
            return
        channels = 1 if msg.encoding=='mono8' else 3
        a=np.frombuffer(msg.data,np.uint8).reshape(msg.height,msg.step)[:,:msg.width*channels].reshape(msg.height,msg.width,channels)
        bgr=(cv2.cvtColor(a,cv2.COLOR_RGB2BGR) if msg.encoding=='rgb8' else
             cv2.cvtColor(a,cv2.COLOR_GRAY2BGR) if channels==1 else a.copy())
        bgr, _ = center_crop(bgr, np.eye(3), self.processing_width, self.processing_height)
        bgr = np.ascontiguousarray(bgr)
        for pts,mid in zip(corners,ids):
            polygon=np.round(np.asarray(pts).reshape(4,2)).astype(np.int32)
            color=(60,220,60) if mid in inliers else (0,170,255)
            cv2.polylines(bgr,[polygon],True,color,2)
            cv2.putText(bgr,str(mid),tuple(polygon[0]),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2)
        cv2.putText(bgr,'accepted markers: %d | alignment: %s'%(len(inliers),'ready' if self.alignment.ready else 'learning'),(16,30),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,220,80),2)
        # Display retains the raw image's stamp; estimated poses use the corrected measurement stamp.
        if self.debug_pub.get_num_connections():
            out=Image();out.header=msg.header;out.height=bgr.shape[0];out.width=bgr.shape[1]
            out.encoding='bgr8';out.step=bgr.shape[1]*3;out.data=bgr.tobytes()
            self.debug_pub.publish(out)
        if self.jpeg_pub.get_num_connections():
            ok,jpeg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,self.jpeg_quality])
            if ok:
                out=CompressedImage();out.header=msg.header;out.format='bgr8; jpeg compressed bgr8';out.data=jpeg.tobytes()
                self.jpeg_pub.publish(out)
        self.debug_count += 1

    def publish_status(self,_event):
        with self.lock:
            estimate=self.alignment.estimator.estimate()
            ready=self.alignment.ready
            samples=self.alignment.pair_count
        fresh=self.last_valid_receipt is not None and rospy.get_time()-self.last_valid_receipt<=self.max_age
        if not fresh:
            self.visible_pub.publish(False)
        done=list(self.completed); durations=list(self.durations)
        rate=(len(done)-1)/(done[-1]-done[0]) if len(done)>1 and time.monotonic()-done[-1]<1 else 0.
        status={'alignment_ready':ready,'alignment_mode':'frozen_for_this_session' if ready else 'learning',
                'alignment_pairs':samples,'marker_fresh':fresh,'last_error':self.last_error,
                'processing_width':self.processing_width,'processing_height':self.processing_height,'center_crop':True,
                'images_received':self.received,'images_processed':self.processed,'valid_poses':self.accepted,
                'processing_hz':rate,'processing_ms_mean':float(np.mean(durations)) if durations else None,
                'processing_ms_p95':float(np.percentile(durations,95)) if durations else None,
                'estimate_rate_limit_hz':self.estimate_hz,'detected_image_rate_limit_hz':self.debug_hz,
                'detected_images_published':self.debug_count,'image_time_offset_s':self.time_offset,
                'alignment_quality':None if estimate is None else {k:estimate[k] for k in ('inlier_count','translation_std_m','rotation_std_deg')}}
        self.status_pub.publish(String(data=json.dumps(status)))


if __name__=='__main__':
    PhysicalPadEstimator()
    rospy.spin()
