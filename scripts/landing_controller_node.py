#!/usr/bin/env python3
"""Marker-only horizontal PD and constant-rate vertical landing controller."""

import math
import threading

import rospy
from aruco_landing.yaw_control import yaw_feedback
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistStamped
from std_msgs.msg import Bool, Float64, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse


class LandingController:
    WAITING = "WAITING_FOR_MARKER"
    DESCENDING = "DESCENDING"
    TOUCHDOWN = "TOUCHDOWN"
    ABORTED = "ABORTED_MARKER_LOSS"
    DISABLED = "DISABLED"

    def __init__(self):
        rospy.init_node("landing_controller")
        self.lock = threading.Lock()
        self.rate_hz = float(rospy.get_param("~update_rate_hz", 60.0))
        self.h_min = float(rospy.get_param("~landing_altitude_min_m", 0.20))
        self.touchdown_height_reference = str(
            rospy.get_param("~touchdown_height_reference", "camera")
        ).lower()
        self.horizontal_reference = str(
            rospy.get_param("~horizontal_reference", "camera")
        ).lower()
        self.descent_speed = float(rospy.get_param("~descent_speed_mps", 0.50))
        self.speed_limit = float(
            rospy.get_param("~horizontal_speed_limit_mps", 2.0)
        )
        self.kp = float(rospy.get_param("~kp_xy", 1.0))
        self.kd = float(rospy.get_param("~kd_xy", 0.15))
        self.derivative_filter_tau = float(
            rospy.get_param("~derivative_filter_tau_s", 0.10)
        )
        self.derivative_speed_limit = float(
            rospy.get_param("~derivative_speed_limit_mps", 2.0)
        )
        self.latency_compensation_enabled = bool(
            rospy.get_param("~latency_compensation_enabled", True)
        )
        self.max_pose_prediction = float(
            rospy.get_param("~max_pose_prediction_s", 0.20)
        )
        self.touchdown_stop_lead = float(
            rospy.get_param("~touchdown_stop_lead_s", 0.08)
        )
        self.loss_timeout = float(
            rospy.get_param("~marker_loss_timeout_s", 0.30)
        )
        self.yaw_reference = float(rospy.get_param("~yaw_reference_rad", 0.0))
        self.yaw_enabled = bool(rospy.get_param("~yaw_control_enabled", True))
        self.yaw_kp = float(rospy.get_param("~yaw_kp", 1.0))
        self.yaw_rate_limit = float(rospy.get_param("~yaw_rate_limit_rad_s", .35))
        self.yaw_deadband = math.radians(float(rospy.get_param("~yaw_deadband_deg", 1.0)))
        yaw_feedback([0.,0.,0.,1.], self.yaw_reference, self.yaw_kp,
                     self.yaw_rate_limit, self.yaw_deadband)
        self.enabled = bool(rospy.get_param("~enabled", False))
        # Preview re-evaluates each pose so repeated manual passes do not latch
        # touchdown/abort. Real landing retains its terminal state machine.
        self.preview_only = bool(rospy.get_param("~preview_only", False))
        self.validate_parameters()

        self.pose = None
        self.vehicle_pose = None
        self.camera_pose = None
        self.visible = False
        self.last_valid_receipt = None
        self.previous_error = None
        self.previous_pose_stamp = None
        self.error_derivative = (0.0, 0.0)
        self.state = self.DISABLED if not self.enabled else self.WAITING

        pose_topic = rospy.get_param(
            "~vehicle_pose_topic", "/landing/vehicle_pose_pad"
        )
        visible_topic = rospy.get_param(
            "~target_visible_topic", "/landing/target_visible"
        )
        command_topic = rospy.get_param("~command_topic", "/landing/cmd_vel_pad")
        if self.preview_only and not rospy.resolve_name(command_topic).startswith("/landing/shadow/"):
            raise ValueError("preview_only requires a /landing/shadow/ command topic")
        self.command_publisher = rospy.Publisher(
            command_topic, TwistStamped, queue_size=1
        )
        self.yaw_publisher = rospy.Publisher(
            "/landing/yaw_cmd", Float64, queue_size=1, latch=True
        )
        self.yaw_error_publisher = rospy.Publisher(
            "/landing/yaw_error", Float64, queue_size=1
        )
        self.active_publisher = rospy.Publisher(
            "/landing/controller/active", Bool, queue_size=1, latch=True
        )
        self.abort_publisher = rospy.Publisher(
            "/landing/controller/abort", Bool, queue_size=1, latch=True
        )
        self.touchdown_publisher = rospy.Publisher(
            "/landing/controller/touchdown", Bool, queue_size=1, latch=True
        )
        self.state_publisher = rospy.Publisher(
            "/landing/controller/state", String, queue_size=1, latch=True
        )
        self.pose_subscriber = rospy.Subscriber(
            pose_topic, PoseWithCovarianceStamped, self.pose_callback,
            queue_size=2, tcp_nodelay=True
        )
        camera_pose_topic = rospy.get_param(
            "~camera_pose_topic", "/landing/camera_pose_pad"
        )
        self.camera_pose_subscriber = rospy.Subscriber(
            camera_pose_topic, PoseWithCovarianceStamped, self.camera_pose_callback,
            queue_size=2, tcp_nodelay=True
        )
        self.visible_subscriber = rospy.Subscriber(
            visible_topic, Bool, self.visible_callback, queue_size=2,
            tcp_nodelay=True
        )
        self.reset_service = rospy.Service(
            "~reset", Trigger, self.reset_callback
        )
        self.enable_service = rospy.Service(
            "~enable", SetBool, self.enable_callback
        )
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.update)
        rospy.loginfo(
            "landing controller ready: %.1f Hz, Kp=%.3f Kd=%.3f vd=%.3f, "
            "horizontal=%s, touchdown=%s z<=%.3fm",
            self.rate_hz, self.kp, self.kd, self.descent_speed,
            self.horizontal_reference, self.touchdown_height_reference, self.h_min
        )

    def validate_parameters(self):
        if self.rate_hz <= 0.0 or self.h_min < 0.0:
            raise ValueError("invalid update rate or touchdown altitude")
        if self.descent_speed <= 0.0 or self.speed_limit <= 0.0:
            raise ValueError("descent speed and horizontal speed limit must be positive")
        if self.kp < 0.0 or self.kd < 0.0 or self.loss_timeout <= 0.0:
            raise ValueError("PD gains must be nonnegative and loss timeout positive")
        if self.derivative_filter_tau < 0.0 or self.derivative_speed_limit <= 0.0:
            raise ValueError("invalid derivative filter or speed limit")
        if self.max_pose_prediction < 0.0 or self.touchdown_stop_lead < 0.0:
            raise ValueError("pose prediction and touchdown stop lead must be nonnegative")
        if self.touchdown_height_reference not in ("camera", "vehicle"):
            raise ValueError("touchdown_height_reference must be camera or vehicle")
        if self.horizontal_reference not in ("camera", "vehicle"):
            raise ValueError("horizontal_reference must be camera or vehicle")

    def pose_callback(self, message):
        with self.lock:
            self.vehicle_pose = message
            if self.horizontal_reference == "vehicle":
                self.update_horizontal_pose_locked(message)

    def update_horizontal_pose_locked(self, message):
        position = message.pose.pose.position
        error = (-position.x, -position.y)
        stamp = message.header.stamp.to_sec()
        if (self.previous_error is not None and self.previous_pose_stamp is not None
                and stamp > self.previous_pose_stamp):
            dt = stamp - self.previous_pose_stamp
            if dt <= 0.25:
                raw_x = (error[0] - self.previous_error[0]) / dt
                raw_y = (error[1] - self.previous_error[1]) / dt
                alpha = (
                    1.0 if self.derivative_filter_tau == 0.0
                    else dt / (self.derivative_filter_tau + dt)
                )
                filtered_x = (
                    (1.0 - alpha) * self.error_derivative[0] + alpha * raw_x
                )
                filtered_y = (
                    (1.0 - alpha) * self.error_derivative[1] + alpha * raw_y
                )
                derivative_speed = math.hypot(filtered_x, filtered_y)
                if derivative_speed > self.derivative_speed_limit:
                    scale = self.derivative_speed_limit / derivative_speed
                    filtered_x *= scale
                    filtered_y *= scale
                self.error_derivative = (filtered_x, filtered_y)
        self.previous_error = error
        self.previous_pose_stamp = stamp
        self.pose = message
        self.last_valid_receipt = rospy.get_time()

    def visible_callback(self, message):
        with self.lock:
            self.visible = bool(message.data)

    def camera_pose_callback(self, message):
        with self.lock:
            self.camera_pose = message
            if self.horizontal_reference == "camera":
                self.update_horizontal_pose_locked(message)

    def reset_locked(self):
        self.pose = None
        self.vehicle_pose = None
        self.camera_pose = None
        self.visible = False
        self.last_valid_receipt = None
        self.previous_error = None
        self.previous_pose_stamp = None
        self.error_derivative = (0.0, 0.0)
        self.state = self.WAITING if self.enabled else self.DISABLED

    def reset_callback(self, _request):
        with self.lock:
            self.reset_locked()
        return TriggerResponse(success=True, message="landing controller reset")

    def enable_callback(self, request):
        with self.lock:
            self.enabled = bool(request.data)
            self.reset_locked()
        return SetBoolResponse(success=True, message="enabled" if self.enabled else "disabled")

    def zero_command(self, stamp, frame_id="landing_pad"):
        command = TwistStamped()
        command.header.stamp = stamp
        command.header.frame_id = frame_id
        return command

    def update(self, event):
        now_sec = rospy.get_time()
        with self.lock:
            state = self.state
            pose = self.pose
            vehicle_pose = self.vehicle_pose
            camera_pose = self.camera_pose
            visible = self.visible
            last_valid = self.last_valid_receipt
            derivative = self.error_derivative
            pose_stamp = self.previous_pose_stamp
            enabled = self.enabled

            if self.preview_only:
                state = self.WAITING
                if (enabled and pose is not None and visible and last_valid is not None
                        and now_sec - last_valid <= self.loss_timeout):
                    state = self.DESCENDING
            if not enabled:
                state = self.DISABLED
            elif state == self.WAITING and pose is not None and visible and not self.preview_only:
                # The first valid marker estimate activates both horizontal
                # feedback and vertical descent; there is no alignment gate.
                state = self.DESCENDING
            elif state == self.DESCENDING:
                if last_valid is None or now_sec - last_valid > self.loss_timeout:
                    state = self.ABORTED

            command = self.zero_command(event.current_real)
            if pose is not None:
                command.header.frame_id = pose.header.frame_id

            if state == self.DESCENDING and pose is not None and visible:
                position = pose.pose.pose.position
                error_x, error_y = -position.x, -position.y
                if self.latency_compensation_enabled and pose_stamp is not None:
                    pose_age = max(0.0, min(now_sec - pose_stamp,
                                            self.max_pose_prediction))
                    error_x += derivative[0] * pose_age
                    error_y += derivative[1] * pose_age
                vx = self.kp * error_x + self.kd * derivative[0]
                vy = self.kp * error_y + self.kd * derivative[1]
                speed = math.hypot(vx, vy)
                if speed > self.speed_limit:
                    scale = self.speed_limit / speed
                    vx *= scale
                    vy *= scale
                command.twist.linear.x = vx
                command.twist.linear.y = vy
                height_pose = (
                    camera_pose
                    if self.touchdown_height_reference == "camera"
                    else vehicle_pose
                )
                predicted_height = None
                if height_pose is not None:
                    predicted_height = height_pose.pose.pose.position.z
                    height_stamp = height_pose.header.stamp.to_sec()
                    if self.latency_compensation_enabled and height_stamp > 0.0:
                        vertical_prediction_s = min(
                            max(0.0, now_sec - height_stamp), self.max_pose_prediction
                        ) + self.touchdown_stop_lead
                        predicted_height -= self.descent_speed * vertical_prediction_s
                if predicted_height is not None and predicted_height <= self.h_min:
                    state = self.TOUCHDOWN
                    command = self.zero_command(event.current_real, pose.header.frame_id)
                else:
                    command.twist.linear.z = -self.descent_speed

            yaw_error = float('nan')
            if (self.yaw_enabled and state == self.DESCENDING and vehicle_pose is not None
                    and vehicle_pose.header.frame_id == command.header.frame_id
                    and -.05 <= now_sec-vehicle_pose.header.stamp.to_sec() <= self.loss_timeout):
                q = vehicle_pose.pose.pose.orientation
                try:
                    yaw_error, command.twist.angular.z = yaw_feedback(
                        [q.x,q.y,q.z,q.w], self.yaw_reference, self.yaw_kp,
                        self.yaw_rate_limit, self.yaw_deadband)
                except ValueError as error:
                    rospy.logwarn_throttle(2., 'yaw command withheld: %s', error)
            self.state = state

        self.yaw_error_publisher.publish(Float64(data=yaw_error))
        self.command_publisher.publish(command)
        self.yaw_publisher.publish(Float64(data=self.yaw_reference))
        self.active_publisher.publish(Bool(data=state == self.DESCENDING))
        self.abort_publisher.publish(Bool(data=state == self.ABORTED))
        self.touchdown_publisher.publish(Bool(data=state == self.TOUCHDOWN))
        self.state_publisher.publish(String(data=state))


if __name__ == "__main__":
    LandingController()
    rospy.spin()
